"""
Tests for live inference system.

Tests cover:
- Engine: model loading, prediction, parity with isolated_inference
- Feed: buffer format, CSV compatibility
- Tracker: TP/SL/timeout logic, stats, pending trade handling
- Dashboard: HTTP endpoints, data serialization
"""

import sys
import os
import json
import csv
import urllib.request
import time
import logging

import numpy as np
import pandas as pd
import pytest
from conftest import TEST_CSV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.utils.seed import set_seed

logging.getLogger('daikoku').setLevel(logging.WARNING)


# ============================================================
# Tracker tests (no network/GPU required)
# ============================================================

class TestTradeTracker:
    """Test tracker logic with synthetic data."""

    def _make_buffer(self, n=100):
        """Create synthetic OHLCV DataFrame for ATR computation."""
        set_seed(config.SEED)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=n, freq='15min'),
            'Open': prices,
            'High': prices + np.random.uniform(1, 3, n),
            'Low': prices - np.random.uniform(1, 3, n),
            'Close': prices + np.random.uniform(-1, 1, n),
            'Volume': np.random.uniform(100, 1000, n),
        })
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        return df

    def _make_pred(self, direction, confidence=0.8):
        """Create a prediction dict (3-class: 0=Bear, 1=Uncertain, 2=Bull)."""
        probs = np.zeros(3)
        probs[direction] = confidence
        # Spread remaining probability across other classes
        remaining = 1 - confidence
        for i in range(3):
            if i != direction:
                probs[i] = remaining / 2
        names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
        pred = {
            'pred': direction,
            'confidence': confidence,
            'probs': probs,
            'class_name': names[direction],
            'signal_ts': '2024-01-01 00:00',
        }
        return pred

    def test_bull_tp(self):
        """Bull trade hits take profit."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        assert trade is not None
        assert trade['direction'] == 'long'

        # Hit TP
        candle_tp = {'high': trade['tp_level'] + 1, 'low': 99.0, 'close': trade['tp_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_tp)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'tp'

    def test_bull_sl(self):
        """Bull trade hits stop loss."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)

        candle_sl = {'high': 100.5, 'low': trade['sl_level'] - 1, 'close': trade['sl_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_sl)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'sl'

    def test_bear_tp(self):
        """Bear trade hits take profit."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(0))
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        assert trade['direction'] == 'short'

        # Bear TP = entry - ATR*mult_tp (price goes down)
        candle_tp = {'high': 100.5, 'low': trade['tp_level'] - 1, 'close': trade['tp_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_tp)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'tp'

    def test_bear_sl(self):
        """Bear trade hits stop loss."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(0))
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)

        # Bear SL = entry + ATR*mult_sl (price goes up)
        candle_sl = {'high': trade['sl_level'] + 1, 'low': 99.0, 'close': trade['sl_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_sl)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'sl'

    def test_timeout(self):
        """Trade times out after max_horizon candles."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=3)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        neutral = {'high': 101.0, 'low': 99.0, 'close': 100.0,
                   'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        for i in range(3):
            closed = tracker.update(neutral)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'timeout'
        assert closed[0]['hit_candle'] == 3

    def test_intra_candle_no_elapsed_no_timeout(self):
        """intra_candle=True checks TP/SL but does not increment elapsed or trigger timeout."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=3)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        neutral = {'high': 101.0, 'low': 99.0, 'close': 100.0,
                   'timestamp': '2024-01-01 00:20', 'volume': 500.0}

        # 50 intra-candle updates should NOT cause timeout (max_horizon=3)
        for _ in range(50):
            closed = tracker.update(neutral, intra_candle=True)
            assert len(closed) == 0

        assert tracker.open_trades[0]['candles_elapsed'] == 0

    def test_intra_candle_detects_sl(self):
        """intra_candle=True still detects SL hit."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)

        candle_sl = {'high': 100.5, 'low': trade['sl_level'] - 1, 'close': trade['sl_level'],
                     'timestamp': '2024-01-01 00:20', 'volume': 500.0}
        closed = tracker.update(candle_sl, intra_candle=True)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'sl'

    def test_uncertain_pred_rejected(self):
        """Uncertain prediction (pred==1) is rejected by tracker."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=5)
        tracker.set_pending(self._make_pred(2))  # Bull = accepted
        assert tracker.pending_trade is not None
        tracker.set_pending(self._make_pred(1))  # Uncertain = rejected
        assert tracker.pending_trade is None

    def test_uncertain_pred_not_stored(self):
        """Uncertain prediction is never stored as pending."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=5)
        tracker.set_pending(self._make_pred(1))  # Uncertain
        assert tracker.pending_trade is None

    def test_stats_empty(self):
        """Stats on empty tracker."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=5)
        stats = tracker.stats
        assert stats['total'] == 0
        assert stats['winrate'] == 0.0

    def test_stats_after_trades(self):
        """Stats after multiple trades."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()

        # Open and close a Bull TP
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        candle_tp = {'high': trade['tp_level'] + 1, 'low': 99.0, 'close': trade['tp_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        tracker.update(candle_tp)

        # Open and close a Bear SL
        tracker.set_pending(self._make_pred(0))
        trade2 = tracker.open_from_pending(candle, df)
        candle_sl = {'high': trade2['sl_level'] + 1, 'low': 99.0, 'close': trade2['sl_level'],
                     'timestamp': '2024-01-01 00:45', 'volume': 500.0}
        tracker.update(candle_sl)

        stats = tracker.stats
        assert stats['total'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert stats['winrate'] == 50.0
        assert stats['bull_total'] == 1
        assert stats['bear_total'] == 1

    def test_csv_export(self):
        """CSV export produces correct fields."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()
        tracker.set_pending(self._make_pred(2))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        candle_tp = {'high': trade['tp_level'] + 1, 'low': 99.0, 'close': trade['tp_level'],
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        tracker.update(candle_tp)

        rows = tracker.trades_to_csv_rows()
        assert len(rows) == 1
        expected_keys = {'signal_ts', 'pair', 'direction', 'entry_price', 'tp_level',
                         'sl_level', 'atr', 'close_ts', 'outcome', 'hit_candle', 'confidence'}
        assert set(rows[0].keys()) == expected_keys

    def test_multiple_open_trades(self):
        """Multiple trades can be open simultaneously."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()

        # Open first trade
        tracker.set_pending(self._make_pred(2, 0.9))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        # Open second trade (different signal)
        tracker.set_pending(self._make_pred(0, 0.7))
        candle2 = {'open': 101.0, 'high': 103.0, 'low': 100.0, 'close': 102.0,
                   'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        tracker.open_from_pending(candle2, df)

        assert len(tracker.open_trades) == 2
        assert tracker.stats['open'] == 2


# ============================================================
# Mode-specific tracker tests
# ============================================================

class TestTradeTrackerModes:
    """Test tracker behavior for different prediction_target modes."""

    def _make_buffer(self, n=100):
        """Create synthetic OHLCV DataFrame for ATR computation."""
        set_seed(config.SEED)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=n, freq='15min'),
            'Open': prices,
            'High': prices + np.random.uniform(1, 3, n),
            'Low': prices - np.random.uniform(1, 3, n),
            'Close': prices + np.random.uniform(-1, 1, n),
            'Volume': np.random.uniform(100, 1000, n),
        })
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        return df

    def _make_pred_binary(self, direction, confidence=0.8):
        """Create a binary prediction dict (2-class: 0=negative, 1=positive)."""
        probs = np.zeros(2)
        probs[direction] = confidence
        probs[1 - direction] = 1 - confidence
        names = {0: 'Class0', 1: 'Class1'}
        return {
            'pred': direction,
            'confidence': confidence,
            'probs': probs,
            'class_name': names[direction],
            'signal_ts': '2024-01-01 00:00',
        }

    def test_bear_mode_pred1_opens_short(self):
        """In bear mode, pred=1 (Bear) should open a SHORT trade."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=0,
            prediction_target='bear')
        df = self._make_buffer()

        pred = self._make_pred_binary(1, 0.8)
        pred['class_name'] = 'Bear'
        tracker.set_pending(pred)
        assert tracker.pending_trade is not None

        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        assert trade is not None
        assert trade['direction'] == 'short'
        # Short TP is below entry
        assert trade['tp_level'] < trade['entry_price']
        # Short SL is above entry
        assert trade['sl_level'] > trade['entry_price']

    def test_bear_mode_pred0_rejected(self):
        """In bear mode, pred=0 (Not-Bear) should be rejected."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=0,
            prediction_target='bear')

        pred = self._make_pred_binary(0, 0.8)
        pred['class_name'] = 'Not-Bear'
        tracker.set_pending(pred)
        assert tracker.pending_trade is None

    def test_bull_mode_pred1_opens_long(self):
        """In bull mode, pred=1 (Bull) should open a LONG trade."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=0,
            prediction_target='bull')
        df = self._make_buffer()

        pred = self._make_pred_binary(1, 0.8)
        pred['class_name'] = 'Bull'
        tracker.set_pending(pred)
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        assert trade is not None
        assert trade['direction'] == 'long'
        assert trade['tp_level'] > trade['entry_price']
        assert trade['sl_level'] < trade['entry_price']

    def test_uncertain_mode_all_rejected(self):
        """In uncertain mode, all predictions should be rejected (no trades)."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=1,
            prediction_target='uncertain')

        # pred=0 (Directional) has no trade direction → rejected
        pred0 = self._make_pred_binary(0, 0.8)
        pred0['class_name'] = 'Directional'
        tracker.set_pending(pred0)
        assert tracker.pending_trade is None

        # pred=1 (Uncertain) is reject_class → rejected
        pred1 = self._make_pred_binary(1, 0.8)
        pred1['class_name'] = 'Uncertain'
        tracker.set_pending(pred1)
        assert tracker.pending_trade is None

    def test_close1_mode_tp_after_1_candle(self):
        """In close1 mode, trade closes after 1 candle based on direction."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=None,
            prediction_target='close1')
        df = self._make_buffer()

        # pred=1 (Up) → long
        pred = self._make_pred_binary(1, 0.8)
        pred['class_name'] = 'Up'
        tracker.set_pending(pred)
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        assert trade is not None
        assert trade['direction'] == 'long'
        assert trade['tp_level'] == 0.0  # No TP/SL in closeN mode
        assert trade['sl_level'] == 0.0

        # Price goes up → tp
        candle_up = {'open': 100.0, 'high': 102.0, 'low': 99.5, 'close': 101.0,
                     'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_up)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'tp'
        assert closed[0]['hit_candle'] == 1

    def test_close1_mode_sl_when_wrong_direction(self):
        """In close1 mode, trade returns 'sl' if price goes wrong way."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=None,
            prediction_target='close1')
        df = self._make_buffer()

        # pred=1 (Up) → long, but price goes down
        pred = self._make_pred_binary(1, 0.8)
        pred['class_name'] = 'Up'
        tracker.set_pending(pred)
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        candle_down = {'open': 100.0, 'high': 100.5, 'low': 98.0, 'close': 99.0,
                       'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(candle_down)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'sl'

    def test_close2_mode_closes_after_2_candles(self):
        """In close2 mode, trade stays open for 2 candles before closing."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=None,
            prediction_target='close2')
        df = self._make_buffer()

        pred = self._make_pred_binary(0, 0.8)
        pred['class_name'] = 'Down'
        tracker.set_pending(pred)
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        # Candle 1: not yet
        c1 = {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 99.5,
              'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        closed = tracker.update(c1)
        assert len(closed) == 0

        # Candle 2: close! Short + price went down → tp
        c2 = {'open': 99.5, 'high': 100.0, 'low': 98.0, 'close': 98.5,
              'timestamp': '2024-01-01 00:45', 'volume': 500.0}
        closed = tracker.update(c2)
        assert len(closed) == 1
        assert closed[0]['outcome'] == 'tp'
        assert closed[0]['hit_candle'] == 2

    def test_close1_no_intra_candle_close(self):
        """In closeN mode, intra_candle=True should NOT trigger close."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=None,
            prediction_target='close1')
        df = self._make_buffer()

        pred = self._make_pred_binary(1, 0.8)
        pred['class_name'] = 'Up'
        tracker.set_pending(pred)
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        # Many intra-candle updates should not close
        for _ in range(20):
            closed = tracker.update(
                {'open': 101.0, 'high': 103.0, 'low': 99.0, 'close': 102.0,
                 'timestamp': '2024-01-01 00:20', 'volume': 500.0},
                intra_candle=True)
            assert len(closed) == 0

    def test_close1_both_directions_trade(self):
        """In close1 mode, both pred=0 (Down/short) and pred=1 (Up/long) generate trades."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(
            atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1,
            max_horizon=10, num_classes=2, reject_class=None,
            prediction_target='close1')
        df = self._make_buffer()

        # pred=0 → short
        pred0 = self._make_pred_binary(0, 0.8)
        pred0['class_name'] = 'Down'
        tracker.set_pending(pred0)
        assert tracker.pending_trade is not None
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade0 = tracker.open_from_pending(candle, df)
        assert trade0['direction'] == 'short'

        # Close it
        c1 = {'open': 100.0, 'high': 100.5, 'low': 98.0, 'close': 99.0,
              'timestamp': '2024-01-01 00:30', 'volume': 500.0}
        tracker.update(c1)

        # pred=1 → long
        pred1 = self._make_pred_binary(1, 0.8)
        pred1['class_name'] = 'Up'
        tracker.set_pending(pred1)
        assert tracker.pending_trade is not None
        candle2 = {'open': 99.0, 'high': 101.0, 'low': 98.5, 'close': 100.0,
                   'timestamp': '2024-01-01 00:45', 'volume': 500.0}
        trade1 = tracker.open_from_pending(candle2, df)
        assert trade1['direction'] == 'long'


# ============================================================
# Dashboard tests (no GPU required)
# ============================================================

class TestDashboard:
    """Test dashboard HTTP serving and data serialization."""

    def test_html_served(self):
        """Dashboard serves valid HTML at /."""
        from modules.inference.dashboard import LiveDashboard
        d = LiveDashboard(17780, 'TEST/USDT', '5m')
        d.start()
        try:
            resp = urllib.request.urlopen('http://localhost:17780/')
            html = resp.read().decode()
            assert 'lightweight-charts' in html
            assert 'Live Inference' in html
            assert 'data.json' in html
        finally:
            d.stop()

    def test_json_empty_state(self):
        """Dashboard returns valid JSON with empty state."""
        from modules.inference.dashboard import LiveDashboard
        d = LiveDashboard(17781, 'TEST/USDT', '5m')
        d.start()
        try:
            resp = urllib.request.urlopen('http://localhost:17781/data.json')
            data = json.loads(resp.read())
            assert data['pair'] == 'TEST/USDT'
            assert data['timeframe'] == '5m'
            assert isinstance(data['candles'], list)
            assert isinstance(data['markers'], list)
        finally:
            d.stop()

    def test_json_with_data(self):
        """Dashboard serializes candles and markers correctly."""
        from modules.inference.dashboard import LiveDashboard
        d = LiveDashboard(17782, 'BTC/USDT', '15m')
        d.start()
        try:
            candles = pd.DataFrame({
                'Open time': pd.date_range('2024-01-01', periods=3, freq='15min'),
                'Open': [100.0, 101.0, 102.0],
                'High': [102.0, 103.0, 104.0],
                'Low': [99.0, 100.0, 101.0],
                'Close': [101.0, 102.0, 103.0],
            })
            preds = [
                {'timestamp': pd.Timestamp('2024-01-01 00:15'), 'pred': 2, 'confidence': 0.75, 'class_name': 'Bull'},
                {'timestamp': pd.Timestamp('2024-01-01 00:30'), 'pred': 0, 'confidence': 0.65, 'class_name': 'Bear'},
            ]
            d.update(candles, preds, [], [], {'total': 0, 'wins': 0, 'losses': 0, 'timeouts': 0, 'winrate': 0, 'open': 0})

            resp = urllib.request.urlopen('http://localhost:17782/data.json')
            data = json.loads(resp.read())
            assert len(data['candles']) == 3
            assert len(data['markers']) == 2
            # Bull marker
            assert data['markers'][0]['shape'] == 'arrowUp'
            assert data['markers'][0]['color'] == '#26a69a'
            # Bear marker
            assert data['markers'][1]['shape'] == 'arrowDown'
            assert data['markers'][1]['color'] == '#ef5350'
        finally:
            d.stop()

    def test_bull_marker_is_directional(self):
        """Bull predictions create directional markers (green arrows)."""
        from modules.inference.dashboard import LiveDashboard
        d = LiveDashboard(17783, 'BTC/USDT', '15m')
        d.start()
        try:
            candles = pd.DataFrame({
                'Open time': pd.date_range('2024-01-01', periods=2, freq='15min'),
                'Open': [100.0, 101.0], 'High': [102.0, 103.0],
                'Low': [99.0, 100.0], 'Close': [101.0, 102.0],
            })
            preds = [{'timestamp': pd.Timestamp('2024-01-01'), 'pred': 2, 'confidence': 0.75, 'class_name': 'Bull'}]
            d.update(candles, preds, [], [], {'total': 0, 'wins': 0, 'losses': 0, 'timeouts': 0, 'winrate': 0, 'open': 0})

            resp = urllib.request.urlopen('http://localhost:17783/data.json')
            data = json.loads(resp.read())
            assert len(data['markers']) == 1
            assert data['markers'][0]['directional'] == True
            assert data['markers'][0]['color'] == '#26a69a'
        finally:
            d.stop()


# ============================================================
# Feed tests (requires network)
# ============================================================

class TestFeedReadCsv:
    """Test CandleFeed.read_csv static method."""

    def test_read_csv_test_dataset(self):
        """read_csv loads test_Dataset.csv with correct dtypes."""
        from modules.inference.feed import CandleFeed

        test_csv = TEST_CSV
        if not os.path.exists(test_csv):
            pytest.skip(f"Test dataset not found: {test_csv}")

        df = CandleFeed.read_csv(test_csv)

        assert 'Open time' in df.columns
        assert str(df['Open time'].dtype).startswith('datetime64')
        assert df['Close'].dtype == np.float64
        assert df['Volume'].dtype == np.float64
        assert len(df) > 0

    def test_read_csv_roundtrip(self, tmp_path):
        """Save a buffer then read it back with read_csv: dtypes match."""
        from modules.inference.feed import CandleFeed

        df_orig = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'Open': np.array([100, 101, 102, 103, 104], dtype=np.float64),
            'High': np.array([102, 103, 104, 105, 106], dtype=np.float64),
            'Low': np.array([98, 99, 100, 101, 102], dtype=np.float64),
            'Close': np.array([101, 102, 103, 104, 105], dtype=np.float64),
            'Volume': np.array([1000, 1100, 1200, 1300, 1400], dtype=np.float64),
        })

        csv_path = str(tmp_path / 'buffer.csv')
        df_save = df_orig.copy()
        df_save['Open time'] = df_save['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_save.to_csv(csv_path, index=False)

        df_loaded = CandleFeed.read_csv(csv_path)

        assert len(df_loaded) == 5
        assert df_loaded['Close'].dtype == np.float64
        np.testing.assert_array_almost_equal(
            df_loaded['Close'].values, df_orig['Close'].values, decimal=2
        )


class TestFeed:
    """Test feed utility functions (no network for some)."""

    def test_timeframe_to_ms(self):
        from modules.inference.feed import timeframe_to_ms
        assert timeframe_to_ms('15m') == 15 * 60 * 1000
        assert timeframe_to_ms('1h') == 60 * 60 * 1000
        assert timeframe_to_ms('4h') == 4 * 60 * 60 * 1000

    def test_timeframe_to_seconds(self):
        from modules.inference.feed import timeframe_to_seconds
        assert timeframe_to_seconds('15m') == 900
        assert timeframe_to_seconds('1h') == 3600

    def test_invalid_timeframe(self):
        from modules.inference.feed import timeframe_to_ms
        with pytest.raises(ValueError):
            timeframe_to_ms('2h')

    def test_invalid_pair_format(self):
        from modules.inference.feed import _find_symbol, _instantiate_exchange
        exchange = _instantiate_exchange('binance')
        with pytest.raises(ValueError):
            _find_symbol(exchange, 'BTCUSDT')  # Missing /

    def test_unsupported_exchange(self):
        from modules.inference.feed import _instantiate_exchange
        with pytest.raises(ValueError):
            _instantiate_exchange('nonexistent')


# ============================================================
# Engine tests (basic, no GPU needed for import check)
# ============================================================

class TestEngineImports:
    """Test engine module structure."""

    def test_class_names(self):
        from modules.data.labeling import TARGET_CLASS_NAMES
        triple = TARGET_CLASS_NAMES["triple"]
        assert triple[0] == 'Bear'
        assert triple[1] == 'Uncertain'
        assert triple[2] == 'Bull'
        assert len(triple) == 3

    def test_engine_class_exists(self):
        from modules.inference.engine import PredictionEngine
        assert callable(PredictionEngine)

    def test_load_model_exists(self):
        from modules.inference.engine import load_model
        assert callable(load_model)


# ============================================================
# Config tests
# ============================================================

class TestLiveConfig:
    """Test LIVE_* config parameters."""

    def test_live_params_exist(self):
        import config
        assert hasattr(config, 'LIVE_EXCHANGE')
        assert hasattr(config, 'LIVE_SYMBOL')
        assert hasattr(config, 'LIVE_TIMEFRAME')
        assert hasattr(config, 'LIVE_BUFFER_SIZE')
        assert hasattr(config, 'LIVE_CHECKPOINT')
        assert hasattr(config, 'LIVE_HTTP_PORT')
        assert hasattr(config, 'LIVE_RETRY_TIMEOUT')
        assert hasattr(config, 'LIVE_RETRY_MAX')
        assert hasattr(config, 'LIVE_COMPARE_EVERY')
        assert hasattr(config, 'LIVE_OUTPUT_DIR')

    def test_live_params_types(self):
        import config
        assert isinstance(config.LIVE_EXCHANGE, str)
        assert isinstance(config.LIVE_SYMBOL, str)
        assert isinstance(config.LIVE_TIMEFRAME, str)
        assert isinstance(config.LIVE_BUFFER_SIZE, int)
        assert isinstance(config.LIVE_HTTP_PORT, int)
        assert isinstance(config.LIVE_RETRY_MAX, int)

    def test_live_buffer_minimum(self):
        """Buffer size must be large enough for dual-TF."""
        import config
        # Minimum: 850 (secondary 800 + warmup)
        assert config.LIVE_BUFFER_SIZE >= 850


# ============================================================
# Timestamp & CSV consistency tests
# ============================================================

class TestInferenceHelpers:
    """Test utility functions from inference.py."""

    def test_pair_tag(self):
        """_pair_tag converts 'BTC/USDT' to 'BTCUSDT'."""
        from inference import _pair_tag
        assert _pair_tag('BTC/USDT') == 'BTCUSDT'
        assert _pair_tag('ETH/USDT') == 'ETHUSDT'

    def test_force_csv_creates_file(self, tmp_path):
        """_force_csv creates a CSV with header only."""
        from inference import _force_csv
        csv_path = str(tmp_path / 'test.csv')
        fields = ['a', 'b', 'c']
        _force_csv(csv_path, fields)

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert 'a,b,c' in lines[0]

    def test_force_csv_overwrites(self, tmp_path):
        """_force_csv overwrites existing content."""
        from inference import _force_csv
        csv_path = str(tmp_path / 'test.csv')
        fields = ['x']

        # Write some data first
        with open(csv_path, 'w') as f:
            f.write('old,data\n1,2\n3,4\n')

        _force_csv(csv_path, fields)

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert 'x' in lines[0]

    def test_ensure_csv_creates_if_missing(self, tmp_path):
        """_ensure_csv creates file if it doesn't exist."""
        from inference import _ensure_csv
        csv_path = str(tmp_path / 'new.csv')
        _ensure_csv(csv_path, ['col1', 'col2'])
        assert os.path.exists(csv_path)

    def test_ensure_csv_keeps_existing(self, tmp_path):
        """_ensure_csv does not overwrite existing file."""
        from inference import _ensure_csv
        csv_path = str(tmp_path / 'existing.csv')

        with open(csv_path, 'w') as f:
            f.write('a,b\n1,2\n')

        _ensure_csv(csv_path, ['x', 'y'])

        with open(csv_path) as f:
            content = f.read()
        assert 'a,b' in content  # Original header preserved

    def test_append_csv(self, tmp_path):
        """_append_csv adds a single row to existing CSV."""
        from inference import _force_csv, _append_csv
        csv_path = str(tmp_path / 'test.csv')
        fields = ['name', 'value']
        _force_csv(csv_path, fields)

        _append_csv(csv_path, {'name': 'foo', 'value': '42'}, fields)
        _append_csv(csv_path, {'name': 'bar', 'value': '99'}, fields)

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows

    def test_load_trades_nonexistent(self):
        """_load_trades returns 0 for non-existent file."""
        from inference import _load_trades
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        count = _load_trades('/nonexistent/trades.csv', tracker)
        assert count == 0

    def test_parse_args_defaults(self):
        """parse_args returns correct defaults."""
        import config as cfg
        from inference import parse_args
        from unittest.mock import patch
        with patch('sys.argv', ['inference.py']):
            args = parse_args()
        assert args.pair == cfg.LIVE_SYMBOL
        assert args.timeframe == cfg.LIVE_TIMEFRAME
        assert args.checkpoint == cfg.LIVE_CHECKPOINT
        assert args.fresh is False

    def test_parse_args_custom(self):
        """parse_args parses custom arguments."""
        from inference import parse_args
        from unittest.mock import patch
        with patch('sys.argv', ['inference.py', '--pair', 'ETH/USDT', '--fresh']):
            args = parse_args()
        assert args.pair == 'ETH/USDT'
        assert args.fresh is True


class TestTimestampConsistency:
    """Test _fmt_ts, _load_predictions, buffer append-only logic."""

    def test_fmt_ts_string(self):
        """_fmt_ts formats string timestamp."""
        from inference import _fmt_ts
        assert _fmt_ts('2026-03-05 09:30:00') == '2026-03-05 09:30:00'

    def test_fmt_ts_pd_timestamp(self):
        """_fmt_ts formats pd.Timestamp."""
        from inference import _fmt_ts
        ts = pd.Timestamp('2026-03-05 10:00:00')
        assert _fmt_ts(ts) == '2026-03-05 10:00:00'

    def test_load_predictions(self, tmp_path):
        """_load_predictions loads naive UTC timestamps."""
        from inference import _load_predictions
        csv_path = tmp_path / 'pred.csv'
        csv_path.write_text(
            'timestamp,pair,open,high,low,close,volume,pred_class,confidence,prob_bear,prob_uncertain,prob_bull\n'
            '2026-03-05 09:30:00,BTC/USDT,100,102,99,101,500,2,0.6253,0.1747,0.2000,0.6253\n'
            '2026-03-05 09:45:00,BTC/USDT,101,103,100,102,600,0,0.5811,0.5811,0.2000,0.2189\n'
        )
        class_names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
        preds = _load_predictions(str(csv_path), class_names)
        assert len(preds) == 2
        assert preds[0]['pred'] == 2  # Bull
        assert preds[1]['pred'] == 0  # Bear
        assert abs(preds[0]['confidence'] - 0.6253) < 1e-4

    def test_load_predictions_nonexistent_file(self):
        """_load_predictions returns empty list for missing file."""
        from inference import _load_predictions
        class_names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
        assert _load_predictions('/nonexistent/path.csv', class_names) == []

    def test_append_csv_fmt_ts_integration(self, tmp_path):
        """CSV written with _fmt_ts produces naive UTC format."""
        from inference import _fmt_ts, _append_csv, _ensure_csv
        csv_path = str(tmp_path / 'test.csv')
        fields = ['timestamp', 'value']
        _ensure_csv(csv_path, fields)

        _append_csv(csv_path, {'timestamp': _fmt_ts('2026-03-05 10:00:00'), 'value': '42'}, fields)
        _append_csv(csv_path, {'timestamp': _fmt_ts('2026-03-05 10:15:00'), 'value': '43'}, fields)

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert '2026-03-05 10:00:00' in lines[1]
        assert '+' not in lines[1]


class TestBufferAppendOnly:
    """Test feed save_buffer/load_buffer append-only behavior."""

    def _make_feed_buffer(self, n=10, start='2024-01-01'):
        """Create a synthetic buffer DataFrame."""
        df = pd.DataFrame({
            'Open time': pd.date_range(start, periods=n, freq='15min'),
            'Open': np.linspace(100, 110, n).astype(np.float64),
            'High': np.linspace(102, 112, n).astype(np.float64),
            'Low': np.linspace(98, 108, n).astype(np.float64),
            'Close': np.linspace(101, 111, n).astype(np.float64),
            'Volume': np.full(n, 500.0, dtype=np.float64),
        })
        return df

    def test_save_buffer_first_write(self, tmp_path):
        """First save writes full buffer with header."""
        from modules.inference.feed import CandleFeed
        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer = self._make_feed_buffer(5)
        feed._last_saved_ts = None
        csv_path = str(tmp_path / 'buffer.csv')

        feed.save_buffer(csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 5
        assert list(df.columns) == ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Check naive UTC format
        assert '+' not in df['Open time'].iloc[0]

    def test_save_buffer_append_only(self, tmp_path):
        """Second save only appends new candles."""
        from modules.inference.feed import CandleFeed
        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer = self._make_feed_buffer(5)
        feed._last_saved_ts = None
        csv_path = str(tmp_path / 'buffer.csv')

        # First save: 5 candles
        feed.save_buffer(csv_path)

        # Add 3 more candles to buffer
        new_candles = self._make_feed_buffer(3, start='2024-01-01 01:15')
        feed.buffer = pd.concat([feed.buffer, new_candles], ignore_index=True)

        # Second save: should append only 3
        feed.save_buffer(csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 8  # 5 + 3

    def test_save_buffer_no_duplicate(self, tmp_path):
        """Save with no new candles doesn't duplicate."""
        from modules.inference.feed import CandleFeed
        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer = self._make_feed_buffer(5)
        feed._last_saved_ts = None
        csv_path = str(tmp_path / 'buffer.csv')

        feed.save_buffer(csv_path)
        # Save again without changes
        feed.save_buffer(csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == 5

    def test_load_buffer_trims_to_buffer_size(self, tmp_path):
        """load_buffer keeps only buffer_size candles in memory."""
        from modules.inference.feed import CandleFeed
        # First, create a CSV with 20 candles
        df = self._make_feed_buffer(20)
        csv_path = str(tmp_path / 'buffer.csv')
        df_out = df.copy()
        df_out['Open time'] = df_out['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_out.to_csv(csv_path, index=False)

        # Create feed with buffer_size=10
        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer_size = 10
        feed._last_saved_ts = None

        count = feed.load_buffer(csv_path)
        assert count == 10
        assert len(feed.buffer) == 10
        # Should have the last 10 candles
        assert feed._last_saved_ts is not None

    def test_load_buffer_small_csv(self, tmp_path):
        """load_buffer with CSV smaller than buffer_size keeps all."""
        from modules.inference.feed import CandleFeed
        df = self._make_feed_buffer(5)
        csv_path = str(tmp_path / 'buffer.csv')
        df_out = df.copy()
        df_out['Open time'] = df_out['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_out.to_csv(csv_path, index=False)

        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer_size = 100
        feed._last_saved_ts = None

        count = feed.load_buffer(csv_path)
        assert count == 5
        assert len(feed.buffer) == 5

    def test_save_then_load_roundtrip(self, tmp_path):
        """Save buffer then load it back: data integrity check."""
        from modules.inference.feed import CandleFeed
        feed = CandleFeed.__new__(CandleFeed)
        feed.buffer = self._make_feed_buffer(10)
        feed._last_saved_ts = None
        csv_path = str(tmp_path / 'buffer.csv')

        feed.save_buffer(csv_path)

        # Load back
        feed2 = CandleFeed.__new__(CandleFeed)
        feed2.buffer_size = 100
        feed2._last_saved_ts = None
        feed2.load_buffer(csv_path)

        assert len(feed2.buffer) == 10
        # Values should match (Close column)
        np.testing.assert_array_almost_equal(
            feed.buffer['Close'].values,
            feed2.buffer['Close'].values,
            decimal=2,
        )


# ============================================================
# Live simulation test (requires GPU + real data + model)
# ============================================================

class TestLiveSimulation:
    """Simulate live candle-by-candle inference, verify reproducibility via _run_comparison."""

    def test_live_determinism_10_candles(self, tmp_path, monkeypatch):
        """
        Simulate 10 candles arriving one by one.
        Each candle shifts the buffer by 1 and triggers a prediction.
        After each prediction, _run_comparison re-infers and asserts 0 divergences.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip('CUDA not available')
        except ImportError:
            pytest.skip('torch not installed')

        if not os.path.exists('models/latest.pt'):
            pytest.skip('Model models/latest.pt not found')
        if not os.path.exists('live/buffer_BTCUSDT_15m.csv'):
            pytest.skip('Buffer live/buffer_BTCUSDT_15m.csv not found')

        import config
        from modules.inference.engine import PredictionEngine
        from modules.data.loader import get_raw_data
        from inference import _run_comparison, _fmt_ts

        BUFFER_SIZE = 1800
        monkeypatch.setattr(config, 'LIVE_BUFFER_SIZE', BUFFER_SIZE)

        df_full = get_raw_data('live/buffer_BTCUSDT_15m.csv')
        engine = PredictionEngine('models/latest.pt', device='auto')

        N_PREDS = 10
        assert len(df_full) >= BUFFER_SIZE + N_PREDS, (
            f"Not enough data: {len(df_full)} < {BUFFER_SIZE + N_PREDS}"
        )

        buffer_csv = str(tmp_path / 'buffer.csv')

        for i in range(N_PREDS):
            # Sliding window: 1 new candle each iteration
            end = BUFFER_SIZE + i + 1
            start = end - BUFFER_SIZE
            df_buffer = df_full.iloc[start:end].copy().reset_index(drop=True)

            result = engine.predict(df_buffer)
            assert result is not None, f"Prediction {i+1}/{N_PREDS} returned None"

            ts = df_full.iloc[end - 1]['Open time']
            pred_log = [{
                'timestamp': _fmt_ts(ts),
                'pred': result['pred'],
                'confidence': result['confidence'],
                'probs': result['probs'],
                'class_name': result['class_name'],
            }]

            # Save CSV (append-only like live: all rows up to current position)
            df_save = df_full.iloc[:end].copy()
            df_save['Open time'] = df_save['Open time'].apply(_fmt_ts)
            df_save.to_csv(buffer_csv, index=False)

            # Compare this ONE prediction
            compare_state = {'total_ok': 0, 'total_errors': 0, 'error_details': []}
            alerts = []
            _run_comparison(buffer_csv, pred_log, engine, alerts, compare_state,
                            buffer_size=BUFFER_SIZE, num_classes=engine.num_classes,
                            class_names=engine.class_names)

            assert compare_state['total_errors'] == 0, (
                f"Prediction {i+1}: {compare_state['error_details']}"
            )
            assert compare_state['total_ok'] == 1


    def test_comparison_detects_corrupted_predictions(self, tmp_path, monkeypatch):
        """
        Simulate 3 candles one by one, corrupt each prediction differently.
        _run_comparison must detect each corrupted candle as error with details.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip('CUDA not available')
        except ImportError:
            pytest.skip('torch not installed')

        if not os.path.exists('models/latest.pt'):
            pytest.skip('Model models/latest.pt not found')
        if not os.path.exists('live/buffer_BTCUSDT_15m.csv'):
            pytest.skip('Buffer live/buffer_BTCUSDT_15m.csv not found')

        import config
        from modules.inference.engine import PredictionEngine
        from modules.data.loader import get_raw_data
        from inference import _run_comparison, _fmt_ts

        BUFFER_SIZE = 1800
        monkeypatch.setattr(config, 'LIVE_BUFFER_SIZE', BUFFER_SIZE)

        df_full = get_raw_data('live/buffer_BTCUSDT_15m.csv')
        engine = PredictionEngine('models/latest.pt', device='auto')

        N_PREDS = 3
        assert len(df_full) >= BUFFER_SIZE + N_PREDS

        # Corruption functions: one per prediction
        corruptions = [
            lambda p: p.__setitem__('pred', (p['pred'] + 1) % 3),        # flip class
            lambda p: p.__setitem__('confidence', p['confidence'] + 0.1), # alter confidence
            lambda p: p['probs'].__setitem__(0, p['probs'][0] + 0.05),   # alter proba
        ]
        expected_errors = ['Error.*Class', 'Error.*Conf', 'Error.*Proba']

        buffer_csv = str(tmp_path / 'buffer.csv')
        all_alerts = []

        for i in range(N_PREDS):
            end = BUFFER_SIZE + i + 1
            start = end - BUFFER_SIZE
            df_buffer = df_full.iloc[start:end].copy().reset_index(drop=True)

            result = engine.predict(df_buffer)
            assert result is not None

            ts = df_full.iloc[end - 1]['Open time']
            pred_log = [{
                'timestamp': _fmt_ts(ts),
                'pred': result['pred'],
                'confidence': result['confidence'],
                'probs': list(result['probs']),
                'class_name': result['class_name'],
            }]

            # Corrupt this prediction
            corruptions[i](pred_log[0])

            # Save CSV (append-only)
            df_save = df_full.iloc[:end].copy()
            df_save['Open time'] = df_save['Open time'].apply(_fmt_ts)
            df_save.to_csv(buffer_csv, index=False)

            # Compare this ONE corrupted prediction
            compare_state = {'total_ok': 0, 'total_errors': 0, 'error_details': []}
            alerts = []
            _run_comparison(buffer_csv, pred_log, engine, alerts, compare_state,
                            buffer_size=BUFFER_SIZE, num_classes=engine.num_classes,
                            class_names=engine.class_names)

            assert compare_state['total_errors'] == 1, (
                f"Prediction {i+1}: expected 1 error, got {compare_state['total_errors']}. "
                f"Details: {compare_state['error_details']}"
            )
            assert compare_state['total_ok'] == 0
            assert len(compare_state['error_details']) == 1

            # Check error message matches expected corruption type
            import re
            assert re.search(expected_errors[i], compare_state['error_details'][0]), (
                f"Prediction {i+1}: expected '{expected_errors[i]}' in "
                f"'{compare_state['error_details'][0]}'"
            )

            # Collect alerts
            all_alerts.extend(alerts)

        # Each comparison with 1 error should produce 1 alert
        assert len(all_alerts) == 3
        for alert in all_alerts:
            assert '[COMPARE]' in alert


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
