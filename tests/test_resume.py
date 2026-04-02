"""
Tests for live inference resume/persistence system.

Tests cover:
- Tracker: save_state / load_state / load_closed_trades
- Feed: load_buffer / fill_gap (mocked)
- inference: _load_predictions / _load_trades / _ensure_csv / --fresh logic
- Dashboard: conf field in trades + filtered stats
"""

import sys
import os
import json
import csv
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.utils.seed import set_seed


# ============================================================
# Tracker persistence tests
# ============================================================

class TestTrackerPersistence:
    """Test save_state / load_state / load_closed_trades."""

    def _make_buffer(self, n=100):
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
        remaining = 1 - confidence
        for i in range(3):
            if i != direction:
                probs[i] = remaining / 2
        names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
        return {
            'pred': direction,
            'confidence': confidence,
            'probs': probs,
            'class_name': names[direction],
            'signal_ts': '2024-01-01 00:00',
        }

    def test_save_load_state_with_open_trades(self):
        """Save state with open trades + pending, reload into new tracker."""
        from modules.inference.tracker import TradeTracker

        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()

        # Open a trade
        tracker.set_pending(self._make_pred(2, 0.85))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        tracker.open_from_pending(candle, df)

        # Set a new pending
        tracker.set_pending(self._make_pred(0, 0.7))

        assert len(tracker.open_trades) == 1
        assert tracker.pending_trade is not None

        # Save state
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_path = f.name
        try:
            tracker.save_state(state_path)
            assert os.path.exists(state_path)

            # Load into new tracker
            tracker2 = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
            result = tracker2.load_state(state_path)

            assert result is True
            assert len(tracker2.open_trades) == 1
            assert tracker2.open_trades[0]['direction'] == 'long'  # Bull = long
            assert tracker2.open_trades[0]['entry_price'] == 100.0
            assert tracker2.pending_trade is not None
            assert tracker2.pending_trade['pred'] == 0
        finally:
            os.unlink(state_path)

    def test_save_load_state_empty(self):
        """Save/load with no open trades."""
        from modules.inference.tracker import TradeTracker

        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_path = f.name
        try:
            tracker.save_state(state_path)

            tracker2 = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
            tracker2.load_state(state_path)
            assert len(tracker2.open_trades) == 0
            assert tracker2.pending_trade is None
        finally:
            os.unlink(state_path)

    def test_load_state_nonexistent(self):
        """load_state returns False for missing file."""
        from modules.inference.tracker import TradeTracker
        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        result = tracker.load_state('/tmp/nonexistent_state_xyz.json')
        assert result is False

    def test_load_closed_trades(self):
        """Load closed trades from CSV-like rows."""
        from modules.inference.tracker import TradeTracker

        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)

        rows = [
            {
                'signal_ts': '2024-01-01 00:00',
                'direction': 'Bull',
                'entry_price': '100.50',
                'tp_level': '105.00',
                'sl_level': '98.00',
                'atr': '2.25',
                'close_ts': '2024-01-01 01:00',
                'outcome': 'tp',
                'hit_candle': '4',
                'confidence': '0.8500',
            },
            {
                'signal_ts': '2024-01-01 02:00',
                'direction': 'Bear',
                'entry_price': '103.00',
                'tp_level': '99.00',
                'sl_level': '106.00',
                'atr': '2.00',
                'close_ts': '2024-01-01 03:00',
                'outcome': 'sl',
                'hit_candle': '3',
                'confidence': '0.7200',
            },
        ]

        tracker.load_closed_trades(rows)

        assert len(tracker.closed_trades) == 2
        assert tracker.closed_trades[0]['direction'] == 'long'  # Bull = long
        assert tracker.closed_trades[0]['entry_price'] == 100.50
        assert tracker.closed_trades[0]['outcome'] == 'tp'
        assert tracker.closed_trades[0]['confidence'] == 0.85

        assert tracker.closed_trades[1]['direction'] == 'short'  # Bear = short
        assert tracker.closed_trades[1]['outcome'] == 'sl'

        # Stats should reflect loaded trades
        stats = tracker.stats
        assert stats['total'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert stats['winrate'] == 50.0

    def test_gap_replay_resolves_open_trade(self):
        """Open trade gets resolved by gap candles during replay."""
        from modules.inference.tracker import TradeTracker

        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
        df = self._make_buffer()

        # Open a Bull trade
        tracker.set_pending(self._make_pred(2, 0.9))  # Bull = class 2
        candle = {'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0,
                  'timestamp': '2024-01-01 00:15', 'volume': 500.0}
        trade = tracker.open_from_pending(candle, df)
        tp_level = trade['tp_level']

        # Save state
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_path = f.name
        try:
            tracker.save_state(state_path)

            # Simulate restart: new tracker, load state
            tracker2 = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
            tracker2.load_state(state_path)

            assert len(tracker2.open_trades) == 1

            # Simulate gap candles that hit TP
            gap_candle = {
                'high': tp_level + 5, 'low': 99.0, 'close': tp_level,
                'timestamp': '2024-01-01 00:30', 'volume': 500.0
            }
            closed = tracker2.update(gap_candle)

            assert len(closed) == 1
            assert closed[0]['outcome'] == 'tp'
            assert len(tracker2.open_trades) == 0
            assert len(tracker2.closed_trades) == 1
        finally:
            os.unlink(state_path)


# ============================================================
# Feed persistence tests
# ============================================================

class TestFeedPersistence:
    """Test load_buffer and fill_gap."""

    def _make_buffer_df(self, n=50):
        """Create synthetic buffer DataFrame."""
        set_seed(config.SEED)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=n, freq='15min', tz='UTC'),
            'Open': prices.astype(np.float64),
            'High': (prices + np.random.uniform(1, 3, n)).astype(np.float64),
            'Low': (prices - np.random.uniform(1, 3, n)).astype(np.float64),
            'Close': (prices + np.random.uniform(-1, 1, n)).astype(np.float64),
            'Volume': np.random.uniform(100, 1000, n).astype(np.float64),
        })
        return df

    def test_load_buffer_from_csv(self):
        """load_buffer correctly reads a saved CSV."""
        from modules.inference.feed import CandleFeed

        df = self._make_buffer_df()

        tmpdir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmpdir, 'buffer.csv')
            df_out = df.copy()
            df_out['Open time'] = df_out['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_out.to_csv(csv_path, index=False)

            # Create feed without filling buffer (we mock the exchange init)
            # We need to test load_buffer in isolation
            # Patch: create feed object and manually set attributes
            feed = object.__new__(CandleFeed)
            feed.buffer = pd.DataFrame()
            feed.buffer_size = 1000
            feed.tf_ms = 15 * 60 * 1000
            feed.tf_seconds = 900

            n_loaded = feed.load_buffer(csv_path)

            assert n_loaded == 50
            assert len(feed.buffer) == 50
            assert feed.buffer['Open time'].dtype.kind == 'M'  # datetime
            assert feed.buffer['Open'].dtype == np.float64
            assert feed.buffer['Close'].dtype == np.float64
        finally:
            shutil.rmtree(tmpdir)

    def test_load_buffer_roundtrip(self):
        """save_buffer -> load_buffer preserves data."""
        from modules.inference.feed import CandleFeed

        df = self._make_buffer_df(30)

        tmpdir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmpdir, 'buffer.csv')

            # Create feed and set buffer
            feed1 = object.__new__(CandleFeed)
            feed1.buffer = df.copy()
            feed1.buffer_size = 1000
            feed1.tf_ms = 15 * 60 * 1000
            feed1.tf_seconds = 900
            feed1._last_saved_ts = None

            # Save
            feed1.save_buffer(csv_path)

            # Load into new feed
            feed2 = object.__new__(CandleFeed)
            feed2.buffer = pd.DataFrame()
            feed2.buffer_size = 1000
            feed2.tf_ms = 15 * 60 * 1000
            feed2.tf_seconds = 900

            feed2.load_buffer(csv_path)

            assert len(feed2.buffer) == 30
            # Check values match (within float32 precision)
            np.testing.assert_array_almost_equal(
                feed2.buffer['Close'].values,
                df['Close'].values,
                decimal=2
            )
        finally:
            shutil.rmtree(tmpdir)


# ============================================================
# inference helper tests
# ============================================================

class TestLiveInferenceHelpers:
    """Test _load_predictions, _load_trades, _ensure_csv."""

    def test_ensure_csv_creates_new(self):
        """_ensure_csv creates file if missing."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from inference import _ensure_csv, _force_csv

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'test.csv')
            fields = ['a', 'b', 'c']
            _ensure_csv(path, fields)

            assert os.path.exists(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == fields
        finally:
            shutil.rmtree(tmpdir)

    def test_ensure_csv_preserves_existing(self):
        """_ensure_csv does NOT overwrite existing file."""
        from inference import _ensure_csv, _append_csv

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'test.csv')
            fields = ['a', 'b']

            # Create with data
            _ensure_csv(path, fields)
            _append_csv(path, {'a': '1', 'b': '2'}, fields)

            # Call ensure again — should NOT erase
            _ensure_csv(path, fields)

            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]['a'] == '1'
        finally:
            shutil.rmtree(tmpdir)

    def test_load_predictions(self):
        """Load predictions from CSV (3-class format)."""
        from inference import _load_predictions, _force_csv, _append_csv

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'preds.csv')
            fields = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume',
                       'pred_class', 'confidence', 'prob_bear', 'prob_uncertain', 'prob_bull']
            _force_csv(path, fields)

            _append_csv(path, {
                'timestamp': '2024-01-01 00:15:00+00:00',
                'pair': 'BTC/USDT',
                'open': '100.0', 'high': '102.0', 'low': '99.0', 'close': '101.0', 'volume': '500.0',
                'pred_class': '2',
                'confidence': '0.8500',
                'prob_bear': '0.0750', 'prob_uncertain': '0.0750', 'prob_bull': '0.8500',
            }, fields)

            _append_csv(path, {
                'timestamp': '2024-01-01 00:30:00+00:00',
                'pair': 'BTC/USDT',
                'open': '101.0', 'high': '103.0', 'low': '100.0', 'close': '102.0', 'volume': '600.0',
                'pred_class': '0',
                'confidence': '0.7000',
                'prob_bear': '0.7000', 'prob_uncertain': '0.1500', 'prob_bull': '0.1500',
            }, fields)

            class_names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
            preds = _load_predictions(path, class_names)

            assert len(preds) == 2
            assert preds[0]['pred'] == 2  # Bull
            assert preds[0]['class_name'] == 'Bull'
            assert preds[0]['confidence'] == 0.85
            assert len(preds[0]['probs']) == 3

            assert preds[1]['pred'] == 0  # Bear
            assert preds[1]['class_name'] == 'Bear'
            assert preds[1]['confidence'] == 0.7
        finally:
            shutil.rmtree(tmpdir)

    def test_load_predictions_empty(self):
        """Load from nonexistent file returns empty list."""
        from inference import _load_predictions
        class_names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
        preds = _load_predictions('/tmp/nonexistent_preds_xyz.csv', class_names)
        assert preds == []

    def test_load_trades(self):
        """Load trades from CSV into tracker."""
        from inference import _load_trades, _force_csv, _append_csv
        from modules.inference.tracker import TradeTracker

        tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'trades.csv')
            fields = ['signal_ts', 'pair', 'direction', 'entry_price', 'tp_level', 'sl_level',
                       'atr', 'close_ts', 'outcome', 'hit_candle', 'confidence']
            _force_csv(path, fields)

            _append_csv(path, {
                'signal_ts': '2024-01-01 00:00', 'pair': 'BTC/USDT', 'direction': 'Bull',
                'entry_price': '100.50', 'tp_level': '105.00', 'sl_level': '98.00',
                'atr': '2.25', 'close_ts': '2024-01-01 01:00', 'outcome': 'tp',
                'hit_candle': '4', 'confidence': '0.8500',
            }, fields)

            n = _load_trades(path, tracker)

            assert n == 1
            assert len(tracker.closed_trades) == 1
            assert tracker.closed_trades[0]['outcome'] == 'tp'
            assert tracker.stats['wins'] == 1
        finally:
            shutil.rmtree(tmpdir)


# ============================================================
# Dashboard conf field tests
# ============================================================

class TestDashboardConfField:
    """Test that conf is included in trade data."""

    def test_closed_trade_has_conf(self):
        """Closed trades in JSON include conf field."""
        from modules.inference.dashboard import _DashboardState

        state = _DashboardState('BTC/USDT', '15m')

        candles = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=2, freq='15min'),
            'Open': [100.0, 101.0], 'High': [102.0, 103.0],
            'Low': [99.0, 100.0], 'Close': [101.0, 102.0],
        })

        closed_trades = [{
            'entry_ts': pd.Timestamp('2024-01-01 00:00'),
            'close_ts': pd.Timestamp('2024-01-01 01:00'),
            'entry_price': 100.0, 'tp_level': 105.0, 'sl_level': 98.0,
            'direction': 2, 'outcome': 'tp', 'confidence': 0.85,
        }]

        open_trades = [{
            'entry_ts': pd.Timestamp('2024-01-01 00:30'),
            'entry_price': 101.0, 'tp_level': 106.0, 'sl_level': 99.0,
            'direction': 0, 'confidence': 0.72,
        }]

        state.update(candles, [], closed_trades, open_trades,
                     {'total': 1, 'wins': 1, 'losses': 0, 'timeouts': 0, 'winrate': 100, 'open': 1})

        data = state.get_json()

        # Closed trade has conf
        assert len(data['trades']) == 1
        assert 'conf' in data['trades'][0]
        assert data['trades'][0]['conf'] == 0.85

        # Open trade has conf
        assert len(data['open_trades']) == 1
        assert 'conf' in data['open_trades'][0]
        assert data['open_trades'][0]['conf'] == 0.72

    def test_trade_without_confidence_defaults_to_zero(self):
        """Trade missing confidence key defaults to 0."""
        from modules.inference.dashboard import _DashboardState

        state = _DashboardState('BTC/USDT', '15m')

        candles = pd.DataFrame({
            'Open time': pd.date_range('2024-01-01', periods=1, freq='15min'),
            'Open': [100.0], 'High': [102.0], 'Low': [99.0], 'Close': [101.0],
        })

        # Trade without confidence key
        closed_trades = [{
            'entry_ts': pd.Timestamp('2024-01-01 00:00'),
            'close_ts': pd.Timestamp('2024-01-01 01:00'),
            'entry_price': 100.0, 'tp_level': 105.0, 'sl_level': 98.0,
            'direction': 2, 'outcome': 'tp',
        }]

        state.update(candles, [], closed_trades, [],
                     {'total': 1, 'wins': 1, 'losses': 0, 'timeouts': 0, 'winrate': 100, 'open': 0})

        data = state.get_json()
        assert data['trades'][0]['conf'] == 0


# ============================================================
# Integration: full resume cycle
# ============================================================

class TestFullResumeCycle:
    """End-to-end test: create session data, simulate resume."""

    def _make_buffer(self, n=100):
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

    def test_full_cycle_predictions_and_trades(self):
        """
        Simulate: write predictions + trades CSVs + state JSON,
        then load them back and verify consistency.
        """
        from inference import _force_csv, _append_csv, _load_predictions, _load_trades
        from modules.inference.tracker import TradeTracker

        tmpdir = tempfile.mkdtemp()
        try:
            # --- Setup CSVs ---
            pred_fields = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume',
                           'pred_class', 'confidence', 'prob_bear', 'prob_uncertain', 'prob_bull']
            trade_fields = ['signal_ts', 'pair', 'direction', 'entry_price', 'tp_level', 'sl_level',
                            'atr', 'close_ts', 'outcome', 'hit_candle', 'confidence']

            pred_csv = os.path.join(tmpdir, 'preds.csv')
            trade_csv = os.path.join(tmpdir, 'trades.csv')
            state_json = os.path.join(tmpdir, 'state.json')

            _force_csv(pred_csv, pred_fields)
            _force_csv(trade_csv, trade_fields)

            # Write 3 predictions (3-class: 0=Bear, 1=Uncertain, 2=Bull)
            for i, (pred_cls, conf) in enumerate([(2, 0.85), (0, 0.70), (2, 0.90)]):
                probs = [0.0, 0.0, 0.0]
                probs[pred_cls] = conf
                remaining = 1 - conf
                for j in range(3):
                    if j != pred_cls:
                        probs[j] = remaining / 2
                _append_csv(pred_csv, {
                    'timestamp': f'2024-01-01 {i:02d}:15:00+00:00',
                    'pair': 'BTC/USDT',
                    'open': '100.0', 'high': '102.0', 'low': '99.0', 'close': '101.0', 'volume': '500.0',
                    'pred_class': str(pred_cls),
                    'confidence': f'{conf:.4f}',
                    'prob_bear': f'{probs[0]:.4f}',
                    'prob_uncertain': f'{probs[1]:.4f}',
                    'prob_bull': f'{probs[2]:.4f}',
                }, pred_fields)

            # Write 2 closed trades
            _append_csv(trade_csv, {
                'signal_ts': '2024-01-01 00:00', 'pair': 'BTC/USDT', 'direction': 'Bull',
                'entry_price': '100.50', 'tp_level': '105.00', 'sl_level': '98.00',
                'atr': '2.25', 'close_ts': '2024-01-01 01:00', 'outcome': 'tp',
                'hit_candle': '4', 'confidence': '0.8500',
            }, trade_fields)
            _append_csv(trade_csv, {
                'signal_ts': '2024-01-01 01:00', 'pair': 'BTC/USDT', 'direction': 'Bear',
                'entry_price': '103.00', 'tp_level': '99.00', 'sl_level': '106.00',
                'atr': '2.00', 'close_ts': '2024-01-01 02:00', 'outcome': 'sl',
                'hit_candle': '3', 'confidence': '0.7000',
            }, trade_fields)

            # --- Simulate resume ---
            # 1. Load predictions
            class_names = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
            preds = _load_predictions(pred_csv, class_names)
            assert len(preds) == 3
            assert preds[0]['class_name'] == 'Bull'
            assert preds[2]['confidence'] == 0.90

            # 2. Load trades into tracker
            tracker = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
            n_trades = _load_trades(trade_csv, tracker)
            assert n_trades == 2
            assert tracker.stats['total'] == 2
            assert tracker.stats['wins'] == 1
            assert tracker.stats['losses'] == 1

            # 3. Create an open trade and save state
            df = self._make_buffer()
            pred = {
                'pred': 2, 'confidence': 0.90,
                'probs': np.array([0.05, 0.05, 0.90]),
                'class_name': 'Bull',
                'signal_ts': '2024-01-01 02:00',
            }
            tracker.set_pending(pred)
            candle = {'open': 102.0, 'high': 104.0, 'low': 101.0, 'close': 103.0,
                      'timestamp': '2024-01-01 02:15', 'volume': 500.0}
            tracker.open_from_pending(candle, df)

            assert len(tracker.open_trades) == 1
            tracker.save_state(state_json)

            # 4. Reload state into fresh tracker (with closed trades already loaded)
            tracker2 = TradeTracker(atr_period=14, atr_multiplier_tp=2, atr_multiplier_sl=1, max_horizon=10)
            _load_trades(trade_csv, tracker2)
            tracker2.load_state(state_json)

            assert len(tracker2.closed_trades) == 2  # from CSV
            assert len(tracker2.open_trades) == 1     # from state
            assert tracker2.open_trades[0]['entry_price'] == 102.0

            # 5. Simulate gap replay: candle that hits TP
            tp_level = tracker2.open_trades[0]['tp_level']
            gap_candle = {
                'high': tp_level + 5, 'low': 101.0, 'close': tp_level,
                'timestamp': '2024-01-01 02:30', 'volume': 500.0
            }
            closed = tracker2.update(gap_candle)
            assert len(closed) == 1
            assert closed[0]['outcome'] == 'tp'

            # Now 3 closed trades total
            assert len(tracker2.closed_trades) == 3
            assert tracker2.stats['total'] == 3
            assert tracker2.stats['wins'] == 2  # 2 TPs

        finally:
            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
