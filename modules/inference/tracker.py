"""
Trade tracker for live inference.

Tracks open trades with mode-aware exit logic:
- triple/bull/bear: TP/SL/timeout barriers identical to labeling.py
- closeN: close after N candles, outcome based on price direction (no TP/SL)
- uncertain: no trades (all predictions rejected)

Entry = Open of the candle following the signal.
"""

import json
import os

import numpy as np
import pandas as pd
from datetime import datetime

from modules.data.labeling import compute_atr, TRADE_DIRECTION, parse_close_horizon
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class TradeTracker:
    """
    Tracks directional predictions as virtual trades.

    For triple/bull/bear modes — mirrors labeling.py Triple Barrier:
    - Long: TP = entry + ATR * mult_tp, SL = entry - ATR * mult_sl
    - Short: TP = entry - ATR * mult_tp, SL = entry + ATR * mult_sl
    - Timeout after max_horizon candles

    For closeN modes — simple directional close:
    - Close after N candles (N parsed from prediction_target)
    - Outcome = 'tp' if price moved in predicted direction, 'sl' otherwise
    - No TP/SL barriers

    Timing: prediction arrives at candle N close. Entry = Open of N+1.
    So we store a "pending_trade" and open it when next candle arrives.
    """

    def __init__(self, atr_period, atr_multiplier_tp, atr_multiplier_sl, max_horizon,
                 num_classes=3, reject_class=1, prediction_target='triple'):
        self.atr_period = atr_period
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
        self.max_horizon = max_horizon
        self.num_classes = num_classes
        self.reject_class = reject_class
        self.prediction_target = prediction_target

        # Direction map from labeling.py: {class_id: "long"/"short"}
        self.trade_direction_map = TRADE_DIRECTION.get(prediction_target, {})

        # Close horizon for closeN modes (None for barrier modes)
        self.close_horizon = parse_close_horizon(prediction_target)

        self.pending_trade = None  # Waiting for next candle Open
        self.open_trades = []      # Active trades
        self.closed_trades = []    # Resolved trades

    def set_pending(self, prediction):
        """
        Store a directional prediction to be opened on next candle.
        Non-tradeable predictions are rejected based on reject_class and trade_direction_map.

        Args:
            prediction: dict from PredictionEngine.predict() with keys:
                        pred, confidence, probs, class_name
                        Plus 'signal_ts' (timestamp of signal candle)
        """
        pred_class = prediction['pred']

        # Reject non-tradeable predictions
        if pred_class == self.reject_class:
            logger.debug(f"Trade rejected: class {pred_class} (reject_class={self.reject_class})")
            self.pending_trade = None
            return

        # Reject if class has no trade direction
        if pred_class not in self.trade_direction_map:
            logger.debug(f"Trade rejected: class {pred_class} has no trade direction for {self.prediction_target}")
            self.pending_trade = None
            return

        self.pending_trade = prediction
        logger.debug(f"Pending trade: {prediction['class_name']} conf={prediction['confidence']:.3f}")

    def open_from_pending(self, candle, df_buffer):
        """
        Open the pending trade using the latest candle's Open as entry.

        Args:
            candle: dict with 'open', 'timestamp', etc. from feed
            df_buffer: full buffer DataFrame for ATR computation

        Returns:
            Opened trade dict or None
        """
        if self.pending_trade is None:
            return None

        pred = self.pending_trade
        self.pending_trade = None

        # Compute ATR from buffer
        atr_array = compute_atr(df_buffer, period=self.atr_period)
        atr_value = atr_array[-1]
        if np.isnan(atr_value):
            logger.warning("ATR is NaN, cannot open trade")
            return None

        entry_price = candle['open']
        pred_class = pred['pred']
        direction = self.trade_direction_map[pred_class]  # "long" or "short"

        if self.close_horizon is not None:
            # closeN mode: no TP/SL barriers
            tp_level = 0.0
            sl_level = 0.0
        elif direction == 'long':
            tp_level = entry_price + self.atr_multiplier_tp * atr_value
            sl_level = entry_price - self.atr_multiplier_sl * atr_value
        else:  # short
            tp_level = entry_price - self.atr_multiplier_tp * atr_value
            sl_level = entry_price + self.atr_multiplier_sl * atr_value

        trade = {
            'signal_ts': pred.get('signal_ts', candle['timestamp']),
            'entry_ts': candle['timestamp'],
            'pred_class': pred_class,
            'direction': direction,  # "long" or "short"
            'direction_name': pred['class_name'],
            'entry_price': entry_price,
            'tp_level': tp_level,
            'sl_level': sl_level,
            'atr': float(atr_value),
            'confidence': pred['confidence'],
            'candles_elapsed': 0,
            'probs': pred['probs'].tolist() if hasattr(pred['probs'], 'tolist') else list(pred['probs']),
        }

        self.open_trades.append(trade)
        if self.close_horizon is not None:
            logger.info(f"Trade opened: {trade['direction_name']} ({direction}) @ {entry_price:.2f} "
                        f"close after {self.close_horizon} candles ATR={atr_value:.2f}")
        else:
            logger.info(f"Trade opened: {trade['direction_name']} ({direction}) @ {entry_price:.2f} "
                        f"TP={tp_level:.2f} SL={sl_level:.2f} ATR={atr_value:.2f}")
        return trade

    def update(self, candle, intra_candle=False):
        """
        Check all open trades against latest candle for TP/SL/timeout/closeN.

        Args:
            candle: dict with 'high', 'low', 'close', 'timestamp'
            intra_candle: if True, only check TP/SL (no elapsed increment, no timeout)
                          For closeN mode, intra_candle updates are ignored.

        Returns:
            list of closed trade dicts (may be empty)
        """
        newly_closed = []
        still_open = []

        for trade in self.open_trades:
            if not intra_candle:
                trade['candles_elapsed'] += 1

            if self.close_horizon is not None:
                outcome = self._check_close_horizon(trade, candle, intra_candle=intra_candle)
            else:
                outcome = self._check_barriers(trade, candle, intra_candle=intra_candle)

            if outcome is not None:
                trade['close_ts'] = candle['timestamp']
                trade['outcome'] = outcome
                trade['hit_candle'] = trade['candles_elapsed']
                self.closed_trades.append(trade)
                newly_closed.append(trade)
                logger.info(f"Trade closed: {trade['direction_name']} -> {outcome} "
                            f"after {trade['hit_candle']} candles "
                            f"(conf={trade['confidence']:.3f})")
            else:
                still_open.append(trade)

        self.open_trades = still_open
        return newly_closed

    def _check_close_horizon(self, trade, candle, intra_candle=False):
        """
        Check if closeN horizon is reached.

        Returns 'tp' if price moved in predicted direction, 'sl' otherwise, or None if not yet.
        No intra-candle checks for closeN mode.
        """
        if intra_candle:
            return None

        if trade['candles_elapsed'] >= self.close_horizon:
            close_price = candle['close']
            entry_price = trade['entry_price']
            direction = trade['direction']  # "long" or "short"

            if direction == 'long':
                return 'tp' if close_price > entry_price else 'sl'
            else:  # short
                return 'tp' if close_price < entry_price else 'sl'

        return None

    def _check_barriers(self, trade, candle, intra_candle=False):
        """
        Check if candle hits TP, SL or timeout.

        Returns 'tp', 'sl', 'timeout', or None if still open.
        Pessimistic: if both TP and SL hit on same candle, returns 'sl'.
        Note: labeling.py treats "both hit" as Uncertain (no winner).
        Here we choose 'sl' (pessimistic) for conservative live tracking.
        If intra_candle=True, skip timeout check (only TP/SL).
        """
        high = candle['high']
        low = candle['low']
        direction = trade['direction']  # "long" or "short"

        if direction == 'long':
            tp_hit = high >= trade['tp_level']
            sl_hit = low <= trade['sl_level']
            if tp_hit and sl_hit:
                return 'sl'  # Pessimistic: both barriers hit = loss
            if tp_hit:
                return 'tp'
            if sl_hit:
                return 'sl'
        else:  # short
            tp_hit = low <= trade['tp_level']
            sl_hit = high >= trade['sl_level']
            if tp_hit and sl_hit:
                return 'sl'  # Pessimistic: both barriers hit = loss
            if tp_hit:
                return 'tp'
            if sl_hit:
                return 'sl'

        if not intra_candle and trade['candles_elapsed'] >= self.max_horizon:
            return 'timeout'

        return None

    @property
    def stats(self):
        """Compute trade statistics."""
        if not self.closed_trades:
            return {
                'total': 0, 'wins': 0, 'losses': 0, 'timeouts': 0,
                'winrate': 0.0, 'open': len(self.open_trades),
            }

        wins = sum(1 for t in self.closed_trades if t['outcome'] == 'tp')
        losses = sum(1 for t in self.closed_trades if t['outcome'] == 'sl')
        timeouts = sum(1 for t in self.closed_trades if t['outcome'] == 'timeout')
        total = len(self.closed_trades)
        resolved = wins + losses

        long_trades = [t for t in self.closed_trades if t['direction'] == 'long']
        short_trades = [t for t in self.closed_trades if t['direction'] == 'short']

        long_wins = sum(1 for t in long_trades if t['outcome'] == 'tp')
        short_wins = sum(1 for t in short_trades if t['outcome'] == 'tp')

        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'winrate': wins / resolved * 100 if resolved > 0 else 0.0,
            'winrate_incl_timeout': wins / total * 100 if total > 0 else 0.0,
            'open': len(self.open_trades),
            'long_total': len(long_trades),
            'long_wins': long_wins,
            'short_total': len(short_trades),
            'short_wins': short_wins,
            # Legacy aliases for dashboard compatibility
            'bull_total': len(long_trades),
            'bull_wins': long_wins,
            'bear_total': len(short_trades),
            'bear_wins': short_wins,
        }

    def load_closed_trades(self, rows):
        """
        Restore closed trades from CSV rows (list of dicts).
        Each row must have: signal_ts, direction, entry_price, tp_level, sl_level,
        atr, close_ts, outcome, hit_candle, confidence.
        """
        for row in rows:
            direction_name = row.get('direction', '')
            # Infer direction from TRADE_DIRECTION map or name heuristic
            direction = self._infer_direction(direction_name)
            trade = {
                'signal_ts': row['signal_ts'],
                'entry_ts': row.get('entry_ts', row['signal_ts']),
                'direction': direction,
                'direction_name': direction_name,
                'entry_price': float(row['entry_price']),
                'tp_level': float(row['tp_level']),
                'sl_level': float(row['sl_level']),
                'atr': float(row['atr']),
                'close_ts': row['close_ts'],
                'outcome': row['outcome'],
                'hit_candle': int(row['hit_candle']),
                'confidence': float(row['confidence']),
                'candles_elapsed': int(row['hit_candle']),
                'probs': [],
            }
            self.closed_trades.append(trade)
        if rows:
            logger.info(f"Restored {len(rows)} closed trades from CSV")

    def _infer_direction(self, direction_name):
        """Infer 'long'/'short' from class name using trade_direction_map."""
        for cls_id, dir_str in self.trade_direction_map.items():
            from modules.data.labeling import TARGET_CLASS_NAMES
            class_names = TARGET_CLASS_NAMES.get(self.prediction_target, {})
            if class_names.get(cls_id) == direction_name:
                return dir_str
        # Fallback: heuristic based on name
        name_lower = direction_name.lower()
        if name_lower in ('bull', 'up', 'long'):
            return 'long'
        return 'short'

    def save_state(self, path):
        """Save open trades + pending to JSON for resume."""
        def _serialize(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        state = {
            'open_trades': self.open_trades,
            'pending_trade': self.pending_trade,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, default=_serialize, indent=2)
        logger.info(f"State saved: {len(self.open_trades)} open trades, "
                     f"pending={'yes' if self.pending_trade else 'no'} -> {path}")

    def load_state(self, path):
        """Restore open trades + pending from JSON."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.open_trades = state.get('open_trades', [])
        self.pending_trade = state.get('pending_trade')
        logger.info(f"State loaded: {len(self.open_trades)} open trades, "
                     f"pending={'yes' if self.pending_trade else 'no'} from {path}")
        return True

    def trades_to_csv_rows(self):
        """Format closed trades for CSV export."""
        rows = []
        for t in self.closed_trades:
            rows.append({
                'signal_ts': t['signal_ts'],
                'pair': '',  # Set by caller
                'direction': t['direction_name'],
                'entry_price': t['entry_price'],
                'tp_level': t['tp_level'],
                'sl_level': t['sl_level'],
                'atr': t['atr'],
                'close_ts': t['close_ts'],
                'outcome': t['outcome'],
                'hit_candle': t['hit_candle'],
                'confidence': t['confidence'],
            })
        return rows
