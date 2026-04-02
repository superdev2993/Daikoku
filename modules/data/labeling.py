"""
Data labeling module - Triple Barrier labeling on raw OHLC data
"""

import traceback

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from modules.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Class labels for PyTorch (indices must start at 0)
# ============================================================================
LABEL_BEARISH = 0    # Profitable short (price hit short TP before short SL)
LABEL_UNCERTAIN = 1  # No profitable trade opportunity
LABEL_BULLISH = 2    # Profitable long (price hit long TP before long SL)


def compute_atr(df, period=None):
    """
    Compute Average True Range using Wilder smoothing.

    TR = max(H-L, |H-Close_prev|, |L-Close_prev|)
    ATR[period-1] = mean(TR[0:period])
    ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period  for i >= period

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Smoothing period (default: from config.ATR_PERIOD)

    Returns:
        numpy array same length as df. First period-1 values are NaN.
    """
    if period is None:
        period = config.ATR_PERIOD

    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)
    n = len(df)

    # True Range (vectorized)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    # Wilder smoothing (sequential)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def compute_barriers(close, atr_value, atr_multiplier_tp=None, atr_multiplier_sl=None):
    """
    Compute asymmetric barriers for long and short scenarios.

    Long:  TP = close + tp_mult × ATR,  SL = close - sl_mult × ATR
    Short: TP = close - tp_mult × ATR,  SL = close + sl_mult × ATR

    Args:
        close: Close price at prediction point
        atr_value: ATR value at prediction point
        atr_multiplier_tp: Take profit distance multiplier (default: from config)
        atr_multiplier_sl: Stop loss distance multiplier (default: from config)

    Returns:
        Tuple: (long_tp, long_sl, short_tp, short_sl)
    """
    if atr_multiplier_tp is None:
        atr_multiplier_tp = config.ATR_MULTIPLIER_TP
    if atr_multiplier_sl is None:
        atr_multiplier_sl = config.ATR_MULTIPLIER_SL

    long_tp = close + atr_multiplier_tp * atr_value
    long_sl = close - atr_multiplier_sl * atr_value
    short_tp = close - atr_multiplier_tp * atr_value
    short_sl = close + atr_multiplier_sl * atr_value

    return long_tp, long_sl, short_tp, short_sl



def get_labels(df, window_size=None, atr_multiplier_tp=None, atr_multiplier_sl=None,
               max_horizon=None, atr_period=None):
    """
    Compute asymmetric triple barrier labels for entire dataset.

    CRITICAL: Works on RAW data, not transformed data!

    For each valid index i, two scenarios are evaluated:
    - Long:  did price hit +TP_mult×ATR BEFORE -SL_mult×ATR?  → BULLISH
    - Short: did price hit -TP_mult×ATR BEFORE +SL_mult×ATR?  → BEARISH
    - Neither → UNCERTAIN

    With TP_mult > SL_mult, this encodes a favorable R:R ratio directly
    into the labels. The two scenarios are mutually exclusive.

    Args:
        df: Raw DataFrame with OHLC data
        window_size: Window size (default: from config)
        atr_multiplier_tp: Take profit multiplier (default: from config)
        atr_multiplier_sl: Stop loss multiplier (default: from config)
        max_horizon: Maximum horizon (default: from config)
        atr_period: ATR smoothing period (default: from config)

    Returns:
        Tuple: (labels, atr_array, confidence, long_pnl_R, short_pnl_R)
        - labels: numpy array {0, 1, 2} aligned with valid indices
            - 0: LABEL_BEARISH (profitable short)
            - 1: LABEL_UNCERTAIN (no profitable trade)
            - 2: LABEL_BULLISH (profitable long)
        - atr_array: numpy array of ATR values for each labeled sample
        - confidence: numpy array [0,1] per sample. For directional labels:
            conf = 1 - (hit_candle / max_horizon). For uncertain: fixed at 1.0.
        - long_pnl_R: numpy array of exact P&L in R-multiples for long trades
        - short_pnl_R: numpy array of exact P&L in R-multiples for short trades
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if atr_multiplier_tp is None:
        atr_multiplier_tp = config.ATR_MULTIPLIER_TP
    if atr_multiplier_sl is None:
        atr_multiplier_sl = config.ATR_MULTIPLIER_SL
    if max_horizon is None:
        max_horizon = config.MAX_HORIZON
    if atr_period is None:
        atr_period = config.ATR_PERIOD

    rr_ratio = atr_multiplier_tp / atr_multiplier_sl if atr_multiplier_sl > 0 else float('inf')
    logger.info("Computing asymmetric triple barrier labels...")
    logger.debug(f"  Window size: {window_size}")
    logger.debug(f"  ATR period: {atr_period}")
    logger.debug(f"  ATR multiplier TP: {atr_multiplier_tp}")
    logger.debug(f"  ATR multiplier SL: {atr_multiplier_sl}")
    logger.debug(f"  R:R ratio: {rr_ratio:.1f}:1")
    logger.debug(f"  Max horizon: {max_horizon} candles")

    # Compute ATR once for the entire dataset
    atr_full = compute_atr(df, period=atr_period)

    # Pre-extract numpy arrays (avoid df.iloc / iterrows)
    high_arr = df['High'].values.astype(np.float64)
    low_arr = df['Low'].values.astype(np.float64)
    open_arr = df['Open'].values.astype(np.float64)
    close_arr = df['Close'].values.astype(np.float64)

    n_samples = len(df)

    # Start from window_size (need history for feature windows)
    # End at n_samples - max_horizon (need future for barrier)
    valid_start = window_size
    valid_end = n_samples - max_horizon

    if valid_end <= valid_start:
        raise ValueError(f"Not enough data: need at least {window_size + max_horizon} candles, got {n_samples}")

    n_labels = valid_end - valid_start
    logger.debug(f"  Valid range: [{valid_start}, {valid_end})")
    logger.debug(f"  Computing labels for {n_labels} samples...")

    labels = np.full(n_labels, LABEL_UNCERTAIN, dtype=np.int8)
    atr_values = np.empty(n_labels, dtype=np.float32)
    confidence = np.ones(n_labels, dtype=np.float32)
    long_pnl_R = np.zeros(n_labels, dtype=np.float32)
    short_pnl_R = np.zeros(n_labels, dtype=np.float32)

    for idx in range(n_labels):
        i = valid_start + idx
        atr_val = atr_full[i]

        # Guard against NaN ATR (first period-1 values)
        if np.isnan(atr_val):
            atr_values[idx] = 0.0
            continue

        if atr_val == 0.0:
            raise ValueError(f"ATR is exactly 0 at index {i} — flat price data cannot produce valid labels")

        atr_values[idx] = atr_val

        # Realistic entry: Open of next candle (signal at Close[i], act at Open[i+1])
        entry = open_arr[i + 1]

        # Future candles starting from execution candle (its H/L can already hit barriers)
        h = high_arr[i + 1: i + 1 + max_horizon]
        l = low_arr[i + 1: i + 1 + max_horizon]

        # === Long scenario: TP up, SL down ===
        long_tp = entry + atr_multiplier_tp * atr_val
        long_sl = entry - atr_multiplier_sl * atr_val

        long_tp_hits = h >= long_tp
        long_sl_hits = l <= long_sl
        long_any = long_tp_hits | long_sl_hits

        long_win = False
        long_hit_time = 0
        if long_any.any():
            first = np.argmax(long_any)
            if long_tp_hits[first] and not long_sl_hits[first]:
                long_win = True
                long_hit_time = first

        # === Short scenario: TP down, SL up ===
        short_tp = entry - atr_multiplier_tp * atr_val
        short_sl = entry + atr_multiplier_sl * atr_val

        short_tp_hits = l <= short_tp
        short_sl_hits = h >= short_sl
        short_any = short_tp_hits | short_sl_hits

        short_win = False
        short_hit_time = 0
        if short_any.any():
            first = np.argmax(short_any)
            if short_tp_hits[first] and not short_sl_hits[first]:
                short_win = True
                short_hit_time = first

        # Exact P&L in R-multiples for each direction
        # Long P&L
        if long_win:
            long_pnl_R[idx] = rr_ratio              # TP hit → +RR
        elif long_sl_hits.any():
            long_pnl_R[idx] = -1.0                   # SL hit → -1R
        else:
            long_pnl_R[idx] = (close_arr[i + max_horizon] - entry) / (atr_multiplier_sl * atr_val)

        # Short P&L
        if short_win:
            short_pnl_R[idx] = rr_ratio
        elif short_sl_hits.any():
            short_pnl_R[idx] = -1.0
        else:
            short_pnl_R[idx] = (entry - close_arr[i + max_horizon]) / (atr_multiplier_sl * atr_val)

        # Assign label: if both directions win, earliest TP hit wins.
        # If same candle hits both TPs (possible when TP < SL), label as Uncertain.
        if long_win and short_win:
            if long_hit_time < short_hit_time:
                labels[idx] = LABEL_BULLISH
                confidence[idx] = 1.0 - (long_hit_time / max_horizon)
            elif short_hit_time < long_hit_time:
                labels[idx] = LABEL_BEARISH
                confidence[idx] = 1.0 - (short_hit_time / max_horizon)
            # else: same candle → stays UNCERTAIN (default)
        elif long_win:
            labels[idx] = LABEL_BULLISH
            confidence[idx] = 1.0 - (long_hit_time / max_horizon)
        elif short_win:
            labels[idx] = LABEL_BEARISH
            confidence[idx] = 1.0 - (short_hit_time / max_horizon)

    # Apply confidence curve and per-dataset normalization
    dir_mask = (labels == LABEL_BEARISH) | (labels == LABEL_BULLISH)
    if dir_mask.any():
        if config.LABEL_CONFIDENCE_CURVE == "sqrt":
            confidence[dir_mask] = np.sqrt(confidence[dir_mask])
        # Normalize so mean(directional confidence) = 1.0 (preserves inter-class balance)
        dir_mean = confidence[dir_mask].mean()
        if dir_mean > 0:
            confidence[dir_mask] /= dir_mean

    # Log label distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"\n  Label distribution (R:R = {rr_ratio:.1f}:1):")
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        label_name = {
            LABEL_BEARISH: "Bearish (short)",
            LABEL_UNCERTAIN: "Uncertain",
            LABEL_BULLISH: "Bullish (long)"
        }[label]
        logger.info(f"    {label:2d} ({label_name:14s}): {count:6d} ({pct:5.2f}%)")

    # Log confidence stats per class
    for lbl, name in [(LABEL_BEARISH, "Bear"), (LABEL_UNCERTAIN, "Uncertain"), (LABEL_BULLISH, "Bull")]:
        mask = labels == lbl
        if mask.any():
            conf_vals = confidence[mask]
            logger.debug(f"    {name:10s} confidence: mean={conf_vals.mean():.3f}, min={conf_vals.min():.3f}, max={conf_vals.max():.3f}")

    logger.info(f"✓ Asymmetric barrier labels computed: {len(labels)} samples")

    return labels, atr_values, confidence, long_pnl_R, short_pnl_R


# ============================================================================
# Prediction target remapping (post-labeling)
# ============================================================================
TARGET_CLASS_NAMES = {
    "triple": {0: "Bear", 1: "Uncertain", 2: "Bull"},
    "bull": {0: "Not-Bull", 1: "Bull"},
    "bear": {0: "Not-Bear", 1: "Bear"},
    "uncertain": {0: "Directional", 1: "Uncertain"},
    "close1": {0: "Down", 1: "Up"},
    "close2": {0: "Down", 1: "Up"},
    "close3": {0: "Down", 1: "Up"},
}

REJECT_CLASS = {
    "triple": 1,     # Reject Uncertain
    "bull": 0,        # Reject Not-Bull (don't trade)
    "bear": 0,        # Reject Not-Bear (don't trade)
    "uncertain": 1,   # Reject Uncertain (don't trade)
    "close1": None,   # Trade both directions
    "close2": None,   # Trade both directions
    "close3": None,   # Trade both directions
}

# Trade direction per class: maps prediction class → "long" or "short"
# Classes absent from the dict are non-tradable (reject)
TRADE_DIRECTION = {
    "triple":    {0: "short", 2: "long"},
    "bull":      {1: "long"},
    "bear":      {1: "short"},
    "uncertain": {},                        # No directional info
    "close1":    {0: "short", 1: "long"},
    "close2":    {0: "short", 1: "long"},
    "close3":    {0: "short", 1: "long"},
}


def parse_close_horizon(prediction_target):
    """Parse closeN prediction target into horizon value.

    Args:
        prediction_target: e.g. "close1", "close2", "close3"

    Returns:
        int horizon if valid closeN target, None otherwise
    """
    if prediction_target and prediction_target.startswith("close"):
        suffix = prediction_target[5:]
        if suffix.isdigit():
            return int(suffix)
    return None


def get_labels_next_close(df, horizon=1, window_size=None, atr_period=None):
    """
    Compute binary next-close labels: Up=1 if Close[i+horizon] > Close[i], else Down=0.

    CRITICAL: Works on RAW data, not transformed data!

    Args:
        df: Raw DataFrame with OHLC data
        horizon: Number of candles ahead to compare (1, 2, 3...)
        window_size: Window size (default: from config)
        atr_period: ATR smoothing period (default: from config)

    Returns:
        Tuple: (labels, atr_array, confidence, None, None)
        - labels: numpy array {0, 1} aligned with valid indices
            - 0: Down (Close[i+horizon] <= Close[i])
            - 1: Up (Close[i+horizon] > Close[i])
        - atr_array: numpy array of ATR values for each labeled sample
        - confidence: numpy array of ones (no confidence notion for next-close)
        - long_pnl_R: None (no triple barrier P&L for next-close)
        - short_pnl_R: None (no triple barrier P&L for next-close)
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if atr_period is None:
        atr_period = config.ATR_PERIOD

    logger.info(f"Computing next-close labels (horizon={horizon})...")

    close = df['Close'].values.astype(np.float64)
    n_samples = len(df)

    # Compute ATR (needed for log_atr feature)
    atr_full = compute_atr(df, period=atr_period)

    valid_start = window_size
    valid_end = n_samples - horizon

    if valid_end <= valid_start:
        raise ValueError(f"Not enough data: need at least {window_size + horizon} candles, got {n_samples}")

    n_labels = valid_end - valid_start
    logger.debug(f"  Valid range: [{valid_start}, {valid_end})")
    logger.debug(f"  Computing labels for {n_labels} samples...")

    # Vectorized: compare close[i] with close[i + horizon]
    current_close = close[valid_start:valid_end]
    future_close = close[valid_start + horizon:valid_end + horizon]
    labels = (future_close > current_close).astype(np.int8)

    # ATR slice aligned with labels
    atr_values = atr_full[valid_start:valid_end].astype(np.float32)

    # No confidence notion for next-close
    confidence = np.ones(n_labels, dtype=np.float32)

    # Log distribution
    n_up = int(labels.sum())
    n_down = n_labels - n_up
    logger.info(f"\n  Label distribution (next-close, horizon={horizon}):")
    logger.info(f"     0 (Down          ): {n_down:6d} ({n_down / n_labels * 100:5.2f}%)")
    logger.info(f"     1 (Up            ): {n_up:6d} ({n_up / n_labels * 100:5.2f}%)")

    logger.info(f"✓ Next-close labels computed: {n_labels} samples")

    return labels, atr_values, confidence, None, None


def remap_labels(labels, prediction_target):
    """Remap triple-barrier labels {0,1,2} based on prediction target mode.

    Convention: class 1 = positive (what we detect), class 0 = everything else.
    Note: closeN targets never pass through remap (labels already binary).

    Args:
        labels: numpy array with values in {0, 1, 2}
        prediction_target: one of "triple", "bull", "bear", "uncertain"

    Returns:
        tuple: (remapped_labels, num_classes)
    """
    if prediction_target == "triple":
        return labels, 3

    new_labels = np.zeros_like(labels)

    if prediction_target == "bull":
        new_labels[labels == 2] = 1      # Bull -> 1, rest -> 0
    elif prediction_target == "bear":
        new_labels[labels == 0] = 1      # Bear -> 1, rest -> 0
    elif prediction_target == "uncertain":
        new_labels[labels == 1] = 1      # Uncertain -> 1, rest -> 0
    else:
        raise ValueError(f"Unknown PREDICTION_TARGET: {prediction_target}")

    return new_labels, 2


if __name__ == "__main__":
    # Test labeling
    print("\n=== Testing labeling.py ===\n")

    from modules.data.loader import get_raw_data

    path = sys.argv[1] if len(sys.argv) > 1 else config.INPUT_FILES[0]

    try:
        # Load data
        df = get_raw_data(path)
        print(f"✅ Loaded {len(df)} rows from {path}")

        # Compute labels
        labels, atr_array, confidence, _, _ = get_labels(df)

        print(f"\n✅ Labeling complete:")
        print(f"  - Total labels: {len(labels)}")
        print(f"  - Label values: {np.unique(labels)}")
        print(f"  - Shape: {labels.shape}")
        print(f"  - ATR array shape: {atr_array.shape}")
        print(f"  - ATR stats: min={atr_array.min():.4f}, max={atr_array.max():.4f}, mean={atr_array.mean():.4f}")
        print(f"  - Confidence shape: {confidence.shape}")
        print(f"  - Confidence stats: min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}")

        # Show some examples
        print(f"\n  First 20 labels: {labels[:20]}")
        print(f"  Last 20 labels: {labels[-20:]}")
        print(f"\n  First 20 ATR values: {atr_array[:20]}")
        print(f"  Last 20 ATR values: {atr_array[-20:]}")
        print(f"\n  First 20 confidence: {confidence[:20]}")
        print(f"  Last 20 confidence: {confidence[-20:]}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
