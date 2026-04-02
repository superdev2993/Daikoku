"""
Data transformation module - Log transformation, normalization
"""

import numpy as np
import pandas as pd
import os

import config
from modules.utils.logger import get_logger
from modules.data.regime import compute_regime_filter

logger = get_logger(__name__)


def compute_time_features(df):
    """
    Extract cyclical time features from Open time column.

    Returns 4 features bounded in [-1, 1], NOT to be normalized:
    - sin(2π × hour/24), cos(2π × hour/24)
    - sin(2π × weekday/7), cos(2π × weekday/7)

    Args:
        df: DataFrame with 'Open time' datetime column

    Returns:
        numpy array of shape (n_rows, 4)
    """
    hours = df['Open time'].dt.hour + df['Open time'].dt.minute / 60.0
    weekdays = df['Open time'].dt.weekday  # 0=Monday, 6=Sunday

    time_features = np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7),
    ])

    logger.debug(f"✓ 4 cyclical time features extracted (shape: {time_features.shape})")
    return time_features


def compute_log_series(df):
    """
    Compute 5 log-transformed series from OHLCV data

    Args:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        DataFrame with 5 columns:
        - log_price: log(Close)
        - log_open: log(Open)
        - log_wick_high: log(High / max(Open, Close))
        - log_wick_low: log(min(Open, Close) / Low)
        - log_volume: log(Volume + epsilon)
    """
    logger.info("Computing log series...")

    # Extract OHLCV
    O = df['Open'].values
    H = df['High'].values
    L = df['Low'].values
    C = df['Close'].values
    V = df['Volume'].values

    # Compute 5 series
    log_price = np.log(C)
    log_open = np.log(O)

    # Wicks computation
    body_max = np.maximum(O, C)
    body_min = np.minimum(O, C)

    # log(High / max(Open, Close))
    log_wick_high = np.log(H / body_max)

    # log(min(Open, Close) / Low)
    log_wick_low = np.log(body_min / L)

    # Epsilon to avoid log(0) if Volume=0
    epsilon = 1e-10
    log_volume = np.log(V + epsilon)

    # Create DataFrame (CRITICAL: Order must match everywhere)
    result = pd.DataFrame({
        'log_price': log_price,
        'log_open': log_open,
        'log_wick_high': log_wick_high,
        'log_wick_low': log_wick_low,
        'log_volume': log_volume
    })

    logger.debug(f"✓ 5 log series computed (shape: {result.shape})")
    logger.debug(f"  Volume stats: min={V.min():.2f}, max={V.max():.2f}, mean={V.mean():.2f}")
    return result



def _compute_base_features(df, regime_length=None):
    """
    Compute base log features (5 OHLCV) and time/regime features from raw DataFrame.

    Zigzag and structural features are computed per-window in the Dataset layer
    to avoid look-ahead bias.

    Shared core between transform_pipeline and apply_transform_from_checkpoint.

    Args:
        df: DataFrame with OHLCV data
        regime_length: HMA period for regime filter. None = use config.REGIME_LENGTH

    Returns:
        tuple: (log_data, time_features, regime_features)
            - log_data: numpy array (n, 5) of log-transformed OHLCV features
            - time_features: numpy array (n, 4) of cyclical time features
            - regime_features: numpy array (n, 2) of HMA regime features
    """
    if regime_length is None:
        regime_length = config.REGIME_LENGTH

    # Extract cyclical time features from original timestamps
    time_features = compute_time_features(df)

    # Compute 5 log series (OHLCV)
    log_series = compute_log_series(df)

    # Clean inf/nan values from log transformation
    # CRITICAL: use ffill instead of dropna() to preserve row alignment
    # dropna() removes rows, breaking alignment with labels, raw_hlc, time/regime features
    n_bad = (np.isinf(log_series.values) | np.isnan(log_series.values)).any(axis=1).sum()
    if n_bad > 0:
        logger.warning(f"Found {n_bad} rows with inf/nan in log series, forward-filling...")
        log_series = log_series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        logger.debug(f"After cleaning: {len(log_series)} rows (preserved alignment)")

    # Convert to numpy
    log_data = log_series.values if isinstance(log_series, pd.DataFrame) else log_series

    logger.debug(f"Log data: {log_data.shape[0]} rows, {log_data.shape[1]} features")

    # Compute regime filter features
    logger.info("Computing regime filter (HMA)...")
    trend, voltrend = compute_regime_filter(df, length=regime_length)
    # Scale to [-1, +1] (original is [-10, +10])
    regime_features = np.column_stack([trend / 10.0, voltrend / 10.0])
    logger.debug(f"✓ 2 regime features computed (shape: {regime_features.shape})")

    return log_data, time_features, regime_features


def _append_context_features(data, time_features_all, window_size, regime_features_all=None):
    """
    Append time (4) and regime (2) features to normalized data.

    Structural features (5) and zigzag interpolation (1) are computed per-window
    in the Dataset layer to avoid look-ahead bias.

    Shared tail between transform_pipeline and apply_transform_from_checkpoint.

    Args:
        data: Normalized data array (n_after_skip, 5 or 6)
        time_features_all: Full time features array (original length)
        window_size: Window size (for alignment offset)
        regime_features_all: Optional regime features array (original length, 2 cols)

    Returns:
        numpy array with time + regime features appended
    """
    # Add cyclical time features (NOT normalized)
    logger.debug("Adding cyclical time features (sin/cos hour + weekday)...")
    time_features_aligned = time_features_all[window_size:]

    min_len = min(len(data), len(time_features_aligned))
    data = data[:min_len]
    time_features_aligned = time_features_aligned[:min_len]

    n_before = data.shape[1]
    data = np.column_stack([data, time_features_aligned])
    logger.debug(f"✓ Time features added: {n_before} + 4 = {data.shape[1]} total features")

    # Add regime features (NOT normalized, already scaled to [-1, +1])
    if regime_features_all is not None:
        logger.debug("Adding regime features (trend + voltrend)...")
        regime_aligned = regime_features_all[window_size:]
        min_len = min(len(data), len(regime_aligned))
        data = data[:min_len]
        regime_aligned = regime_aligned[:min_len]
        # Replace NaN (warmup) with 0
        regime_aligned = np.nan_to_num(regime_aligned, nan=0.0)
        data = np.column_stack([data, regime_aligned])
        logger.debug(f"✓ Regime features added: 2 features (total: {data.shape[1]})")

    return data


def transform_pipeline(df, atr_array=None, split_ratio=None, window_size=None):
    """
    Complete transformation pipeline

    Pipeline order:
    1. Compute 5 log series (OHLCV)
    2. Add log_atr as 6th feature if provided
    3. Skip initial window_size rows
    4. Add cyclical time features (4, NOT normalized)
    5. Add regime features (2, NOT normalized)

    Per-window features (zigzag + 5 structural) are added in Dataset layer.

    Args:
        df: Raw DataFrame with OHLCV data
        atr_array: Optional ATR values (adds 6th normalized feature)
        split_ratio: Unused (kept for API compatibility)
        window_size: Window size to skip at start (default: from config)

    Returns:
        Dictionary containing:
        - data: Transformed data array (n, 12) = 6 norm + 4 time + 2 regime
        - normalization_params: Always None (per-window normalization in Dataset)
        - n_features: Total feature count (12)
        - n_norm_features: Features to normalize per-window (5 or 6)
        - split_ratio, window_size, normalize_mode
    """
    if split_ratio is None:
        split_ratio = config.TRAIN_TEST_SPLIT

    if window_size is None:
        window_size = config.WINDOW_SIZE

    logger.info("=" * 60)
    logger.info("Starting transformation pipeline")
    logger.info("=" * 60)

    # Step 1: Compute base log features (5 OHLCV) + regime
    # Zigzag + structural computed per-window in Dataset to avoid look-ahead bias
    log_data, time_features_all, regime_features = _compute_base_features(df)

    # Validation: Should have 5 features (5 OHLCV, no zigzag)
    assert log_data.shape[1] == 5, f"Expected 5 features, got {log_data.shape[1]}"

    # Determine expected number of features (5 base + optional ATR)
    expected_features = 6 if atr_array is not None else 5

    # Step 2: Skip initial window
    if len(log_data) <= window_size:
        raise ValueError(f"Not enough data after transformation. Need > {window_size}, got {len(log_data)}")

    data_after_skip = log_data[window_size:]
    logger.debug(f"After skip: {len(data_after_skip)} rows")

    # Add log_atr as 6th feature AFTER skip (for correct alignment)
    if atr_array is not None:
        logger.debug("Adding log_atr as 6th feature...")

        # Apply log transformation (epsilon to avoid log(0))
        epsilon = 1e-10
        log_atr = np.log(atr_array + epsilon)

        # Alignment: data_after_skip[0] corresponds to index window_size of original DF
        # atr_array[0] corresponds to index window_size of original DF
        # Direct alignment (no nan_offset)
        log_atr_aligned = log_atr

        # Truncate to common length
        min_len = min(len(data_after_skip), len(log_atr_aligned))
        data_after_skip = data_after_skip[:min_len]
        log_atr_aligned = log_atr_aligned[:min_len]

        # Concatenate: (n, 5) + (n, 1) → (n, 6)
        data_after_skip = np.column_stack([data_after_skip, log_atr_aligned])

        logger.debug(f"✓ Added log_atr: new shape {data_after_skip.shape}")
        logger.debug(f"  log_atr stats: min={log_atr_aligned.min():.4f}, max={log_atr_aligned.max():.4f}, mean={log_atr_aligned.mean():.4f}")

    # Step 3: Window normalization — skip global normalization, each window normalizes itself in Dataset
    n_norm_features = data_after_skip.shape[1]  # Features to normalize per-window (6 or 7)
    logger.info("Window normalization mode: skipping global normalization")
    logger.debug(f"  Each window will normalize its first {n_norm_features} features independently")
    all_data_normalized = data_after_skip
    normalization_params = None

    # Validation: normalized OHLCV features count
    assert all_data_normalized.shape[1] == expected_features, f"Expected {expected_features} OHLCV features after normalization, got {all_data_normalized.shape[1]}"

    # Step 4: Append time + regime features (structural computed per-window in Dataset)
    all_data_final = _append_context_features(
        all_data_normalized, time_features_all, window_size,
        regime_features_all=regime_features
    )
    total_features = all_data_final.shape[1]

    # Final validation: norm + time + regime (no structural — computed per-window)
    expected_total = expected_features + 4 + 2  # norm + time + regime
    assert all_data_final.shape[1] == expected_total, f"Expected {expected_total} total features, got {all_data_final.shape[1]}"

    logger.info("=" * 60)
    logger.info("Transformation pipeline complete")
    logger.info(f"  - Total transformed data: {len(all_data_final)} rows")
    logger.info(f"  - Features: {total_features} ({expected_features} normalized + 4 time + 2 regime)")
    logger.info(f"  - Note: zigzag (1) + structural (5) = 6 features computed per-window in Dataset")
    logger.info("=" * 60)

    return {
        'data': all_data_final,
        'normalization_params': normalization_params,
        'n_features': total_features,
        'split_ratio': split_ratio,
        'window_size': window_size,
        'normalize_mode': "window",
        'n_norm_features': n_norm_features  # Features to normalize per-window (6 or 7)
    }


def apply_transform_from_checkpoint(df, checkpoint_params):
    """
    Apply pre-calibrated transformation for inference/evaluation.

    Uses parameters from checkpoint instead of recalculating them.

    Args:
        df: Raw DataFrame with OHLCV data
        checkpoint_params: Dict containing:
            - WINDOW_SIZE, REGIME_LENGTH, ATR_PERIOD
            - n_norm_features: number of normalized features (5 or 6)
            - normalization_params: legacy, typically None

    Returns:
        numpy array: Transformed data (n, 12) = 6 norm + 4 time + 2 regime
    """
    logger.info("=" * 60)
    logger.info("Applying pre-calibrated transformation (inference mode)")
    logger.info("=" * 60)

    # Extract parameters from checkpoint
    normalization_params = checkpoint_params.get('normalization_params')
    window_size = checkpoint_params['WINDOW_SIZE']

    # Detect number of normalized features from checkpoint (5 or 6 global)
    # Checkpoint may store 7 (assembled: 5 OHLCV + log_zz + log_atr) — log_zz is per-window
    if normalization_params is not None:
        n_features = len(normalization_params['mean'])
    else:
        n_features = checkpoint_params['n_norm_features']
    if n_features == 7:
        # Per-window pipeline: 7th feature (log_zz) added per-window in Dataset, not here
        n_features = 6
    if n_features not in [5, 6]:
        raise ValueError(f"Checkpoint incompatible: expected 5 or 6 global norm features, got {n_features}")
    logger.debug(f"Checkpoint expects {n_features} global normalized features")

    # Step 1: Compute base log features (5 OHLCV) + regime (use checkpoint's REGIME_LENGTH)
    regime_length = checkpoint_params['REGIME_LENGTH']
    log_data, time_features_all, regime_features = _compute_base_features(df, regime_length=regime_length)

    # Step 2: Skip initial window
    if len(log_data) <= window_size:
        raise ValueError(f"Not enough data after transformation. Need > {window_size}, got {len(log_data)}")

    data_after_skip = log_data[window_size:]
    logger.debug(f"After skip: {len(data_after_skip)} rows")

    # Step 2b: Add log_atr as 6th feature if checkpoint expects it
    if n_features == 6:
        from modules.data.labeling import compute_atr
        atr_period = checkpoint_params['ATR_PERIOD']
        logger.debug(f"Adding log_atr as 6th feature (checkpoint expects 6, ATR_PERIOD={atr_period})...")
        atr_full = compute_atr(df, period=atr_period)
        atr_slice = atr_full[window_size:]
        # Replace NaN with 0 (should not happen if window_size > atr_period)
        atr_slice = np.nan_to_num(atr_slice, nan=0.0)
        log_atr = np.log(atr_slice + 1e-10)
        # Direct alignment (no nan_offset)
        min_len = min(len(data_after_skip), len(log_atr))
        data_after_skip = data_after_skip[:min_len]
        log_atr = log_atr[:min_len]
        data_after_skip = np.column_stack([data_after_skip, log_atr])
        logger.debug(f"✓ Added log_atr: new shape {data_after_skip.shape}")

    # Step 3: Window normalization — skip global normalization (per-window at dataset level)
    normalized_data = data_after_skip
    logger.info("Window normalization mode: skipping global normalization (per-window at dataset level)")

    # Step 4: Append time + regime features (structural computed per-window in Dataset)
    data_final = _append_context_features(
        normalized_data, time_features_all, window_size,
        regime_features_all=regime_features
    )

    logger.info("=" * 60)
    logger.info(f"Transformation complete: {len(data_final)} rows, {data_final.shape[1]} features")
    logger.info("=" * 60)

    return data_final


if __name__ == "__main__":
    import sys
    import traceback
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Test transformation pipeline
    print("\n=== Testing transform.py ===\n")

    from modules.data.loader import get_raw_data

    path = sys.argv[1] if len(sys.argv) > 1 else config.INPUT_FILES[0]

    try:
        # Load data
        df = get_raw_data(path)
        print(f"✅ Loaded {len(df)} rows from {path}")

        # Transform
        result = transform_pipeline(df)

        print(f"\n✅ Transformation complete:")
        print(f"  - Data shape: {result['data'].shape}")
        print(f"  - Split ratio: {result['split_ratio']}")
        print(f"  - Normalization params: {result['normalization_params'] is not None}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
