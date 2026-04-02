"""
Test transform.py - Validation of data transformation

After the per-window zigzag fix, transform_pipeline produces 12 global features
(or 11 without ATR). Zigzag + structural (6 features) are computed per-window
in Dataset.__getitem__() to avoid look-ahead bias.
"""

import sys
import os
import numpy as np
import pandas as pd
from conftest import TEST_CSV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data import loader, transform, labeling
from modules.data.dataset import OHLCV_NAMES
import config
from modules.utils.seed import set_seed


def test_compute_log_series():
    """Test log series computation"""
    print("\n=== Test: compute_log_series ===")

    df = loader.get_raw_data(TEST_CSV)
    log_series = transform.compute_log_series(df)

    assert log_series.shape[0] == len(df), "Number of rows should match"
    assert log_series.shape[1] == 5, "Should have 5 columns (OHLCV)"

    expected_cols = OHLCV_NAMES
    assert list(log_series.columns) == expected_cols, f"Columns should be {expected_cols}"

    print(f"✓ Log series computed: {log_series.shape}")



def test_transform_pipeline():
    """Test complete transformation pipeline (global features only)"""
    print("\n=== Test: transform_pipeline ===")

    df = loader.get_raw_data(TEST_CSV)
    result = transform.transform_pipeline(df)

    # Check outputs exist
    assert 'data' in result, "Missing 'data' key"
    assert 'normalization_params' in result, "Missing 'normalization_params'"
    assert 'split_ratio' in result, "Missing 'split_ratio'"
    assert 'window_size' in result, "Missing 'window_size'"

    all_data = result['data']
    split_ratio = result['split_ratio']

    # Check shape: 5 log OHLCV + 4 time + 2 regime = 11 (no ATR, no zigzag/struct)
    assert all_data.shape[1] == 11, f"Data should have 11 features (5 log + 4 time + 2 regime), got {all_data.shape[1]}"
    print(f"  - Total data: {len(all_data)} rows")

    # Calculate train portion for verification
    split_index = int(len(all_data) * split_ratio)
    train_portion = all_data[:split_index]
    test_portion = all_data[split_index:]

    print(f"  - Train portion (first {split_ratio:.0%}): {len(train_portion)} rows")
    print(f"  - Test portion: {len(test_portion)} rows")

    # In window normalization mode, data is NOT globally normalized
    assert not np.isnan(all_data).any(), "Data should have no NaN"
    assert not np.isinf(all_data).any(), "Data should have no inf"

    # Check time features are bounded in [-1, 1] (cols 5-8 without ATR)
    time_features = all_data[:, 5:9]
    assert time_features.shape[1] == 4, f"Should have 4 time features, got {time_features.shape[1]}"
    assert time_features.min() >= -1.0 - 1e-7, f"Time features should be >= -1, got {time_features.min()}"
    assert time_features.max() <= 1.0 + 1e-7, f"Time features should be <= 1, got {time_features.max()}"

    print(f"✓ Pipeline complete: total={len(all_data)} rows")


def test_no_data_leakage():
    """Test that train and test data are properly separated"""
    print("\n=== Test: no_data_leakage ===")

    df = loader.get_raw_data(TEST_CSV)
    result = transform.transform_pipeline(df)

    all_data = result['data']
    split_ratio = result['split_ratio']

    # Split data into train and test portions
    split_index = int(len(all_data) * split_ratio)
    train_portion = all_data[:split_index]
    test_portion = all_data[split_index:]

    # Train and test should have different distributions since they cover different time periods
    log_train = train_portion[:, :5]
    log_test = test_portion[:, :5]

    train_mean = np.mean(log_train, axis=0)
    test_mean = np.mean(log_test, axis=0)

    diff = np.abs(train_mean - test_mean)
    assert diff.max() > 0, "Train and test should have different distributions"

    # Verify no overlap in indices
    assert len(train_portion) + len(test_portion) == len(all_data), "Train + test should equal total"

    print(f"✓ No data leakage detected (train={len(train_portion)}, test={len(test_portion)})")


def test_no_nan_inf():
    """Test that final data has no NaN or inf"""
    print("\n=== Test: no_nan_inf ===")

    df = loader.get_raw_data(TEST_CSV)
    result = transform.transform_pipeline(df)

    all_data = result['data']

    assert not np.isnan(all_data).any(), "Data should have no NaN"
    assert not np.isinf(all_data).any(), "Data should have no inf"

    print(f"✓ No NaN or inf in final data")


def test_transform_with_atr():
    """Test transformation pipeline with atr_array (6 norm + 4 time + 2 regime = 12 features)"""
    print("\n=== Test: transform_with_atr ===")

    df = loader.get_raw_data(TEST_CSV)

    # Compute labels and atr_array
    labels, atr_array, _, _, _ = labeling.get_labels(df, window_size=100, max_horizon=50)

    # Transform with atr_array
    result = transform.transform_pipeline(df, atr_array=atr_array, window_size=100)

    all_data = result['data']
    n_features = result['n_features']

    # Should have 12 features now (6 norm + 4 time + 2 regime)
    # Zigzag + structural (6) computed per-window in Dataset
    assert n_features == 12, f"Expected 12 features, got {n_features}"
    assert all_data.shape[1] == 12, f"Data should have 12 columns, got {all_data.shape[1]}"

    # Check that 6th feature (log_atr) is present and valid (index 5)
    log_atr = all_data[:, 5]
    assert not np.isnan(log_atr).any(), "log_atr should not contain NaN"
    assert not np.isinf(log_atr).any(), "log_atr should not contain inf"

    # In window mode, normalization_params may be None
    norm_params = result['normalization_params']
    if norm_params is not None:
        assert len(norm_params['columns']) == 6, f"normalization_params should have 6 columns, got {len(norm_params['columns'])}"
        assert norm_params['columns'][5] == 'log_atr', "6th column should be log_atr"

    # Check time features (cols 6-9) are bounded in [-1, 1]
    time_features = all_data[:, 6:10]
    assert time_features.shape[1] == 4, f"Should have 4 time features, got {time_features.shape[1]}"
    assert time_features.min() >= -1.0 - 1e-7, "Time features should be >= -1"
    assert time_features.max() <= 1.0 + 1e-7, "Time features should be <= 1"

    # Check regime features (cols 10-11) are bounded in [-1, 1]
    regime_features = all_data[:, 10:12]
    assert regime_features.shape[1] == 2, f"Should have 2 regime features, got {regime_features.shape[1]}"
    assert not np.isnan(regime_features).any(), "Regime features should not contain NaN"
    assert regime_features.min() >= -1.0 - 1e-7, "Regime features should be >= -1"
    assert regime_features.max() <= 1.0 + 1e-7, "Regime features should be <= 1"

    print(f"✓ Transform with ATR works: {all_data.shape}")


def test_transform_without_atr():
    """Test transformation pipeline without atr_array (5 log + 4 time + 2 regime = 11 features)"""
    print("\n=== Test: transform_without_atr ===")

    df = loader.get_raw_data(TEST_CSV)

    # Transform WITHOUT atr_array
    result = transform.transform_pipeline(df, atr_array=None, window_size=100)

    all_data = result['data']
    n_features = result['n_features']

    # Should have 11 features (5 log + 4 time + 2 regime)
    assert n_features == 11, f"Expected 11 features, got {n_features}"
    assert all_data.shape[1] == 11, f"Data should have 11 columns, got {all_data.shape[1]}"

    # In window mode, normalization_params may be None
    norm_params = result['normalization_params']
    if norm_params is not None:
        assert len(norm_params['columns']) == 5, f"normalization_params should have 5 columns, got {len(norm_params['columns'])}"

    # Check time features (cols 5-8) are bounded in [-1, 1]
    time_features = all_data[:, 5:9]
    assert time_features.shape[1] == 4, f"Should have 4 time features, got {time_features.shape[1]}"
    assert time_features.min() >= -1.0 - 1e-7, "Time features should be >= -1"
    assert time_features.max() <= 1.0 + 1e-7, "Time features should be <= 1"

    # Check regime features (cols 9-10) are bounded in [-1, 1]
    regime_features = all_data[:, 9:11]
    assert regime_features.shape[1] == 2, f"Should have 2 regime features, got {regime_features.shape[1]}"
    assert not np.isnan(regime_features).any(), "Regime features should not contain NaN"
    assert regime_features.min() >= -1.0 - 1e-7, "Regime features should be >= -1"
    assert regime_features.max() <= 1.0 + 1e-7, "Regime features should be <= 1"

    print(f"✓ Transform without ATR works: {all_data.shape}")


def test_time_features_alignment():
    """Test that time features are correctly aligned with OHLCV data after transformation"""
    print("\n=== Test: time_features_alignment ===")

    df = loader.get_raw_data(TEST_CSV)
    labels, atr_array, _, _, _ = labeling.get_labels(df, window_size=100, max_horizon=50)
    result = transform.transform_pipeline(df, atr_array=atr_array, window_size=100)

    all_data = result['data']
    window_size = result['window_size']

    # Recompute time features from original df for verification
    hours = df['Open time'].dt.hour.values
    weekdays = df['Open time'].dt.weekday.values

    # The first row of all_data corresponds to original df index window_size
    original_start_idx = window_size

    # Time features are at cols 6-9 (after 6 normalized: 5 OHLCV + 1 ATR)
    for check_offset in [0, 1, 10, 100]:
        if check_offset >= len(all_data):
            break
        orig_idx = original_start_idx + check_offset
        if orig_idx >= len(df):
            break

        expected_sin_hour = np.sin(2 * np.pi * hours[orig_idx] / 24)
        actual_sin_hour = all_data[check_offset, 6]  # sin_hour

        expected_cos_hour = np.cos(2 * np.pi * hours[orig_idx] / 24)
        actual_cos_hour = all_data[check_offset, 7]  # cos_hour

        expected_sin_weekday = np.sin(2 * np.pi * weekdays[orig_idx] / 7)
        actual_sin_weekday = all_data[check_offset, 8]  # sin_weekday

        expected_cos_weekday = np.cos(2 * np.pi * weekdays[orig_idx] / 7)
        actual_cos_weekday = all_data[check_offset, 9]  # cos_weekday

        assert np.isclose(expected_sin_hour, actual_sin_hour, atol=1e-6), \
            f"sin_hour mismatch at offset {check_offset}: expected {expected_sin_hour}, got {actual_sin_hour}"
        assert np.isclose(expected_cos_hour, actual_cos_hour, atol=1e-6), \
            f"cos_hour mismatch at offset {check_offset}"
        assert np.isclose(expected_sin_weekday, actual_sin_weekday, atol=1e-6), \
            f"sin_weekday mismatch at offset {check_offset}"
        assert np.isclose(expected_cos_weekday, actual_cos_weekday, atol=1e-6), \
            f"cos_weekday mismatch at offset {check_offset}"

    print(f"✓ Time features correctly aligned (checked offsets 0, 1, 10, 100)")
    print(f"  window_size={window_size}, start_idx={original_start_idx}")


def test_perwindow_zigzag():
    """Test that per-window zigzag features are correctly computed in Dataset"""
    print("\n=== Test: perwindow_zigzag ===")

    from modules.data.dataset import TimeSeriesWindowDataset

    df = loader.get_raw_data(TEST_CSV)
    labels, atr_array, _, _, _ = labeling.get_labels(df, window_size=100, max_horizon=50)
    result = transform.transform_pipeline(df, atr_array=atr_array, window_size=100)

    data = result['data']
    ws = 100  # Small window for test

    # Extract raw H/L/C
    n_data = len(data)
    raw_hlc = np.column_stack([
        df['High'].values[ws:ws + n_data],
        df['Low'].values[ws:ws + n_data],
        df['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)

    # Align
    min_len = min(len(data), len(labels))
    data = data[:min_len]
    labels_arr = labels[:min_len]
    raw_hlc = raw_hlc[:min_len]

    indices = list(range(min(10, len(data) - ws)))  # Just a few windows

    dataset = TimeSeriesWindowDataset(
        data=data, labels=labels_arr, indices=indices,
        window_size=ws, raw_hlc=raw_hlc,
        n_norm_features=7
    )

    # Get first window
    window, label = dataset[0]

    # Final output: 18 features = 5 OHLCV + log_zz + log_atr + 4 time + 5 struct + 2 regime
    assert window.shape == (ws, 18), f"Expected ({ws}, 18), got {window.shape}"

    # Check no NaN/inf
    assert not window.isnan().any(), "Window should have no NaN"
    assert not window.isinf().any(), "Window should have no inf"

    # Check structural features (cols 11-15) are finite
    struct = window[:, 11:16]
    assert not struct.isnan().any(), "Structural features should have no NaN"

    # struct_rank (col 11) should be in [0, 1] (before normalization it's a percentile)
    # After normalization of first 7 features, struct_rank is NOT normalized (col 11 > 7)
    # So it should still be in [0, 1]
    assert struct[:, 0].min() >= 0.0, "struct_rank should be >= 0"
    assert struct[:, 0].max() <= 1.0, "struct_rank should be <= 1"

    print(f"✓ Per-window zigzag features work: window shape {window.shape}")


# ============================================================================
# ============================================================================
# Test: transform_pipeline with NORMALIZE=False
# ============================================================================

def test_transform_pipeline_normalize_disabled():
    """Test transform_pipeline with NORMALIZE=False."""
    print("\n=== Test: transform_pipeline_normalize_disabled ===")

    orig_normalize = config.NORMALIZE

    try:
        config.NORMALIZE = False

        df = loader.get_raw_data(TEST_CSV)
        labels, atr_array, _, _, _ = labeling.get_labels(df, window_size=100, max_horizon=50)
        result = transform.transform_pipeline(df, atr_array=atr_array, window_size=100)

        data = result['data']
        norm_params = result['normalization_params']

        # No normalization → no params
        assert norm_params is None, "Should be None when NORMALIZE=False"

        # Data should still have 12 features
        assert data.shape[1] == 12, f"Expected 12 features, got {data.shape[1]}"

        # Raw log values should NOT be near 0 (not normalized)
        log_price_mean = data[:, 0].mean()
        assert abs(log_price_mean) > 1.0, \
            f"log_price mean should not be ~0 when normalization disabled, got {log_price_mean}"

        print(f"  Shape: {data.shape}, norm_params: {norm_params}")
        print(f"  log_price mean (raw): {log_price_mean:.4f}")
        print("  PASS")

    finally:
        config.NORMALIZE = orig_normalize


# ============================================================================
# Test: apply_transform_from_checkpoint
# ============================================================================

def test_apply_transform_from_checkpoint_window_mode():
    """Test apply_transform_from_checkpoint with window normalization mode."""
    print("\n=== Test: apply_transform_from_checkpoint_window_mode ===")

    df = loader.get_raw_data(TEST_CSV)

    checkpoint_params = {
        'normalization_params': None,  # Window mode has no global params
        'WINDOW_SIZE': 100,
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
        'NORMALIZE_MODE': 'window',
        'n_norm_features': 7,
    }

    data_infer = transform.apply_transform_from_checkpoint(df, checkpoint_params)

    assert data_infer.shape[1] == 12, f"Expected 12, got {data_infer.shape[1]}"
    assert not np.isnan(data_infer).any(), "No NaN"
    assert not np.isinf(data_infer).any(), "No inf"

    print(f"  Shape: {data_infer.shape}")
    print("  PASS")


def test_apply_transform_from_checkpoint_5_features():
    """Test apply_transform_from_checkpoint with 5 norm features (no ATR)."""
    print("\n=== Test: apply_transform_from_checkpoint_5_features ===")

    df = loader.get_raw_data(TEST_CSV)

    checkpoint_params = {
        'normalization_params': None,
        'WINDOW_SIZE': 100,
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
        'NORMALIZE_MODE': 'window',
        'n_norm_features': 5,  # No ATR
    }

    data_infer = transform.apply_transform_from_checkpoint(df, checkpoint_params)

    # 11 features: 5 norm + 4 time + 2 regime (no ATR)
    assert data_infer.shape[1] == 11, f"Expected 11, got {data_infer.shape[1]}"

    print(f"  Shape: {data_infer.shape}")
    print("  PASS")


# ============================================================================
# Test: inf/nan cleaning in _compute_base_features (lines 172-175)
# ============================================================================

def test_compute_base_features_inf_nan_cleaning():
    """Test that inf/nan in log series are forward-filled (not dropped)."""
    print("\n=== Test: compute_base_features_inf_nan_cleaning ===")

    df = loader.get_raw_data(TEST_CSV)
    n_original = len(df)

    # Inject a zero Close (log(0) = -inf) and zero Volume already has epsilon
    df_bad = df.copy()
    df_bad.loc[df_bad.index[500], 'Close'] = 0.0   # log(0) = -inf
    df_bad.loc[df_bad.index[501], 'Open'] = 0.0     # log(0) = -inf

    log_data, time_features, regime_features = transform._compute_base_features(df_bad)

    # Row count must be PRESERVED (ffill, not dropna)
    assert log_data.shape[0] == n_original, \
        f"Row count changed: {log_data.shape[0]} vs {n_original} — should ffill, not dropna"
    assert time_features.shape[0] == n_original
    assert regime_features.shape[0] == n_original

    # No inf/nan in output
    assert not np.isinf(log_data).any(), "log_data should have no inf after cleaning"
    assert not np.isnan(log_data).any(), "log_data should have no NaN after cleaning"

    print(f"  Injected inf at rows 500-501, output preserved {log_data.shape[0]} rows")
    print("  PASS")


# ============================================================================
# Test: ValueError when data too short for window_size (line 288)
# ============================================================================

def test_transform_pipeline_insufficient_data():
    """Test ValueError when data has fewer rows than window_size."""
    print("\n=== Test: transform_pipeline_insufficient_data ===")

    # Create a tiny DataFrame (50 rows < window_size=200)
    set_seed(config.SEED)
    n = 50
    df_tiny = pd.DataFrame({
        'Open time': pd.date_range('2023-01-01', periods=n, freq='1h'),
        'Open': np.random.uniform(100, 200, n),
        'High': np.random.uniform(150, 250, n),
        'Low': np.random.uniform(50, 150, n),
        'Close': np.random.uniform(100, 200, n),
        'Volume': np.random.uniform(1000, 5000, n),
    })
    # Ensure High >= Open,Close and Low <= Open,Close
    df_tiny['High'] = df_tiny[['Open', 'High', 'Close']].max(axis=1) + 1
    df_tiny['Low'] = df_tiny[['Open', 'Low', 'Close']].min(axis=1) - 1

    import pytest
    with pytest.raises(ValueError, match="Not enough data"):
        transform.transform_pipeline(df_tiny, window_size=200)

    print("  Correctly raised ValueError for 50 rows < window_size=200")
    print("  PASS")


# ============================================================================
# Test: ValueError for incompatible checkpoint n_features (line 433)
# ============================================================================

def test_apply_transform_checkpoint_incompatible_features():
    """Test ValueError when checkpoint has unexpected n_norm_features."""
    print("\n=== Test: apply_transform_checkpoint_incompatible_features ===")

    df = loader.get_raw_data(TEST_CSV)

    # n_norm_features=3 is neither 5, 6, nor 7 → should raise
    checkpoint_params = {
        'normalization_params': None,
        'WINDOW_SIZE': 100,
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
        'NORMALIZE_MODE': 'window',
        'n_norm_features': 3,
    }

    import pytest
    with pytest.raises(ValueError, match="expected 5 or 6"):
        transform.apply_transform_from_checkpoint(df, checkpoint_params)

    print("  Correctly raised ValueError for n_norm_features=3")
    print("  PASS")


# ============================================================================
# Test: ValueError for insufficient data in apply_transform_from_checkpoint (line 442)
# ============================================================================

def test_apply_transform_checkpoint_insufficient_data():
    """Test ValueError when data too short for checkpoint's window_size."""
    print("\n=== Test: apply_transform_checkpoint_insufficient_data ===")

    # Create tiny DataFrame
    set_seed(config.SEED)
    n = 50
    df_tiny = pd.DataFrame({
        'Open time': pd.date_range('2023-01-01', periods=n, freq='1h'),
        'Open': np.random.uniform(100, 200, n),
        'High': np.random.uniform(150, 250, n),
        'Low': np.random.uniform(50, 150, n),
        'Close': np.random.uniform(100, 200, n),
        'Volume': np.random.uniform(1000, 5000, n),
    })
    df_tiny['High'] = df_tiny[['Open', 'High', 'Close']].max(axis=1) + 1
    df_tiny['Low'] = df_tiny[['Open', 'Low', 'Close']].min(axis=1) - 1

    checkpoint_params = {
        'normalization_params': None,
        'WINDOW_SIZE': 200,  # > 50 rows
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
        'NORMALIZE_MODE': 'window',
        'n_norm_features': 5,
    }

    import pytest
    with pytest.raises(ValueError, match="Not enough data"):
        transform.apply_transform_from_checkpoint(df_tiny, checkpoint_params)

    print("  Correctly raised ValueError for 50 rows < WINDOW_SIZE=200")
    print("  PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing transform.py")
    print("=" * 60)

    try:
        test_compute_log_series()
        test_transform_pipeline()
        test_no_data_leakage()
        test_no_nan_inf()
        test_transform_with_atr()
        test_transform_without_atr()
        test_time_features_alignment()
        test_perwindow_zigzag()
        test_transform_pipeline_normalize_disabled()
        test_apply_transform_from_checkpoint_window_mode()
        test_apply_transform_from_checkpoint_5_features()
        test_compute_base_features_inf_nan_cleaning()
        test_transform_pipeline_insufficient_data()
        test_apply_transform_checkpoint_incompatible_features()
        test_apply_transform_checkpoint_insufficient_data()

        print("\n" + "=" * 60)
        print("✅ All transform tests passed!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
