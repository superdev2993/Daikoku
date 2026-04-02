"""
Test aggregation.py - Validation of candle aggregation for multi-timeframe
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from conftest import TEST_CSV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.utils.seed import set_seed
from modules.data.aggregation import aggregate_candles
from modules.data import loader


# ============================================================================
# Helpers
# ============================================================================

def make_test_df(n_rows=100, start_hour=0):
    """Create a synthetic 1h OHLCV DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='1h')
    set_seed(config.SEED)
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    open_ = close + np.random.randn(n_rows) * 0.2
    high = np.maximum(open_, close) + np.abs(np.random.randn(n_rows)) * 0.3
    low = np.minimum(open_, close) - np.abs(np.random.randn(n_rows)) * 0.3
    volume = np.random.rand(n_rows) * 1000 + 100

    df = pd.DataFrame({
        'Open time': dates,
        'Open': open_.astype(np.float32),
        'High': high.astype(np.float32),
        'Low': low.astype(np.float32),
        'Close': close.astype(np.float32),
        'Volume': volume.astype(np.float32),
    })
    return df


# ============================================================================
# Tests
# ============================================================================

def test_aggregate_structured():
    """Test structured alignment groups by temporal boundaries."""
    df = make_test_df(48)  # 48h = 2 days
    result, end_ts, _ = aggregate_candles(df, divisor=4, align="structured")

    # 48 hours with divisor 4 = 12 groups
    assert len(result) == 12, f"Expected 12 candles, got {len(result)}"
    assert len(end_ts) == 12, f"Expected 12 end_timestamps, got {len(end_ts)}"

    # Check that each group starts at hour % 4 == 0
    hours = result['Open time'].dt.hour
    assert (hours % 4 == 0).all(), f"Structured groups should start at hours divisible by 4: {hours.tolist()}"

    # End timestamps must be >= start timestamps
    for i in range(len(result)):
        assert end_ts[i] >= result['Open time'].iloc[i], \
            f"Group {i}: end_ts {end_ts[i]} < start_ts {result['Open time'].iloc[i]}"

    print(f"✓ test_aggregate_structured passed ({len(result)} candles)")


def test_aggregate_unstructured():
    """Test unstructured alignment groups by consecutive blocks."""
    df = make_test_df(50)
    result, end_ts, _ = aggregate_candles(df, divisor=4, align="unstructured")

    # 50 / 4 = 12 full groups + 1 incomplete (2 candles) = 13
    assert len(result) == 13, f"Expected 13 candles, got {len(result)}"
    assert len(end_ts) == 13, f"Expected 13 end_timestamps, got {len(end_ts)}"

    print(f"✓ test_aggregate_unstructured passed ({len(result)} candles)")


def test_incomplete_candle():
    """Test that last candle can be incomplete (< divisor source candles)."""
    df = make_test_df(10)
    result, end_ts, _ = aggregate_candles(df, divisor=4, align="unstructured")

    # 10 / 4 = 2 full + 1 incomplete (2 candles) = 3
    assert len(result) == 3, f"Expected 3 candles, got {len(result)}"

    # Last candle should aggregate only last 2 candles
    last_group = df.iloc[8:10]
    assert result.iloc[-1]['Open'] == last_group.iloc[0]['Open']
    assert result.iloc[-1]['Close'] == last_group.iloc[-1]['Close']
    assert result.iloc[-1]['High'] == last_group['High'].max()
    assert result.iloc[-1]['Low'] == last_group['Low'].min()

    print(f"✓ test_incomplete_candle passed")


def test_ohlcv_integrity():
    """Test OHLCV integrity: High >= Low, Volume = sum, etc."""
    df = make_test_df(200)
    result, _, _ = aggregate_candles(df, divisor=4, align="structured")

    # High >= Low everywhere
    assert (result['High'] >= result['Low']).all(), "High must be >= Low"

    # High >= Open and Close
    assert (result['High'] >= result['Open']).all(), "High must be >= Open"
    assert (result['High'] >= result['Close']).all(), "High must be >= Close"

    # Low <= Open and Close
    assert (result['Low'] <= result['Open']).all(), "Low must be <= Open"
    assert (result['Low'] <= result['Close']).all(), "Low must be <= Close"

    # Volume should be positive
    assert (result['Volume'] > 0).all(), "Volume must be positive"

    # Spot check: first candle
    first_group = df.iloc[:4]  # First 4h group (assuming starts at 00:00)
    first_hour = first_group['Open time'].dt.hour.iloc[0]
    group_start = (first_hour // 4) * 4
    group_mask = (df['Open time'].dt.hour >= group_start) & (df['Open time'].dt.hour < group_start + 4) & (df['Open time'].dt.date == df['Open time'].dt.date.iloc[0])
    group = df[group_mask]

    if len(group) > 0:
        assert result.iloc[0]['High'] == group['High'].max(), "First candle High should be max of group"
        assert result.iloc[0]['Low'] == group['Low'].min(), "First candle Low should be min of group"
        assert np.isclose(result.iloc[0]['Volume'], group['Volume'].sum(), rtol=1e-5), "Volume should be sum"

    print(f"✓ test_ohlcv_integrity passed")


def test_divisor_values():
    """Test with various divisor values."""
    df = make_test_df(240)  # 10 days of 1h data

    for divisor in [2, 3, 4, 6, 8]:
        result, _, _ = aggregate_candles(df, divisor=divisor, align="unstructured")
        expected = (len(df) + divisor - 1) // divisor  # Ceiling division
        assert len(result) == expected, f"Divisor {divisor}: expected {expected}, got {len(result)}"

        # OHLCV integrity
        assert (result['High'] >= result['Low']).all(), f"Divisor {divisor}: High < Low found"

    print(f"✓ test_divisor_values passed (divisors: 2, 3, 4, 6, 8)")


def test_divisor_validation():
    """Test that invalid divisor raises ValueError."""
    df = make_test_df(10)

    with pytest.raises(ValueError):
        aggregate_candles(df, divisor=1, align="structured")

    with pytest.raises(ValueError):
        aggregate_candles(df, divisor=0, align="structured")

    print(f"✓ test_divisor_validation passed")


def test_align_validation():
    """Test that invalid alignment mode raises ValueError."""
    df = make_test_df(10)

    with pytest.raises(ValueError):
        aggregate_candles(df, divisor=4, align="invalid")

    print(f"✓ test_align_validation passed")


def test_volume_sum():
    """Test that aggregated volume equals sum of source candles."""
    df = make_test_df(100)
    result, _, _ = aggregate_candles(df, divisor=4, align="unstructured")

    # Check total volume matches
    total_source = df['Volume'].sum()
    total_agg = result['Volume'].sum()
    assert np.isclose(total_source, total_agg, rtol=1e-4), \
        f"Total volume mismatch: source={total_source:.2f}, agg={total_agg:.2f}"

    print(f"✓ test_volume_sum passed")


def test_partial_ohlcv():
    """Test that partial_ohlcv returns correct expanding aggregation per raw candle."""
    df = make_test_df(12)  # 3 groups of 4
    result, _, partial = aggregate_candles(df, divisor=4, align="unstructured")

    assert partial.shape == (12, 5), f"Expected (12, 5), got {partial.shape}"

    # Check first group (raw candles 0-3)
    # At raw candle 0: Open=first, High=H[0], Low=L[0], Close=C[0], Volume=V[0]
    assert partial[0, 0] == df['Open'].iloc[0], "Partial Open should be first candle's Open"
    assert partial[0, 1] == df['High'].iloc[0], "Partial High at position 0"
    assert partial[0, 2] == df['Low'].iloc[0], "Partial Low at position 0"
    assert partial[0, 3] == df['Close'].iloc[0], "Partial Close should be current candle"
    assert np.isclose(partial[0, 4], df['Volume'].iloc[0], rtol=1e-5), "Partial Volume at position 0"

    # At raw candle 1: expanding within group
    assert partial[1, 0] == df['Open'].iloc[0], "Partial Open still first candle"
    assert partial[1, 1] == max(df['High'].iloc[0], df['High'].iloc[1]), "Partial High cummax"
    assert partial[1, 2] == min(df['Low'].iloc[0], df['Low'].iloc[1]), "Partial Low cummin"
    assert partial[1, 3] == df['Close'].iloc[1], "Partial Close is current"
    assert np.isclose(partial[1, 4], df['Volume'].iloc[0] + df['Volume'].iloc[1], rtol=1e-5), "Partial Volume cumsum"

    # At raw candle 3 (last of group): should match the complete aggregated candle
    assert partial[3, 0] == result.iloc[0]['Open'], "Complete: Open matches"
    assert np.isclose(partial[3, 1], result.iloc[0]['High'], rtol=1e-5), "Complete: High matches"
    assert np.isclose(partial[3, 2], result.iloc[0]['Low'], rtol=1e-5), "Complete: Low matches"
    assert partial[3, 3] == result.iloc[0]['Close'], "Complete: Close matches"
    assert np.isclose(partial[3, 4], result.iloc[0]['Volume'], rtol=1e-4), "Complete: Volume matches"

    # At raw candle 4 (first of group 1): resets
    assert partial[4, 0] == df['Open'].iloc[4], "Group 1: Open resets"
    assert partial[4, 1] == df['High'].iloc[4], "Group 1: High resets"
    assert partial[4, 3] == df['Close'].iloc[4], "Group 1: Close is current"

    print(f"✓ test_partial_ohlcv passed")


def test_with_real_data():
    """Test aggregation on real dataset if available."""
    test_file = TEST_CSV
    if not os.path.exists(test_file):
        pytest.skip(f"Test data not found: {test_file}")

    df = loader.get_raw_data(test_file)
    result, _, _ = aggregate_candles(df, divisor=4, align="structured")

    assert len(result) > 0, "Aggregated data should not be empty"
    assert (result['High'] >= result['Low']).all(), "OHLCV integrity failed"

    print(f"✓ test_with_real_data passed ({len(df)} → {len(result)} candles)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing aggregation.py")
    print("=" * 60)

    try:
        test_aggregate_structured()
        test_aggregate_unstructured()
        test_incomplete_candle()
        test_ohlcv_integrity()
        test_divisor_values()
        test_divisor_validation()
        test_align_validation()
        test_volume_sum()
        test_partial_ohlcv()
        test_with_real_data()

        print("\n" + "=" * 60)
        print("All aggregation tests passed!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\nTest failed: {e}\n")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
