"""
Test labeling.py - Validation of triple barrier labeling
"""

import sys
import os
import numpy as np
import pandas as pd
from conftest import TEST_CSV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data import loader, labeling
from modules.data.labeling import remap_labels, TARGET_CLASS_NAMES, REJECT_CLASS
import config


def test_compute_atr():
    """Test ATR computation with Wilder smoothing"""
    print("\n=== Test: compute_atr ===")

    # Create test data with known values
    test_data = pd.DataFrame({
        'Open':  [100, 102, 101, 103, 102, 104, 103, 105],
        'High':  [105, 106, 104, 107, 106, 108, 107, 109],
        'Low':   [95,  96,  97,  98,  97,  99,  98, 100],
        'Close': [102, 101, 103, 102, 104, 103, 105, 104]
    })

    atr = labeling.compute_atr(test_data, period=3)

    # First 2 values (period-1) should be NaN
    assert np.isnan(atr[0]), "atr[0] should be NaN"
    assert np.isnan(atr[1]), "atr[1] should be NaN"

    # atr[2] should be mean of TR[0:3]
    # TR[0] = H[0]-L[0] = 10 (no previous close)
    # TR[1] = max(H[1]-L[1], |H[1]-C[0]|, |L[1]-C[0]|) = max(10, 4, 6) = 10
    # TR[2] = max(H[2]-L[2], |H[2]-C[1]|, |L[2]-C[1]|) = max(7, 3, 4) = 7
    # atr[2] = (10+10+7)/3 = 9.0
    assert abs(atr[2] - 9.0) < 1e-6, f"atr[2] expected 9.0, got {atr[2]}"

    # Subsequent values should use Wilder smoothing
    assert not np.isnan(atr[3]), "atr[3] should not be NaN"
    assert atr[3] > 0, "atr[3] should be positive"

    # All values after period-1 should be non-NaN and positive
    for i in range(3, len(atr)):
        assert not np.isnan(atr[i]), f"atr[{i}] should not be NaN"
        assert atr[i] > 0, f"atr[{i}] should be positive"

    print(f"✓ ATR computation correct: {atr}")


def test_compute_barriers():
    """Test asymmetric barrier calculation"""
    print("\n=== Test: compute_barriers ===")

    close = 100
    atr_value = 5.0

    # Asymmetric: TP=2.0, SL=1.0
    long_tp, long_sl, short_tp, short_sl = labeling.compute_barriers(close, atr_value, 2.0, 1.0)

    # Long: TP = 100 + 2.0×5.0 = 110, SL = 100 - 1.0×5.0 = 95
    assert abs(long_tp - 110.0) < 1e-6, f"long_tp expected 110.0, got {long_tp}"
    assert abs(long_sl - 95.0) < 1e-6, f"long_sl expected 95.0, got {long_sl}"
    # Short: TP = 100 - 2.0×5.0 = 90, SL = 100 + 1.0×5.0 = 105
    assert abs(short_tp - 90.0) < 1e-6, f"short_tp expected 90.0, got {short_tp}"
    assert abs(short_sl - 105.0) < 1e-6, f"short_sl expected 105.0, got {short_sl}"

    print(f"✓ Barriers correct: long=({long_tp},{long_sl}), short=({short_tp},{short_sl})")


def _make_test_df(n_history, future_rows):
    """
    Build a minimal DataFrame for triple barrier testing.

    Args:
        n_history: Number of flat history candles before prediction point
        future_rows: List of dicts with Open/High/Low/Close for future candles

    Returns:
        DataFrame with n_history + 1 (prediction) + len(future_rows) candles
    """
    rows = []
    # Flat history (needed for window_size)
    for _ in range(n_history):
        rows.append({'Open': 100, 'High': 100, 'Low': 100, 'Close': 100, 'Volume': 1000})
    # Prediction candle (Close=100, ATR will be ~0 from flat history,
    # so we use atr_multiplier to control barriers)
    rows.append({'Open': 100, 'High': 100, 'Low': 100, 'Close': 100, 'Volume': 1000})
    # Future candles
    for fr in future_rows:
        rows.append({
            'Open': fr.get('Open', 100),
            'High': fr['High'],
            'Low': fr['Low'],
            'Close': fr.get('Close', 100),
            'Volume': 1000
        })
    df = pd.DataFrame(rows)
    # Add Open time (required by some code paths)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')
    return df


def test_triple_barrier_tp_hit():
    """Test triple barrier when TP is hit"""
    print("\n=== Test: triple_barrier_tp_hit ===")

    # 10 flat candles (history), then future candles with High reaching 110
    # With flat data, ATR ≈ 0. We use large amplitude in future to guarantee hit.
    # Instead, use a dataset where ATR is meaningful:
    # Oscillating history to get ATR ~ 5, then TP at close + mult * ATR
    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        # Oscillating to create ATR
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Future: price goes UP → TP hit (High well above close + ATR*mult)
    for _ in range(10):
        rows.append({'Open': 100, 'High': 120, 'Low': 99, 'Close': 110, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    # Should have 1 label (prediction point at index n_hist, future = 10 candles)
    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    assert labels[0] == labeling.LABEL_BULLISH, f"Expected BULLISH, got {labels[0]}"

    print(f"✓ TP hit detected correctly: label={labels[0]} (BULLISH)")


def test_triple_barrier_sl_hit():
    """Test triple barrier when SL is hit"""
    print("\n=== Test: triple_barrier_sl_hit ===")

    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Future: price goes DOWN → SL hit (Low well below close - ATR*mult)
    for _ in range(10):
        rows.append({'Open': 100, 'High': 101, 'Low': 80, 'Close': 90, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    assert labels[0] == labeling.LABEL_BEARISH, f"Expected BEARISH, got {labels[0]}"

    print(f"✓ SL hit detected correctly: label={labels[0]} (BEARISH)")


def test_triple_barrier_both_hit():
    """Test triple barrier when both TP and SL hit in same candle"""
    print("\n=== Test: triple_barrier_both_hit ===")

    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Future: huge candle that hits BOTH barriers
    for _ in range(10):
        rows.append({'Open': 100, 'High': 120, 'Low': 80, 'Close': 100, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    assert labels[0] == labeling.LABEL_UNCERTAIN, f"Expected UNCERTAIN (both hit), got {labels[0]}"

    print(f"✓ Both hit in same candle handled correctly: label={labels[0]} (UNCERTAIN)")


def test_triple_barrier_neither_hit():
    """Test triple barrier when neither is hit"""
    print("\n=== Test: triple_barrier_neither_hit ===")

    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Future: flat candles, neither barrier hit
    for _ in range(10):
        rows.append({'Open': 100, 'High': 101, 'Low': 99, 'Close': 100, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    assert labels[0] == labeling.LABEL_UNCERTAIN, f"Expected UNCERTAIN (neither hit), got {labels[0]}"

    print(f"✓ Neither hit handled correctly: label={labels[0]} (UNCERTAIN)")


def test_triple_barrier_gap_up():
    """Test triple barrier with gap up — entry at Open[N+1], not Close[N].
    Gap up means entry is already high, so TP is harder to reach from there.
    With entry=115, ATR~10, TP=125. High=120 < 125 → no TP on gap candle.
    Subsequent candles at 100 → price drops below SL=105? No, Low=99 < 105 → SL hit for long.
    Short scenario: entry=115, TP=105, subsequent Low=99 < 105 → short TP hit = BEARISH.
    """
    print("\n=== Test: triple_barrier_gap_up ===")

    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Gap up: Open jumps to 115, High at 120 → but entry is now 115
    rows.append({'Open': 115, 'High': 120, 'Low': 114, 'Close': 118, 'Volume': 1000})
    for _ in range(9):
        rows.append({'Open': 100, 'High': 101, 'Low': 99, 'Close': 100, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    # Entry=Open[N+1]=115. Price reverts to ~100 → short TP hit (115-ATR) before long TP
    assert labels[0] == labeling.LABEL_BEARISH, f"Expected BEARISH (gap up then revert → short TP), got {labels[0]}"

    print(f"✓ Gap up with realistic entry correctly detected as BEARISH: label={labels[0]}")


def test_triple_barrier_gap_down():
    """Test triple barrier with gap down — entry at Open[N+1], not Close[N].
    Gap down means entry is already low, so TP for long is easier from there.
    Entry=85, ATR~10, long TP=95. Subsequent candles High=101 > 95 → long TP hit = BULLISH.
    """
    print("\n=== Test: triple_barrier_gap_down ===")

    n_hist = 20
    rows = []
    for i in range(n_hist + 1):
        rows.append({
            'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000
        })
    # Gap down: Open drops to 85 → but entry is now 85
    rows.append({'Open': 85, 'High': 86, 'Low': 80, 'Close': 82, 'Volume': 1000})
    for _ in range(9):
        rows.append({'Open': 100, 'High': 101, 'Low': 99, 'Close': 100, 'Volume': 1000})

    df = pd.DataFrame(rows)
    df['Open time'] = pd.date_range('2020-01-01', periods=len(df), freq='h')

    labels, atr, _, _, _ = labeling.get_labels(df, window_size=n_hist, max_horizon=10, atr_period=5, atr_multiplier_tp=1.0, atr_multiplier_sl=1.0)

    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    # Entry=Open[N+1]=85. Price reverts to ~100 → long TP hit (85+ATR) before short TP
    assert labels[0] == labeling.LABEL_BULLISH, f"Expected BULLISH (gap down then revert → long TP), got {labels[0]}"

    print(f"✓ Gap down with realistic entry correctly detected as BULLISH: label={labels[0]}")


def test_get_labels():
    """Test complete labeling on real data"""
    print("\n=== Test: get_labels ===")

    # Load real data
    df = loader.get_raw_data(TEST_CSV)
    print(f"  Loaded {len(df)} rows")

    # Compute labels and atr_array
    labels, atr_array, confidence, _, _ = labeling.get_labels(df, window_size=100, max_horizon=50)

    # Check labels output
    assert isinstance(labels, np.ndarray), "Labels should be numpy array"
    assert labels.dtype == np.int8, "Labels should be int8"
    assert set(np.unique(labels)).issubset({0, 1, 2}), "Labels should be {0, 1, 2}"

    # Check atr_array output
    assert isinstance(atr_array, np.ndarray), "atr_array should be numpy array"
    assert atr_array.dtype == np.float32, "atr_array should be float32"
    assert (atr_array > 0).all(), "All ATR values should be positive"
    assert not np.isnan(atr_array).any(), "atr_array should not contain NaN"

    # Check lengths match
    assert len(labels) == len(atr_array), f"Labels and atr_array lengths must match: {len(labels)} vs {len(atr_array)}"

    # Check length
    expected_length = len(df) - 100 - 50  # window_size - max_horizon
    assert len(labels) == expected_length, f"Expected {expected_length} labels, got {len(labels)}"

    # Check distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {
        labeling.LABEL_BEARISH: "Bearish",
        labeling.LABEL_UNCERTAIN: "Uncertain",
        labeling.LABEL_BULLISH: "Bullish"
    }
    print(f"  Label distribution:")
    for label, count in zip(unique, counts):
        print(f"    {label} ({label_names[label]}): {count}")

    print(f"  ATR stats: min={atr_array.min():.4f}, max={atr_array.max():.4f}, mean={atr_array.mean():.4f}")

    # Check confidence output
    assert isinstance(confidence, np.ndarray), "confidence should be numpy array"
    assert confidence.dtype == np.float32, "confidence should be float32"
    assert len(confidence) == len(labels), f"confidence length mismatch: {len(confidence)} vs {len(labels)}"
    assert (confidence >= 0).all(), "confidence should be >= 0"

    # After normalization: directional mean should be ~1.0, uncertain = 1.0
    dir_mask = (labels == labeling.LABEL_BULLISH) | (labels == labeling.LABEL_BEARISH)
    unc_mask = labels == labeling.LABEL_UNCERTAIN
    if dir_mask.any():
        dir_mean = confidence[dir_mask].mean()
        assert abs(dir_mean - 1.0) < 0.01, f"Directional confidence mean should be ~1.0, got {dir_mean:.4f}"
        print(f"  Confidence: directional mean={dir_mean:.3f}, min={confidence[dir_mask].min():.3f}, max={confidence[dir_mask].max():.3f}")
    if unc_mask.any():
        assert (confidence[unc_mask] == 1.0).all(), "Uncertain confidence should be 1.0"

    print(f"✓ get_labels works correctly: {len(labels)} labels, {len(atr_array)} ATR values, {len(confidence)} confidence values")


# ============================================================================
# remap_labels tests
# ============================================================================

def test_remap_labels_triple():
    """Triple mode: labels unchanged, num_classes=3."""
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    remapped, nc = remap_labels(labels, "triple")
    assert nc == 3
    np.testing.assert_array_equal(remapped, labels)


def test_remap_labels_bull():
    """Bull mode: Bull(2)->1, rest->0, num_classes=2."""
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    remapped, nc = remap_labels(labels, "bull")
    assert nc == 2
    expected = np.array([0, 0, 1, 0, 0, 1], dtype=np.int8)
    np.testing.assert_array_equal(remapped, expected)


def test_remap_labels_bear():
    """Bear mode: Bear(0)->1, rest->0, num_classes=2."""
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    remapped, nc = remap_labels(labels, "bear")
    assert nc == 2
    expected = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    np.testing.assert_array_equal(remapped, expected)


def test_remap_labels_uncertain():
    """Uncertain mode: Uncertain(1)->1, rest->0, num_classes=2."""
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    remapped, nc = remap_labels(labels, "uncertain")
    assert nc == 2
    expected = np.array([0, 1, 0, 0, 1, 0], dtype=np.int8)
    np.testing.assert_array_equal(remapped, expected)


def test_remap_labels_invalid():
    """Invalid mode should raise ValueError."""
    labels = np.array([0, 1, 2], dtype=np.int8)
    try:
        remap_labels(labels, "invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown PREDICTION_TARGET" in str(e)


def test_target_class_names():
    """TARGET_CLASS_NAMES should have correct structure for all modes."""
    assert len(TARGET_CLASS_NAMES) == 7
    assert len(TARGET_CLASS_NAMES["triple"]) == 3
    assert len(TARGET_CLASS_NAMES["bull"]) == 2
    assert len(TARGET_CLASS_NAMES["bear"]) == 2
    assert len(TARGET_CLASS_NAMES["uncertain"]) == 2
    assert len(TARGET_CLASS_NAMES["close1"]) == 2
    assert len(TARGET_CLASS_NAMES["close2"]) == 2
    assert len(TARGET_CLASS_NAMES["close3"]) == 2


def test_reject_class():
    """REJECT_CLASS should map modes to correct reject classes."""
    assert REJECT_CLASS["triple"] == 1
    assert REJECT_CLASS["bull"] == 0
    assert REJECT_CLASS["bear"] == 0
    assert REJECT_CLASS["uncertain"] == 1


# ============================================================================
# Next-close labeling tests
# ============================================================================

def test_get_labels_next_close_basic():
    """get_labels_next_close returns correct Up/Down binary labels."""
    from modules.data.labeling import get_labels_next_close

    # Build a small DataFrame with known close direction
    n = 60
    df = pd.DataFrame({
        'Open':  np.linspace(100, 110, n),
        'High':  np.linspace(102, 112, n),
        'Low':   np.linspace(98, 108, n),
        'Close': np.linspace(100, 110, n),  # monotonic increase → all Up
        'Open time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'Volume': np.full(n, 1000.0),
    })

    labels, atr_values, confidence, pnl_long, pnl_short = get_labels_next_close(
        df, horizon=1, window_size=10, atr_period=5
    )

    # All closes increase → all labels should be 1 (Up)
    assert np.all(labels == 1)
    assert len(labels) == n - 10 - 1  # n - window_size - horizon
    assert pnl_long is None
    assert pnl_short is None
    assert len(atr_values) == len(labels)
    np.testing.assert_array_equal(confidence, np.ones(len(labels)))


def test_get_labels_next_close_down():
    """get_labels_next_close detects Down labels on decreasing data."""
    from modules.data.labeling import get_labels_next_close

    n = 60
    df = pd.DataFrame({
        'Open':  np.linspace(110, 100, n),
        'High':  np.linspace(112, 102, n),
        'Low':   np.linspace(108, 98, n),
        'Close': np.linspace(110, 100, n),  # monotonic decrease → all Down
        'Open time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'Volume': np.full(n, 1000.0),
    })

    labels, _, _, _, _ = get_labels_next_close(df, horizon=1, window_size=10, atr_period=5)

    assert np.all(labels == 0)


def test_get_labels_next_close_horizon_2():
    """get_labels_next_close with horizon=2 compares close[i+2] vs close[i]."""
    from modules.data.labeling import get_labels_next_close

    n = 60
    # Alternating: up, down, up, down...
    closes = np.array([100 + (i % 2) for i in range(n)], dtype=float)
    df = pd.DataFrame({
        'Open':  closes,
        'High':  closes + 1,
        'Low':   closes - 1,
        'Close': closes,
        'Open time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'Volume': np.full(n, 1000.0),
    })

    labels, _, _, _, _ = get_labels_next_close(df, horizon=2, window_size=10, atr_period=5)

    # close[i+2] - close[i] = same parity → 0 → Down=0 for all
    assert len(labels) == n - 10 - 2
    assert np.all(labels == 0)


def test_get_labels_next_close_too_short():
    """get_labels_next_close raises ValueError on insufficient data."""
    from modules.data.labeling import get_labels_next_close
    import pytest

    df = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [102, 103, 104],
        'Low': [98, 99, 100],
        'Close': [101, 102, 103],
        'Open time': pd.date_range('2024-01-01', periods=3, freq='1h'),
        'Volume': [1000, 1000, 1000],
    })

    with pytest.raises(ValueError, match="Not enough data"):
        get_labels_next_close(df, horizon=1, window_size=10, atr_period=5)


def test_parse_close_horizon():
    """parse_close_horizon extracts horizon from closeN targets."""
    from modules.data.labeling import parse_close_horizon

    assert parse_close_horizon("close1") == 1
    assert parse_close_horizon("close3") == 3
    assert parse_close_horizon("close10") == 10
    assert parse_close_horizon("triple") is None
    assert parse_close_horizon("bull") is None
    assert parse_close_horizon("closeX") is None
    assert parse_close_horizon(None) is None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing labeling.py")
    print("=" * 60)

    try:
        test_compute_atr()
        test_compute_barriers()
        test_triple_barrier_tp_hit()
        test_triple_barrier_sl_hit()
        test_triple_barrier_both_hit()
        test_triple_barrier_neither_hit()
        test_triple_barrier_gap_up()
        test_triple_barrier_gap_down()
        test_get_labels()
        test_remap_labels_triple()
        test_remap_labels_bull()
        test_remap_labels_bear()
        test_remap_labels_uncertain()
        test_remap_labels_invalid()
        test_target_class_names()
        test_reject_class()

        print("\n" + "=" * 60)
        print("✅ All labeling tests passed!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
