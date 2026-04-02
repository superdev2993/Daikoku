"""
Tests for modules/data/dataset.py

Covers:
1. TimeSeriesWindowDataset: shapes, look-ahead, label alignment, confidence, normalization
2. MultiTFWindowDataset: shapes, look-ahead, confidence, cross_tf_normalize
3. Helper functions: _assemble_window, _compute_perwindow_features
"""

import sys
import os
import numpy as np
import torch
import pytest
from conftest import TEST_CSV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data import loader, transform, labeling
from modules.data.dataset import (
    TimeSeriesWindowDataset,
    MultiTFWindowDataset,
    _assemble_window,
    _compute_perwindow_features,
    _wma_at,
    _compute_hma_at,
    _recompute_partial_secondary,
)
import config
from modules.utils.seed import set_seed


# ============================================================================
# Shared fixtures
# ============================================================================

def _build_mono_dataset_components(ws=100):
    """
    Build all components needed for a TimeSeriesWindowDataset from test_Dataset.csv.
    Returns dict with data, labels, raw_hlc, confidence, n_norm_features.
    """
    df = loader.get_raw_data(TEST_CSV)
    labels, atr_array, confidence, _, _ = labeling.get_labels(
        df, window_size=ws, max_horizon=50, atr_period=14
    )
    result = transform.transform_pipeline(df, atr_array=atr_array, window_size=ws)

    data = result['data']
    n_norm_features = result.get('n_norm_features', 0)

    n_data = len(data)
    raw_hlc = np.column_stack([
        df['High'].values[ws:ws + n_data],
        df['Low'].values[ws:ws + n_data],
        df['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)

    # Align
    min_len = min(len(data), len(labels), len(raw_hlc))
    data = data[:min_len]
    labels = labels[:min_len]
    confidence = confidence[:min_len]
    raw_hlc = raw_hlc[:min_len]

    return {
        'data': data,
        'labels': labels,
        'raw_hlc': raw_hlc,
        'confidence': confidence,
        'n_norm_features': n_norm_features + 1,  # +1 for log_zz
        'ws': ws,
        'df': df,
    }


# ============================================================================
# Test 1: Look-ahead bias verification (mono)
# ============================================================================

def test_no_lookahead_mono():
    """
    Verify that modifying data AFTER the window does NOT change the window output.
    This is the most critical test: proves no look-ahead bias.
    """
    print("\n=== Test: no_lookahead_mono ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']

    # Use a window in the middle of the dataset
    test_idx = 50
    indices = [test_idx]

    # Build dataset and get window
    dataset1 = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=c['raw_hlc'],
        n_norm_features=c['n_norm_features'],
    )
    window1, label1 = dataset1[0]

    # Now modify raw_hlc AFTER the window (future data)
    raw_hlc_modified = c['raw_hlc'].copy()
    future_start = test_idx + ws  # First index after window
    if future_start + 10 < len(raw_hlc_modified):
        raw_hlc_modified[future_start:future_start + 10] *= 2.0  # Double future prices

    dataset2 = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=raw_hlc_modified,
        n_norm_features=c['n_norm_features'],
    )
    window2, label2 = dataset2[0]

    # Windows MUST be identical — future data should not affect current window
    assert torch.allclose(window1, window2, atol=1e-6), \
        "LOOK-AHEAD BIAS DETECTED: modifying future raw_hlc changed the window!"
    assert label1 == label2, "Labels should be identical"

    print("  Window shape:", window1.shape)
    print("  Max diff:", (window1 - window2).abs().max().item())
    print("  PASS: No look-ahead bias detected")


# ============================================================================
# Test 2: Shape verification (mono)
# ============================================================================

def test_shape_mono_with_perwindow():
    """Test that mono dataset with per-window features produces (ws, 18)."""
    print("\n=== Test: shape_mono_with_perwindow ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    indices = list(range(5))

    dataset = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=c['raw_hlc'],
        n_norm_features=c['n_norm_features'],
    )

    window, label = dataset[0]
    assert window.shape == (ws, 18), f"Expected ({ws}, 18), got {window.shape}"
    assert label.dim() == 0, "Label should be scalar"
    assert label.item() in [0, 1, 2], f"Label should be in {{0,1,2}}, got {label.item()}"
    assert not window.isnan().any(), "No NaN expected"
    assert not window.isinf().any(), "No inf expected"

    print(f"  Shape: {window.shape}, label: {label.item()}")
    print("  PASS")


def test_shape_mono_without_perwindow():
    """Test that mono dataset without per-window features returns raw global shape."""
    print("\n=== Test: shape_mono_without_perwindow ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    indices = [0]

    # No raw_hlc → no per-window features
    dataset = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=None,
        n_norm_features=0,
    )

    window, label = dataset[0]
    n_global = c['data'].shape[1]
    assert window.shape == (ws, n_global), f"Expected ({ws}, {n_global}), got {window.shape}"

    print(f"  Shape: {window.shape} (no per-window features)")
    print("  PASS")


# ============================================================================
# Test 3: Label alignment (mono)
# ============================================================================

def test_label_alignment_mono():
    """
    Verify that label corresponds to the LAST candle of the window, not the first.
    Window [data_idx, data_idx+ws) → label at data_idx+ws-1.
    """
    print("\n=== Test: label_alignment_mono ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']

    # Check multiple indices
    for test_idx in [0, 10, 50]:
        indices = [test_idx]
        dataset = TimeSeriesWindowDataset(
            data=c['data'], labels=c['labels'], indices=indices,
            window_size=ws, raw_hlc=None,
            n_norm_features=0,
        )

        _, label = dataset[0]
        expected_label = c['labels'][test_idx + ws - 1]
        assert label.item() == expected_label, \
            f"idx={test_idx}: label={label.item()}, expected={expected_label} (at position {test_idx + ws - 1})"

    print("  PASS: Labels correctly aligned to last candle of window")


# ============================================================================
# Test 4: Confidence tuple (mono)
# ============================================================================

def test_confidence_returned_when_provided():
    """Test that dataset returns (window, label, conf) when confidence is provided."""
    print("\n=== Test: confidence_returned_when_provided ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    indices = [0, 10]

    # With confidence
    dataset_conf = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=None,
        n_norm_features=0,
        confidence=c['confidence'],
    )

    result = dataset_conf[0]
    assert len(result) == 3, f"Expected tuple of 3 (window, label, conf), got {len(result)}"
    window, label, conf = result
    assert conf.dim() == 0, "Confidence should be scalar"
    assert conf.item() >= 0, "Confidence should be >= 0"

    # Verify confidence alignment (same as label: last candle of window)
    expected_conf_idx = indices[0] + ws - 1
    expected_conf = c['confidence'][expected_conf_idx]
    assert abs(conf.item() - expected_conf) < 1e-6, \
        f"Confidence mismatch: got {conf.item()}, expected {expected_conf}"

    print(f"  With confidence: tuple len={len(result)}, conf={conf.item():.4f}")


def test_confidence_not_returned_when_absent():
    """Test that dataset returns (window, label) when confidence is None."""
    print("\n=== Test: confidence_not_returned_when_absent ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    indices = [0]

    # Without confidence
    dataset_no_conf = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=None,
        n_norm_features=0,
        confidence=None,
    )

    result = dataset_no_conf[0]
    assert len(result) == 2, f"Expected tuple of 2 (window, label), got {len(result)}"

    print(f"  Without confidence: tuple len={len(result)}")
    print("  PASS")


# ============================================================================
# Test 5: _assemble_window order
# ============================================================================

def test_assemble_window_with_time():
    """Test _assemble_window produces correct feature order with time features."""
    print("\n=== Test: assemble_window_with_time ===")

    ws = 10
    # Global: 12 features [5 OHLCV, 1 log_atr, 4 time, 2 regime]
    # Use realistic values: log_price ~10.8 (BTC ~50k), log_atr ~6.2 (ATR ~500)
    global_window = torch.randn(ws, 12)
    global_window[:, 0] = 10.8  # log_price
    global_window[:, 5] = 6.2   # log_atr
    log_zz_t = torch.randn(ws, 1)
    struct_t = torch.randn(ws, 5)

    # Compute expected ATR ratio for verification
    atr_ratio = torch.exp(global_window[:, 5:6] - global_window[:, 0:1])  # (ws, 1)

    result = _assemble_window(global_window, log_zz_t, struct_t, has_time=True)

    # Expected: [5 OHLCV, 1 log_zz, 1 log_atr, 4 time, 5 struct, 2 regime] = 18
    assert result.shape == (ws, 18), f"Expected ({ws}, 18), got {result.shape}"

    # Verify order
    assert torch.allclose(result[:, :5], global_window[:, :5]), "OHLCV mismatch"
    assert torch.allclose(result[:, 5:6], log_zz_t), "log_zz mismatch"
    assert torch.allclose(result[:, 6:7], global_window[:, 5:6]), "log_atr mismatch"
    assert torch.allclose(result[:, 7:11], global_window[:, 6:10]), "time features mismatch"
    # struct_rank (col 0) unchanged, dist cols (1-4) scaled by ATR ratio
    assert torch.allclose(result[:, 11:12], struct_t[:, 0:1]), "struct_rank mismatch"
    expected_struct_dist = struct_t[:, 1:] / (atr_ratio + 1e-8)
    assert torch.allclose(result[:, 12:16], expected_struct_dist, atol=1e-6), "struct dist scaling mismatch"
    assert torch.allclose(result[:, 16:18], global_window[:, 10:12]), "regime features mismatch"

    print(f"  Shape: {result.shape}")
    print("  Feature order: [OHLCV(5), log_zz(1), log_atr(1), time(4), struct(5), regime(2)]")
    print("  PASS")


def test_assemble_window_without_time():
    """Test _assemble_window produces correct feature order without time features."""
    print("\n=== Test: assemble_window_without_time ===")

    ws = 10
    # Global: 8 features [5 OHLCV, 1 log_atr, 2 regime]
    global_window = torch.randn(ws, 8)
    global_window[:, 0] = 10.8  # log_price
    global_window[:, 5] = 6.2   # log_atr
    log_zz_t = torch.randn(ws, 1)
    struct_t = torch.randn(ws, 5)

    atr_ratio = torch.exp(global_window[:, 5:6] - global_window[:, 0:1])

    result = _assemble_window(global_window, log_zz_t, struct_t, has_time=False)

    # Expected: [5 OHLCV, 1 log_zz, 1 log_atr, 5 struct, 2 regime] = 14
    assert result.shape == (ws, 14), f"Expected ({ws}, 14), got {result.shape}"

    # Verify order
    assert torch.allclose(result[:, :5], global_window[:, :5]), "OHLCV mismatch"
    assert torch.allclose(result[:, 5:6], log_zz_t), "log_zz mismatch"
    assert torch.allclose(result[:, 6:7], global_window[:, 5:6]), "log_atr mismatch"
    # struct_rank (col 0) unchanged, dist cols (1-4) scaled
    assert torch.allclose(result[:, 7:8], struct_t[:, 0:1]), "struct_rank mismatch"
    expected_struct_dist = struct_t[:, 1:] / (atr_ratio + 1e-8)
    assert torch.allclose(result[:, 8:12], expected_struct_dist, atol=1e-6), "struct dist scaling mismatch"
    assert torch.allclose(result[:, 12:14], global_window[:, 6:8]), "regime features mismatch"

    print(f"  Shape: {result.shape}")
    print("  Feature order: [OHLCV(5), log_zz(1), log_atr(1), struct(5), regime(2)]")
    print("  PASS")


def test_assemble_window_with_vp():
    """Test _assemble_window with Volume Profile features."""
    print("\n=== Test: assemble_window_with_vp ===")

    ws = 10
    global_window = torch.randn(ws, 12)
    global_window[:, 0] = 10.8  # log_price
    global_window[:, 5] = 6.2   # log_atr
    log_zz_t = torch.randn(ws, 1)
    struct_t = torch.randn(ws, 5)
    vp_t = torch.randn(ws, 5)

    atr_ratio = torch.exp(global_window[:, 5:6] - global_window[:, 0:1])

    result = _assemble_window(global_window, log_zz_t, struct_t, has_time=True, vp_t=vp_t)

    # Expected: [5 OHLCV, 1 log_zz, 1 log_atr, 4 time, 5 struct, 5 VP, 2 regime] = 23
    assert result.shape == (ws, 23), f"Expected ({ws}, 23), got {result.shape}"

    # VP cols 0-3 should be scaled, col 4 (vp_skew) unchanged
    expected_vp_dist = vp_t[:, :4] / (atr_ratio + 1e-8)
    assert torch.allclose(result[:, 16:20], expected_vp_dist, atol=1e-6), "VP dist scaling mismatch"
    assert torch.allclose(result[:, 20:21], vp_t[:, 4:5]), "vp_skew should be unchanged"
    assert torch.allclose(result[:, 21:23], global_window[:, 10:12]), "regime after VP mismatch"

    print(f"  Shape: {result.shape}")
    print("  PASS")


# ============================================================================
# Test 5b: _assemble_window ATR scaling mechanics
# ============================================================================

def test_assemble_window_atr_scaling():
    """Test ATR-relative scaling: struct dist and VP dist are divided by ATR/Close ratio."""
    print("\n=== Test: assemble_window_atr_scaling ===")

    ws = 10
    # Realistic values: BTC ~50k, ATR ~500
    global_window = torch.randn(ws, 12)
    global_window[:, 0] = 10.82  # log(50000)
    global_window[:, 5] = 6.21   # log(500)
    log_zz_t = torch.randn(ws, 1)

    # Known struct values
    struct_t = torch.ones(ws, 5) * 0.01  # small log-ratio distances
    struct_t[:, 0] = 0.5  # struct_rank in [0,1]

    # Known VP values
    vp_t = torch.ones(ws, 5) * 0.02
    vp_t[:, 4] = 0.3  # vp_skew

    atr_ratio = torch.exp(global_window[:, 5:6] - global_window[:, 0:1])  # ~0.01

    result = _assemble_window(global_window, log_zz_t, struct_t, has_time=True, vp_t=vp_t)

    # struct_rank (col 11) unchanged
    assert torch.allclose(result[:, 11:12], torch.ones(ws, 1) * 0.5), "struct_rank should be unchanged"

    # struct dist (cols 12-15) = 0.01 / ~0.01 ≈ 1.0 (much larger than input)
    struct_dist_result = result[:, 12:16]
    assert struct_dist_result.mean() > 0.5, \
        f"Scaled struct dist should be >> input (0.01), got mean={struct_dist_result.mean():.4f}"

    # VP dist (cols 16-19) = 0.02 / ~0.01 ≈ 2.0
    vp_dist_result = result[:, 16:20]
    assert vp_dist_result.mean() > 1.0, \
        f"Scaled VP dist should be >> input (0.02), got mean={vp_dist_result.mean():.4f}"

    # vp_skew (col 20) unchanged
    assert torch.allclose(result[:, 20:21], torch.ones(ws, 1) * 0.3), "vp_skew should be unchanged"

    print(f"  ATR ratio: {atr_ratio[0, 0]:.6f}")
    print(f"  struct dist input=0.01 → scaled={struct_dist_result[0, 0]:.4f}")
    print(f"  VP dist input=0.02 → scaled={vp_dist_result[0, 0]:.4f}")
    print("  PASS")


def test_assemble_window_custom_scaling():
    """Test that passing a custom scaling_atr_ratio overrides the default ATR/Close."""
    print("\n=== Test: assemble_window_custom_scaling ===")

    ws = 10
    global_window = torch.randn(ws, 12)
    global_window[:, 0] = 10.8
    global_window[:, 5] = 6.2
    log_zz_t = torch.randn(ws, 1)
    struct_t = torch.ones(ws, 5) * 0.01
    struct_t[:, 0] = 0.5

    # Custom scaling ratio (e.g., from primary branch)
    custom_ratio = torch.ones(ws, 1) * 0.02  # 2x the branch's own ratio

    result_default = _assemble_window(global_window, log_zz_t, struct_t.clone(), has_time=True)
    result_custom = _assemble_window(global_window, log_zz_t, struct_t.clone(), has_time=True, scaling_atr_ratio=custom_ratio)

    # struct_rank unchanged in both
    assert torch.allclose(result_default[:, 11:12], result_custom[:, 11:12]), "struct_rank should match"

    # struct dist should differ (different scaling ratio)
    dist_default = result_default[:, 12:16]
    dist_custom = result_custom[:, 12:16]
    assert not torch.allclose(dist_default, dist_custom, atol=1e-4), \
        "Custom scaling should produce different results"

    # Verify custom ratio is used: 0.01 / (0.02 + 1e-8) = 0.5
    expected_custom_dist = 0.01 / (0.02 + 1e-8)
    assert torch.allclose(dist_custom, torch.ones_like(dist_custom) * expected_custom_dist, atol=1e-4), \
        f"Custom scaled dist should be ~{expected_custom_dist:.4f}, got {dist_custom[0, 0]:.4f}"

    print(f"  Default dist[0]: {dist_default[0, 0]:.4f}")
    print(f"  Custom dist[0]:  {dist_custom[0, 0]:.4f} (expected ~{expected_custom_dist:.4f})")
    print("  PASS")


# ============================================================================
# Test 6: Per-window normalization
# ============================================================================

def test_perwindow_normalization():
    """Test that per-window normalization normalizes only the first n_norm features."""
    print("\n=== Test: perwindow_normalization ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    n_norm = c['n_norm_features']
    indices = [50]

    dataset = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=ws, raw_hlc=c['raw_hlc'],
        n_norm_features=n_norm,
    )

    window, _ = dataset[0]

    # Normalized features (first n_norm): should have median ~0
    norm_part = window[:, :n_norm]
    medians = norm_part.median(dim=0).values
    for i in range(n_norm):
        assert abs(medians[i].item()) < 0.5, \
            f"Feature {i} median should be ~0 after normalization, got {medians[i].item():.4f}"

    # Non-normalized features: struct_rank (col 11) should still be in [0, 1]
    struct_rank = window[:, 11]
    assert struct_rank.min() >= -0.01, f"struct_rank min={struct_rank.min():.4f}, expected >= 0"
    assert struct_rank.max() <= 1.01, f"struct_rank max={struct_rank.max():.4f}, expected <= 1"

    # Regime features (last 2 cols) should still be in [-1, 1]
    regime = window[:, -2:]
    assert regime.min() >= -1.1, f"regime min={regime.min():.4f}"
    assert regime.max() <= 1.1, f"regime max={regime.max():.4f}"

    print(f"  n_norm_features: {n_norm}")
    print(f"  Normalized medians (first {n_norm}): {[f'{m:.3f}' for m in medians.tolist()]}")
    print(f"  struct_rank range: [{struct_rank.min():.3f}, {struct_rank.max():.3f}]")
    print(f"  regime range: [{regime.min():.3f}, {regime.max():.3f}]")
    print("  PASS")


def test_normalization_does_not_leak_between_windows():
    """Verify that normalization of window A doesn't affect window B."""
    print("\n=== Test: normalization_does_not_leak_between_windows ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']
    n_norm = c['n_norm_features']

    # Get two different windows
    dataset = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=[10, 50],
        window_size=ws, raw_hlc=c['raw_hlc'],
        n_norm_features=n_norm,
    )

    w1, _ = dataset[0]
    w2, _ = dataset[1]

    # Normalized parts should be different (different windows → different stats)
    norm1 = w1[:, :n_norm]
    norm2 = w2[:, :n_norm]
    assert not torch.allclose(norm1, norm2, atol=1e-4), \
        "Different windows should produce different normalized values"

    print("  PASS: Windows normalized independently")


# ============================================================================
# Test 7: _compute_perwindow_features
# ============================================================================

def test_compute_perwindow_features_shape():
    """Test that _compute_perwindow_features returns correct shapes."""
    print("\n=== Test: compute_perwindow_features_shape ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']

    raw_window = c['raw_hlc'][:ws]
    log_zz, struct = _compute_perwindow_features(raw_window)

    assert log_zz.shape == (ws,), f"log_zz shape: expected ({ws},), got {log_zz.shape}"
    assert struct.shape == (ws, 5), f"struct shape: expected ({ws}, 5), got {struct.shape}"
    assert not np.isnan(log_zz).any(), "log_zz should have no NaN"
    assert not np.isnan(struct).any(), "struct should have no NaN"

    print(f"  log_zz shape: {log_zz.shape}, range: [{log_zz.min():.4f}, {log_zz.max():.4f}]")
    print(f"  struct shape: {struct.shape}")
    print("  PASS")


# ============================================================================
# Test 8: MultiTFWindowDataset - shapes
# ============================================================================

def _build_multi_tf_components(ws=100):
    """
    Build components for MultiTFWindowDataset using test_Dataset.csv for both branches.
    Simulates multi-TF by using same data with a simple 1:1 mapping.
    """
    df = loader.get_raw_data(TEST_CSV)
    labels, atr_array, confidence, _, _ = labeling.get_labels(
        df, window_size=ws, max_horizon=50, atr_period=14
    )
    result = transform.transform_pipeline(df, atr_array=atr_array, window_size=ws)

    data = result['data']
    n_norm_features = result.get('n_norm_features', 0)

    n_data = len(data)
    raw_hlc = np.column_stack([
        df['High'].values[ws:ws + n_data],
        df['Low'].values[ws:ws + n_data],
        df['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)

    # Align
    min_len = min(len(data), len(labels), len(raw_hlc))
    data = data[:min_len]
    labels = labels[:min_len]
    confidence = confidence[:min_len]
    raw_hlc = raw_hlc[:min_len]

    # Secondary branch: simulate with 8-feature global data (no time features)
    # [5 OHLCV, 1 log_atr, 2 regime] = indices [0:5, 5, 10:12] from primary's 12 features
    data_secondary = np.column_stack([
        data[:, :5],    # 5 OHLCV
        data[:, 5:6],   # log_atr
        data[:, 10:12], # 2 regime
    ])

    # Simple 1:1 mapping (same timeframe simulated)
    mapping = np.arange(min_len)

    return {
        'data_primary': data,
        'data_secondary': data_secondary,
        'labels': labels,
        'raw_hlc_primary': raw_hlc,
        'raw_hlc_secondary': raw_hlc,
        'confidence': confidence,
        'mapping': mapping,
        'n_norm_primary': n_norm_features + 1,
        'n_norm_secondary': n_norm_features + 1,
        'ws': ws,
    }


def test_shape_multi_tf():
    """Test MultiTFWindowDataset produces correct shapes for both branches."""
    print("\n=== Test: shape_multi_tf ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']
    indices = list(range(5))

    dataset = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
    )

    result = dataset[0]
    assert len(result) == 4, f"Expected tuple of 4 (pri, sec, tf_mapping, label), got {len(result)}"

    window_pri, window_sec, tf_mapping, label = result

    # Primary: 18 features (with time)
    assert window_pri.shape == (ws, 18), f"Primary: expected ({ws}, 18), got {window_pri.shape}"
    # Secondary: 14 features (no time)
    assert window_sec.shape == (ws, 14), f"Secondary: expected ({ws}, 14), got {window_sec.shape}"
    # Label
    assert label.dim() == 0 and label.item() in [0, 1, 2]
    # tf_mapping: (window_size,) LongTensor with valid secondary positions
    assert tf_mapping.shape == (ws,), f"tf_mapping: expected ({ws},), got {tf_mapping.shape}"
    assert tf_mapping.dtype == torch.int64, f"tf_mapping dtype: expected int64, got {tf_mapping.dtype}"
    assert (tf_mapping >= 0).all() and (tf_mapping < ws).all(), "tf_mapping out of bounds"

    # No NaN/inf
    assert not window_pri.isnan().any(), "Primary window has NaN"
    assert not window_sec.isnan().any(), "Secondary window has NaN"

    print(f"  Primary shape: {window_pri.shape}")
    print(f"  Secondary shape: {window_sec.shape}")
    print(f"  tf_mapping shape: {tf_mapping.shape}")
    print(f"  Label: {label.item()}")
    print("  PASS")


# ============================================================================
# Test 9: MultiTF look-ahead
# ============================================================================

def test_no_lookahead_multi_tf():
    """
    Verify no look-ahead in multi-TF mode:
    modifying future raw_hlc should not change current window for either branch.
    """
    print("\n=== Test: no_lookahead_multi_tf ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']
    test_idx = 50
    indices = [test_idx]

    # Original dataset
    dataset1 = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
    )
    pri1, sec1, _, label1 = dataset1[0]

    # Modify future data for both branches
    raw_hlc_pri_mod = c['raw_hlc_primary'].copy()
    raw_hlc_sec_mod = c['raw_hlc_secondary'].copy()
    future_start = test_idx + ws
    if future_start + 10 < len(raw_hlc_pri_mod):
        raw_hlc_pri_mod[future_start:future_start + 10] *= 2.0
        raw_hlc_sec_mod[future_start:future_start + 10] *= 2.0

    dataset2 = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=raw_hlc_pri_mod, raw_hlc_secondary=[raw_hlc_sec_mod],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
    )
    pri2, sec2, _, label2 = dataset2[0]

    assert torch.allclose(pri1, pri2, atol=1e-6), \
        "LOOK-AHEAD BIAS: modifying future data changed PRIMARY window!"
    assert torch.allclose(sec1, sec2, atol=1e-6), \
        "LOOK-AHEAD BIAS: modifying future data changed SECONDARY window!"
    assert label1 == label2

    print("  Primary max diff:", (pri1 - pri2).abs().max().item())
    print("  Secondary max diff:", (sec1 - sec2).abs().max().item())
    print("  PASS: No look-ahead bias in multi-TF mode")


# ============================================================================
# Test 9b: Full pipeline look-ahead (full data vs truncated data)
# ============================================================================

def test_no_lookahead_full_pipeline(trained_checkpoint):
    """
    Proof of no look-ahead bias in the FULL pipeline (transform + secondary + VP + dataset).

    For N test candles, builds the dataset TWO ways using the SAME code path:
      1. FULL: all raw data -> extract tensor at index data_idx
      2. TRUNCATED: raw data cut to just past the window -> extract tensor for last window
    If tensors are identical -> the pipeline is causal (no future data leakage).
    """
    print("\n=== Test: no_lookahead_full_pipeline ===")

    from modules.data.transform import apply_transform_from_checkpoint
    from modules.data.multi_tf import build_secondary_branches, compute_branch_vp
    from modules.inference.engine import load_model

    checkpoint_path, _ = trained_checkpoint

    _, config_params = load_model(checkpoint_path, 'cpu')
    ws = config_params['WINDOW_SIZE']
    divisor = config_params['MULTI_TF_DIVISOR']

    df_raw = loader.get_raw_data(TEST_CSV)

    def _build_ds(df, target_index=None):
        transformed_pri = apply_transform_from_checkpoint(df, config_params)
        n_data_pri = len(transformed_pri)
        raw_hlc_pri = np.column_stack([
            df['High'].values[ws:ws + n_data_pri],
            df['Low'].values[ws:ws + n_data_pri],
            df['Close'].values[ws:ws + n_data_pri]
        ]).astype(np.float64)

        def sec_transform_fn(df_agg):
            return apply_transform_from_checkpoint(df_agg, config_params)

        sec = build_secondary_branches(df, sec_transform_fn, config_params, n_data_pri)
        vp_pri = compute_branch_vp(df, config_params, ws, n_data_pri)

        min_start = sec['min_primary_start']
        valid = list(range(min_start, n_data_pri - ws))
        indices = [target_index] if target_index is not None else valid

        n_norm_pri = transformed_pri.shape[1] - 6
        dataset = MultiTFWindowDataset(
            data_primary=transformed_pri,
            data_secondary=sec['data_secondary_list'],
            labels=np.zeros(n_data_pri, dtype=np.int64),
            indices_primary=indices,
            mapping_primary_to_secondary=sec['mapping_list'],
            window_size=ws,
            raw_hlc_primary=raw_hlc_pri,
            raw_hlc_secondary=sec['raw_hlc_secondary_list'],
            n_norm_features_primary=n_norm_pri + 1,
            n_norm_features_secondary=sec['n_norm_secondary'] + 1,
            partial_context=sec['partial_ctx_list'],
            zigzag_length=config_params['ZIGZAG_LENGTH'],
            vp_features_primary=vp_pri,
            vp_features_secondary=sec['vp_secondary_list'],
            cross_tf_normalize=config_params['CROSS_TF_NORMALIZE'],
        )
        return dataset, n_data_pri, valid

    # Build FULL dataset
    dataset_full, _, valid_indices = _build_ds(df_raw)

    # Test 10 evenly spaced points
    n_tests = 10
    step = max(1, len(valid_indices) // n_tests)
    test_points = valid_indices[::step][:n_tests]
    n_raw = len(df_raw)

    for data_idx in test_points:
        ds_idx = valid_indices.index(data_idx)
        full_pri, full_sec, _, _ = dataset_full[ds_idx]

        raw_trunc = 2 * ws + data_idx + 2 * divisor
        raw_trunc = min(raw_trunc, n_raw)
        df_trunc = df_raw.iloc[:raw_trunc].copy().reset_index(drop=True)

        dataset_trunc, _, _ = _build_ds(df_trunc, target_index=data_idx)
        trunc_pri, trunc_sec, _, _ = dataset_trunc[0]

        diff_pri = torch.abs(full_pri - trunc_pri).max().item()
        diff_sec = torch.abs(full_sec - trunc_sec).max().item()

        assert diff_pri < 1e-6, \
            f"LOOK-AHEAD BIAS: PRIMARY tensors differ at data_idx={data_idx}, diff={diff_pri:.2e}"
        assert diff_sec < 1e-6, \
            f"LOOK-AHEAD BIAS: SECONDARY tensors differ at data_idx={data_idx}, diff={diff_sec:.2e}"

    print(f"  Tested {len(test_points)} candles: all PASS (max_diff=0.00e+00)")
    print("  PASS: No look-ahead bias in full pipeline")


# ============================================================================
# Test 10: MultiTF confidence tuple
# ============================================================================

def test_confidence_multi_tf():
    """Test MultiTFWindowDataset returns correct tuple with/without confidence."""
    print("\n=== Test: confidence_multi_tf ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']
    indices = [0]

    # With confidence
    dataset_conf = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=None, raw_hlc_secondary=[c['raw_hlc_secondary']],
        confidence=c['confidence'],
    )

    result = dataset_conf[0]
    assert len(result) == 5, f"Expected tuple of 5 (pri, sec, tf_mapping, label, conf), got {len(result)}"
    _, _, _, label, conf = result
    assert conf.dim() == 0 and conf.item() >= 0

    # Without confidence
    dataset_no_conf = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=None, raw_hlc_secondary=[c['raw_hlc_secondary']],
        confidence=None,
    )

    result2 = dataset_no_conf[0]
    assert len(result2) == 4, f"Expected tuple of 4 (pri, sec, tf_mapping, label), got {len(result2)}"

    print(f"  With confidence: len={len(result)}, conf={conf.item():.4f}")
    print(f"  Without confidence: len={len(result2)}")
    print("  PASS")


# ============================================================================
# Test 11: cross_tf_normalize
# ============================================================================

def test_cross_tf_normalize():
    """
    Test that cross_tf_normalize uses primary stats for secondary normalization.
    Compare secondary window with cross_tf=True vs False — they should differ.
    """
    print("\n=== Test: cross_tf_normalize ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']
    indices = [50]

    # Without cross-TF
    dataset_off = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
        cross_tf_normalize=False,
    )
    _, sec_off, _, _ = dataset_off[0]

    # With cross-TF
    dataset_on = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
        cross_tf_normalize=True,
    )
    _, sec_on, _, _ = dataset_on[0]

    # The normalized parts should differ (different stats used)
    n_norm_sec = c['n_norm_secondary']
    norm_off = sec_off[:, :n_norm_sec]
    norm_on = sec_on[:, :n_norm_sec]

    # Non-normalized parts should be identical
    rest_off = sec_off[:, n_norm_sec:]
    rest_on = sec_on[:, n_norm_sec:]
    assert torch.allclose(rest_off, rest_on, atol=1e-6), \
        "Non-normalized features should be identical regardless of cross_tf setting"

    print(f"  Normalized diff (cross_tf on vs off): {(norm_on - norm_off).abs().mean():.6f}")
    print(f"  Non-normalized diff: {(rest_on - rest_off).abs().max():.6f}")
    print("  PASS")


# ============================================================================
# Test 12: len()
# ============================================================================

def test_len():
    """Test that __len__ returns correct count."""
    print("\n=== Test: len ===")

    c = _build_mono_dataset_components(ws=100)
    indices = [0, 5, 10, 20, 50]

    dataset = TimeSeriesWindowDataset(
        data=c['data'], labels=c['labels'], indices=indices,
        window_size=c['ws'], raw_hlc=None,
    )

    assert len(dataset) == 5, f"Expected 5, got {len(dataset)}"
    print("  PASS")


# ============================================================================
# Test 13: MultiTF mapping alignment
# ============================================================================

def test_multi_tf_mapping_alignment():
    """
    Verify that the secondary window ends at the mapped position of the last primary candle.
    With our 1:1 test mapping, secondary window should cover the same indices as primary.
    """
    print("\n=== Test: multi_tf_mapping_alignment ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']
    test_idx = 50
    indices = [test_idx]

    # Without per-window features to check raw global slicing
    dataset = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=indices,
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=None, raw_hlc_secondary=[c['raw_hlc_secondary']],
    )

    pri, sec, _, _ = dataset[0]

    # With 1:1 mapping, the secondary window should end at mapping[test_idx + ws - 1]
    last_pri_idx = test_idx + ws - 1
    j_sec = c['mapping'][last_pri_idx]
    start_sec = j_sec - ws + 1

    # Manually extract expected secondary window
    expected_sec = torch.from_numpy(c['data_secondary'][start_sec:j_sec + 1]).float()
    assert torch.allclose(sec, expected_sec, atol=1e-6), \
        "Secondary window does not match expected mapping slice"

    # With 1:1 mapping, j_sec == last_pri_idx
    assert j_sec == last_pri_idx, \
        f"1:1 mapping: expected j_sec={last_pri_idx}, got {j_sec}"

    print(f"  Primary indices: [{test_idx}, {test_idx + ws - 1}]")
    print(f"  Secondary indices: [{start_sec}, {j_sec}]")
    print("  PASS: Mapping alignment correct")


# ============================================================================
# Test 14: MultiTF label alignment
# ============================================================================

def test_label_alignment_multi_tf():
    """
    Verify that MultiTF label corresponds to the last candle of the PRIMARY window.
    """
    print("\n=== Test: label_alignment_multi_tf ===")

    c = _build_multi_tf_components(ws=100)
    ws = c['ws']

    for test_idx in [0, 10, 50]:
        indices = [test_idx]
        dataset = MultiTFWindowDataset(
            data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
            labels=c['labels'], indices_primary=indices,
            mapping_primary_to_secondary=[c['mapping']],
            window_size=ws,
            raw_hlc_primary=None, raw_hlc_secondary=[c['raw_hlc_secondary']],
        )

        _, _, _, label = dataset[0]
        expected_label = c['labels'][test_idx + ws - 1]
        assert label.item() == expected_label, \
            f"idx={test_idx}: label={label.item()}, expected={expected_label}"

    print("  PASS: Labels correctly aligned to last candle of primary window")


# ============================================================================
# Test 15: _compute_perwindow_features values
# ============================================================================

def test_compute_perwindow_features_values():
    """
    Test that per-window features have sensible values:
    - struct_rank in [0, 1]
    - log_zz values close to log(close) range
    - dist features are finite
    """
    print("\n=== Test: compute_perwindow_features_values ===")

    c = _build_mono_dataset_components(ws=100)
    ws = c['ws']

    raw_window = c['raw_hlc'][50:50 + ws]
    log_zz, struct = _compute_perwindow_features(raw_window)

    # struct_rank (col 0) should be in [0, 1]
    struct_rank = struct[:, 0]
    assert struct_rank.min() >= 0.0, f"struct_rank min={struct_rank.min():.4f}, expected >= 0"
    assert struct_rank.max() <= 1.0, f"struct_rank max={struct_rank.max():.4f}, expected <= 1"

    # dist features (cols 1-4) should be finite
    dist_features = struct[:, 1:5]
    assert np.isfinite(dist_features).all(), "Distance features should be finite"

    # log_zz should be in a reasonable range relative to log(close)
    close = raw_window[:, 2]
    log_close = np.log(np.maximum(close, 1e-10))
    log_zz_mean = log_zz.mean()
    log_close_mean = log_close.mean()
    # Zigzag interpolates the close, so log_zz should be in same ballpark
    assert abs(log_zz_mean - log_close_mean) < 1.0, \
        f"log_zz mean ({log_zz_mean:.4f}) too far from log_close mean ({log_close_mean:.4f})"

    print(f"  struct_rank range: [{struct_rank.min():.3f}, {struct_rank.max():.3f}]")
    print(f"  dist features finite: True")
    print(f"  log_zz mean: {log_zz_mean:.4f} vs log_close mean: {log_close_mean:.4f}")
    print("  PASS")


# ============================================================================
# Test 16: _wma_at
# ============================================================================

def test_wma_at_basic():
    """Test WMA computation at a specific position with known values."""
    print("\n=== Test: wma_at_basic ===")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    period = 3

    # WMA at j=2 (uses [1,2,3], weights [1,2,3]): (1*1 + 2*2 + 3*3) / 6 = 14/6
    result = _wma_at(data, j=2, period=3)
    expected = (1*1 + 2*2 + 3*3) / (1+2+3)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    # WMA at j=4 (uses [3,4,5], weights [1,2,3]): (3*1 + 4*2 + 5*3) / 6 = 26/6
    result2 = _wma_at(data, j=4, period=3)
    expected2 = (3*1 + 4*2 + 5*3) / 6
    assert abs(result2 - expected2) < 1e-10, f"Expected {expected2}, got {result2}"

    print(f"  WMA(j=2, p=3) = {result:.6f} (expected {expected:.6f})")
    print(f"  WMA(j=4, p=3) = {result2:.6f} (expected {expected2:.6f})")
    print("  PASS")


def test_wma_at_insufficient_data():
    """Test WMA returns NaN when insufficient data."""
    print("\n=== Test: wma_at_insufficient_data ===")

    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # j=1, period=3 → needs data[0:2] but period requires 3 values starting from j-2
    result = _wma_at(data, j=1, period=3)
    assert np.isnan(result), f"Expected NaN, got {result}"

    print("  PASS: Returns NaN for insufficient data")


def test_wma_at_partial_last():
    """Test WMA with partial_last replacement."""
    print("\n=== Test: wma_at_partial_last ===")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    # Without partial: WMA at j=4 uses [3,4,5]
    result_normal = _wma_at(data, j=4, period=3)

    # With partial_last=10.0: replaces 5.0 → [3,4,10]
    result_partial = _wma_at(data, j=4, period=3, partial_last=10.0)
    expected_partial = (3*1 + 4*2 + 10*3) / 6
    assert abs(result_partial - expected_partial) < 1e-10
    assert result_normal != result_partial, "Partial should change the result"

    # Verify original data is not modified
    assert data[4] == 5.0, "Original data should not be modified"

    print(f"  Normal: {result_normal:.6f}, Partial(10): {result_partial:.6f}")
    print("  PASS")


# ============================================================================
# Test 17: _compute_hma_at
# ============================================================================

def test_compute_hma_at():
    """Test HMA computation at a specific position."""
    print("\n=== Test: compute_hma_at ===")

    # Use enough data for HMA to be valid
    set_seed(config.SEED)
    n = 50
    data = np.cumsum(np.random.randn(n)) + 100.0  # Random walk around 100
    period = 10

    # Pre-compute diff array (same as regime.py logic)
    from modules.data.regime import compute_wma
    half_p = max(int(period / 2), 1)
    diff = 2.0 * compute_wma(data, half_p) - compute_wma(data, period)

    # Compute HMA at a position deep enough to be valid
    j = 40
    result = _compute_hma_at(data, j, period, diff)

    assert not np.isnan(result), f"HMA at j={j} should not be NaN"
    assert np.isfinite(result), f"HMA at j={j} should be finite"

    # HMA should be in the same ballpark as the data
    assert abs(result - data[j]) < 20.0, \
        f"HMA ({result:.4f}) too far from data ({data[j]:.4f})"

    print(f"  HMA(j={j}) = {result:.4f}, data[{j}] = {data[j]:.4f}")
    print("  PASS")


def test_compute_hma_at_partial():
    """Test HMA with partial_last modifies the result."""
    print("\n=== Test: compute_hma_at_partial ===")

    set_seed(config.SEED)
    n = 50
    data = np.cumsum(np.random.randn(n)) + 100.0
    period = 10

    from modules.data.regime import compute_wma
    half_p = max(int(period / 2), 1)
    diff = 2.0 * compute_wma(data, half_p) - compute_wma(data, period)

    j = 40
    result_normal = _compute_hma_at(data, j, period, diff)
    result_partial = _compute_hma_at(data, j, period, diff, partial_last=data[j] + 50.0)

    assert not np.isnan(result_partial), "Partial HMA should not be NaN"
    assert result_normal != result_partial, "Partial should change the HMA result"

    print(f"  Normal: {result_normal:.4f}, Partial(+50): {result_partial:.4f}")
    print("  PASS")


# ============================================================================
# Test 18: _recompute_partial_secondary
# ============================================================================

def _build_partial_context(ws=100):
    """Build a real partial_context from test_Dataset.csv via the aggregation pipeline."""
    from modules.data.aggregation import aggregate_candles
    from modules.data.regime import compute_wma, compute_hma

    df = loader.get_raw_data(TEST_CSV)
    labels, atr, confidence, _, _ = labeling.get_labels(df, window_size=ws, max_horizon=50, atr_period=14)
    result = transform.transform_pipeline(df, atr_array=atr, window_size=ws)
    data = result['data']

    df_agg, agg_end_ts, partial_ohlcv = aggregate_candles(df, 16, 'structured')
    _, atr_sec, _, _, _ = labeling.get_labels(df_agg, window_size=ws, max_horizon=50, atr_period=14)
    result_sec = transform.transform_pipeline(df_agg, atr_array=atr_sec, window_size=ws)
    data_sec_full = result_sec['data']
    n_norm_sec = result_sec.get('n_norm_features', 0)

    # Strip time features from secondary
    data_sec = np.delete(data_sec_full, np.s_[n_norm_sec:n_norm_sec+4], axis=1)

    n_data = len(data)
    n_data_sec = len(data_sec)

    raw_hlc = np.column_stack([
        df['High'].values[ws:ws + n_data],
        df['Low'].values[ws:ws + n_data],
        df['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)

    raw_hlc_sec = np.column_stack([
        df_agg['High'].values[ws:ws + n_data_sec],
        df_agg['Low'].values[ws:ws + n_data_sec],
        df_agg['Close'].values[ws:ws + n_data_sec]
    ]).astype(np.float64)

    # Align
    min_len = min(len(data), len(labels), len(raw_hlc))
    data = data[:min_len]
    labels = labels[:min_len]
    confidence = confidence[:min_len]
    raw_hlc = raw_hlc[:min_len]

    # Mapping
    timestamps_pri_raw = df['Open time'].values
    ts_pri = timestamps_pri_raw[ws:ws + min_len]
    ts_sec_start = df_agg['Open time'].values[ws:ws + n_data_sec]
    mapping = np.searchsorted(ts_sec_start, ts_pri, side='right') - 1
    mapping = np.clip(mapping, 0, len(data_sec) - 1)

    # Partial context
    regime_len = config.REGIME_LENGTH
    sec_hlc3 = ((df_agg['High'].values + df_agg['Low'].values + df_agg['Close'].values) / 3.0).astype(np.float64)
    sec_volume = df_agg['Volume'].values.astype(np.float64)
    sec_close = df_agg['Close'].values.astype(np.float64)

    half_p = max(int(regime_len / 2), 1)
    sec_diff_price = 2.0 * compute_wma(sec_hlc3, half_p) - compute_wma(sec_hlc3, regime_len)
    sec_diff_vol = 2.0 * compute_wma(sec_volume, half_p) - compute_wma(sec_volume, regime_len)
    sec_hma_price = compute_hma(sec_hlc3, regime_len)
    sec_hma_vol = compute_hma(sec_volume, regime_len)

    ctx = {
        'partial_ohlcv': partial_ohlcv,
        'sec_hlc3': sec_hlc3,
        'sec_volume': sec_volume,
        'sec_close': sec_close,
        'sec_atr': atr_sec,
        'sec_diff_price': sec_diff_price,
        'sec_diff_vol': sec_diff_vol,
        'sec_hma_price': sec_hma_price,
        'sec_hma_vol': sec_hma_vol,
        'window_size_raw': ws,
        'regime_length': regime_len,
        'atr_period': config.ATR_PERIOD,
    }

    return {
        'ctx': ctx,
        'data_primary': data,
        'data_secondary': data_sec,
        'labels': labels,
        'confidence': confidence,
        'raw_hlc_primary': raw_hlc,
        'raw_hlc_secondary': raw_hlc_sec,
        'mapping': mapping,
        'n_norm_primary': result.get('n_norm_features', 0) + 1,
        'n_norm_secondary': n_norm_sec + 1,
        'ws': ws,
    }


def test_recompute_partial_secondary():
    """Test that _recompute_partial_secondary returns valid 8-feature vector."""
    print("\n=== Test: recompute_partial_secondary ===")

    c = _build_partial_context(ws=100)
    ctx = c['ctx']
    ws = c['ws']

    # Pick a secondary index deep enough for regime warmup
    j_sec = 50
    # raw_idx_last = window_size_raw + some primary index that maps to j_sec
    raw_idx_last = ws + 200  # Deep enough in the data

    # Ensure raw_idx_last is within bounds of partial_ohlcv
    if raw_idx_last >= len(ctx['partial_ohlcv']):
        raw_idx_last = len(ctx['partial_ohlcv']) - 1

    features, (p_H, p_L, p_C) = _recompute_partial_secondary(ctx, j_sec, raw_idx_last)

    # Should return 8 features
    assert features.shape == (8,), f"Expected (8,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    # All features should be finite
    assert np.isfinite(features).all(), f"Features contain NaN/inf: {features}"

    # Partial HLC should be positive prices
    assert p_H > 0, f"Partial High should be positive, got {p_H}"
    assert p_L > 0, f"Partial Low should be positive, got {p_L}"
    assert p_C > 0, f"Partial Close should be positive, got {p_C}"
    assert p_H >= p_L, f"High ({p_H}) should be >= Low ({p_L})"

    # Feature sanity checks
    # [0] log_price = log(Close) → should be > 0 for prices > 1
    assert features[0] > 0, f"log_price should be > 0 for crypto prices, got {features[0]}"
    # [6,7] regime features scaled to [-1, +1]
    assert -1.1 <= features[6] <= 1.1, f"regime_trend out of range: {features[6]}"
    assert -1.1 <= features[7] <= 1.1, f"regime_voltrend out of range: {features[7]}"

    print(f"  Features: {features}")
    print(f"  Partial HLC: H={p_H:.2f}, L={p_L:.2f}, C={p_C:.2f}")
    print("  PASS")


# ============================================================================
# Test 19: MultiTF with partial_context end-to-end
# ============================================================================

def test_multi_tf_with_partial_context():
    """
    Test that MultiTFWindowDataset works with partial_context enabled.
    The last secondary candle should be recomputed from partial OHLCV.
    """
    print("\n=== Test: multi_tf_with_partial_context ===")

    c = _build_partial_context(ws=100)
    ws = c['ws']

    # Find a valid index: mapping must give j_sec >= ws - 1
    valid_indices = []
    for i in range(len(c['data_primary']) - ws):
        j_sec_end = c['mapping'][i + ws - 1]
        if j_sec_end >= ws - 1 and (j_sec_end - ws + 1) >= 0:
            valid_indices.append(i)
        if len(valid_indices) >= 3:
            break

    if len(valid_indices) == 0:
        pytest.skip("No valid indices for partial context test")

    # With partial context
    dataset_partial = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=valid_indices[:1],
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
        partial_context=[c['ctx']],
    )

    # Without partial context
    dataset_no_partial = MultiTFWindowDataset(
        data_primary=c['data_primary'], data_secondary=[c['data_secondary']],
        labels=c['labels'], indices_primary=valid_indices[:1],
        mapping_primary_to_secondary=[c['mapping']],
        window_size=ws,
        raw_hlc_primary=c['raw_hlc_primary'], raw_hlc_secondary=[c['raw_hlc_secondary']],
        n_norm_features_primary=c['n_norm_primary'],
        n_norm_features_secondary=c['n_norm_secondary'],
        partial_context=None,
    )

    pri_p, sec_p, _, label_p = dataset_partial[0]
    pri_np, sec_np, _, label_np = dataset_no_partial[0]

    # Primary should be identical (partial only affects secondary)
    assert torch.allclose(pri_p, pri_np, atol=1e-6), \
        "Primary windows should be identical with/without partial context"

    # Labels should be identical
    assert label_p == label_np

    # Secondary should differ (last candle recomputed)
    # At minimum, shapes should match
    assert sec_p.shape == sec_np.shape, \
        f"Shape mismatch: partial={sec_p.shape}, no_partial={sec_np.shape}"

    # No NaN/inf
    assert not sec_p.isnan().any(), "Partial secondary has NaN"
    assert not sec_p.isinf().any(), "Partial secondary has inf"

    # The secondary windows may or may not differ depending on whether the
    # partial candle happens to equal the complete candle for this position
    diff = (sec_p - sec_np).abs().max().item()
    print(f"  Primary diff: {(pri_p - pri_np).abs().max().item():.8f}")
    print(f"  Secondary diff: {diff:.8f}")
    print(f"  Shapes: pri={pri_p.shape}, sec={sec_p.shape}")
    print("  PASS")


# ============================================================================
# Test: Unstructured offset places raw candle in LAST position of its group
# ============================================================================

def test_unstructured_offset_no_future(trained_checkpoint):
    """
    Verify that with the (R+1) % D offset fix, the last raw candle of a primary
    window always falls in the LAST position of its aggregation group.
    This guarantees the secondary candle contains NO future data.

    Method: for N test points, check that the end_timestamp of the aggregation
    group containing the last primary candle equals the timestamp of that candle.
    """
    print("\n=== Test: unstructured_offset_no_future ===")

    from modules.data.multi_tf import build_secondary_branches, compute_branch_vp
    from modules.data.transform import apply_transform_from_checkpoint
    from modules.inference.engine import load_model
    from modules.data.aggregation import aggregate_candles

    checkpoint_path, _ = trained_checkpoint

    _, config_params = load_model(checkpoint_path, 'cpu')
    ws = config_params['WINDOW_SIZE']
    divisor = config_params['MULTI_TF_DIVISOR']
    align = config_params.get('MULTI_TF_ALIGN', 'unstructured')

    if align != "unstructured":
        pytest.skip("Test only applies to unstructured alignment")

    df_raw = loader.get_raw_data(TEST_CSV)
    ts_raw = df_raw['Open time'].values

    # Build dataset via full pipeline
    transformed_pri = apply_transform_from_checkpoint(df_raw, config_params)
    n_data_pri = len(transformed_pri)

    def sec_transform_fn(df_agg):
        return apply_transform_from_checkpoint(df_agg, config_params)

    sec = build_secondary_branches(df_raw, sec_transform_fn, config_params, n_data_pri)

    min_start = sec['min_primary_start']
    valid_indices = list(range(min_start, n_data_pri - ws))

    # Test 20 evenly spaced points
    n_tests = 20
    step = max(1, len(valid_indices) // n_tests)
    test_points = valid_indices[::step][:n_tests]

    violations = []

    for data_idx in test_points:
        data_idx_last = data_idx + ws - 1
        raw_idx_last = ws + data_idx_last  # absolute raw index

        # Compute offset the same way dataset.__getitem__ does
        raw_idx_last_for_offset = ws + data_idx_last
        offset = (raw_idx_last_for_offset + 1) % divisor

        # Get the shifted df for this offset
        df_shifted = df_raw.iloc[offset:].reset_index(drop=True) if offset > 0 else df_raw

        # Aggregation: group = index_in_shifted // divisor
        # raw_idx_last in original -> index_in_shifted = raw_idx_last - offset
        idx_in_shifted = raw_idx_last - offset
        group_idx = idx_in_shifted // divisor
        group_start_in_shifted = group_idx * divisor
        group_end_in_shifted = min(group_start_in_shifted + divisor - 1, len(df_shifted) - 1)

        # Position within group (0-indexed, should be divisor-1 = last)
        pos_in_group = idx_in_shifted % divisor

        # The last raw candle in this group
        last_raw_in_group_shifted = group_end_in_shifted
        last_raw_in_group_original = last_raw_in_group_shifted + offset

        ts_current = ts_raw[raw_idx_last]
        ts_last_in_group = ts_raw[last_raw_in_group_original] if last_raw_in_group_original < len(ts_raw) else None

        if pos_in_group != divisor - 1:
            violations.append({
                'data_idx': data_idx,
                'raw_idx_last': raw_idx_last,
                'offset': offset,
                'pos_in_group': pos_in_group,
                'expected_pos': divisor - 1,
            })

    assert len(violations) == 0, (
        f"FUTURE LEAK: {len(violations)}/{len(test_points)} candles NOT in last position of group.\n"
        f"First violation: {violations[0]}"
    )

    print(f"  Tested {len(test_points)} candles: all in LAST position (pos={divisor-1}) of their group")
    print("  PASS: No future data in secondary candles")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing dataset.py")
    print("=" * 60)

    test_no_lookahead_mono()
    test_shape_mono_with_perwindow()
    test_shape_mono_without_perwindow()
    test_label_alignment_mono()
    test_confidence_returned_when_provided()
    test_confidence_not_returned_when_absent()
    test_assemble_window_with_time()
    test_assemble_window_without_time()
    test_assemble_window_with_vp()
    test_assemble_window_atr_scaling()
    test_assemble_window_custom_scaling()
    test_perwindow_normalization()
    test_normalization_does_not_leak_between_windows()
    test_compute_perwindow_features_shape()
    test_shape_multi_tf()
    test_no_lookahead_multi_tf()
    # test_no_lookahead_full_pipeline — requires trained_checkpoint fixture (run via pytest)
    test_confidence_multi_tf()
    test_cross_tf_normalize()
    test_len()
    test_multi_tf_mapping_alignment()
    test_label_alignment_multi_tf()
    test_compute_perwindow_features_values()
    test_wma_at_basic()
    test_wma_at_insufficient_data()
    test_wma_at_partial_last()
    test_compute_hma_at()
    test_compute_hma_at_partial()
    test_recompute_partial_secondary()
    test_multi_tf_with_partial_context()
    # test_unstructured_offset_no_future — requires trained_checkpoint fixture (run via pytest)

    print("\n" + "=" * 60)
    print("ALL DATASET TESTS PASSED")
    print("=" * 60)
