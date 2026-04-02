"""
Tests for modules/data/partial_context.py

Covers: build_partial_context with real aggregated data.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


from conftest import TEST_CSV


@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not os.path.exists(TEST_CSV):
        pytest.skip(f"Test dataset not found: {TEST_CSV}")


def _get_aggregated_data():
    """Load and aggregate test data for partial_context tests."""
    from modules.data.loader import get_raw_data
    from modules.data.aggregation import aggregate_candles
    from modules.data.transform import transform_pipeline

    df_raw = get_raw_data(TEST_CSV)
    divisor = config.MULTI_TF_DIVISOR
    align = config.MULTI_TF_ALIGN
    df_agg, _, partial_ohlcv = aggregate_candles(df_raw, divisor, align)

    # Transform secondary to get n_data_sec
    result = transform_pipeline(df_agg)
    n_data_sec = len(result['data'])

    return df_agg, partial_ohlcv, n_data_sec


def test_build_partial_context_keys():
    """Verify build_partial_context returns all 12 expected keys."""
    from modules.data.partial_context import build_partial_context

    df_agg, partial_ohlcv, n_data_sec = _get_aggregated_data()
    ws = config.WINDOW_SIZE

    config_params = {
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
    }

    ctx = build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec)

    expected_keys = {
        'partial_ohlcv', 'sec_hlc3', 'sec_volume', 'sec_close',
        'sec_atr', 'sec_diff_price', 'sec_diff_vol',
        'sec_hma_price', 'sec_hma_vol',
        'window_size_raw', 'regime_length', 'atr_period',
    }
    assert set(ctx.keys()) == expected_keys


def test_build_partial_context_atr_shape():
    """Verify sec_atr is correctly sliced to n_data_sec length."""
    from modules.data.partial_context import build_partial_context

    df_agg, partial_ohlcv, n_data_sec = _get_aggregated_data()
    ws = config.WINDOW_SIZE

    config_params = {
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
    }

    ctx = build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec)

    assert len(ctx['sec_atr']) == n_data_sec
    assert ctx['sec_atr'].dtype == np.float64


def test_build_partial_context_hma_shape():
    """Verify HMA intermediates have the same length as full aggregated data."""
    from modules.data.partial_context import build_partial_context

    df_agg, partial_ohlcv, n_data_sec = _get_aggregated_data()
    ws = config.WINDOW_SIZE

    config_params = {
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
    }

    ctx = build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec)

    n_agg = len(df_agg)
    assert len(ctx['sec_hlc3']) == n_agg
    assert len(ctx['sec_volume']) == n_agg
    assert len(ctx['sec_close']) == n_agg
    assert len(ctx['sec_diff_price']) == n_agg
    assert len(ctx['sec_hma_price']) == n_agg


def test_build_partial_context_no_nan_in_atr():
    """Verify sec_atr has no NaN after slicing."""
    from modules.data.partial_context import build_partial_context

    df_agg, partial_ohlcv, n_data_sec = _get_aggregated_data()
    ws = config.WINDOW_SIZE

    config_params = {
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
    }

    ctx = build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec)

    assert not np.isnan(ctx['sec_atr']).any(), "sec_atr contains NaN"


def test_build_partial_context_scalar_params():
    """Verify scalar params are stored correctly."""
    from modules.data.partial_context import build_partial_context

    df_agg, partial_ohlcv, n_data_sec = _get_aggregated_data()
    ws = config.WINDOW_SIZE

    config_params = {
        'REGIME_LENGTH': config.REGIME_LENGTH,
        'ATR_PERIOD': config.ATR_PERIOD,
    }

    ctx = build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec)

    assert ctx['window_size_raw'] == ws
    assert ctx['regime_length'] == config.REGIME_LENGTH
    assert ctx['atr_period'] == config.ATR_PERIOD
