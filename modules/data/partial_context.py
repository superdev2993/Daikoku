"""
Shared partial_context builder for real-time secondary branch simulation.

Used by main.py, evaluator.py, and engine.py to build the partial_context
dict passed to MultiTFWindowDataset. Eliminates code duplication.
"""

import numpy as np

from modules.data.regime import compute_wma, compute_hma
from modules.data.labeling import compute_atr


def build_partial_context(df_agg, partial_ohlcv, config_params, ws, n_data_sec):
    """
    Build the partial_context dict for MultiTFWindowDataset.

    Pre-computes HMA intermediates and ATR on the COMPLETE aggregated data,
    then slices ATR to align with the transformed secondary data space.

    Args:
        df_agg: DataFrame of aggregated (secondary) candles with OHLCV columns
        partial_ohlcv: array (N_raw, 5) from aggregate_candles(), expanding partial per raw candle
        config_params: dict with keys 'REGIME_LENGTH', 'ATR_PERIOD'
        ws: WINDOW_SIZE (transform offset)
        n_data_sec: number of secondary data points (len of transformed secondary data)

    Returns:
        dict: partial_context ready for MultiTFWindowDataset
    """
    regime_len = config_params['REGIME_LENGTH']
    atr_period = config_params['ATR_PERIOD']

    sec_hlc3 = ((df_agg['High'].values + df_agg['Low'].values + df_agg['Close'].values) / 3.0).astype(np.float64)
    sec_volume = df_agg['Volume'].values.astype(np.float64)
    sec_close = df_agg['Close'].values.astype(np.float64)

    # ATR computed on full df_agg, then sliced to data space
    atr_full = compute_atr(df_agg, period=atr_period)
    sec_atr = atr_full[ws:ws + n_data_sec].astype(np.float64)

    # HMA intermediates on complete aggregated data
    half_p = max(int(regime_len / 2), 1)
    sec_diff_price = 2.0 * compute_wma(sec_hlc3, half_p) - compute_wma(sec_hlc3, regime_len)
    sec_diff_vol = 2.0 * compute_wma(sec_volume, half_p) - compute_wma(sec_volume, regime_len)
    sec_hma_price = compute_hma(sec_hlc3, regime_len)
    sec_hma_vol = compute_hma(sec_volume, regime_len)

    return {
        'partial_ohlcv': partial_ohlcv,
        'sec_hlc3': sec_hlc3,
        'sec_volume': sec_volume,
        'sec_close': sec_close,
        'sec_atr': sec_atr,
        'sec_diff_price': sec_diff_price,
        'sec_diff_vol': sec_diff_vol,
        'sec_hma_price': sec_hma_price,
        'sec_hma_vol': sec_hma_vol,
        'window_size_raw': ws,
        'regime_length': regime_len,
        'atr_period': atr_period,
    }
