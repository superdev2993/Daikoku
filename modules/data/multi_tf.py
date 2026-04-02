"""
Shared secondary branch pipeline for multi-timeframe data preparation.

Factored out from main.py, evaluator.py, and engine.py to eliminate ~430 lines
of duplicated code. Handles both structured (single pass) and unstructured
(N-offset) alignment modes.
"""

import numpy as np

from modules.data.aggregation import aggregate_candles
from modules.data.partial_context import build_partial_context
from modules.data.volume_profile import compute_vp_features


def compute_branch_vp(df_source, config_params, ws, n_data):
    """
    Compute Volume Profile features for any data branch (primary or secondary).

    Args:
        df_source: DataFrame with High, Low, Close, Volume columns
        config_params: dict with keys VP_LOOKBACK, VP_BINS, VP_VA_PCT
        ws: WINDOW_SIZE (transform offset)
        n_data: number of transformed data points

    Returns:
        ndarray (n_data, 5)
    """
    vp_lookback = config_params['VP_LOOKBACK']
    vp_raw_start = max(0, ws - vp_lookback + 1)
    h = df_source['High'].values[vp_raw_start:ws + n_data]
    lo = df_source['Low'].values[vp_raw_start:ws + n_data]
    c = df_source['Close'].values[vp_raw_start:ws + n_data]
    v = df_source['Volume'].values[vp_raw_start:ws + n_data]

    vp = compute_vp_features(h, lo, c, v, vp_lookback, config_params['VP_BINS'], config_params['VP_VA_PCT'])

    if len(vp) < n_data:
        pad = np.zeros((n_data - len(vp), 5), dtype=np.float32)
        vp = np.concatenate([pad, vp], axis=0)

    return vp


def build_secondary_branches(df_raw, transform_fn, config_params, n_data_primary):
    """
    Build all secondary branch data for MultiTFWindowDataset.

    Handles both structured (single aggregation) and unstructured (D offsets).

    Args:
        df_raw: Raw primary DataFrame (OHLCV + Open time)
        transform_fn: Callable(df_agg: DataFrame) -> np.ndarray
                       Must return the FULL transformed array (including time+regime cols).
        config_params: dict with keys:
            MULTI_TF_DIVISOR, MULTI_TF_ALIGN, WINDOW_SIZE,
            VP_LOOKBACK, VP_BINS, VP_VA_PCT,
            REGIME_LENGTH, ATR_PERIOD
        n_data_primary: len(transformed_primary) -- for timestamp mapping

    Returns:
        dict with keys:
            data_secondary_list:     list[ndarray]
            raw_hlc_secondary_list:  list[ndarray]
            vol_secondary_list:      list[ndarray] — aggregated Volume per offset
            partial_ctx_list:        list[dict]
            mapping_list:            list[ndarray]
            vp_secondary_list:       list[ndarray]
            n_norm_secondary:        int
            min_primary_start:            int
    """
    divisor = config_params['MULTI_TF_DIVISOR']
    align = config_params['MULTI_TF_ALIGN']
    ws = config_params['WINDOW_SIZE']

    offsets = range(divisor) if align == "unstructured" else range(1)

    data_sec_list = []
    raw_hlc_sec_list = []
    vol_sec_list = []
    partial_ctx_list = []
    mapping_list = []
    vp_sec_list = []
    min_primary_starts = []
    n_norm_secondary = None

    for k in offsets:
        df_shifted = df_raw.iloc[k:].reset_index(drop=True) if k > 0 else df_raw

        df_agg_k, _, partial_ohlcv_k = aggregate_candles(df_shifted, divisor, align)

        # Transform secondary via caller-provided callable
        transformed_full_k = transform_fn(df_agg_k)

        # Compute n_norm: total cols minus 6 (4 time + 2 regime)
        n_norm_k = transformed_full_k.shape[1] - 6

        # Strip 4 time features
        data_sec_k = np.delete(transformed_full_k, np.s_[n_norm_k:n_norm_k + 4], axis=1)

        # Use offset 0 as reference for n_norm
        if k == 0:
            n_norm_secondary = n_norm_k

        n_data_sec_k = len(data_sec_k)

        # Raw H/L/C for per-window zigzag
        raw_hlc_k = np.column_stack([
            df_agg_k['High'].values[ws:ws + n_data_sec_k],
            df_agg_k['Low'].values[ws:ws + n_data_sec_k],
            df_agg_k['Close'].values[ws:ws + n_data_sec_k]
        ]).astype(np.float64)

        # Aggregated Volume for partial VP recomputation
        vol_k = df_agg_k['Volume'].values[ws:ws + n_data_sec_k].astype(np.float64)

        # VP features for this offset
        vp_k = compute_branch_vp(df_agg_k, config_params, ws, n_data_sec_k)

        # Timestamp-based mapping: primary -> secondary
        ts_pri = df_raw['Open time'].values[ws:ws + n_data_primary]
        ts_sec_k = df_agg_k['Open time'].values[ws:ws + n_data_sec_k]
        mapping_k = np.searchsorted(ts_sec_k, ts_pri, side='right') - 1
        mapping_k = np.clip(mapping_k, 0, n_data_sec_k - 1)

        # Min valid primary start for this offset
        min_start_k = 0
        for i in range(n_data_primary - ws):
            j_sec_end = mapping_k[i + ws - 1]
            if j_sec_end >= ws - 1:
                min_start_k = i
                break
        min_primary_starts.append(min_start_k)

        # Partial context for this offset
        if align == "unstructured":
            partial_ctx_k = None  # Offset fix makes last candle always complete
        else:
            partial_ctx_k = build_partial_context(df_agg_k, partial_ohlcv_k, config_params, ws, n_data_sec_k)

        data_sec_list.append(data_sec_k)
        raw_hlc_sec_list.append(raw_hlc_k)
        vol_sec_list.append(vol_k)
        partial_ctx_list.append(partial_ctx_k)
        mapping_list.append(mapping_k)
        vp_sec_list.append(vp_k)

    # Conservative: use MAX of all min starts
    min_primary_start = max(min_primary_starts)

    return {
        'data_secondary_list': data_sec_list,
        'raw_hlc_secondary_list': raw_hlc_sec_list,
        'vol_secondary_list': vol_sec_list,
        'partial_ctx_list': partial_ctx_list,
        'mapping_list': mapping_list,
        'vp_secondary_list': vp_sec_list,
        'n_norm_secondary': n_norm_secondary,
        'min_primary_start': min_primary_start,
    }
