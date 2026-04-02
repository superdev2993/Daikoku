"""
PyTorch Dataset classes for time series sliding windows.

Provides:
- TimeSeriesWindowDataset: Single-timeframe sliding window dataset
- MultiTFWindowDataset: Multi-timeframe dual-branch dataset
- ChunkSampler: Custom sampler for chronological chunk-based shuffling

Per-window zigzag: log_zz and 5 structural features are computed inside
__getitem__() on each 200-candle window independently, avoiding look-ahead bias.
"""

import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import numpy as np

from modules.data.zigzag import compute_zigzag_interpolation, compute_structural_features
from modules.data.volume_profile import compute_single_vp
import config as cfg


# ============================================================================
# Feature name constants — single source of truth, matches _assemble_window order
# ============================================================================

OHLCV_NAMES = ['log_price', 'log_open', 'log_wick_high', 'log_wick_low', 'log_volume']
ZIGZAG_NAMES = ['log_zz']
ATR_NAMES = ['log_atr']
TIME_NAMES = ['sin_hour', 'cos_hour', 'sin_weekday', 'cos_weekday']
STRUCT_NAMES = ['struct_rank', 'dist_res_1', 'dist_res_2', 'dist_sup_1', 'dist_sup_2']
VP_NAMES = ['vp_poc', 'vp_vah', 'vp_val', 'vp_width', 'vp_skew']
REGIME_NAMES = ['regime_trend', 'regime_voltrend']
NORM_NAMES = OHLCV_NAMES + ZIGZAG_NAMES + ATR_NAMES  # 7 features z-scored per window


def get_feature_names(has_time=True, has_vp=False):
    """Return feature names in assembly order (matches _assemble_window)."""
    names = OHLCV_NAMES + ZIGZAG_NAMES + ATR_NAMES
    if has_time:
        names = names + TIME_NAMES
    names = names + STRUCT_NAMES
    if has_vp:
        names = names + VP_NAMES
    names = names + REGIME_NAMES
    return names


# ============================================================================
# Helpers for on-the-fly partial secondary candle recomputation
# ============================================================================

def _wma_at(data, j, period, partial_last=None):
    """
    Compute WMA at position j using data[j-period+1 : j+1].

    If partial_last is provided, it replaces data[j] for this computation.
    Returns NaN if insufficient data.
    """
    if j < period - 1:
        return np.nan
    window = data[j - period + 1: j + 1].copy()
    if partial_last is not None:
        window[-1] = partial_last
    weights = np.arange(1, period + 1, dtype=np.float64)
    return float(np.dot(window, weights) / weights.sum())


def _compute_hma_at(data, j, period, diff_precomputed, partial_last=None):
    """
    Compute HMA at position j with optional partial value for data[j].

    Uses pre-computed diff array for positions < j (complete data),
    and recomputes diff at j using partial_last.

    Args:
        data: Full data array (e.g. hlc3 or volume), indexed in df_agg space
        j: Position to compute HMA at (df_agg index)
        period: HMA period (REGIME_LENGTH)
        diff_precomputed: Pre-computed 2*WMA(n/2) - WMA(n) array, same indexing as data
        partial_last: If provided, replaces data[j] for WMA computation
    """
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)

    # Compute diff at j with partial value
    wma_half = _wma_at(data, j, half_period, partial_last=partial_last)
    wma_full = _wma_at(data, j, period, partial_last=partial_last)

    if np.isnan(wma_half) or np.isnan(wma_full):
        return np.nan

    diff_j = 2.0 * wma_half - wma_full

    # Build diff window: pre-computed for [j-sqrt+1, ..., j-1] + partial diff at j
    start = j - sqrt_period + 1
    if start < 0:
        return np.nan

    diff_window = np.empty(sqrt_period, dtype=np.float64)
    diff_window[:-1] = diff_precomputed[start: j]
    diff_window[-1] = diff_j

    if np.any(np.isnan(diff_window)):
        return np.nan

    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
    return float(np.dot(diff_window, weights) / weights.sum())


def _recompute_partial_secondary(ctx, j_sec, raw_idx_last):
    """
    Recompute all 8 global features for the last secondary candle using partial OHLCV.

    Returns numpy array of 8 features:
    [log_price, log_open, log_wick_high, log_wick_low, log_volume, log_atr,
     regime_trend_scaled, regime_voltrend_scaled]

    Also returns partial (H, L, C) for raw_hlc replacement.
    """
    ws = ctx['window_size_raw']
    agg_idx = ws + j_sec  # df_agg index

    # Partial OHLCV at this raw position
    p_O = float(ctx['partial_ohlcv'][raw_idx_last, 0])
    p_H = float(ctx['partial_ohlcv'][raw_idx_last, 1])
    p_L = float(ctx['partial_ohlcv'][raw_idx_last, 2])
    p_C = float(ctx['partial_ohlcv'][raw_idx_last, 3])
    p_V = float(ctx['partial_ohlcv'][raw_idx_last, 4])

    # 5 log features
    body_max = max(p_O, p_C)
    body_min = min(p_O, p_C)
    log_price = np.log(max(p_C, 1e-10))
    log_open = np.log(max(p_O, 1e-10))
    log_wick_high = np.log(p_H / body_max) if body_max > 0 else 0.0
    log_wick_low = np.log(body_min / p_L) if p_L > 0 else 0.0
    log_volume = np.log(p_V + 1e-10)

    # ATR (Wilder EMA update)
    atr_period = ctx['atr_period']
    if j_sec > 0:
        atr_prev = float(ctx['sec_atr'][j_sec - 1])
        c_prev = float(ctx['sec_close'][agg_idx - 1])
    else:
        atr_prev = abs(p_H - p_L)
        c_prev = p_C

    tr = max(p_H - p_L, abs(p_H - c_prev), abs(p_L - c_prev))
    atr_partial = (atr_prev * (atr_period - 1) + tr) / atr_period
    log_atr = np.log(atr_partial + 1e-10)

    # Regime (HMA-based oscillator)
    regime_len = ctx['regime_length']
    hlc3_partial = (p_H + p_L + p_C) / 3.0
    vol_partial = p_V

    hma_price_j = _compute_hma_at(
        ctx['sec_hlc3'], agg_idx, regime_len,
        ctx['sec_diff_price'], partial_last=hlc3_partial
    )
    hma_vol_j = _compute_hma_at(
        ctx['sec_volume'], agg_idx, regime_len,
        ctx['sec_diff_vol'], partial_last=vol_partial
    )

    # Regime oscillator: sum(sign(hma[j] - hma[j-lag])) for lag 1..length
    hma_price_full = ctx['sec_hma_price']
    hma_vol_full = ctx['sec_hma_vol']

    trend = 0.0
    voltrend = 0.0
    for lag in range(1, regime_len + 1):
        k = agg_idx - lag
        if k >= 0:
            if not np.isnan(hma_price_j) and not np.isnan(hma_price_full[k]):
                trend += np.sign(hma_price_j - hma_price_full[k])
            if not np.isnan(hma_vol_j) and not np.isnan(hma_vol_full[k]):
                voltrend += np.sign(hma_vol_j - hma_vol_full[k])

    trend = trend * 10.0 / regime_len
    voltrend = voltrend * 10.0 / regime_len
    # Scale to [-1, +1] (same as transform.py: regime / 10.0)
    regime_trend_scaled = trend / 10.0
    regime_voltrend_scaled = voltrend / 10.0

    features = np.array([
        log_price, log_open, log_wick_high, log_wick_low, log_volume,
        log_atr, regime_trend_scaled, regime_voltrend_scaled
    ], dtype=np.float32)

    return features, (p_H, p_L, p_C)


def _compute_perwindow_features(raw_hlc_window, zigzag_length=None):
    """
    Compute zigzag interpolation and structural features on a single window.

    Args:
        raw_hlc_window: numpy array (window_size, 3) with [High, Low, Close]
        zigzag_length: Fractal radius. None = use config.ZIGZAG_LENGTH

    Returns:
        tuple: (log_zz, struct_features)
            - log_zz: numpy array (window_size,) log of zigzag interpolation
            - struct_features: numpy array (window_size, 5) structural features
    """
    if zigzag_length is None:
        zigzag_length = cfg.ZIGZAG_LENGTH

    high = raw_hlc_window[:, 0]
    low = raw_hlc_window[:, 1]
    close = raw_hlc_window[:, 2]
    ws = len(close)

    zz_interp, pivot_indices, pivot_prices, _ = compute_zigzag_interpolation(
        high, low, close, length=zigzag_length
    )
    log_zz = np.log(np.maximum(zz_interp, 1e-10)).astype(np.float32)

    struct_features = compute_structural_features(
        close, pivot_indices, pivot_prices,
        length=zigzag_length, lookback=ws
    ).astype(np.float32)

    return log_zz, struct_features


def _assemble_window(global_window, log_zz_t, struct_t, has_time, vp_t=None, scaling_atr_ratio=None):
    """
    Assemble the full feature tensor from global + per-window features.

    Global layout (with time):  [5 OHLCV, log_atr, 4 time, 2 regime] = 12
    Global layout (no time):    [5 OHLCV, log_atr, 2 regime] = 8

    Without VP:
      With time:   [5 OHLCV, log_zz, log_atr, 4 time, 5 struct, 2 regime] = 18
      No time:     [5 OHLCV, log_zz, log_atr, 5 struct, 2 regime] = 14

    With VP:
      With time:   [5 OHLCV, log_zz, log_atr, 4 time, 5 struct, 5 VP, 2 regime] = 23
      No time:     [5 OHLCV, log_zz, log_atr, 5 struct, 5 VP, 2 regime] = 19

    ATR-relative scaling: distance features (struct cols 1-4, VP cols 0-3)
    are divided by ATR/Close ratio to express them in ATR units.
    struct_rank (col 0) and vp_skew (col 4) are NOT scaled.

    Args:
        global_window: tensor (ws, n_global_features)
        log_zz_t: tensor (ws, 1)
        struct_t: tensor (ws, 5)
        has_time: bool, whether global data includes 4 time features
        vp_t: tensor (ws, 5) or None — Volume Profile features
        scaling_atr_ratio: tensor (ws, 1) or None — ATR/Close ratio for scaling.
            If None, computed from this branch's own log_atr and log_price.
            Pass primary ATR ratio for secondary branch (multi-TF).

    Returns:
        tensor (ws, 18/23 or 14/19)
    """
    ohlcv = global_window[:, :5]
    log_atr = global_window[:, 5:6]

    # ATR-relative scaling: convert log-ratio distances to ATR units
    if scaling_atr_ratio is None:
        scaling_atr_ratio = torch.exp(log_atr - global_window[:, 0:1])  # (ws, 1)

    # Scale structural distance features (cols 1-4, not struct_rank at col 0)
    struct_t = struct_t.clone()
    struct_t[:, 1:] = struct_t[:, 1:] / (scaling_atr_ratio + 1e-8)

    # Scale VP distance features (cols 0-3, not vp_skew at col 4)
    if vp_t is not None:
        vp_t = vp_t.clone()
        vp_t[:, :4] = vp_t[:, :4] / (scaling_atr_ratio + 1e-8)

    if has_time:
        time_feat = global_window[:, 6:10]
        regime = global_window[:, 10:12]
        parts = [ohlcv, log_zz_t, log_atr, time_feat, struct_t]
    else:
        regime = global_window[:, 6:8]
        parts = [ohlcv, log_zz_t, log_atr, struct_t]

    if vp_t is not None:
        parts.append(vp_t)
    parts.append(regime)

    return torch.cat(parts, dim=1)


class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch Dataset for time series sliding windows.

    Creates windows of shape (WINDOW_SIZE, n_features) from transformed data.
    Computes zigzag + structural features per-window to avoid look-ahead bias.
    Labels are already in {0, 1, 2} format (no remapping needed).

    Uses indices to handle train/test split without copying data.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, indices: list, window_size: int,
                 raw_hlc: np.ndarray = None,
                 n_norm_features: int = 0,
                 zigzag_length: int = None,
                 vp_features: np.ndarray = None,
                 confidence: np.ndarray = None,
                 long_pnl_R: np.ndarray = None,
                 short_pnl_R: np.ndarray = None):
        """
        Args:
            data: ALL transformed data, shape (n_samples, 12) — global features
            labels: ALL labels {0, 1, 2}, shape (n_valid_samples,)
            indices: List of valid indices to use (e.g., [0, 1, 2, ..., 1724] for train)
            window_size: Window size (default: 200)
            raw_hlc: Raw High/Low/Close prices aligned with data, shape (n_samples, 3).
                     Required for per-window zigzag computation.
            n_norm_features: Number of first features to normalize per-window (7 after assembly)
            zigzag_length: Fractal radius for zigzag. None = use config.ZIGZAG_LENGTH
            vp_features: Precomputed VP features aligned with data, shape (n_samples, 5).
                         vp_features[i] aligns with data[i]. None = VP disabled.
            confidence: Per-sample confidence weights aligned with labels, shape (n_valid_samples,).
                        None = not returned in __getitem__ (backward compat for evaluation).
            long_pnl_R: Exact P&L in R-multiples for long trades, shape (n_valid_samples,).
                        Used for edge metrics. None = not available (closeN modes).
            short_pnl_R: Exact P&L in R-multiples for short trades, shape (n_valid_samples,).
                         Used for edge metrics. None = not available (closeN modes).

        Note:
            For each index i in indices:
            - Window: data[i:i+window_size]
            - Label: labels[i]
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.window_size = window_size
        self.indices = list(indices)
        self.n_norm_features = n_norm_features
        self.zigzag_length = zigzag_length if zigzag_length is not None else cfg.ZIGZAG_LENGTH
        # Raw H/L/C for per-window zigzag (keep as numpy for zigzag functions)
        self.raw_hlc = raw_hlc
        self.has_perwindow = raw_hlc is not None
        # Volume Profile (precomputed, aligned with data)
        self.vp_features = torch.from_numpy(vp_features).float() if vp_features is not None else None
        self.has_vp = vp_features is not None
        # Label confidence weights
        self.confidence = torch.from_numpy(confidence).float() if confidence is not None else None
        self.has_confidence = confidence is not None
        # Exact P&L arrays for edge metrics — not returned by __getitem__
        self.long_pnl_R = torch.from_numpy(long_pnl_R).float() if long_pnl_R is not None else None
        self.short_pnl_R = torch.from_numpy(short_pnl_R).float() if short_pnl_R is not None else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index in self.indices (NOT in data!)

        Returns:
            Without confidence: (window, label)
            With confidence: (window, label, conf)
            - window: Tensor of shape (window_size, n_features)
            - label: Scalar tensor in {0, 1, 2}
            - conf: Scalar tensor in [0, 1] (only if confidence was provided)

        Note:
            Label corresponds to the LAST candle of the window.
            Window [T, T+199] -> Label predicts future after T+199
        """
        # Get actual data index
        data_idx = self.indices[idx]

        # Extract global features window
        global_window = self.data[data_idx : data_idx + self.window_size]

        if self.has_perwindow:
            # Compute per-window zigzag + structural features
            raw_window = self.raw_hlc[data_idx : data_idx + self.window_size]
            log_zz, struct_features = _compute_perwindow_features(raw_window, zigzag_length=self.zigzag_length)

            log_zz_t = torch.from_numpy(log_zz).unsqueeze(1)  # (ws, 1)
            struct_t = torch.from_numpy(struct_features)        # (ws, 5)

            # Slice precomputed Volume Profile features
            vp_t = None
            if self.has_vp:
                vp_t = self.vp_features[data_idx : data_idx + self.window_size]

            # Assemble: [5 OHLCV, log_zz, log_atr, 4 time, 5 struct, (5 VP), 2 regime]
            window = _assemble_window(global_window, log_zz_t, struct_t, has_time=True, vp_t=vp_t)
        else:
            window = global_window

        # Per-window normalization: normalize first n_norm_features using window's own median/IQR
        if self.n_norm_features > 0:
            window = window.clone()  # Avoid modifying shared data
            norm_part = window[:, :self.n_norm_features]
            median = norm_part.median(dim=0).values
            q75 = torch.quantile(norm_part, 0.75, dim=0)
            q25 = torch.quantile(norm_part, 0.25, dim=0)
            iqr = q75 - q25
            window[:, :self.n_norm_features] = (norm_part - median) / (iqr + 1e-8)

        # Label for the LAST candle of the window (not the first!)
        # This ensures the model predicts the future of the last seen candle
        label = self.labels[data_idx + self.window_size - 1]

        if self.has_confidence:
            conf = self.confidence[data_idx + self.window_size - 1]
            return window, label, conf
        return window, label


class MultiTFWindowDataset(Dataset):
    """
    PyTorch Dataset for multi-timeframe sliding windows (dual branch).

    Creates aligned window pairs from primary and secondary data.
    Computes zigzag + structural features per-window on both branches.
    The secondary window ends at the same timestamp as the primary window.
    Labels come from the primary branch (predict primary direction).
    """

    def __init__(
        self,
        data_primary: np.ndarray,
        data_secondary: list,
        labels: np.ndarray,
        indices_primary: list,
        mapping_primary_to_secondary: list,
        window_size: int,
        raw_hlc_primary: np.ndarray = None,
        raw_hlc_secondary: list = None,
        vol_secondary: list = None,
        n_norm_features_primary: int = 0,
        n_norm_features_secondary: int = 0,
        partial_context: list = None,
        zigzag_length: int = None,
        vp_features_primary: np.ndarray = None,
        vp_features_secondary: list = None,
        vp_params: dict = None,
        cross_tf_normalize: bool = False,
        confidence: np.ndarray = None,
        long_pnl_R: np.ndarray = None,
        short_pnl_R: np.ndarray = None,
    ):
        """
        Args:
            data_primary: All primary transformed data, shape (n1, 12) — global features
            data_secondary: List of D ndarray, each shape (n4_k, 8) — one per offset
            labels: All primary labels {0, 1, 2}
            indices_primary: Valid primary start indices for windows
            mapping_primary_to_secondary: List of D ndarray — mapping per offset
            window_size: Window size (200)
            raw_hlc_primary: Raw H/L/C for primary branch, shape (n1, 3)
            raw_hlc_secondary: List of D ndarray — raw H/L/C per offset
            n_norm_features_primary: Features to normalize per-window for primary (7 after assembly)
            n_norm_features_secondary: Features to normalize per-window for secondary (7 after assembly)
            partial_context: List of D dicts for real-time partial secondary candle recomputation.
            zigzag_length: Fractal radius for zigzag. None = use config.ZIGZAG_LENGTH
            vp_features_primary: Precomputed VP for primary, shape (n1, 5). None = disabled.
            vp_features_secondary: List of D ndarray — VP per offset. None = disabled.
            cross_tf_normalize: If True, normalize secondary using primary's median/IQR stats.
            confidence: Per-sample confidence weights aligned with labels, shape (n_valid_samples,).
                        None = not returned in __getitem__ (backward compat for evaluation).
            long_pnl_R: Exact P&L in R-multiples for long trades, shape (n_valid_samples,).
                        Used for edge metrics. None = not available (closeN modes).
            short_pnl_R: Exact P&L in R-multiples for short trades, shape (n_valid_samples,).
                         Used for edge metrics. None = not available (closeN modes).
        """
        self.data_primary = torch.from_numpy(data_primary).float()
        self.labels = torch.from_numpy(labels).long()
        self.window_size = window_size
        self.indices = list(indices_primary)
        self.n_norm_features_primary = n_norm_features_primary
        self.n_norm_features_secondary = n_norm_features_secondary
        self.zigzag_length = zigzag_length if zigzag_length is not None else cfg.ZIGZAG_LENGTH
        # Raw H/L/C for per-window zigzag (keep as numpy)
        self.raw_hlc_primary = raw_hlc_primary
        self.has_perwindow = raw_hlc_primary is not None
        self.cross_tf_normalize = cross_tf_normalize
        # Label confidence weights
        self.confidence = torch.from_numpy(confidence).float() if confidence is not None else None
        self.has_confidence = confidence is not None
        # Exact P&L arrays for edge metrics — not returned by __getitem__
        self.long_pnl_R = torch.from_numpy(long_pnl_R).float() if long_pnl_R is not None else None
        self.short_pnl_R = torch.from_numpy(short_pnl_R).float() if short_pnl_R is not None else None

        # N-offset secondary data (lists of D arrays)
        self.divisor = len(data_secondary)
        self.data_secondary_list = [torch.from_numpy(d).float() for d in data_secondary]
        self.mapping_list = mapping_primary_to_secondary
        self.raw_hlc_secondary_list = raw_hlc_secondary
        self.vol_secondary_list = vol_secondary
        self.partial_context_list = partial_context
        self.vp_params = vp_params

        # Volume Profile (primary: single array, secondary: list of D arrays)
        self.vp_features_primary = torch.from_numpy(vp_features_primary).float() if vp_features_primary is not None else None
        self.has_vp_primary = vp_features_primary is not None
        if vp_features_secondary is not None:
            self.vp_secondary_list = [torch.from_numpy(v).float() if v is not None else None for v in vp_features_secondary]
        else:
            self.vp_secondary_list = [None] * self.divisor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (window_primary, window_secondary, label)
                - window_primary: (window_size, 18) — with time features
                - window_secondary: (window_size, 14) — without time features
                - label: scalar in {0, 1, 2}

        When partial_context is set, the last secondary candle's features are
        recomputed on-the-fly from partial OHLCV (real-time simulation: only
        raw candles observed up to the current primary timestamp are used).

        N-offset selection: for unstructured aggregation with D offsets,
        selects the offset-specific secondary data based on the raw position
        of the last primary candle modulo D.
        """
        i_pri = self.indices[idx]

        # Primary window (global features)
        global_primary = self.data_primary[i_pri : i_pri + self.window_size]

        # Select offset-specific secondary data
        data_idx_last = i_pri + self.window_size - 1
        if self.divisor > 1:
            raw_idx_last_for_offset = self.window_size + data_idx_last
            offset = (raw_idx_last_for_offset + 1) % self.divisor
        else:
            offset = 0

        data_sec = self.data_secondary_list[offset]
        mapping = self.mapping_list[offset]
        ctx = self.partial_context_list[offset] if self.partial_context_list is not None else None
        raw_hlc_sec = self.raw_hlc_secondary_list[offset] if self.raw_hlc_secondary_list is not None else None
        vp_sec = self.vp_secondary_list[offset]

        # Secondary window: find the secondary index corresponding to the last primary candle
        j_sec = mapping[data_idx_last]
        start_sec = j_sec - self.window_size + 1
        global_secondary = data_sec[start_sec : j_sec + 1]

        # Compute within-window temporal mapping: pri position → sec position
        pri_data_indices = np.arange(i_pri, i_pri + self.window_size)
        sec_data_indices = mapping[pri_data_indices]
        window_mapping = np.clip(sec_data_indices - start_sec, 0, self.window_size - 1).astype(np.int64)
        tf_mapping = torch.from_numpy(window_mapping)  # (window_size,) LongTensor

        # === PARTIAL SECONDARY CANDLE REPLACEMENT (real-time simulation) ===
        partial_hlc = None
        partial_vol = None
        if ctx is not None:
            raw_idx_last = ctx['window_size_raw'] + data_idx_last
            # Adjust for offset: partial_ohlcv_k has N-k rows (k candles dropped from start)
            raw_idx_for_partial = raw_idx_last - offset

            # Recompute all 8 global features for the last secondary candle
            partial_features, partial_hlc = _recompute_partial_secondary(
                ctx, j_sec, raw_idx_for_partial
            )
            partial_vol = float(ctx['partial_ohlcv'][raw_idx_for_partial, 4])

            # Replace last candle's global features (clone to avoid modifying shared data)
            global_secondary = global_secondary.clone()
            global_secondary[-1] = torch.from_numpy(partial_features).float()

        if self.has_perwindow:
            # Primary per-window zigzag
            raw_pri = self.raw_hlc_primary[i_pri : i_pri + self.window_size]
            log_zz_pri, struct_pri = _compute_perwindow_features(raw_pri, zigzag_length=self.zigzag_length)
            log_zz_pri_t = torch.from_numpy(log_zz_pri).unsqueeze(1)
            struct_pri_t = torch.from_numpy(struct_pri)

            # Slice precomputed Volume Profile features (primary)
            vp_pri_t = None
            if self.has_vp_primary:
                vp_pri_t = self.vp_features_primary[i_pri : i_pri + self.window_size]

            window_primary = _assemble_window(global_primary, log_zz_pri_t, struct_pri_t, has_time=True, vp_t=vp_pri_t)

            # Primary ATR/Close ratio for scaling secondary distance features
            pri_scaling_atr = torch.exp(global_primary[:, 5:6] - global_primary[:, 0:1])  # (ws, 1)

            # Secondary per-window zigzag (use partial H/L/C for last candle)
            raw_sec_window = raw_hlc_sec[start_sec : j_sec + 1]
            if partial_hlc is not None:
                raw_sec_window = raw_sec_window.copy()  # Don't modify shared array
                raw_sec_window[-1] = [partial_hlc[0], partial_hlc[1], partial_hlc[2]]
            log_zz_sec, struct_sec = _compute_perwindow_features(raw_sec_window, zigzag_length=self.zigzag_length)
            log_zz_sec_t = torch.from_numpy(log_zz_sec).unsqueeze(1)
            struct_sec_t = torch.from_numpy(struct_sec)

            # Slice precomputed Volume Profile features (secondary)
            vp_sec_t = None
            if vp_sec is not None:
                vp_sec_t = vp_sec[start_sec : j_sec + 1]

                # Patch VP for last secondary candle (avoid future data in structured mode)
                if partial_hlc is not None and self.vol_secondary_list is not None:
                    vp_sec_t = vp_sec_t.clone()
                    lookback = self.vp_params['VP_LOOKBACK']
                    vp_start = max(0, j_sec - lookback + 1)
                    w_high = raw_hlc_sec[vp_start:j_sec + 1, 0].copy()
                    w_low = raw_hlc_sec[vp_start:j_sec + 1, 1].copy()
                    w_vol = self.vol_secondary_list[offset][vp_start:j_sec + 1].copy()
                    w_high[-1] = partial_hlc[0]
                    w_low[-1] = partial_hlc[1]
                    current_close = partial_hlc[2]
                    w_vol[-1] = partial_vol
                    patched_vp = compute_single_vp(
                        w_high, w_low, w_vol, current_close,
                        self.vp_params['VP_BINS'], self.vp_params['VP_VA_PCT']
                    )
                    vp_sec_t[-1] = torch.from_numpy(patched_vp).float()

            window_secondary = _assemble_window(global_secondary, log_zz_sec_t, struct_sec_t, has_time=False, vp_t=vp_sec_t, scaling_atr_ratio=pri_scaling_atr)
        else:
            window_primary = global_primary
            window_secondary = global_secondary

        # Per-window normalization
        pri_median, pri_iqr = None, None
        if self.n_norm_features_primary > 0:
            window_primary = window_primary.clone()
            norm_part = window_primary[:, :self.n_norm_features_primary]
            pri_median = norm_part.median(dim=0).values
            q75 = torch.quantile(norm_part, 0.75, dim=0)
            q25 = torch.quantile(norm_part, 0.25, dim=0)
            pri_iqr = q75 - q25
            window_primary[:, :self.n_norm_features_primary] = (norm_part - pri_median) / (pri_iqr + 1e-8)

        if self.n_norm_features_secondary > 0:
            window_secondary = window_secondary.clone()
            norm_part = window_secondary[:, :self.n_norm_features_secondary]
            if self.cross_tf_normalize and pri_median is not None:
                # Cross-TF: normalize secondary using primary's stats
                median = pri_median[:self.n_norm_features_secondary]
                iqr = pri_iqr[:self.n_norm_features_secondary]
            else:
                median = norm_part.median(dim=0).values
                q75 = torch.quantile(norm_part, 0.75, dim=0)
                q25 = torch.quantile(norm_part, 0.25, dim=0)
                iqr = q75 - q25
            window_secondary[:, :self.n_norm_features_secondary] = (norm_part - median) / (iqr + 1e-8)

        # Label for the LAST candle of the primary window
        label = self.labels[i_pri + self.window_size - 1]

        if self.has_confidence:
            conf = self.confidence[i_pri + self.window_size - 1]
            return window_primary, window_secondary, tf_mapping, label, conf
        return window_primary, window_secondary, tf_mapping, label


class ChunkSampler(Sampler):
    """
    Custom sampler that splits each file's indices into N chronological chunks,
    groups corresponding chunks across files, and shuffles within each group.

    With ConcatDataset([file1_train, file2_train, ...]) and n_chunks=3:
        file1: [0..999]    → chunk0=[0..332]     chunk1=[333..665]     chunk2=[666..999]
        file2: [1000..1799] → chunk0=[1000..1265] chunk1=[1266..1532]  chunk2=[1533..1799]

        Group 0: [0..332] + [1000..1265] → shuffled together
        Group 1: [333..665] + [1266..1532] → shuffled together
        Group 2: [666..999] + [1533..1799] → shuffled together

        Iteration: Group 0 → Group 1 → Group 2

    When n_chunks=1: one group = all indices shuffled = identical to shuffle=True.
    """

    def __init__(self, dataset, n_chunks=1, seed=42):
        """
        Args:
            dataset: ConcatDataset or single Dataset
            n_chunks: Number of chronological chunks per file (1 = full shuffle)
            seed: Base random seed for reproducibility
        """
        super().__init__(dataset)
        self.n_chunks = max(1, n_chunks)
        self.seed = seed
        self._epoch = 0
        self._total_len = len(dataset)

        # Determine per-file index ranges
        if isinstance(dataset, ConcatDataset):
            boundaries = [0] + list(dataset.cumulative_sizes)
        else:
            boundaries = [0, len(dataset)]

        # Build groups: group[k] = list of indices from chunk k of all files
        self.groups = [[] for _ in range(self.n_chunks)]
        for f in range(len(boundaries) - 1):
            start = boundaries[f]
            end = boundaries[f + 1]
            file_len = end - start
            chunk_size = file_len // self.n_chunks
            remainder = file_len % self.n_chunks

            offset = start
            for k in range(self.n_chunks):
                # Distribute remainder to first chunks
                c_size = chunk_size + (1 if k < remainder else 0)
                self.groups[k].extend(range(offset, offset + c_size))
                offset += c_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        for group in self.groups:
            indices = list(group)
            rng.shuffle(indices)
            yield from indices
        self._epoch += 1

    def __len__(self):
        return self._total_len
