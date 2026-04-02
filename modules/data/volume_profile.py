"""
Volume Profile — sliding VP features computation (standard CME methodology).

Computes 5 features at each timestep from a sliding lookback window:
  vp_poc   : log(POC_price / Close) — distance to Point of Control
  vp_vah   : log(VA_High / Close)   — distance to Value Area High
  vp_val   : log(VA_Low / Close)    — distance to Value Area Low
  vp_width : log(VA_High / VA_Low)  — Value Area width (log ratio)
  vp_skew  : weighted skewness of the VP distribution

Each position uses a LOCAL bin grid (min/max of its lookback window),
ensuring consistent bin resolution regardless of dataset size.

Volume distribution: proportional to candle/bin overlap (standard).
Value Area: CME expansion from POC (not sort-top-N).
Batch computation via Numba JIT for performance.
"""

import math
import numpy as np
from numba import njit


# ============================================================================
# Numba internals
# ============================================================================

@njit(cache=True)
def _searchsorted_right(arr, val):
    """Equivalent to np.searchsorted(arr, val, side='right')."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _compute_vp_batch(high, low, close, volume, lookback, n_bins, va_pct):
    """
    Numba-compiled sliding VP features (standard CME methodology).

    For output position i (0-based), VP is computed on candles [i, i+lookback).
    Features are relative to close[i + lookback - 1] (the "current" candle).
    """
    n = len(high)
    n_output = n - lookback + 1
    results = np.empty((n_output, 5), np.float32)

    for i in range(n_output):
        # Window min/max
        price_min = high[i]
        price_max = high[i]
        for j in range(lookback):
            if low[i + j] < price_min:
                price_min = low[i + j]
            if high[i + j] > price_max:
                price_max = high[i + j]

        current_close = close[i + lookback - 1]

        if price_max <= price_min or current_close <= 0.0:
            for f in range(5):
                results[i, f] = 0.0
            continue

        # Bin edges (equivalent to np.linspace)
        step = (price_max - price_min) / n_bins
        bin_edges = np.empty(n_bins + 1, np.float64)
        for b in range(n_bins + 1):
            bin_edges[b] = price_min + b * step
        bin_edges[n_bins] = price_max  # exact endpoint

        # Bin centers
        bin_centers = np.empty(n_bins, np.float64)
        for b in range(n_bins):
            bin_centers[b] = (bin_edges[b] + bin_edges[b + 1]) * 0.5

        # Build histogram — proportional volume distribution (standard)
        histogram = np.zeros(n_bins, np.float64)
        for j in range(lookback):
            c_low = low[i + j]
            c_high = high[i + j]
            c_vol = volume[i + j]

            if c_vol <= 0.0:
                continue

            candle_range = c_high - c_low

            if candle_range <= 0.0:
                # Doji: all volume to bin containing this price
                idx = _searchsorted_right(bin_edges, c_low) - 1
                if idx < 0:
                    idx = 0
                if idx > n_bins - 1:
                    idx = n_bins - 1
                histogram[idx] += c_vol
                continue

            # Bins this candle touches
            lo_idx = _searchsorted_right(bin_edges, c_low) - 1
            if lo_idx < 0:
                lo_idx = 0
            if lo_idx > n_bins - 1:
                lo_idx = n_bins - 1
            hi_idx = _searchsorted_right(bin_edges, c_high) - 1
            if hi_idx < 0:
                hi_idx = 0
            if hi_idx > n_bins - 1:
                hi_idx = n_bins - 1

            # Distribute proportionally to overlap
            for b in range(lo_idx, hi_idx + 1):
                overlap = min(c_high, bin_edges[b + 1]) - max(c_low, bin_edges[b])
                if overlap > 0.0:
                    histogram[b] += c_vol * (overlap / candle_range)

        total_vol = 0.0
        for b in range(n_bins):
            total_vol += histogram[b]

        if total_vol <= 0.0:
            for f in range(5):
                results[i, f] = 0.0
            continue

        # POC = bin with max volume
        poc_idx = 0
        poc_max = histogram[0]
        for b in range(1, n_bins):
            if histogram[b] > poc_max:
                poc_max = histogram[b]
                poc_idx = b
        poc_price = bin_centers[poc_idx]

        # Value Area — CME expansion from POC
        accumulated = histogram[poc_idx]
        target = total_vol * va_pct
        va_up = poc_idx + 1
        va_dn = poc_idx - 1

        while accumulated < target:
            up_vol = histogram[va_up] if va_up < n_bins else -1.0
            dn_vol = histogram[va_dn] if va_dn >= 0 else -1.0

            if up_vol < 0.0 and dn_vol < 0.0:
                break

            if up_vol >= dn_vol:  # tie: prefer up (standard)
                accumulated += up_vol
                va_up += 1
            else:
                accumulated += dn_vol
                va_dn -= 1

        # Included bins: [va_dn+1 .. va_up-1]
        va_low_price = bin_edges[va_dn + 1]
        va_high_price = bin_edges[va_up]

        # Features (log-ratios)
        vp_poc = math.log(poc_price / current_close) if poc_price > 0.0 else 0.0
        vp_vah = math.log(va_high_price / current_close) if va_high_price > 0.0 else 0.0
        vp_val = math.log(va_low_price / current_close) if va_low_price > 0.0 else 0.0
        vp_width = math.log(va_high_price / va_low_price) if va_high_price > va_low_price else 0.0

        # Skewness: weighted 3rd moment of VP distribution
        vp_norm = np.empty(n_bins, np.float64)
        for b in range(n_bins):
            vp_norm[b] = histogram[b] / total_vol

        mean_price = 0.0
        for b in range(n_bins):
            mean_price += vp_norm[b] * bin_centers[b]

        variance = 0.0
        for b in range(n_bins):
            diff = bin_centers[b] - mean_price
            variance += vp_norm[b] * diff * diff

        std_price = math.sqrt(variance) if variance > 0.0 else 0.0

        vp_skew = 0.0
        if std_price > 0.0:
            for b in range(n_bins):
                diff = (bin_centers[b] - mean_price) / std_price
                vp_skew += vp_norm[b] * diff * diff * diff

        results[i, 0] = np.float32(vp_poc)
        results[i, 1] = np.float32(vp_vah)
        results[i, 2] = np.float32(vp_val)
        results[i, 3] = np.float32(vp_width)
        results[i, 4] = np.float32(vp_skew)

    return results


# ============================================================================
# Public API
# ============================================================================

def compute_vp_features(high, low, close, volume, lookback, n_bins, va_pct):
    """
    Compute sliding Volume Profile features with per-window local bin grid.

    For output position i (0-based), the VP is computed on
    input candles [i, i + lookback). Features are relative to
    close[i + lookback - 1] (the "current" candle at that position).

    Uses standard CME methodology:
      - Volume distributed proportionally to candle/bin overlap
      - Value Area computed via expansion from POC

    Args:
        high, low, close, volume: arrays of shape (n,)
            where n >= lookback (ideally n = n_output + lookback - 1)
        lookback: number of candles in VP window
        n_bins: number of horizontal price bins
        va_pct: Value Area percentage (e.g. 0.70)

    Returns:
        features: np.ndarray of shape (n_output, 5), dtype float32
            where n_output = n - lookback + 1
            Columns: [vp_poc, vp_vah, vp_val, vp_width, vp_skew]
    """
    n = len(high)
    n_output = n - lookback + 1

    if n_output <= 0:
        return np.zeros((0, 5), dtype=np.float32)

    return _compute_vp_batch(high, low, close, volume, lookback, n_bins, va_pct)


def compute_single_vp(w_high, w_low, w_vol, current_close, n_bins, va_pct):
    """
    Compute VP features for a single window (pure numpy, no numba).

    Used by dataset.py for real-time single-window patching in __getitem__.
    Same algorithm as the batch version (standard CME methodology).

    Args:
        w_high, w_low, w_vol: arrays of shape (lookback,)
        current_close: scalar — close price of the last candle
        n_bins: number of horizontal price bins
        va_pct: Value Area percentage (e.g. 0.70)

    Returns:
        np.ndarray of shape (5,), dtype float32
    """
    price_min = w_low.min()
    price_max = w_high.max()

    if price_max <= price_min or current_close <= 0:
        return np.zeros(5, dtype=np.float32)

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    # Build histogram — proportional volume distribution (standard)
    histogram = np.zeros(n_bins, dtype=np.float64)
    lookback = len(w_high)
    for j in range(lookback):
        c_low, c_high, c_vol = w_low[j], w_high[j], w_vol[j]
        if c_vol <= 0:
            continue
        candle_range = c_high - c_low
        if candle_range <= 0:
            # Doji
            idx = min(max(int(np.searchsorted(bin_edges, c_low, side='right')) - 1, 0), n_bins - 1)
            histogram[idx] += c_vol
            continue
        lo_idx = min(max(int(np.searchsorted(bin_edges, c_low, side='right')) - 1, 0), n_bins - 1)
        hi_idx = min(max(int(np.searchsorted(bin_edges, c_high, side='right')) - 1, 0), n_bins - 1)
        for b in range(lo_idx, hi_idx + 1):
            overlap = min(c_high, bin_edges[b + 1]) - max(c_low, bin_edges[b])
            if overlap > 0:
                histogram[b] += c_vol * (overlap / candle_range)

    total_vol = histogram.sum()
    if total_vol <= 0:
        return np.zeros(5, dtype=np.float32)

    # POC = bin with max volume
    poc_idx = np.argmax(histogram)
    poc_price = bin_centers[poc_idx]

    # Value Area — CME expansion from POC
    accumulated = histogram[poc_idx]
    target = total_vol * va_pct
    va_up = poc_idx + 1
    va_dn = poc_idx - 1

    while accumulated < target:
        up_vol = histogram[va_up] if va_up < n_bins else -1.0
        dn_vol = histogram[va_dn] if va_dn >= 0 else -1.0
        if up_vol < 0 and dn_vol < 0:
            break
        if up_vol >= dn_vol:
            accumulated += up_vol
            va_up += 1
        else:
            accumulated += dn_vol
            va_dn -= 1

    va_low_price = bin_edges[va_dn + 1]
    va_high_price = bin_edges[va_up]

    # Features
    vp_poc = np.log(poc_price / current_close) if poc_price > 0 else 0.0
    vp_vah = np.log(va_high_price / current_close) if va_high_price > 0 else 0.0
    vp_val = np.log(va_low_price / current_close) if va_low_price > 0 else 0.0
    vp_width = np.log(va_high_price / va_low_price) if va_high_price > va_low_price else 0.0

    # Skewness
    vp_norm = histogram / total_vol
    mean_price = np.dot(vp_norm, bin_centers)
    variance = np.dot(vp_norm, (bin_centers - mean_price) ** 2)
    std_price = np.sqrt(variance) if variance > 0 else 0.0
    if std_price > 0:
        vp_skew = float(np.dot(vp_norm, ((bin_centers - mean_price) / std_price) ** 3))
    else:
        vp_skew = 0.0

    return np.array([vp_poc, vp_vah, vp_val, vp_width, vp_skew], dtype=np.float32)
