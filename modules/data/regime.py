"""
Market regime filter based on Hull Moving Average (HMA).

Computes trend and volume-trend oscillators by comparing HMA values
across a window. Inspired by "Regime Filter [BigBeluga]" (TradingView).

Single parameter `length` controls both HMA smoothing and sign accumulation,
matching the original TradingView script behavior.

Output: two arrays (trend, voltrend) bounded [-10, +10].
"""

import numpy as np


def compute_wma(data, period):
    """Weighted Moving Average using np.convolve."""
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights = weights / weights.sum()
    # 'valid' convolution, left-padded with NaN to preserve original length
    conv = np.convolve(data, weights[::-1], mode='valid')
    result = np.full(len(data), np.nan)
    result[period - 1:] = conv
    return result


def compute_hma(data, period):
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)

    wma_half = compute_wma(data, half_period)
    wma_full = compute_wma(data, period)

    # 2*WMA(n/2) - WMA(n)
    diff = 2.0 * wma_half - wma_full

    # Final WMA on the diff series (NaNs propagate through warmup)
    hma = compute_wma(diff, sqrt_period)
    return hma


def compute_regime_filter(df, length):
    """
    Compute trend and volume-trend regime oscillators.

    Single `length` parameter controls both HMA period and sign accumulation
    window, matching the original BigBeluga TradingView script.

    Args:
        df: DataFrame with columns High, Low, Close, Volume
        length: HMA period and lookback window (single parameter)

    Returns:
        (trend, voltrend) - two numpy arrays of shape (n,), bounded [-10, +10].
        NaN where warmup is insufficient.
    """
    hlc3 = (df['High'].values + df['Low'].values + df['Close'].values) / 3.0
    volume = df['Volume'].values.astype(np.float64)

    hma_price = compute_hma(hlc3, length)
    hma_vol = compute_hma(volume, length)

    n = len(hlc3)
    trend = np.zeros(n, dtype=np.float64)
    voltrend = np.zeros(n, dtype=np.float64)

    # Vectorized loop over lags (not over bars)
    for lag in range(1, length + 1):
        # sign(hma_now - hma_shifted_by_lag)
        shifted_price = np.roll(hma_price, lag)
        shifted_vol = np.roll(hma_vol, lag)

        # Mark rolled positions as NaN to avoid wrap-around artifacts
        shifted_price[:lag] = np.nan
        shifted_vol[:lag] = np.nan

        trend += np.where(np.isnan(hma_price) | np.isnan(shifted_price), 0,
                          np.sign(hma_price - shifted_price))
        voltrend += np.where(np.isnan(hma_vol) | np.isnan(shifted_vol), 0,
                             np.sign(hma_vol - shifted_vol))

    # Normalize to [-10, +10]
    trend = trend * 10.0 / length
    voltrend = voltrend * 10.0 / length

    # Set warmup region to NaN (2*length: HMA warmup + lookback)
    warmup = 2 * length
    trend[:warmup] = np.nan
    voltrend[:warmup] = np.nan

    return trend, voltrend
