"""
Fractal ZigZag feature extraction - Causal, with forced alternation.

Two modes:
1. Global interpolation: piecewise-linear zigzag line
2. Structural features (5): position + distances to S/R pivots (spatial awareness)

Causality: A pivot at index T is only visible at T + length (confirmation delay).
Alternation: Consecutive pivots always alternate high/low (TradingView style).
"""

import numpy as np


def detect_pivots(high, low, length):
    """
    Detect causal alternating zigzag pivots from fractal extrema.

    Causality: A pivot at index T is only visible at T + length (confirmation delay).
    Alternation: Consecutive pivots always alternate high/low (TradingView style).

    Args:
        high: numpy array of High prices
        low: numpy array of Low prices
        length: Fractal radius (pivot confirmed when it's the extremum
                in a window of 2*length+1 candles).

    Returns:
        tuple: (pivot_indices, pivot_prices, pivot_types)
            - pivot_indices: list of confirmed pivot indices
            - pivot_prices: list of confirmed pivot prices
            - pivot_types: list of pivot types (1=high, -1=low)
    """
    n = len(high)

    # --- Step 1: Detect raw fractals ---
    is_pivot_high = np.zeros(n, dtype=bool)
    is_pivot_low = np.zeros(n, dtype=bool)

    for i in range(length, n - length):
        window_high = high[i - length: i + length + 1]
        window_low = low[i - length: i + length + 1]

        if high[i] == np.max(window_high):
            is_pivot_high[i] = True
        if low[i] == np.min(window_low):
            is_pivot_low[i] = True

    # --- Step 2: Build alternating pivot sequence (causal) ---
    confirmed_pivots = []  # List of (index, price, type) with forced alternation

    for i in range(length, n):
        # Check if a pivot was confirmed at (i - length)
        check_idx = i - length

        new_pivot = None
        if is_pivot_high[check_idx] and is_pivot_low[check_idx]:
            # Both high and low at same candle: pick the one that alternates
            if len(confirmed_pivots) == 0 or confirmed_pivots[-1][2] == -1:
                new_pivot = (check_idx, high[check_idx], 1)
            else:
                new_pivot = (check_idx, low[check_idx], -1)
        elif is_pivot_high[check_idx]:
            new_pivot = (check_idx, high[check_idx], 1)
        elif is_pivot_low[check_idx]:
            new_pivot = (check_idx, low[check_idx], -1)

        # Skip corrupted data (price = 0)
        if new_pivot is not None and new_pivot[1] <= 0:
            new_pivot = None

        # Enforce alternation
        if new_pivot is not None:
            if len(confirmed_pivots) == 0:
                confirmed_pivots.append(new_pivot)
            elif new_pivot[2] != confirmed_pivots[-1][2]:
                # Different type: normal alternation, append
                confirmed_pivots.append(new_pivot)
            else:
                # Same type: keep the more extreme one (replace last)
                if new_pivot[2] == 1 and new_pivot[1] > confirmed_pivots[-1][1]:
                    confirmed_pivots[-1] = new_pivot
                elif new_pivot[2] == -1 and new_pivot[1] < confirmed_pivots[-1][1]:
                    confirmed_pivots[-1] = new_pivot

    pivot_indices = [p[0] for p in confirmed_pivots]
    pivot_prices = [p[1] for p in confirmed_pivots]
    pivot_types = [p[2] for p in confirmed_pivots]

    return pivot_indices, pivot_prices, pivot_types


def compute_zigzag_interpolation(high, low, close, length):
    """
    Compute causal piecewise-linear zigzag interpolation.

    At each timestep i, the value is the linear interpolation between
    the two most recent confirmed pivots. Before 2 pivots are confirmed,
    the close price is used as fallback.

    Args:
        high, low, close: numpy arrays of prices
        length: Fractal radius

    Returns:
        tuple: (zz_interp, pivot_indices, pivot_prices, pivot_types)
            - zz_interp: numpy array (n,) with interpolated zigzag values
    """
    pivot_indices, pivot_prices, pivot_types = detect_pivots(
        high, low, length
    )

    n = len(close)
    zz_interp = np.copy(close)  # Fallback = close price

    if len(pivot_indices) < 2:
        return zz_interp, pivot_indices, pivot_prices, pivot_types

    # Build causal interpolation: at each position, use only confirmed pivots
    # Pivot at index pi is confirmed at pi + length
    pivot_confirm = [(pi + length, pp) for pi, pp in zip(pivot_indices, pivot_prices)]

    # Fill interpolation segment by segment
    confirmed_so_far = []
    pc_idx = 0  # Next pivot to confirm

    for i in range(n):
        # Confirm any pivots that become visible at time i
        while pc_idx < len(pivot_confirm) and pivot_confirm[pc_idx][0] <= i:
            confirmed_so_far.append(pivot_confirm[pc_idx])
            pc_idx += 1

        if len(confirmed_so_far) >= 2:
            # Interpolate between last two confirmed pivots
            t1, p1 = confirmed_so_far[-2]
            t2, p2 = confirmed_so_far[-1]
            if i <= t2:
                # Between or at the last two pivots
                if t2 != t1:
                    alpha = (i - t1) / (t2 - t1)
                    zz_interp[i] = p1 + alpha * (p2 - p1)
                else:
                    zz_interp[i] = p2
            else:
                # After last confirmed pivot: hold at last pivot value
                zz_interp[i] = p2
        elif len(confirmed_so_far) == 1:
            zz_interp[i] = confirmed_so_far[0][1]

    return zz_interp, pivot_indices, pivot_prices, pivot_types


def compute_structural_features(close, pivot_indices, pivot_prices, length, lookback):
    """
    Compute 5 structural features from zigzag pivots (spatial awareness).

    For each candle at time i, uses confirmed pivots within [i-lookback, i].
    All features are stationary by construction (log ratios and percentile rank).

    Features:
        struct_rank:  Percentile rank of close among window pivots [0, 1].
                      0.0 = below all pivots (crash), 1.0 = above all (discovery).
        dist_res_1:   Log distance to nearest resistance (pivot above).
                      Positive = inside structure, negative = discovery.
        dist_res_2:   Log distance to 2nd nearest resistance.
                      Negative when < 2 pivots above (edge of structure).
        dist_sup_1:   Log distance to nearest support (pivot below).
                      Positive = inside structure, negative = crash.
        dist_sup_2:   Log distance to 2nd nearest support.
                      Negative when < 2 pivots below (edge of structure).

    Args:
        close: numpy array of close prices
        pivot_indices: list of pivot indices from detect_pivots
        pivot_prices: list of pivot prices
        length: zigzag fractal radius (confirmation delay = length candles)
        lookback: rolling window size for structural context

    Returns:
        numpy array (n, 5) with structural features
    """
    n = len(close)
    n_pivots = len(pivot_indices)

    # Confirmation times: pivot at index pi is confirmed at pi + length
    confirm_times = [pi + length for pi in pivot_indices]

    out = np.zeros((n, 5))

    # Sliding window pointers for O(n + p) complexity
    next_confirm = 0      # Next pivot to confirm
    window_start_ptr = 0  # Oldest pivot still in lookback window

    for i in range(n):
        price = close[i]
        if price <= 0:
            continue

        # Confirm new pivots visible at time i
        while next_confirm < n_pivots and confirm_times[next_confirm] <= i:
            next_confirm += 1

        # Advance window start pointer (evict old pivots)
        window_start = max(0, i - lookback)
        while window_start_ptr < next_confirm and pivot_indices[window_start_ptr] < window_start:
            window_start_ptr += 1

        # No pivots in window → all zeros
        if window_start_ptr >= next_confirm:
            continue

        # Collect pivot prices in current window
        prices_in_window = [pivot_prices[j] for j in range(window_start_ptr, next_confirm)]

        # Separate pivots above and below current price
        above = sorted([p for p in prices_in_window if p > price])    # ascending (nearest first)
        below = sorted([p for p in prices_in_window if p <= price],
                        reverse=True)                                  # descending (nearest first)

        # --- struct_rank: fraction of pivots at or below price ---
        out[i, 0] = len(below) / len(prices_in_window)

        # --- dist_res_1: nearest resistance ---
        if len(above) >= 1:
            out[i, 1] = np.log(above[0] / price)
        else:
            # Discovery: all pivots below → distance to highest (negative)
            out[i, 1] = np.log(max(prices_in_window) / price)

        # --- dist_res_2: 2nd nearest resistance ---
        if len(above) >= 2:
            out[i, 2] = np.log(above[1] / price)
        else:
            # Edge of structure or discovery → negative
            out[i, 2] = -abs(out[i, 1])

        # --- dist_sup_1: nearest support ---
        if len(below) >= 1:
            out[i, 3] = np.log(price / below[0])
        else:
            # Crash: all pivots above → distance to lowest (negative)
            out[i, 3] = np.log(price / min(prices_in_window))

        # --- dist_sup_2: 2nd nearest support ---
        if len(below) >= 2:
            out[i, 4] = np.log(price / below[1])
        else:
            # Edge of structure or crash → negative
            out[i, 4] = -abs(out[i, 3])

    return out
