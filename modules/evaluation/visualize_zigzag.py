"""
ZigZag visualization script - Candlestick + zigzag interpolation + log signals.

Usage:
    python modules/evaluation/visualize_zigzag.py [--file FILE] [--start START] [--count COUNT] [--length LENGTH]

Examples:
    python modules/evaluation/visualize_zigzag.py
    python modules/evaluation/visualize_zigzag.py --start 5000 --count 300
    python modules/evaluation/visualize_zigzag.py --file data/BTC_15min_bin.csv --length 3
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from modules.data.loader import get_raw_data
from modules.data.zigzag import compute_zigzag_interpolation


def plot_candlesticks(ax, n, opens, highs, lows, closes):
    """Draw candlestick chart. Green=bullish, Red=bearish."""
    width = 0.6
    for i in range(n):
        if closes[i] >= opens[i]:
            color = '#26a69a'
            body_bottom = opens[i]
            body_height = closes[i] - opens[i]
        else:
            color = '#ef5350'
            body_bottom = closes[i]
            body_height = opens[i] - closes[i]

        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5)
        if body_height < 1e-10:
            body_height = (highs[i] - lows[i]) * 0.01
        ax.add_patch(plt.Rectangle(
            (i - width / 2, body_bottom), width, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5
        ))


def visualize_zigzag(file_path, start_idx, count, length):
    """Main visualization: candlestick + zigzag interp + log(zz) vs log(close)."""
    # Load data
    df = get_raw_data(file_path)
    print(f"Loaded {len(df)} candles from {file_path}")

    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    close = df['Close'].values.astype(np.float64)

    # Compute zigzag interpolation on FULL dataset
    zz_interp, pivot_indices, pivot_prices, pivot_types = compute_zigzag_interpolation(
        high, low, close, length=length
    )
    print(f"ZigZag: {len(pivot_indices)} pivots (length={length})")

    # Compute log signals
    log_zz = np.log(np.maximum(zz_interp, 1e-10))
    log_close = np.log(np.maximum(close, 1e-10))

    # --- Display range ---
    end_idx = min(start_idx + count, len(df))
    if start_idx >= len(df):
        print(f"start_idx={start_idx} exceeds data length {len(df)}")
        return

    display_slice = slice(start_idx, end_idx)
    dates = df['Open time'].values[display_slice]
    opens = df['Open'].values[display_slice].astype(np.float64)
    highs_disp = high[display_slice]
    lows_disp = low[display_slice]
    closes_disp = close[display_slice]
    zz_disp = zz_interp[display_slice]
    n_display = end_idx - start_idx

    # Filter pivots in display range
    visible_pivots = [(pi - start_idx, pp, pt) for pi, pp, pt
                      in zip(pivot_indices, pivot_prices, pivot_types)
                      if start_idx <= pi < end_idx]

    # --- Figure: 2 panels ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 1.5],
                              sharex=True)
    fig.suptitle(
        f"ZigZag Fractal (length={length}) — {file_path}\n"
        f"Candles {start_idx}–{end_idx} | {len(visible_pivots)} pivots",
        fontsize=12
    )

    # --- Panel 1: Candlestick + ZigZag interpolation ---
    ax1 = axes[0]
    plot_candlesticks(ax1, n_display, opens, highs_disp, lows_disp, closes_disp)

    # Zigzag interpolation line
    ax1.plot(range(n_display), zz_disp, color='#1976d2', linewidth=1.5, alpha=0.7,
             label='ZZ interp')

    # Pivot markers
    for pi, pp, pt in visible_pivots:
        if pt == 1:
            ax1.plot(pi, pp, 'v', color='#ef5350', markersize=7, zorder=5)
        else:
            ax1.plot(pi, pp, '^', color='#26a69a', markersize=7, zorder=5)

    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, n_display)

    # X-axis dates
    tick_step = max(1, n_display // 15)
    tick_pos = list(range(0, n_display, tick_step))
    tick_lbl = [str(dates[i])[:13] for i in tick_pos if i < len(dates)]
    ax1.set_xticks(tick_pos[:len(tick_lbl)])
    ax1.set_xticklabels(tick_lbl, rotation=45, fontsize=7)

    # --- Panel 2: log(zigzag) vs log(close) ---
    ax2 = axes[1]
    ax2.plot(range(n_display), log_close[display_slice], color='gray', linewidth=0.6, alpha=0.5,
             label='log(close)')
    ax2.plot(range(n_display), log_zz[display_slice], color='#1976d2', linewidth=1.2,
             label='log(zz_interp)')
    ax2.set_ylabel('Log price')
    ax2.set_xlabel('Candle index')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    output_path = os.path.join(project_root, 'evaluation', 'zigzag_visualization.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ZigZag fractal features")
    parser.add_argument("--file", type=str, default=config.INPUT_FILES[0],
                        help="Data file path")
    parser.add_argument("--start", type=int, default=1000,
                        help="Start candle index")
    parser.add_argument("--count", type=int, default=400,
                        help="Number of candles to display")
    parser.add_argument("--length", type=int, default=config.ZIGZAG_LENGTH,
                        help="Fractal radius")

    args = parser.parse_args()
    visualize_zigzag(args.file, args.start, args.count, args.length)
