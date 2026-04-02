"""
Analyze label distribution for any dataset with configurable labeling parameters.

Usage:
    python -m modules.tools.analyze_labels data/BTC_1h_agg.csv
    python -m modules.tools.analyze_labels data/BTC_1h_agg.csv --atr-tp 2 --atr-sl 1 --atr-period 24 --horizon 48
    python -m modules.tools.analyze_labels data/BTC_1h_agg.csv data/ETH_1h_bin.csv --atr-tp 3 --atr-sl 1
"""
import sys
import os
import argparse
import time
import numpy as np

import config
from modules.data.loader import get_raw_data
from modules.data.labeling import get_labels, compute_atr


def analyze_dataset(filepath, atr_period, atr_multiplier_tp, atr_multiplier_sl, max_horizon):
    """Full label analysis for a single dataset."""
    rr = atr_multiplier_tp / atr_multiplier_sl if atr_multiplier_sl > 0 else float('inf')
    print(f"\n{'='*70}")
    print(f"  {filepath}")
    print(f"  ATR_PERIOD={atr_period}  TP={atr_multiplier_tp}  SL={atr_multiplier_sl}  R:R={rr:.1f}:1  MAX_HORIZON={max_horizon}")
    print(f"{'='*70}")

    # Load
    start = time.time()
    df = get_raw_data(filepath)
    load_time = time.time() - start
    print(f"\n  Loaded {len(df)} rows in {load_time:.1f}s")

    # Compute labels with overridden params
    original_atr_period = config.ATR_PERIOD
    original_atr_tp = config.ATR_MULTIPLIER_TP
    original_atr_sl = config.ATR_MULTIPLIER_SL
    original_horizon = config.MAX_HORIZON
    config.ATR_PERIOD = atr_period
    config.ATR_MULTIPLIER_TP = atr_multiplier_tp
    config.ATR_MULTIPLIER_SL = atr_multiplier_sl
    config.MAX_HORIZON = max_horizon

    start = time.time()
    labels, _, _, _, _ = get_labels(df)
    label_time = time.time() - start

    config.ATR_PERIOD = original_atr_period
    config.ATR_MULTIPLIER_TP = original_atr_tp
    config.ATR_MULTIPLIER_SL = original_atr_sl
    config.MAX_HORIZON = original_horizon

    # Distribution
    total = len(labels)
    bear = np.sum(labels == 0)
    uncertain = np.sum(labels == 1)
    bull = np.sum(labels == 2)

    print(f"  Labeled {total} samples in {label_time:.1f}s\n")
    print(f"  {'Class':<20} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Bear (0)':<20} {bear:>8d} {bear/total*100:>7.2f}%")
    print(f"  {'Uncertain (1)':<20} {uncertain:>8d} {uncertain/total*100:>7.2f}%")
    print(f"  {'Bull (2)':<20} {bull:>8d} {bull/total*100:>7.2f}%")

    # Balance ratio
    min_class = min(bear, uncertain, bull)
    max_class = max(bear, uncertain, bull)
    ratio = min_class / max_class if max_class > 0 else 0
    print(f"\n  Balance ratio (min/max): {ratio:.3f}  {'OK' if ratio > 0.5 else 'DESEQUILIBRE'}")

    # Both-hit analysis — uses long scenario barriers for analysis
    atr_full = compute_atr(df, period=atr_period)
    window_size = config.WINDOW_SIZE
    valid_start = window_size
    valid_end = len(df) - max_horizon

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    indices = np.arange(valid_start, valid_end)
    atr_vals = atr_full[indices]

    nan_mask = np.isnan(atr_vals)
    nan_count = int(np.sum(nan_mask))

    valid_mask = ~nan_mask
    valid_idx = indices[valid_mask]
    valid_closes = closes[valid_idx]
    valid_atr = atr_vals[valid_mask]

    # Long scenario barriers for both-hit analysis
    tp_barriers = valid_closes + atr_multiplier_tp * valid_atr
    sl_barriers = valid_closes - atr_multiplier_sl * valid_atr

    n = len(valid_idx)
    status = np.zeros(n, dtype=np.int8)  # 0=unresolved, 1=tp, 2=sl, 3=both
    resolved = np.zeros(n, dtype=bool)

    for k in range(1, max_horizon + 1):
        if np.all(resolved):
            break
        future_idx = valid_idx + k
        in_bounds = future_idx < len(df)
        can_check = ~resolved & in_bounds
        if not np.any(can_check):
            continue
        fut_highs = highs[future_idx[can_check]]
        fut_lows = lows[future_idx[can_check]]
        tp_hit = fut_highs >= tp_barriers[can_check]
        sl_hit = fut_lows <= sl_barriers[can_check]
        both = tp_hit & sl_hit
        tp_alone = tp_hit & ~sl_hit
        sl_alone = sl_hit & ~tp_hit
        check_pos = np.where(can_check)[0]
        status[check_pos[both]] = 3
        status[check_pos[tp_alone]] = 1
        status[check_pos[sl_alone]] = 2
        resolved[check_pos[tp_hit | sl_hit]] = True

    tp_only = int(np.sum(status == 1))
    sl_only = int(np.sum(status == 2))
    both_hit = int(np.sum(status == 3))
    neither = nan_count + int(np.sum(~resolved))

    analyzed = both_hit + tp_only + sl_only + neither
    print(f"\n  Long barrier hit analysis ({analyzed} samples):")
    print(f"    TP only:   {tp_only:>7d} ({tp_only/analyzed*100:5.2f}%)")
    print(f"    SL only:   {sl_only:>7d} ({sl_only/analyzed*100:5.2f}%)")
    print(f"    Neither:   {neither:>7d} ({neither/analyzed*100:5.2f}%)")
    print(f"    Both hit:  {both_hit:>7d} ({both_hit/analyzed*100:5.2f}%)  <- ambiguous")


def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution for datasets")
    parser.add_argument("files", nargs="+", help="CSV file(s) to analyze")
    parser.add_argument("--atr-period", type=int, default=config.ATR_PERIOD,
                        help=f"ATR smoothing period (default: {config.ATR_PERIOD})")
    parser.add_argument("--atr-tp", type=float, default=config.ATR_MULTIPLIER_TP,
                        help=f"ATR TP multiplier (default: {config.ATR_MULTIPLIER_TP})")
    parser.add_argument("--atr-sl", type=float, default=config.ATR_MULTIPLIER_SL,
                        help=f"ATR SL multiplier (default: {config.ATR_MULTIPLIER_SL})")
    parser.add_argument("--horizon", type=int, default=config.MAX_HORIZON,
                        help=f"Max candles to wait (default: {config.MAX_HORIZON})")
    args = parser.parse_args()

    for filepath in args.files:
        analyze_dataset(filepath, args.atr_period, args.atr_tp, args.atr_sl, args.horizon)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
