#!/usr/bin/env python3
"""
Evaluate model on each INPUT_FILE separately.
Reports winrate, profit factor, edge/trade, and trade count on test split.
Supports fee deduction in R-multiples.

Usage:
    python -m modules.tools.eval_per_file
    python -m modules.tools.eval_per_file --fee 0.001
"""

import argparse
import numpy as np
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import config
from modules.evaluation.evaluator import ModelEvaluator
from modules.data.loader import get_raw_data
from modules.data.labeling import compute_atr, TRADE_DIRECTION


def compute_trading_stats(y_pred, long_pnl_R, short_pnl_R, prediction_target,
                          fee_in_R=None):
    """Compute winrate, profit factor, edge, trade count from P&L arrays.

    Uses TRADE_DIRECTION to determine which predictions map to long/short.
    Supports all prediction targets (triple, bull, bear, closeN).
    """
    trade_dir = TRADE_DIRECTION.get(prediction_target, {})
    if not trade_dir:
        return None

    # Build long/short masks from TRADE_DIRECTION
    long_mask = np.zeros(len(y_pred), dtype=bool)
    short_mask = np.zeros(len(y_pred), dtype=bool)
    for cls, direction in trade_dir.items():
        if direction == "long":
            long_mask |= (y_pred == cls)
        elif direction == "short":
            short_mask |= (y_pred == cls)

    # Collect P&L and fees for each trade
    pnls_list = []
    fees_list = []

    if long_mask.any():
        pnls_list.append(long_pnl_R[long_mask])
        if fee_in_R is not None:
            fees_list.append(fee_in_R[long_mask])
    if short_mask.any():
        pnls_list.append(short_pnl_R[short_mask])
        if fee_in_R is not None:
            fees_list.append(fee_in_R[short_mask])

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    if not pnls_list:
        return {"n_trades": 0, "n_long": 0, "n_short": 0,
                "winrate": 0, "pf": 0, "edge_per_trade": 0, "total_R": 0,
                "total_fees_R": 0, "avg_fee_R": 0,
                "net_winrate": 0, "net_pf": 0, "net_edge": 0, "net_total_R": 0}

    all_pnl = np.concatenate(pnls_list)
    all_fees = np.concatenate(fees_list) if fees_list else np.zeros(len(all_pnl))

    # Net P&L = gross P&L - fee per trade
    net_pnl = all_pnl - all_fees

    n_trades = len(all_pnl)

    # Gross stats
    n_wins = (all_pnl > 0).sum()
    winrate = n_wins / n_trades
    gross_gain = all_pnl[all_pnl > 0].sum() if (all_pnl > 0).any() else 0
    gross_loss = abs(all_pnl[all_pnl < 0].sum()) if (all_pnl < 0).any() else 0
    pf = gross_gain / gross_loss if gross_loss > 0 else float('inf')
    total_R = all_pnl.sum()
    edge_per_trade = total_R / n_trades

    # Net stats (after fees)
    net_wins = (net_pnl > 0).sum()
    net_winrate = net_wins / n_trades
    net_gain = net_pnl[net_pnl > 0].sum() if (net_pnl > 0).any() else 0
    net_loss = abs(net_pnl[net_pnl < 0].sum()) if (net_pnl < 0).any() else 0
    net_pf = net_gain / net_loss if net_loss > 0 else float('inf')
    net_total_R = net_pnl.sum()
    net_edge = net_total_R / n_trades

    total_fees = all_fees.sum()
    avg_fee = total_fees / n_trades

    return {
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "winrate": winrate,
        "pf": pf,
        "edge_per_trade": edge_per_trade,
        "total_R": total_R,
        "net_winrate": net_winrate,
        "net_pf": net_pf,
        "net_edge": net_edge,
        "net_total_R": net_total_R,
        "total_fees_R": total_fees,
        "avg_fee_R": avg_fee,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on each INPUT_FILE separately")
    parser.add_argument('--fee', type=float, default=0.001,
                        help='Total fee per trade as fraction (default: 0.001 = 0.1%%)')
    args = parser.parse_args()
    fee_pct = args.fee

    checkpoint = config.EVAL_CHECKPOINT
    prediction_target = config.PREDICTION_TARGET
    trade_dir = TRADE_DIRECTION.get(prediction_target, {})

    print(f"Checkpoint: {checkpoint}")
    print(f"Prediction target: {prediction_target}")
    print(f"Trade directions: {trade_dir}")
    print(f"ATR TP:SL = {config.ATR_MULTIPLIER_TP}:{config.ATR_MULTIPLIER_SL}")
    print(f"Fee: {fee_pct*100:.2f}% per trade")
    print("=" * 120)

    if not trade_dir:
        print(f"No tradable directions for target '{prediction_target}', nothing to evaluate.")
        return

    results = []

    for fpath in config.INPUT_FILES:
        fname = Path(fpath).stem
        print(f"\n--- {fname} ---")

        try:
            evaluator = ModelEvaluator(checkpoint_path=checkpoint)
            evaluator.load_checkpoint()

            dataset, y_true, timestamps, split_info = evaluator.prepare_data(
                data_path=fpath, split="test", inference_only=False
            )

            y_pred, y_probs = evaluator.predict(dataset)

            long_pnl = evaluator._dataset_long_pnl
            short_pnl = evaluator._dataset_short_pnl

            if long_pnl is None:
                print(f"  No P&L arrays available, skipping")
                continue

            # Compute fee in R from raw data
            df_raw = get_raw_data(fpath)
            ws = evaluator.config_params['WINDOW_SIZE']
            atr_p = evaluator.config_params['ATR_PERIOD']

            close = df_raw['Close'].values.astype(np.float64)
            atr = compute_atr(df_raw, period=atr_p)

            start_idx = split_info.get('raw_start_idx', ws)
            n_test = len(y_pred)

            raw_positions = np.arange(start_idx, start_idx + n_test)
            raw_positions = np.clip(raw_positions, 0, len(close) - 1)

            close_at = close[raw_positions]
            atr_at = atr[raw_positions]
            safe_atr = np.where(atr_at > 0, atr_at, 1e-10)
            fee_in_R = fee_pct * close_at / safe_atr

            stats = compute_trading_stats(y_pred, long_pnl, short_pnl, prediction_target,
                                          fee_in_R=fee_in_R)
            if stats is None:
                print(f"  No tradable directions, skipping")
                continue

            stats["file"] = fname
            results.append(stats)

            print(f"  Trades: {stats['n_trades']} (long={stats['n_long']}, short={stats['n_short']})")
            print(f"  Gross: WR={stats['winrate']:.1%}  PF={stats['pf']:.3f}  Edge={stats['edge_per_trade']:.4f}R  Total={stats['total_R']:.1f}R")
            print(f"  Fees:  avg={stats['avg_fee_R']:.4f}R/trade  total={stats['total_fees_R']:.1f}R")
            print(f"  Net:   WR={stats['net_winrate']:.1%}  PF={stats['net_pf']:.3f}  Edge={stats['net_edge']:.4f}R  Total={stats['net_total_R']:.1f}R")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Summary table
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 120)
    print(f"{'File':<20s} {'Trades':>6s} | {'WR':>6s} {'PF':>6s} {'Edge/T':>7s} {'Total':>8s} | {'AvgFee':>7s} {'Fees':>7s} | {'NetWR':>6s} {'NetPF':>6s} {'NetEdge':>8s} {'NetTotal':>9s}")
    print("-" * 120)

    sum_trades = 0
    sum_gross = 0
    sum_fees = 0
    sum_net = 0

    for r in results:
        print(f"{r['file']:<20s} {r['n_trades']:>6d} | {r['winrate']:>5.1%} {r['pf']:>6.3f} {r['edge_per_trade']:>7.4f} {r['total_R']:>8.1f} | {r['avg_fee_R']:>7.4f} {r['total_fees_R']:>7.1f} | {r['net_winrate']:>5.1%} {r['net_pf']:>6.3f} {r['net_edge']:>8.4f} {r['net_total_R']:>9.1f}")
        sum_trades += r["n_trades"]
        sum_gross += r["total_R"]
        sum_fees += r["total_fees_R"]
        sum_net += r["net_total_R"]

    print("-" * 120)
    if sum_trades > 0:
        print(f"{'TOTAL':<20s} {sum_trades:>6d} | {'':>6s} {'':>6s} {sum_gross/sum_trades:>7.4f} {sum_gross:>8.1f} | {sum_fees/sum_trades:>7.4f} {sum_fees:>7.1f} | {'':>6s} {'':>6s} {sum_net/sum_trades:>8.4f} {sum_net:>9.1f}")


if __name__ == "__main__":
    main()
