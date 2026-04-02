#!/usr/bin/env python3
"""
Build a synthetic ETF OHLCV from multiple altcoin CSVs.

Methodology: Equal-weight portfolio (buy & hold)
- At t=0: invest $1 per coin → q_i = 1 / close_i(0)
- Price: ETF_OHLC(t) = Σ(q_i × OHLC_i(t))
- Volume: ETF_V(t) = Σ(weight_i(t) × volume_i(t) × close_i(t))
  where weight_i(t) = (q_i × close_i(t)) / portfolio_value(t)

No rebalancing. Weights drift naturally with performance.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


def load_and_parse(csv_path: str) -> pd.DataFrame:
    """Load a CSV, parse timestamps, set as index."""
    df = pd.read_csv(csv_path, parse_dates=["Open time"])
    df = df.set_index("Open time").sort_index()
    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_etf(csv_paths: list[str], output_path: str) -> pd.DataFrame:
    """
    Build synthetic ETF from multiple OHLCV CSVs.

    1. Load all CSVs
    2. Inner join on common timestamps
    3. Compute equal-weight portfolio OHLCV
    """
    # --- Load all ---
    dfs = {}
    for path in csv_paths:
        name = Path(path).stem
        df = load_and_parse(path)
        if df.empty:
            print(f"WARNING: {path} is empty, skipping")
            continue
        dfs[name] = df
        print(f"  {name}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")

    if len(dfs) < 2:
        print("ERROR: Need at least 2 valid CSVs")
        sys.exit(1)

    # --- Inner join on common timestamps ---
    common_index = None
    for name, df in dfs.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    common_index = common_index.sort_values()
    print(f"\nCommon timestamps: {len(common_index)}")
    print(f"  From {common_index[0]} to {common_index[-1]}")

    # Align all dataframes to common index
    for name in dfs:
        dfs[name] = dfs[name].loc[common_index]

    n_coins = len(dfs)
    names = list(dfs.keys())

    # --- Compute quantities: q_i = 1 / close_i(0) ---
    quantities = {}
    print(f"\nPortfolio allocation ($1 per coin, {n_coins} coins):")
    for name in names:
        first_close = dfs[name]["Close"].iloc[0]
        q = 1.0 / first_close
        quantities[name] = q
        print(f"  {name}: q = {q:.8f} (first close = {first_close:.4f})")

    # --- Build ETF OHLCV ---
    etf = pd.DataFrame(index=common_index)

    # Price columns: ETF_X(t) = Σ(q_i × X_i(t))
    for col in ["Open", "High", "Low", "Close"]:
        etf[col] = sum(quantities[name] * dfs[name][col] for name in names)

    # Portfolio value at each candle (using Close)
    portfolio_value = etf["Close"]

    # Volume: ETF_V(t) = Σ(weight_i(t) × usd_volume_i(t))
    # where weight_i(t) = (q_i × close_i(t)) / portfolio_value(t)
    #   and usd_volume_i(t) = volume_i(t) × close_i(t)
    etf_volume = pd.Series(0.0, index=common_index)
    for name in names:
        coin_value = quantities[name] * dfs[name]["Close"]  # holding value
        weight = coin_value / portfolio_value               # portfolio weight
        usd_vol = dfs[name]["Volume"] * dfs[name]["Close"]  # USD turnover
        etf_volume += weight * usd_vol

    etf["Volume"] = etf_volume

    # --- Sanity checks ---
    initial_value = etf["Close"].iloc[0]
    final_value = etf["Close"].iloc[-1]
    print(f"\nETF portfolio:")
    print(f"  Initial value: ${initial_value:.4f}")
    print(f"  Final value:   ${final_value:.4f}")
    print(f"  Return:        {(final_value/initial_value - 1)*100:.1f}%")
    print(f"  Candles:       {len(etf)}")

    # Check OHLC consistency
    bad_hl = (etf["High"] < etf["Low"]).sum()
    bad_oh = (etf["High"] < etf["Open"]).sum()
    bad_ol = (etf["Low"] > etf["Open"]).sum()
    bad_ch = (etf["High"] < etf["Close"]).sum()
    bad_cl = (etf["Low"] > etf["Close"]).sum()
    if any([bad_hl, bad_oh, bad_ol, bad_ch, bad_cl]):
        print(f"\n  WARNING: OHLC inconsistencies detected:")
        print(f"    H < L: {bad_hl}, H < O: {bad_oh}, L > O: {bad_ol}, H < C: {bad_ch}, L > C: {bad_cl}")
    else:
        print("  OHLC consistency: OK")

    # --- Save ---
    etf.index.name = "Open time"
    etf.to_csv(output_path)
    print(f"\nSaved to {output_path}")

    return etf


def main():
    parser = argparse.ArgumentParser(
        description="Build synthetic ETF OHLCV from altcoin CSVs"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/Boby-ETF",
        help="Directory containing individual coin CSVs"
    )
    parser.add_argument(
        "--output", type=str, default="data/Boby-ETF/ETFcoins.csv",
        help="Output ETF CSV path"
    )
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=[],
        help="Coin names to exclude (stem of CSV filename)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_files = sorted(input_dir.glob("*_1h_bin.csv"))

    if args.exclude:
        exclude_lower = [e.lower() for e in args.exclude]
        csv_files = [f for f in csv_files if not any(
            e in f.stem.lower() for e in exclude_lower
        )]

    # Exclude the output file itself if it exists in the same dir
    output_stem = Path(args.output).stem
    csv_files = [f for f in csv_files if f.stem != output_stem]

    if not csv_files:
        print(f"ERROR: No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Building ETF from {len(csv_files)} coins:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()

    build_etf([str(f) for f in csv_files], args.output)


if __name__ == "__main__":
    main()
