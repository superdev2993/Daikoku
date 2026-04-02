#!/usr/bin/env python3
"""
Split downloaded CSV files into TestA/TestB/TestC date ranges.

Reads from data/download/, writes to data/TestA/, data/TestB/, data/TestC/.
Source files are NEVER modified.

All periods share the same start date (2020-10-01) with staggered ends:
  TestA: 2020-10-01 00:00:00  →  2025-01-01 00:00:00 (exclusive end)
  TestB: 2020-10-01 00:00:00  →  2025-08-01 00:00:00 (exclusive end)
  TestC: 2020-10-01 00:00:00  →  2026-03-14 00:00:00 (exclusive end)
"""

import os
import pandas as pd
import sys

# Source files to split
SOURCE_DIR = "data/download"
FILES = [
    "BTC_1h_bin.csv",
    "BNB_1h_bin.csv",
    "SOL_1h_bin.csv",
    "LINK_1h_bin.csv",
    "ETH_1h_bin.csv",
    "BTC_15min_bin.csv",
]

# Test period definitions: (start_inclusive, end_exclusive)
# Same start (earliest available), staggered ends (~7 months apart)
PERIODS = {
    "TestA": ("2020-10-01 00:00:00", "2025-01-01 00:00:00"),
    "TestB": ("2020-10-01 00:00:00", "2025-08-01 00:00:00"),
    "TestC": ("2020-10-01 00:00:00", "2026-03-14 00:00:00"),
}


def split_file(filename):
    """Split a single source CSV into the 3 test periods."""
    source_path = os.path.join(SOURCE_DIR, filename)
    if not os.path.exists(source_path):
        print(f"  SKIP: {source_path} not found")
        return

    # Read source
    df = pd.read_csv(source_path)
    original_count = len(df)
    print(f"\n  {filename}: {original_count} rows")

    # Parse timestamps
    df["_ts"] = pd.to_datetime(df["Open time"])

    for period_name, (start, end) in PERIODS.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Filter: start <= timestamp < end
        mask = (df["_ts"] >= start_ts) & (df["_ts"] < end_ts)
        subset = df.loc[mask].drop(columns=["_ts"])

        # Write output
        out_dir = os.path.join("data", period_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        subset.to_csv(out_path, index=False)

        # Report
        first_ts = subset["Open time"].iloc[0] if len(subset) > 0 else "N/A"
        last_ts = subset["Open time"].iloc[-1] if len(subset) > 0 else "N/A"
        print(f"    {period_name}: {len(subset):>6} rows | {first_ts} → {last_ts}")


def verify_integrity():
    """
    Two independent verification methods:
    1. Boundary check: first/last timestamp of each file matches expected range
    2. Row count consistency: rows in overlapping zones must match between periods
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 1: Boundary timestamps")
    print("=" * 70)

    all_ok = True
    for filename in FILES:
        print(f"\n  {filename}:")
        for period_name, (start, end) in PERIODS.items():
            out_path = os.path.join("data", period_name, filename)
            if not os.path.exists(out_path):
                print(f"    {period_name}: FILE MISSING!")
                all_ok = False
                continue

            df = pd.read_csv(out_path)
            if len(df) == 0:
                print(f"    {period_name}: EMPTY FILE!")
                all_ok = False
                continue

            first = pd.Timestamp(df["Open time"].iloc[0])
            last = pd.Timestamp(df["Open time"].iloc[-1])
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)

            # First row must be >= start
            ok_start = first >= start_ts
            # Last row must be < end
            ok_end = last < end_ts

            status = "OK" if (ok_start and ok_end) else "FAIL"
            if not (ok_start and ok_end):
                all_ok = False
            print(f"    {period_name}: [{status}] {first} → {last}  (expect >= {start_ts}, < {end_ts})")

    print("\n" + "=" * 70)
    print("VERIFICATION 2: Overlap row count consistency")
    print("=" * 70)

    # Overlap A∩B: rows where timestamp >= TestB_start AND < TestA_end
    # Overlap B∩C: rows where timestamp >= TestC_start AND < TestB_end
    overlaps = [
        ("A∩B", "TestA", "TestB", PERIODS["TestB"][0], PERIODS["TestA"][1]),
        ("B∩C", "TestB", "TestC", PERIODS["TestC"][0], PERIODS["TestB"][1]),
    ]

    for label, period1, period2, overlap_start, overlap_end in overlaps:
        print(f"\n  Overlap {label}: {overlap_start} → {overlap_end}")
        for filename in FILES:
            path1 = os.path.join("data", period1, filename)
            path2 = os.path.join("data", period2, filename)

            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)

            df1["_ts"] = pd.to_datetime(df1["Open time"])
            df2["_ts"] = pd.to_datetime(df2["Open time"])

            overlap_start_ts = pd.Timestamp(overlap_start)
            overlap_end_ts = pd.Timestamp(overlap_end)

            rows1 = df1[(df1["_ts"] >= overlap_start_ts) & (df1["_ts"] < overlap_end_ts)]
            rows2 = df2[(df2["_ts"] >= overlap_start_ts) & (df2["_ts"] < overlap_end_ts)]

            match = len(rows1) == len(rows2)
            if not match:
                all_ok = False

            # Also verify the actual data values match (not just count)
            if match and len(rows1) > 0:
                vals1 = rows1.drop(columns=["_ts"]).reset_index(drop=True)
                vals2 = rows2.drop(columns=["_ts"]).reset_index(drop=True)
                data_match = vals1.equals(vals2)
                if not data_match:
                    all_ok = False
                    status = "FAIL (data mismatch)"
                else:
                    status = "OK"
            elif match:
                status = "OK (0 rows)"
            else:
                status = f"FAIL ({len(rows1)} vs {len(rows2)})"

            print(f"    {filename}: [{status}] {len(rows1)} overlap rows")

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - review above")
    print("=" * 70)

    return all_ok


if __name__ == "__main__":
    print("Splitting source files into test periods...")
    print(f"Source: {SOURCE_DIR}/")
    for name, (s, e) in PERIODS.items():
        print(f"  {name}: {s} → {e} (exclusive)")

    for filename in FILES:
        split_file(filename)

    # Run verification
    ok = verify_integrity()
    sys.exit(0 if ok else 1)
