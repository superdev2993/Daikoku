"""
Data augmentation via Ornstein-Uhlenbeck process.

Generates N variant CSVs from an original OHLCV file:
- Open/Close: OU process with step = max(PCT% * body, 0.1% * price)
  Continuity enforced: open[i+1] = close[i]
- High/Low: ±PCT% of wick size, protected (high >= max(O,C), low <= min(O,C))
- Volume: ±PCT% uniform random

Usage:
    python -m modules.tools.augment_data
    python -m modules.tools.augment_data --file data/ETH_1h_bin.csv --variants 4
    python -m modules.tools.augment_data --file data/btc_futures.csv --pct 5 --variants 6
"""

import sys
import os

import argparse
import numpy as np
import pandas as pd

import config


# OU mean-reversion speed (internal, not configurable)
LAMBDA_REVERSION = 0.03


def generate_ou_process(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a normalized Ornstein-Uhlenbeck process of length n.
    Values are roughly in [-1, 1] range due to mean reversion.

    Args:
        n: Length of the process
        rng: NumPy random generator

    Returns:
        Array of shape (n,) with OU values
    """
    ou = np.zeros(n)
    for i in range(1, n):
        noise = rng.standard_normal()
        ou[i] = ou[i - 1] * (1 - LAMBDA_REVERSION) + noise
    # Normalize to [-1, 1] range
    max_abs = np.abs(ou).max()
    if max_abs > 0:
        ou = ou / max_abs
    return ou


def augment_ohlcv(df: pd.DataFrame, pct: float, seed: int) -> pd.DataFrame:
    """
    Generate one augmented variant of an OHLCV DataFrame.

    Args:
        df: Original DataFrame with Open, High, Low, Close, Volume columns
        pct: Base percentage parameter (e.g., 5 means 5% wicks/volume, derived for price)
        seed: Random seed for reproducibility

    Returns:
        Augmented DataFrame with same structure
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    open_arr = df["Open"].values.astype(np.float64).copy()
    high_arr = df["High"].values.astype(np.float64).copy()
    low_arr = df["Low"].values.astype(np.float64).copy()
    close_arr = df["Close"].values.astype(np.float64).copy()
    volume_arr = df["Volume"].values.astype(np.float64).copy()

    pct_frac = pct / 100.0          # e.g., 0.05
    price_floor = pct / 5000.0      # e.g., 5 → 0.001 = 0.1%

    # --- Open/Close: Ornstein-Uhlenbeck ---
    ou = generate_ou_process(n, rng)

    new_close = close_arr.copy()
    new_open = open_arr.copy()

    for i in range(n):
        body = abs(close_arr[i] - open_arr[i])
        price = (open_arr[i] + close_arr[i]) / 2.0
        step_scale = max(pct_frac * body, price_floor * price)
        perturbation = ou[i] * step_scale

        if i == 0:
            new_open[i] = open_arr[i] + perturbation
        else:
            new_open[i] = new_close[i - 1]  # Continuity

        new_close[i] = close_arr[i] + perturbation

    # --- High: ±PCT% of upper wick ---
    for i in range(n):
        body_top = max(new_open[i], new_close[i])
        original_body_top = max(open_arr[i], close_arr[i])
        upper_wick = high_arr[i] - original_body_top

        if upper_wick > 0:
            noise = rng.uniform(-pct_frac, pct_frac)
            new_wick = upper_wick * (1 + noise)
            new_wick = max(new_wick, 0)  # Wick cannot be negative
            high_arr[i] = body_top + new_wick
        else:
            high_arr[i] = body_top  # No wick, stays at body top

    # --- Low: ±PCT% of lower wick ---
    for i in range(n):
        body_bottom = min(new_open[i], new_close[i])
        original_body_bottom = min(open_arr[i], close_arr[i])
        lower_wick = original_body_bottom - low_arr[i]

        if lower_wick > 0:
            noise = rng.uniform(-pct_frac, pct_frac)
            new_wick = lower_wick * (1 + noise)
            new_wick = max(new_wick, 0)
            low_arr[i] = body_bottom - new_wick
        else:
            low_arr[i] = body_bottom

    # --- Volume: ±PCT% random ---
    volume_noise = rng.uniform(-pct_frac, pct_frac, size=n)
    new_volume = volume_arr * (1 + volume_noise)
    new_volume = np.maximum(new_volume, 0)  # Volume cannot be negative

    # Build result
    result = df.copy()
    result["Open"] = np.round(new_open, 4)
    result["High"] = np.round(high_arr, 4)
    result["Low"] = np.round(low_arr, 4)
    result["Close"] = np.round(new_close, 4)
    result["Volume"] = np.round(new_volume, 4)

    return result


def generate_augmented_files(
    input_file: str,
    n_variants: int = None,
    pct: float = None,
    output_dir: str = None,
    base_seed: int = None,
) -> list:
    """
    Generate N augmented CSV files from an input file.

    Args:
        input_file: Path to original CSV
        n_variants: Number of variants (default: config.AUGMENT_VARIANTS)
        pct: Percentage parameter (default: config.AUGMENT_PCT)
        output_dir: Output directory (default: same as input)
        base_seed: Base seed for reproducibility (default: config.SEED)

    Returns:
        List of generated file paths
    """
    n_variants = n_variants if n_variants is not None else config.AUGMENT_VARIANTS
    pct = pct if pct is not None else config.AUGMENT_PCT
    base_seed = base_seed if base_seed is not None else config.SEED

    df = pd.read_csv(input_file)
    print(f"Loaded {input_file}: {len(df)} candles")

    if output_dir is None:
        output_dir = os.path.dirname(input_file)

    os.makedirs(output_dir, exist_ok=True)

    # Build output filenames
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    ext = os.path.splitext(input_file)[1]

    generated = []
    for v in range(1, n_variants + 1):
        seed = base_seed + v * 1000  # Distinct seeds per variant
        aug_df = augment_ohlcv(df, pct, seed)

        out_path = os.path.join(output_dir, f"{base_name}_aug{v}{ext}")
        aug_df.to_csv(out_path, index=False)
        print(f"  Generated: {out_path}")
        generated.append(out_path)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate augmented OHLCV datasets")
    parser.add_argument("--file", type=str, default=config.INPUT_FILES[0],
                        help="Input CSV file")
    parser.add_argument("--variants", type=int, default=None,
                        help=f"Number of variants (default: {config.AUGMENT_VARIANTS})")
    parser.add_argument("--pct", type=float, default=None,
                        help=f"Perturbation percentage (default: {config.AUGMENT_PCT})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    generated = generate_augmented_files(
        input_file=args.file,
        n_variants=args.variants,
        pct=args.pct,
        output_dir=args.output_dir,
    )

    print(f"\nDone: {len(generated)} augmented files generated.")
    print("Add them to config.INPUT_FILES to use in training.")


if __name__ == "__main__":
    main()
