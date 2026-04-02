"""
Data loader module - CSV loading and validation
"""

import pandas as pd
import numpy as np
import os

from modules.utils.logger import get_logger

logger = get_logger(__name__)


# Required OHLCV columns
REQUIRED_COLUMNS = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Gap detection thresholds
GAP_THRESHOLD_PCT = 0.2    # single gap > this % triggers count
GAP_RATIO_WARNING_PCT = 0.5  # warning if this % of transitions have gaps


def load_csv(path):
    """
    Load CSV file into pandas DataFrame

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info(f"Loading CSV: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        raise ValueError("CSV file is empty")

    return df


def validate_data(df):
    """
    Validate DataFrame structure and content

    Args:
        df: Input DataFrame

    Returns:
        Validated and cleaned DataFrame

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.debug(f"✓ Required columns present: {REQUIRED_COLUMNS}")

    # Select only required columns (ignore extra columns)
    df = df[REQUIRED_COLUMNS].copy()

    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        raise ValueError(f"Missing values found:\n{null_cols}")

    logger.debug("✓ No missing values")

    # Convert Open time to datetime
    try:
        df['Open time'] = pd.to_datetime(df['Open time'])
    except Exception as e:
        raise ValueError(f"Failed to parse 'Open time' as datetime: {e}")

    # Check duplicate timestamps
    if df['Open time'].duplicated().any():
        raise ValueError("Duplicate timestamps found")

    # Check chronological order
    if not df['Open time'].is_monotonic_increasing:
        raise ValueError("Data is not in chronological order")

    logger.debug("✓ Data in chronological order (no duplicates)")

    # Convert OHLC and Volume to float64
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        try:
            df[col] = df[col].astype(np.float64)
        except Exception as e:
            raise ValueError(f"Failed to convert '{col}' to numeric: {e}")

    logger.debug("✓ OHLC and Volume converted to float64")

    # Validate OHLC logic (High >= Low, etc.)
    invalid_high = (df['High'] < df['Low']).any()
    invalid_open = ((df['Open'] < df['Low']) | (df['Open'] > df['High'])).any()
    invalid_close = ((df['Close'] < df['Low']) | (df['Close'] > df['High'])).any()

    if invalid_high:
        raise ValueError("Invalid data: High < Low")
    if invalid_open:
        raise ValueError("Invalid data: Open outside [Low, High]")
    if invalid_close:
        raise ValueError("Invalid data: Close outside [Low, High]")

    logger.debug("✓ OHLC logic valid")

    # Validate Volume (must be non-negative)
    if (df['Volume'] < 0).any():
        raise ValueError("Invalid data: Volume contains negative values")

    logger.debug("✓ Volume valid (non-negative)")

    # Check price continuity: Close[i] vs Open[i+1] (detect gaps)
    if len(df) > 1:
        close = df['Close'].values
        open_next = df['Open'].values[1:]

        # Detect significant gaps
        gaps_pct = np.abs((open_next - close[:-1]) / close[:-1]) * 100
        n_significant = (gaps_pct > GAP_THRESHOLD_PCT).sum()
        gap_ratio = n_significant / len(gaps_pct) * 100

        if gap_ratio > GAP_RATIO_WARNING_PCT:
            logger.warning(f"⚠️  {n_significant} price gaps > {GAP_THRESHOLD_PCT}% ({gap_ratio:.2f}% of transitions)")
        else:
            logger.debug(f"✓ Price continuity: {n_significant} gaps > {GAP_THRESHOLD_PCT}% ({gap_ratio:.2f}%)")

    logger.info(f"Validation complete: {len(df)} valid rows")
    return df


def load_and_validate(path):
    """
    Load and validate CSV file

    Args:
        path: Path to CSV file

    Returns:
        Validated DataFrame
    """
    df = load_csv(path)
    df = validate_data(df)

    return df


def get_raw_data(path):
    """
    Get raw validated data

    Args:
        path: Path to CSV file

    Returns:
        Validated DataFrame with OHLCV data (Open, High, Low, Close, Volume)
    """
    return load_and_validate(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Test loading
    print("\n=== Testing data loader ===\n")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/BTC_1h_agg.csv"

    try:
        df = get_raw_data(path)
        print(f"\n✅ Successfully loaded {len(df)} rows from {path}")
        print(f"\nFirst rows:\n{df.head()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nDate range: {df['Open time'].min()} to {df['Open time'].max()}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
