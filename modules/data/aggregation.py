"""
Candle aggregation module for multi-timeframe analysis.

Aggregates primary candles into secondary (higher) timeframe candles.
Supports two alignment modes:
- "structured": groups by temporal boundaries (hour % divisor == 0)
- "unstructured": groups by consecutive blocks of N candles
"""

import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.utils.logger import get_logger

logger = get_logger(__name__)


def aggregate_candles(df: pd.DataFrame, divisor: int, align: str = "structured"):
    """
    Aggregate primary candles into secondary (higher) timeframe candles.

    Args:
        df: DataFrame with columns [Open time, Open, High, Low, Close, Volume]
            'Open time' must be datetime.
        divisor: Aggregation factor (N primary candles → 1 secondary candle)
        align: Alignment mode:
            - "structured": group by temporal boundaries (hour % divisor == 0)
            - "unstructured": group by consecutive blocks of N candles

    Returns:
        tuple: (DataFrame, end_timestamps, partial_ohlcv)
            - DataFrame with aggregated candles
            - end_timestamps: numpy array, Open time of LAST raw candle per group.
            - partial_ohlcv: numpy array (N_raw, 5) with expanding partial OHLCV
              per raw candle [Open, cumHigh, cumLow, Close, cumVolume].
        Last candle may be incomplete (< divisor source candles).

    Raises:
        ValueError: If divisor < 2 or input is too small.
    """
    if divisor < 2:
        raise ValueError(f"Divisor must be >= 2, got {divisor}")

    if len(df) < divisor:
        raise ValueError(f"Need at least {divisor} candles, got {len(df)}")

    logger.debug(f"Aggregating {len(df)} candles with divisor={divisor}, align={align}")

    if align == "structured":
        result, end_timestamps, partial_ohlcv = _aggregate_structured(df, divisor)
    elif align == "unstructured":
        result, end_timestamps, partial_ohlcv = _aggregate_unstructured(df, divisor)
    else:
        raise ValueError(f"Unknown alignment mode: {align}. Use 'structured' or 'unstructured'.")

    logger.info(f"✓ Aggregated: {len(df)} → {len(result)} candles (divisor={divisor})")

    return result, end_timestamps, partial_ohlcv


def _aggregate_structured(df: pd.DataFrame, divisor: int) -> tuple:
    """
    Aggregate by temporal boundaries (hour % divisor == 0).

    Groups candles so that each group starts at an hour that is a multiple of divisor.
    E.g. with divisor=4: groups start at hours 0, 4, 8, 12, 16, 20.
    """
    df = df.copy()
    hours = df['Open time'].dt.hour

    # Assign group: floor(hour / divisor) * divisor gives the boundary hour
    # Combine with date for unique group keys
    group_hour = (hours // divisor) * divisor
    group_key = df['Open time'].dt.date.astype(str) + '_' + group_hour.astype(str).str.zfill(2)
    df['_group'] = group_key

    return _aggregate_groups(df)


def _aggregate_unstructured(df: pd.DataFrame, divisor: int) -> tuple:
    """
    Aggregate by consecutive blocks of N candles.

    Simply groups every N consecutive candles together.
    Last group may have fewer than N candles.
    """
    df = df.copy()
    df['_group'] = np.arange(len(df)) // divisor

    return _aggregate_groups(df)


def _aggregate_groups(df: pd.DataFrame):
    """
    Perform OHLCV aggregation on pre-grouped DataFrame.

    For each group:
    - Open time: first candle's open time
    - Open: first candle's open
    - High: max of all highs
    - Low: min of all lows
    - Close: last candle's close
    - Volume: sum of all volumes

    Returns:
        tuple: (aggregated DataFrame, end_timestamps, partial_ohlcv)
            - end_timestamps: numpy array, Open time of the LAST raw candle in each group.
            - partial_ohlcv: numpy array (N_raw, 5) with expanding partial OHLCV
              per raw candle within its group [Open_first, cummax(High), cummin(Low),
              Close_current, cumsum(Volume)]. Used for real-time secondary branch
              feature recomputation.
    """
    agg = df.groupby('_group', sort=False).agg(
        **{
            'Open time': ('Open time', 'first'),
            'Open': ('Open', 'first'),
            'High': ('High', 'max'),
            'Low': ('Low', 'min'),
            'Close': ('Close', 'last'),
            'Volume': ('Volume', 'sum'),
        }
    ).reset_index(drop=True)

    # End timestamp of each group (last raw candle's Open time)
    end_timestamps = df.groupby('_group', sort=False)['Open time'].last().values

    # Expanding partial OHLCV per raw candle within each group.
    # For real-time simulation: at raw candle i, the secondary candle only
    # contains raw candles from the group start up to i (inclusive).
    grouped = df.groupby('_group', sort=False)
    partial_ohlcv = np.column_stack([
        grouped['Open'].transform('first').values,   # Open = first candle's Open
        grouped['High'].cummax().values,              # High = expanding max
        grouped['Low'].cummin().values,               # Low = expanding min
        df['Close'].values,                           # Close = current candle's Close
        grouped['Volume'].cumsum().values,            # Volume = expanding sum
    ]).astype(np.float32)

    # Ensure correct column order
    agg = agg[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Ensure dtypes match input
    agg['Open'] = agg['Open'].astype(np.float32)
    agg['High'] = agg['High'].astype(np.float32)
    agg['Low'] = agg['Low'].astype(np.float32)
    agg['Close'] = agg['Close'].astype(np.float32)
    agg['Volume'] = agg['Volume'].astype(np.float32)

    return agg, end_timestamps, partial_ohlcv
