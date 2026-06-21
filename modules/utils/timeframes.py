"""Shared timeframe constants and conversion utilities."""

TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

TIMEFRAME_TO_MS = {
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '2h': 2 * 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '8h': 8 * 60 * 60 * 1000,
    '12h': 12 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
}


def timeframe_to_ms(timeframe):
    """Convert timeframe string to milliseconds."""
    if timeframe not in TIMEFRAME_TO_MS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {', '.join(TIMEFRAMES)}"
        )
    return TIMEFRAME_TO_MS[timeframe]


def timeframe_to_seconds(timeframe):
    """Convert timeframe string to seconds."""
    return timeframe_to_ms(timeframe) // 1000
