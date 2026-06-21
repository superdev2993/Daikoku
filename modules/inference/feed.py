"""
Live candle feed using ccxt.

Downloads historical buffer then fetches new candles as they close.
Output DataFrame format is identical to loader.get_raw_data():
  columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
"""

import time
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

from modules.utils.logger import get_logger
from modules.utils.ccxt_proxy import merge_proxy_config
from modules.utils.timeframes import timeframe_to_ms, timeframe_to_seconds

logger = get_logger(__name__)


def _instantiate_exchange(name):
    """
    Create a ccxt exchange instance configured for futures/perpetuals.
    Duplicated from modules/tools/download_candles.py (kept separate for isolation).
    """
    if name == 'binance':
        exchange = ccxt.binance(merge_proxy_config({'options': {'defaultType': 'future'}}))
    elif name == 'bybit':
        exchange = ccxt.bybit(merge_proxy_config({'options': {'defaultType': 'future'}}))
    elif name == 'bitget':
        exchange = ccxt.bitget(merge_proxy_config({'options': {'defaultType': 'swap'}}))
    else:
        raise ValueError(f"Unsupported exchange: {name}")

    exchange.enableRateLimit = True
    return exchange


def _find_symbol(exchange, pair):
    """
    Find the perpetual futures symbol for a trading pair.
    Args:
        exchange: ccxt exchange instance
        pair: e.g. "BTC/USDT"
    Returns:
        symbol string usable with fetch_ohlcv
    """
    if '/' not in pair:
        raise ValueError(f"Invalid pair format (use BASE/QUOTE): {pair}")

    base, quote = pair.split('/')
    exchange.load_markets()

    # Try common perpetual formats
    candidates = [
        f"{base}/{quote}:{quote}",   # Binance/Bybit perp
        f"{base}/{quote}",            # Simple format
    ]
    for symbol in candidates:
        if symbol in exchange.markets:
            market = exchange.markets[symbol]
            if market.get('swap') or market.get('future') or market.get('linear'):
                return symbol

    # Fallback: search all markets
    for market in exchange.markets.values():
        if market.get('base') == base and market.get('quote') == quote:
            if market.get('swap') or market.get('future'):
                return market['symbol']

    raise RuntimeError(f"Perpetual symbol for {pair} not found on {exchange.id}")


class CandleFeed:
    """
    Maintains a rolling buffer of OHLCV candles from a crypto exchange.

    The buffer DataFrame has the same format as loader.get_raw_data() output:
      columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
      dtypes:   datetime64, float64, float64, float64, float64, float64
    """

    def __init__(self, exchange_name, pair, timeframe, buffer_size):
        self.exchange = _instantiate_exchange(exchange_name)
        self.pair = pair
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.tf_ms = timeframe_to_ms(timeframe)
        self.tf_seconds = timeframe_to_seconds(timeframe)

        self.symbol = _find_symbol(self.exchange, pair)
        logger.info(f"CandleFeed: {exchange_name} {self.symbol} {timeframe} (buffer={buffer_size})")

        self.buffer = pd.DataFrame(columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self._last_saved_ts = None  # Track last saved timestamp for append-only CSV

    def _deduplicate_and_strip_open(self, all_ohlcv, now_ms):
        """Deduplicate candles by timestamp and remove the last one if still open."""
        seen = {}
        for candle in all_ohlcv:
            seen[candle[0]] = candle
        result = sorted(seen.values(), key=lambda x: x[0])

        if result:
            expected_close_ms = result[-1][0] + self.tf_ms
            if expected_close_ms > now_ms:
                result = result[:-1]

        return result

    def _trim_buffer(self):
        """Trim buffer to buffer_size, keeping most recent candles."""
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer.iloc[-self.buffer_size:].reset_index(drop=True)

    def _ohlcv_to_df(self, ohlcv_list):
        """Convert ccxt ohlcv list to DataFrame matching loader format."""
        df = pd.DataFrame(ohlcv_list, columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Open time'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.drop(columns=['ts'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def fill_buffer(self):
        """
        Download historical candles to fill the buffer.

        Uses paginated fetch going backwards from now.
        Returns the number of candles loaded.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Start far enough back to get buffer_size candles
        since_ms = now_ms - (self.buffer_size + 100) * self.tf_ms

        all_ohlcv = []
        current = since_ms
        limit = 1000  # ccxt default max per request

        logger.info(f"Filling buffer: fetching ~{self.buffer_size} candles...")

        while current < now_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, since=current, limit=limit
                )
            except Exception as e:
                logger.warning(f"Fetch error at {current}: {e}, retrying...")
                time.sleep(2)
                continue

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            current = last_ts + self.tf_ms

            if len(ohlcv) < limit:
                break
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_ohlcv:
            raise RuntimeError("Failed to fetch any historical candles")

        all_ohlcv = self._deduplicate_and_strip_open(all_ohlcv, now_ms)

        # Keep only buffer_size most recent
        if len(all_ohlcv) > self.buffer_size:
            all_ohlcv = all_ohlcv[-self.buffer_size:]

        self.buffer = self._ohlcv_to_df(all_ohlcv)
        logger.info(f"Buffer filled: {len(self.buffer)} candles "
                     f"({self.buffer['Open time'].iloc[0]} to {self.buffer['Open time'].iloc[-1]})")
        return len(self.buffer)

    @staticmethod
    def read_csv(csv_path):
        """Read a buffer CSV into a DataFrame with consistent parsing."""
        df = pd.read_csv(csv_path)
        df['Open time'] = pd.to_datetime(df['Open time'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        return df

    def load_buffer(self, csv_path):
        """Load buffer from saved CSV. Full history stays on disk, last buffer_size in memory."""
        df = CandleFeed.read_csv(csv_path)

        self._last_saved_ts = df['Open time'].iloc[-1]
        total = len(df)

        # Keep only last buffer_size in memory for inference
        if total > self.buffer_size:
            self.buffer = df.iloc[-self.buffer_size:].reset_index(drop=True)
        else:
            self.buffer = df

        logger.info(f"Buffer loaded from {csv_path}: {total} total candles, "
                     f"{len(self.buffer)} in memory "
                     f"({self.buffer['Open time'].iloc[0]} to {self.buffer['Open time'].iloc[-1]})")
        return len(self.buffer)

    def fill_gap(self):
        """
        Download only candles newer than buffer tail.
        Appends to buffer and trims to buffer_size.

        Returns:
            list of new candle dicts (chronological), empty if none.
        """
        if self.buffer.empty:
            raise RuntimeError("Buffer is empty, cannot fill gap")

        last_ts_ms = int(self.buffer['Open time'].iloc[-1].timestamp() * 1000)
        since_ms = last_ts_ms + self.tf_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if since_ms >= now_ms:
            logger.info("Buffer is up-to-date, no gap to fill")
            return []

        all_ohlcv = []
        current = since_ms

        logger.info(f"Filling gap from {pd.Timestamp(since_ms, unit='ms')}...")

        while current < now_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, since=current, limit=1000
                )
            except Exception as e:
                logger.warning(f"Gap fetch error at {current}: {e}, retrying...")
                time.sleep(2)
                continue

            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            current = last_ts + self.tf_ms

            if len(ohlcv) < 1000:
                break
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_ohlcv:
            logger.info("No new candles in gap")
            return []

        all_ohlcv = self._deduplicate_and_strip_open(all_ohlcv, now_ms)

        if not all_ohlcv:
            logger.info("No closed candles in gap")
            return []

        # Convert to DataFrame and append
        new_df = self._ohlcv_to_df(all_ohlcv)
        self.buffer = pd.concat([self.buffer, new_df], ignore_index=True)

        self._trim_buffer()

        # Return new candles as dicts for gap replay
        new_candles = []
        for _, row in new_df.iterrows():
            new_candles.append({
                'timestamp': row['Open time'],
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
            })

        logger.info(f"Gap filled: {len(new_candles)} new candles "
                     f"(buffer now {len(self.buffer)} candles)")
        return new_candles

    def fetch_latest_candle(self, retry_max=5, retry_timeout=10):
        """
        Fetch the most recently closed candle.

        Retries up to retry_max times. Raises alert if takes > retry_timeout seconds.

        Returns:
            dict with candle data, or None if fetch failed.
            Also returns alert flag.
        """
        t_start = time.time()
        alert = False

        for attempt in range(1, retry_max + 1):
            try:
                # Fetch last 2 candles
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=2)
                if not ohlcv or len(ohlcv) < 2:
                    logger.warning(f"Attempt {attempt}: got {len(ohlcv) if ohlcv else 0} candles")
                    time.sleep(2)
                    continue

                # The second-to-last is the most recently CLOSED candle
                # (last one may still be open)
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                closed_candle = None

                for candle in reversed(ohlcv):
                    candle_close_ms = candle[0] + self.tf_ms
                    if candle_close_ms <= now_ms:
                        closed_candle = candle
                        break

                if closed_candle is None:
                    logger.warning(f"Attempt {attempt}: no closed candle found")
                    time.sleep(2)
                    continue

                elapsed = time.time() - t_start
                if elapsed > retry_timeout:
                    alert = True
                    logger.warning(f"Candle fetch took {elapsed:.1f}s (> {retry_timeout}s)")

                # Check if this candle is already in buffer
                candle_ts = closed_candle[0]
                last_buffer_ts = int(self.buffer['Open time'].iloc[-1].timestamp() * 1000)

                if candle_ts <= last_buffer_ts:
                    # Already have this candle
                    return None, alert

                # Append to buffer
                new_df = self._ohlcv_to_df([closed_candle])
                self.buffer = pd.concat([self.buffer, new_df], ignore_index=True)

                self._trim_buffer()

                candle_data = {
                    'timestamp': new_df['Open time'].iloc[0],
                    'open': float(new_df['Open'].iloc[0]),
                    'high': float(new_df['High'].iloc[0]),
                    'low': float(new_df['Low'].iloc[0]),
                    'close': float(new_df['Close'].iloc[0]),
                    'volume': float(new_df['Volume'].iloc[0]),
                }
                logger.info(f"New candle: {candle_data['timestamp']} "
                            f"O={candle_data['open']:.2f} H={candle_data['high']:.2f} "
                            f"L={candle_data['low']:.2f} C={candle_data['close']:.2f}")
                return candle_data, alert

            except Exception as e:
                logger.error(f"Attempt {attempt}: {e}")
                time.sleep(2)

        logger.error(f"Failed to fetch candle after {retry_max} attempts")
        return None, True

    def save_buffer(self, path):
        """Append new candles to history CSV (append-only, never truncates)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self._last_saved_ts is not None:
            # Append only candles newer than last saved
            new_mask = self.buffer['Open time'] > self._last_saved_ts
            if not new_mask.any():
                logger.debug(f"Buffer save: no new candles to append")
                return

            df_new = self.buffer[new_mask].copy()
            df_new['Open time'] = df_new['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_new.to_csv(path, mode='a', header=False, index=False)
            self._last_saved_ts = self.buffer['Open time'].iloc[-1]
            logger.info(f"Buffer appended: {len(df_new)} candles to {path}")
        else:
            # First save: write full buffer with header
            df_out = self.buffer.copy()
            df_out['Open time'] = df_out['Open time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_out.to_csv(path, index=False)
            self._last_saved_ts = self.buffer['Open time'].iloc[-1]
            logger.info(f"Buffer saved: {path} ({len(df_out)} candles)")

    def fetch_open_candle(self):
        """
        Fetch the currently open (unclosed) candle for display only.
        Does NOT modify the buffer.

        Returns:
            dict with candle OHLCV data, or None on error.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1)
            if not ohlcv:
                return None
            c = ohlcv[-1]
            return {
                'timestamp': pd.Timestamp(c[0], unit='ms'),
                'open': float(c[1]),
                'high': float(c[2]),
                'low': float(c[3]),
                'close': float(c[4]),
                'volume': float(c[5]),
            }
        except Exception as e:
            logger.debug(f"fetch_open_candle error: {e}")
            return None

    def next_candle_time(self):
        """Return the expected UTC close time of the next candle."""
        last_ts = self.buffer['Open time'].iloc[-1]
        # Next candle opens at last_ts + tf, closes at last_ts + 2*tf
        next_open = last_ts + pd.Timedelta(seconds=self.tf_seconds)
        next_close = next_open + pd.Timedelta(seconds=self.tf_seconds)
        return next_close
