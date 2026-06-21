#!/usr/bin/env python3
"""
OHLCV download script for crypto pairs from multiple exchanges.

IMPORTANT FOR BITGET:
- Bitget REQUIRES API keys to access historical data
- Without API keys, Bitget only returns ~100 recent candles
- To get full history:
  1. Create an account on Bitget
  2. Generate API keys with read permission
  3. Configure your keys in the API_KEYS variable below
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
from dateutil import parser
from tqdm import tqdm
import math
import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.utils.ccxt_proxy import merge_proxy_config
from modules.utils.timeframes import TIMEFRAMES, timeframe_to_ms

# API key configuration via environment variables.
# Set them in your shell or .env file (never commit secrets to code).
# Example:  export BINANCE_API_KEY=xxx  BINANCE_SECRET=yyy
# WARNING: Bitget REQUIRES API keys to access historical data!
# Without API keys, Bitget only returns ~100 recent candles.
API_KEYS = {
    'binance': {
        'apiKey': os.environ.get('BINANCE_API_KEY', ''),
        'secret': os.environ.get('BINANCE_SECRET', '')
    },
    'bybit': {
        'apiKey': os.environ.get('BYBIT_API_KEY', ''),
        'secret': os.environ.get('BYBIT_SECRET', '')
    },
    'bitget': {
        'apiKey': os.environ.get('BITGET_API_KEY', ''),
        'secret': os.environ.get('BITGET_SECRET', ''),
        'password': os.environ.get('BITGET_PASSWORD', '')
    },
    'kucoin': {
        'apiKey': os.environ.get('KUCOIN_API_KEY', ''),
        'secret': os.environ.get('KUCOIN_SECRET', ''),
        'password': os.environ.get('KUCOIN_PASSWORD', '')
    }
}

SUPPORTED_EXCHANGES = ['binance', 'bybit', 'bitget', 'kucoin']
# Bitget constant: max window 90 days in ms
BITGET_MAX_WINDOW_MS = 90 * 24 * 60 * 60 * 1000

def instantiate_exchange(name):
    """Return a CCXT instance configured for USDT futures/perpetuals if needed."""
    params = {}
    api_config = {}

    # Add API keys if available
    if API_KEYS.get(name, {}).get('apiKey'):
        api_config = API_KEYS[name].copy()

    if name == 'binance':
        params = {'options': {'defaultType': 'future'}}
        exchange = ccxt.binance(merge_proxy_config({**api_config, **params}))
    elif name == 'bybit':
        params = {'options': {'defaultType': 'future'}}
        exchange = ccxt.bybit(merge_proxy_config({**api_config, **params}))
    elif name == 'bitget':
        # Use 'swap' for Bitget perpetual futures
        params = {'options': {'defaultType': 'swap'}}
        exchange = ccxt.bitget(merge_proxy_config({**api_config, **params}))
    elif name == 'kucoin':
        exchange = ccxt.kucoinfutures(merge_proxy_config(api_config))
    else:
        raise ValueError(f"Unsupported exchange: {name}")

    exchange.enableRateLimit = True
    return exchange

def find_symbol_for_pair(exchange, target_base, target_quote='USDT'):
    """Load markets and find the matching perpetual symbol (e.g. ETH/USDT)."""
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"Error loading markets: {e}")
        raise

    # For Bitget, dynamically build probable symbols
    if exchange.id == 'bitget':
        # List of possible Bitget formats
        candidates = [
            f"{target_base}/{target_quote}:{target_quote}", # Standard V2 format
            f"{target_base}{target_quote}_UMCBL",          # Legacy format
            f"{target_base}/{target_quote}"                 # Simple format
        ]
        for symbol in candidates:
            if symbol in exchange.markets:
                market = exchange.markets[symbol]
                if market.get('swap') or market.get('future'):
                    return symbol

    # General search via base/quote properties
    for market in exchange.markets.values():
        if market.get('base') == target_base and market.get('quote') == target_quote:
            if market.get('swap') or market.get('future'):
                return market['symbol']

    raise RuntimeError(f"Perpetual symbol {target_base}/{target_quote} not found for {exchange.id}")

def test_bitget_api(exchange, symbol):
    """Quickly test whether Bitget API keys are working correctly."""
    print("\nTesting Bitget API keys...")
    try:
        # Try to fetch a few old candles (1 month back)
        test_since = int((datetime.now(timezone.utc).timestamp() - 30 * 24 * 60 * 60) * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=test_since, limit=10)

        if ohlcv and len(ohlcv) > 5:
            print("API keys valid - Historical data access confirmed")
            return True
        else:
            print("API keys invalid or insufficient permissions")
            return False
    except Exception as e:
        print(f"Error testing API keys: {e}")
        return False

def get_market_listing_time(exchange, symbol):
    """Return the timestamp (ms) of the first market availability if available."""
    market = exchange.markets.get(symbol)
    if not market:
        return None
    info = market.get('info', {})
    for key in ['listTime', 'openTime', 'launchTime', 'listedAt', 'list_time']:
        list_time = info.get(key)
        if list_time:
            try:
                if isinstance(list_time, str) and list_time.isdigit():
                    return int(list_time)
                if isinstance(list_time, (int, float)):
                    return int(list_time)
            except Exception:
                continue
    return None

def fetch_ohlcv_bitget(exchange, symbol, timeframe, since, until=None, limit=200, sleep=1):
    """Specialized fetch for Bitget with improved handling."""
    all_ohlcv = []
    tf_ms = timeframe_to_ms(timeframe)

    # Check if API keys are needed
    has_api_keys = bool(API_KEYS.get('bitget', {}).get('apiKey'))
    if not has_api_keys:
        print("\nWARNING: No Bitget API keys configured!")
        print("Without API keys, Bitget only returns the ~100 most recent candles.")
        print("Full history is NOT accessible without authentication.")

    # Adjust start date if necessary
    listing = get_market_listing_time(exchange, symbol)
    if listing and since < listing:
        print(f"Adjusting start date to market listing: {datetime.fromtimestamp(listing/1000, tz=timezone.utc).isoformat()}")
        since = listing

    current = since
    max_candles_per_request = 100  # Safer limit for Bitget
    end_ts = until if until else int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"Requested period: {datetime.fromtimestamp(since/1000, tz=timezone.utc).isoformat()} -> {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc).isoformat()}")

    # If no API keys, try to fetch recent data only
    if not has_api_keys:
        print("\nFetching recent data only (no history without API keys)...")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            if ohlcv:
                # Filter to keep only candles within the requested period
                filtered = []
                for candle in ohlcv:
                    ts = candle[0]
                    if ts >= since and ts < end_ts:
                        filtered.append(candle)

                if filtered:
                    print(f"{len(filtered)} candles fetched (within requested period)")
                else:
                    print("Fetched data is outside the requested period")
                    print("   Historical data requires Bitget API keys")

                return filtered
            else:
                print("No data fetched")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    # If API keys are available, proceed normally
    # Estimate total number of requests
    total_candles = (end_ts - since) // tf_ms
    total_requests = math.ceil(total_candles / max_candles_per_request)

    with tqdm(total=total_requests, desc="Bitget", unit="req") as pbar:
        no_data_count = 0
        first_request = True

        while current < end_ts:
            # Calculate time window
            window_end = min(
                current + max_candles_per_request * tf_ms,
                end_ts
            )

            # For very old data, try with a smaller window
            if no_data_count > 5:
                window_end = min(current + 50 * tf_ms, end_ts)

            try:
                # For Bitget, try standard method first
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(current), limit=max_candles_per_request)

                # If no result with standard method, try with params
                if not ohlcv:
                    params = {
                        'startTime': str(int(current)),
                        'endTime': str(int(window_end))
                    }
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=None, limit=max_candles_per_request, params=params)

                # Detect if only recent data is returned (sign of missing API keys)
                if first_request and ohlcv and len(ohlcv) == 100:
                    # Check if all candles are recent
                    oldest_ts = min(candle[0] for candle in ohlcv)
                    if oldest_ts > since + 30 * 24 * 60 * 60 * 1000:  # More than one month after requested date
                        print("\nBitget is only returning recent data!")
                        print("   This confirms that API keys are required for historical data.")
                        # Return only candles within the requested period
                        filtered = []
                        for candle in ohlcv:
                            ts = candle[0]
                            if ts >= since and ts < end_ts:
                                filtered.append(candle)
                        return filtered

                first_request = False

                if not ohlcv:
                    no_data_count += 1
                    if no_data_count > 10:
                        current += 7 * 24 * 60 * 60 * 1000
                        print(f"\nNo data, skipping to {datetime.fromtimestamp(current/1000, tz=timezone.utc).isoformat()}")
                        continue
                    else:
                        current = window_end
                        continue

                no_data_count = 0

                # Filter and add candles
                added = 0
                for candle in ohlcv:
                    ts = candle[0]
                    if ts >= since and ts < end_ts:
                        all_ohlcv.append(candle)
                        added += 1

                # Update progress
                pbar.update(1)

                # Debug info
                if added > 0:
                    first_ts = ohlcv[0][0]
                    last_ts = ohlcv[-1][0]
                    print(f"\r{added} candles: {datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} -> {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}", end='')

                # Advance to next timestamp
                if ohlcv:
                    last_ts = ohlcv[-1][0]
                    current = last_ts + tf_ms
                else:
                    current = window_end

                # If fewer candles than requested
                if len(ohlcv) < max_candles_per_request:
                    if current < end_ts - 7 * 24 * 60 * 60 * 1000:
                        print(f"\nFewer data than expected ({len(ohlcv)} candles)")
                        current += 7 * 24 * 60 * 60 * 1000
                        print(f"Skipping to {datetime.fromtimestamp(current/1000, tz=timezone.utc).isoformat()}")
                    else:
                        print(f"\nEnd of available data")
                        break

                time.sleep(sleep)

            except Exception as e:
                error_msg = str(e)
                if 'authentication' in error_msg.lower() or 'api' in error_msg.lower() or 'permission' in error_msg.lower():
                    print(f"\n\nBitget authentication error: {e}")
                    print("API keys are REQUIRED for historical data!")
                    break
                elif 'rate' in error_msg.lower():
                    print(f"\nRate limit reached, pausing for 30 seconds...")
                    time.sleep(30)
                    continue
                else:
                    print(f"\nBitget error: {e}, retrying...")
                    time.sleep(sleep * 5)
                    current += tf_ms
                    continue

    print(f"\n\nTotal: {len(all_ohlcv)} candles fetched for Bitget")
    return all_ohlcv

def fetch_ohlcv_with_pagination(exchange, symbol, timeframe, since, until=None, limit=1000, sleep=1, total_pages=None):
    """Fetch OHLCV candles with pagination, from 'since' (timestamp ms) to 'until' (timestamp ms)."""
    # Special case for Bitget
    if exchange.id == 'bitget':
        return fetch_ohlcv_bitget(exchange, symbol, timeframe, since, until, limit=200, sleep=sleep)

    # General case for other exchanges
    all_ohlcv = []
    last_since = since

    pbar = None
    if total_pages is not None:
        pbar = tqdm(total=total_pages, desc=f"{exchange.id}", unit="page")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=last_since, limit=limit)
        except Exception as e:
            print(f"Error fetch_ohlcv {exchange.id} {symbol} from {last_since}: {e}, retrying after pause.")
            time.sleep(sleep * 5)
            continue

        if pbar:
            pbar.update(1)

        if not ohlcv:
            break

        for candle in ohlcv:
            ts = candle[0]
            if until and ts >= until:
                if pbar:
                    pbar.close()
                return all_ohlcv
            if ts >= since:
                all_ohlcv.append(candle)

        if len(ohlcv) < limit:
            break

        last_since = ohlcv[-1][0] + 1
        time.sleep(sleep)

    if pbar:
        pbar.close()

    return all_ohlcv

def aggregate_exchanges(exchanges_data):
    df_all = pd.concat([
        df.assign(exchange=name) for name, df in exchanges_data.items()
    ], ignore_index=True)

    if df_all.empty:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    if df_all['timestamp'].dtype != 'datetime64[ns]':
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], unit='ms', utc=True)

    def weighted_avg(group, price_col):
        v = group['volume']
        p = group[price_col]
        if v.sum() == 0:
            return 0
        return (p * v).sum() / v.sum()

    # Handle compatibility with different pandas versions
    try:
        # For pandas >= 2.2.0
        agg = df_all.groupby('timestamp', group_keys=False).apply(
            lambda g: pd.Series({
                'open': weighted_avg(g, 'open'),
                'high': weighted_avg(g, 'high'),
                'low': weighted_avg(g, 'low'),
                'close': weighted_avg(g, 'close'),
                'volume': g['volume'].sum()
            }),
            include_groups=False
        )
    except TypeError:
        # For pandas < 2.2.0
        agg = df_all.groupby('timestamp', group_keys=False).apply(
            lambda g: pd.Series({
                'open': weighted_avg(g, 'open'),
                'high': weighted_avg(g, 'high'),
                'low': weighted_avg(g, 'low'),
                'close': weighted_avg(g, 'close'),
                'volume': g['volume'].sum()
            })
        )

    agg = agg.reset_index().sort_values('timestamp')
    return agg

def prompt_choice(prompt, options):
    while True:
        print(prompt)
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {opt}")
        choice = input("Choose a number: ").strip()
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(options):
                return options[i-1]
        print("Invalid input, try again.")

def prompt_multiple_choices(prompt, options):
    while True:
        print(prompt)
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {opt}")
        choice = input("Choose one or more numbers separated by commas: ").strip()
        parts = [p.strip() for p in choice.split(',') if p.strip()]
        indices, valid = [], True
        for p in parts:
            if p.isdigit():
                i = int(p)
                if 1 <= i <= len(options):
                    indices.append(i-1)
                else:
                    valid = False
            else:
                valid = False
        if valid and indices:
            selected = []
            for i in indices:
                if options[i] not in selected:
                    selected.append(options[i])
            return selected
        print("Invalid input, try again.")

def prompt_date(prompt, default=None):
    while True:
        if default:
            inp = input(f"{prompt} [{default.isoformat()}]: ").strip()
            if not inp:
                return default
        else:
            inp = input(f"{prompt}: ").strip()
        try:
            dt = parser.isoparse(inp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            print("Invalid date format. Example: 2021-01-01T00:00:00Z")

def prompt_input(prompt, default=None, cast_func=None):
    while True:
        if default is not None:
            inp = input(f"{prompt} [{default}]: ").strip()
            if not inp:
                return default
        else:
            inp = input(f"{prompt}: ").strip()
        if cast_func:
            try:
                return cast_func(inp)
            except Exception:
                print("Invalid input, try again.")
        else:
            return inp

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download OHLCV candles from crypto exchanges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (default)
  python download_candles.py

  # CLI mode - single exchange
  python download_candles.py --exchange binance --timeframe 1h --start 2024-01-01 --output data.csv

  # CLI mode - multiple exchanges (aggregated)
  python download_candles.py --exchanges binance,bybit --timeframe 15m --start 2024-01-01 --end 2024-12-31
        '''
    )

    parser.add_argument('--exchange', type=str, choices=SUPPORTED_EXCHANGES,
                        help='Single exchange to download from')
    parser.add_argument('--exchanges', type=str,
                        help='Comma-separated list of exchanges (aggregated)')
    parser.add_argument('--pair', type=str, default='BTC/USDT',
                        help='Trading pair to download (e.g. ETH/USDT, SOL/USDT)')
    parser.add_argument('--timeframe', type=str, choices=TIMEFRAMES,
                        help='Timeframe (15m, 1h, 4h, 1d)')
    parser.add_argument('--start', type=str,
                        help='Start date (ISO format: 2024-01-01 or 2024-01-01T00:00:00Z)')
    parser.add_argument('--end', type=str,
                        help='End date (ISO format, default: now)')
    parser.add_argument('--output', type=str,
                        help='Output CSV filename')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Max candles per API call (default: 1000)')
    parser.add_argument('--sleep', type=float, default=1.0,
                        help='Sleep between API calls in seconds (default: 1.0)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Check if CLI mode (any argument provided)
    cli_mode = any([args.exchange, args.exchanges, args.timeframe, args.start, args.output])

    if cli_mode:
        # CLI mode: use arguments
        if args.exchange and args.exchanges:
            print("Error: Cannot use both --exchange and --exchanges")
            sys.exit(1)

        if not args.timeframe:
            print("Error: --timeframe is required in CLI mode")
            sys.exit(1)

        timeframe = args.timeframe

        if args.exchanges:
            exchanges = [e.strip() for e in args.exchanges.split(',')]
            # Validate exchanges
            invalid = [e for e in exchanges if e not in SUPPORTED_EXCHANGES]
            if invalid:
                print(f"Error: Invalid exchanges: {invalid}")
                sys.exit(1)
        elif args.exchange:
            exchanges = [args.exchange]
        else:
            print("Error: Either --exchange or --exchanges is required in CLI mode")
            sys.exit(1)

        target_pair = args.pair.upper()
        if '/' not in target_pair:
             # Basic handling if user writes "ETHUSDT" instead of "ETH/USDT"
             print(f"Invalid pair format (use BASE/QUOTE, e.g. ETH/USDT). Got: {target_pair}")
             sys.exit(1)

        target_base, target_quote = target_pair.split('/')

        # Parse dates
        earliest = parser.isoparse('2019-08-01T00:00:00Z')
        if args.start:
            since_dt = parser.isoparse(args.start)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
            else:
                since_dt = since_dt.astimezone(timezone.utc)
        else:
            since_dt = earliest

        if since_dt < earliest:
            print(f"Warning: Start date before 2019-08-01, adjusting to {earliest.isoformat()}")
            since_dt = earliest

        if args.end:
            until_dt = parser.isoparse(args.end)
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
            else:
                until_dt = until_dt.astimezone(timezone.utc)
        else:
            until_dt = datetime.now(timezone.utc)

        output = args.output if args.output else 'data/download/btc_usdt_agg.csv'
        limit = args.limit
        sleep = args.sleep

    else:
        # Interactive mode (original behavior)
        print("--- OHLCV Download Configuration ---")
        print("\nIMPORTANT: Bitget REQUIRES API keys to access historical data!")
        print("Without API keys, Bitget only returns ~100 recent candles.")
        print("To get full historical data, set env vars: BITGET_API_KEY, BITGET_SECRET, BITGET_PASSWORD\n")

        target_pair_input = prompt_input("Pair to download (e.g. ETH/USDT, SOL/USDT)", default="BTC/USDT")
        target_pair = target_pair_input.upper()
        if '/' not in target_pair:
             print("Invalid format, defaulting to BTC/USDT.")
             target_pair = "BTC/USDT"

        target_base, target_quote = target_pair.split('/')

        timeframe = prompt_choice("Select timeframe:", TIMEFRAMES)
        exchanges = prompt_multiple_choices("Select exchanges to query:", SUPPORTED_EXCHANGES)

        # Check if Bitget is selected and API keys are configured
        if 'bitget' in exchanges and not API_KEYS.get('bitget', {}).get('apiKey'):
            print("\nWARNING: Bitget is selected but no API keys are configured!")
            print("Without API keys, you will only get about 100 recent candles.")
            print("\nTo get full history:")
            print("1. Create an account on Bitget")
            print("2. Generate API keys with read permission")
            print("3. Set environment variables:")
            print("   export BITGET_API_KEY='your_api_key'")
            print("   export BITGET_SECRET='your_secret'")
            print("   export BITGET_PASSWORD='your_passphrase'")
            cont = input("\nContinue without API keys? (y/n): ").strip().lower()
            if cont != 'y':
                print("Configuration cancelled. Please add your API keys and rerun the script.")
                return

        earliest = parser.isoparse('2019-08-01T00:00:00Z')
        since_dt = prompt_date("Start date ISO (>= 2019-08-01T00:00:00Z)", default=earliest)
        if since_dt < earliest:
            print(f"Start date before 2019-08-01, adjusting to {earliest.isoformat()}")
            since_dt = earliest

        until_default = datetime.now(timezone.utc)
        until_dt = prompt_date("End date ISO", default=until_default)
        if until_dt <= since_dt:
            print("End date must be after start date. Setting end to now UTC.")
            until_dt = until_default

        output = prompt_input("Output CSV filename", default='data/download/btc_usdt_agg.csv')
        limit = prompt_input("Max candles per API call (limit)", default=1000, cast_func=int)
        sleep = prompt_input("Sleep between API calls in seconds", default=1.0, cast_func=float)

    # Common logic for both modes

    since_ts = int(since_dt.timestamp() * 1000)
    until_ts = int(until_dt.timestamp() * 1000)

    exchanges_data = {}
    tf_ms = timeframe_to_ms(timeframe)
    total_intervals = math.ceil((until_ts - since_ts) / tf_ms)
    pages_estimate = math.ceil(total_intervals / limit)

    for name in exchanges:
        print(f"\nProcessing exchange {name}...")
        try:
            ex = instantiate_exchange(name)
        except Exception as e:
            print(f"  Error initializing {name}: {e}")
            continue

        try:
            symbol = find_symbol_for_pair(ex, target_base, target_quote)
        except Exception as e:
            print(f"  Error detecting symbol for {name}: {e}")
            continue

        print(f"  Symbol detected for {target_pair}: {symbol}")

        # Special test for Bitget
        if name == 'bitget' and API_KEYS.get('bitget', {}).get('apiKey'):
            if not test_bitget_api(ex, symbol):
                print("  WARNING: API key test failed. Data will be limited.")

        ohlcv_list = fetch_ohlcv_with_pagination(
            ex, symbol, timeframe, since_ts, until=until_ts,
            limit=limit, sleep=sleep, total_pages=pages_estimate if name != 'bitget' else None
        )

        if not ohlcv_list:
            print(f"  No OHLCV returned for {name}.")
            continue

        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        exchanges_data[name] = df
        print(f"  {len(df)} candles fetched for {name}.")

    if not exchanges_data:
        print("\nNo data fetched from any exchange.")
        return

    print("\nAggregating data...")
    df_agg = aggregate_exchanges(exchanges_data)

    if df_agg.empty:
        print("No data after aggregation.")
        return

    if df_agg['timestamp'].dtype != 'datetime64[ns]':
        df_agg['timestamp'] = pd.to_datetime(df_agg['timestamp'], utc=True)

    df_agg['Open time'] = df_agg['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_agg = df_agg.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    df_out = df_agg[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Create output directory if needed
    import os
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_out.to_csv(output, index=False)
    print(f"\nAggregation complete. CSV file generated: {output}")
    print(f"Total of {len(df_out)} candles saved.")

if __name__ == '__main__':
    main()
