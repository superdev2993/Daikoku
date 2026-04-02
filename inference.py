#!/usr/bin/env python3
"""
Live inference entry point.

Downloads historical candles, runs predictions on each new candle,
tracks virtual trades, and serves a live dashboard.

Usage:
    python inference.py
    python inference.py --pair ETH/USDT --timeframe 15m --port 8888
    python inference.py --checkpoint models/best_model.pt --device cuda
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

import config
from modules.inference.engine import PredictionEngine
from modules.inference.feed import CandleFeed, timeframe_to_seconds
from modules.inference.tracker import TradeTracker
from modules.inference.dashboard import LiveDashboard
from modules.data.labeling import REJECT_CLASS
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Live inference with dashboard')
    parser.add_argument('--pair', type=str, default=config.LIVE_SYMBOL,
                        help=f'Trading pair (default: {config.LIVE_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=config.LIVE_TIMEFRAME,
                        help=f'Timeframe (default: {config.LIVE_TIMEFRAME})')
    parser.add_argument('--checkpoint', type=str, default=config.LIVE_CHECKPOINT,
                        help=f'Model checkpoint (default: {config.LIVE_CHECKPOINT})')
    parser.add_argument('--port', type=int, default=config.LIVE_HTTP_PORT,
                        help=f'HTTP port (default: {config.LIVE_HTTP_PORT})')
    parser.add_argument('--buffer-size', type=int, default=config.LIVE_BUFFER_SIZE,
                        help=f'Buffer size (default: {config.LIVE_BUFFER_SIZE})')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    parser.add_argument('--exchange', type=str, default=config.LIVE_EXCHANGE,
                        help=f'Exchange (default: {config.LIVE_EXCHANGE})')
    parser.add_argument('--fresh', action='store_true',
                        help='Force fresh start: erase all CSVs and state')
    return parser.parse_args()


def _pair_tag(pair):
    """Convert 'BTC/USDT' to 'BTCUSDT' for filenames."""
    return pair.replace('/', '')


def _force_csv(path, fieldnames):
    """Create or overwrite CSV file with header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _ensure_csv(path, fieldnames):
    """Create CSV with header only if it doesn't exist."""
    if not os.path.exists(path):
        _force_csv(path, fieldnames)


def _fmt_ts(ts):
    """Format timestamp as naive UTC string (consistent with training CSVs)."""
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def _append_csv(path, row, fieldnames):
    """Append a single row to CSV."""
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _load_predictions(csv_path, class_names):
    """Load predictions from CSV, return list of dicts for predictions_log."""
    if not os.path.exists(csv_path):
        return []
    predictions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = int(row['pred_class'])
            # Read probabilities from available columns (supports both old and new format)
            probs = []
            # New format: prob_0, prob_1, prob_2
            for col in ['prob_0', 'prob_1', 'prob_2']:
                if col in row and row[col]:
                    probs.append(float(row[col]))
            # Legacy format: prob_bear, prob_uncertain, prob_bull
            if not probs:
                for col in ['prob_bear', 'prob_uncertain', 'prob_bull']:
                    if col in row and row[col]:
                        probs.append(float(row[col]))
            confidence = float(row['confidence'])
            ts = pd.Timestamp(row['timestamp'])
            predictions.append({
                'timestamp': ts,
                'pred': pred,
                'confidence': confidence,
                'probs': probs,
                'class_name': class_names.get(pred, f'Class_{pred}'),
                'signal_ts': ts,
            })
    return predictions


def _format_trade_row(trade, pair):
    """Format a closed trade dict into a CSV row dict."""
    return {
        'signal_ts': _fmt_ts(trade['signal_ts']),
        'pair': pair,
        'direction': trade['direction_name'],
        'entry_price': f"{trade['entry_price']:.2f}",
        'tp_level': f"{trade['tp_level']:.2f}",
        'sl_level': f"{trade['sl_level']:.2f}",
        'atr': f"{trade['atr']:.2f}",
        'close_ts': _fmt_ts(trade['close_ts']),
        'outcome': trade['outcome'],
        'hit_candle': trade['hit_candle'],
        'confidence': f"{trade['confidence']:.4f}",
    }


def _load_trades(csv_path, tracker):
    """Load closed trades from CSV into tracker."""
    if not os.path.exists(csv_path):
        return 0
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if rows:
        tracker.load_closed_trades(rows)
    return len(rows)


def main():
    args = parse_args()

    pair = args.pair
    timeframe = args.timeframe
    tag = _pair_tag(pair)
    tf_seconds = timeframe_to_seconds(timeframe)
    output_dir = config.LIVE_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    pred_csv = os.path.join(output_dir, f'predictions_{tag}_{timeframe}.csv')
    trade_csv = os.path.join(output_dir, f'trades_{tag}_{timeframe}.csv')
    buffer_csv = os.path.join(output_dir, f'buffer_{tag}_{timeframe}.csv')
    state_json = os.path.join(output_dir, f'state_{tag}_{timeframe}.json')

    # num_classes and class names will be derived from engine after loading
    # prob columns: 3 for triple mode, 2 for binary modes
    pred_fields = ['timestamp', 'pair', 'open', 'high', 'low', 'close', 'volume',
                   'pred_class', 'confidence', 'prob_0', 'prob_1', 'prob_2']
    trade_fields = ['signal_ts', 'pair', 'direction', 'entry_price', 'tp_level', 'sl_level',
                    'atr', 'close_ts', 'outcome', 'hit_candle', 'confidence']

    # Determine resume vs fresh start
    can_resume = (not args.fresh
                  and os.path.exists(pred_csv)
                  and os.path.exists(trade_csv))

    if args.fresh:
        # Force fresh: erase everything
        _force_csv(pred_csv, pred_fields)
        _force_csv(trade_csv, trade_fields)
        if os.path.exists(state_json):
            os.remove(state_json)
        print("  Mode: FRESH (all CSVs erased)")
    else:
        # Ensure CSVs exist (create if missing, keep if present)
        _ensure_csv(pred_csv, pred_fields)
        _ensure_csv(trade_csv, trade_fields)

    # --- Initialize components ---
    print(f"{'='*60}")
    print(f"  Live Inference - {pair} {timeframe}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dashboard: http://localhost:{args.port}")
    if can_resume:
        print(f"  Mode: RESUME (loading previous session)")
    print(f"{'='*60}")

    print("\n[1/5] Loading model...")
    engine = PredictionEngine(args.checkpoint, args.device)

    # Derive num_classes and class names from checkpoint
    prediction_target = engine.prediction_target
    num_classes = engine.num_classes
    class_names = engine.class_names

    print("[2/5] Initializing tracker...")
    tracker = TradeTracker(
        atr_period=config.ATR_PERIOD,
        atr_multiplier_tp=config.ATR_MULTIPLIER_TP,
        atr_multiplier_sl=config.ATR_MULTIPLIER_SL,
        max_horizon=config.MAX_HORIZON,
        num_classes=num_classes,
        reject_class=REJECT_CLASS[prediction_target],
        prediction_target=prediction_target,
    )

    # Resume: load previous predictions and trades
    predictions_log = []
    if can_resume:
        predictions_log = _load_predictions(pred_csv, class_names)
        n_trades = _load_trades(trade_csv, tracker)
        print(f"  Resumed {len(predictions_log)} predictions, {n_trades} closed trades")

    # Resume: load open trades + pending from state
    has_state = False
    if can_resume and os.path.exists(state_json):
        has_state = tracker.load_state(state_json)
        if has_state:
            print(f"  Restored {len(tracker.open_trades)} open trades, "
                  f"pending={'yes' if tracker.pending_trade else 'no'}")

    print("[3/5] Loading candles...")
    feed = CandleFeed(args.exchange, pair, timeframe, args.buffer_size)

    # Check if buffer CSV exists and is recent enough (<24h)
    buffer_loaded = False
    if can_resume and os.path.exists(buffer_csv):
        mtime = os.stat(buffer_csv).st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            try:
                feed.load_buffer(buffer_csv)
                buffer_loaded = True
                print(f"  Buffer loaded from CSV ({age_hours:.1f}h old)")
            except Exception as e:
                logger.warning(f"Failed to load buffer CSV: {e}")

    if buffer_loaded:
        # Fill gap: download only missing candles
        gap_candles = feed.fill_gap()
        print(f"  Gap filled: {len(gap_candles)} new candles")

        # Replay gap candles on open trades
        if has_state and gap_candles:
            print(f"  Replaying {len(gap_candles)} gap candles on open trades...")
            for candle in gap_candles:
                # Open pending trade if any
                if tracker.pending_trade is not None:
                    tracker.open_from_pending(candle, feed.buffer)
                # Check TP/SL/timeout
                closed = tracker.update(candle)
                for t in closed:
                    _append_csv(trade_csv, _format_trade_row(t, pair), trade_fields)
                    print(f"  >> Gap trade closed: {t['direction_name']} -> {t['outcome']}")
            print(f"  After replay: {len(tracker.open_trades)} trades still open")
    else:
        # Full download from exchange
        feed.fill_buffer()

    # Clean up state file after loading (one-shot)
    if os.path.exists(state_json):
        os.remove(state_json)

    print("[4/5] Starting dashboard...")
    dashboard = LiveDashboard(args.port, pair, timeframe, num_classes=num_classes,
                              prediction_target=prediction_target)
    dashboard.start()

    print("[5/5] Ready!")

    # Initial dashboard update
    dashboard.update(
        feed.buffer,
        predictions_log,
        tracker.closed_trades,
        tracker.open_trades,
        tracker.stats,
    )

    # --- Main loop ---
    candle_count = 0
    alerts = []
    compare_state = {
        'total_ok': 0,
        'total_errors': 0,
        'error_details': [],
    }

    print(f"\nWaiting for candles... (next close: {feed.next_candle_time()} UTC)")
    print(f"Dashboard: http://localhost:{args.port}\n")

    try:
        while True:
            # Wait until next candle close, refreshing open candle every 60s
            next_close = feed.next_candle_time()
            next_close_utc = next_close.to_pydatetime() if hasattr(next_close, 'to_pydatetime') else next_close
            if next_close_utc.tzinfo is None:
                next_close_utc = next_close_utc.replace(tzinfo=timezone.utc)

            while True:
                now = datetime.now(timezone.utc)
                remaining = (next_close_utc - now).total_seconds() + 5
                if remaining <= 0:
                    break

                # Fetch open candle for live display + early TP/SL check
                live_candle = feed.fetch_open_candle()
                if live_candle and tracker.open_trades:
                    closed_early = tracker.update(live_candle, intra_candle=True)
                    for t in closed_early:
                        _append_csv(trade_csv, _format_trade_row(t, pair), trade_fields)
                        print(f"  >> Trade closed (intra-candle): {t['direction_name']} -> {t['outcome']}")

                dashboard.update(
                    feed.buffer, predictions_log,
                    tracker.closed_trades, tracker.open_trades,
                    tracker.stats, alerts or None,
                    live_candle=live_candle,
                )

                sleep_time = min(config.LIVE_REFRESH_INTERVAL, remaining)
                logger.debug(f"Next close in {remaining:.0f}s, sleeping {sleep_time:.0f}s...")
                time.sleep(sleep_time)

            # Fetch new candle
            candle_data, fetch_alert = feed.fetch_latest_candle(
                retry_max=config.LIVE_RETRY_MAX,
                retry_timeout=config.LIVE_RETRY_TIMEOUT,
            )

            if fetch_alert:
                alerts.append(f"Slow candle fetch at {datetime.now(timezone.utc).strftime('%H:%M')}")

            if candle_data is None:
                # No new candle yet, wait a bit and retry
                logger.debug("No new candle, retrying in 10s...")
                time.sleep(10)
                continue

            candle_count += 1

            # --- Open pending trade on this candle's open ---
            if tracker.pending_trade is not None:
                tracker.open_from_pending(candle_data, feed.buffer)

            # --- Check open trades for TP/SL/timeout ---
            closed = tracker.update(candle_data)

            # Log closed trades to CSV
            for t in closed:
                _append_csv(trade_csv, _format_trade_row(t, pair), trade_fields)

            # --- Run prediction ---
            result = engine.predict(feed.buffer)

            if result:
                result['signal_ts'] = candle_data['timestamp']
                result['timestamp'] = candle_data['timestamp']
                predictions_log.append(result)

                # Console output
                stats = tracker.stats
                stats_str = (f"W:{stats['wins']} L:{stats['losses']} T:{stats['timeouts']} "
                             f"WR:{stats['winrate']:.1f}%") if stats['total'] > 0 else "no trades"

                print(f"[{candle_count}] {candle_data['timestamp']} "
                      f"C={candle_data['close']:.2f} "
                      f"-> {result['class_name']} ({result['confidence']:.3f}) "
                      f"| {stats_str}")

                # Log to CSV
                probs = result['probs']
                pred_row = {
                    'timestamp': _fmt_ts(candle_data['timestamp']),
                    'pair': pair,
                    'open': candle_data['open'],
                    'high': candle_data['high'],
                    'low': candle_data['low'],
                    'close': candle_data['close'],
                    'volume': candle_data['volume'],
                    'pred_class': result['pred'],
                    'confidence': f"{result['confidence']:.4f}",
                    'prob_0': f"{probs[0]:.4f}",
                    'prob_1': f"{probs[1]:.4f}" if len(probs) > 1 else "",
                    'prob_2': f"{probs[2]:.4f}" if len(probs) > 2 else "",
                }
                _append_csv(pred_csv, pred_row, pred_fields)

                # Set as pending trade
                tracker.set_pending(result)
            else:
                print(f"[{candle_count}] {candle_data['timestamp']} "
                      f"C={candle_data['close']:.2f} -> prediction failed (not enough data?)")

            # --- Update dashboard ---
            dashboard.update(
                feed.buffer,
                predictions_log,
                tracker.closed_trades,
                tracker.open_trades,
                tracker.stats,
                alerts or None,
            )

            # --- Periodic comparison ---
            if candle_count % config.LIVE_COMPARE_EVERY == 0:
                logger.info(f"Periodic comparison at candle {candle_count}...")
                try:
                    feed.save_buffer(buffer_csv)
                    _run_comparison(buffer_csv, predictions_log, engine, alerts, compare_state,
                                     buffer_size=args.buffer_size, num_classes=num_classes,
                                     class_names=class_names)
                    # Refresh dashboard immediately with comparison results
                    dashboard.update(
                        feed.buffer, predictions_log,
                        tracker.closed_trades, tracker.open_trades,
                        tracker.stats, alerts or None,
                        compare_details=compare_state['error_details'],
                        compare_state=compare_state,
                    )
                except Exception as e:
                    logger.error(f"Comparison failed: {e}")

    except KeyboardInterrupt:
        print(f"\n\nShutting down...")
        feed.save_buffer(buffer_csv)

        # Save open trades + pending for resume
        if tracker.open_trades or tracker.pending_trade:
            tracker.save_state(state_json)

        stats = tracker.stats
        total_checked = compare_state['total_ok'] + compare_state['total_errors']
        print(f"\n{'='*60}")
        print(f"  Session Summary - {pair} {timeframe}")
        print(f"  Candles processed: {candle_count}")
        print(f"  Predictions: {len(predictions_log)}")
        print(f"  Trades: {stats['total']} ({stats['wins']}W/{stats['losses']}L/{stats['timeouts']}T)")
        print(f"  Winrate: {stats['winrate']:.1f}%")
        print(f"  Open trades: {stats['open']}")
        print(f"  Comparison: {compare_state['total_ok']}/{total_checked} OK, "
              f"{compare_state['total_errors']} error(s)")
        if compare_state['error_details']:
            print(f"  Error details:")
            for detail in compare_state['error_details']:
                print(f"    - {detail}")
        if os.path.exists(state_json):
            print(f"  State saved for resume: {state_json}")
        print(f"{'='*60}")
        print(f"  Logs: {pred_csv}")
        print(f"  Trades: {trade_csv}")
        print(f"  Buffer: {buffer_csv}")
        print(f"{'='*60}")

        # Force exit — dashboard.stop() can block indefinitely
        # (server.shutdown() waits for serve_forever thread acknowledgement).
        # Dashboard thread is daemon=True, so it dies automatically on exit.
        sys.exit(0)


def _run_comparison(buffer_csv, predictions_log, engine, alerts, compare_state,
                     buffer_size, num_classes, class_names):
    """
    Compare ALL live predictions against fresh re-inference on the same candles.
    Runs every LIVE_COMPARE_EVERY candles. Checks every prediction, every time.

    For each prediction in predictions_log:
      1. Find the candle timestamp in the buffer
      2. Take exactly buffer_size candles ending at that timestamp
      3. Re-infer → compare class, confidence, probabilities
      4. Any difference = 1 error for that candle

    If a candle cannot be found or buffer is insufficient → also 1 error.

    Args:
        buffer_size: number of candles per inference slice. Defaults to config.LIVE_BUFFER_SIZE.

    compare_state dict:
        total_ok: int — candles with perfect match
        total_errors: int — candles with any difference
        error_details: list — [detail_str, ...]
    """
    if not predictions_log:
        return

    try:
        df_full = CandleFeed.read_csv(buffer_csv)
    except Exception as e:
        logger.error(f"Cannot load buffer for comparison: {e}")
        return

    # Reset state each run (we compare ALL predictions every time)
    compare_state['total_ok'] = 0
    compare_state['total_errors'] = 0
    compare_state['error_details'] = []

    for live_pred in predictions_log:
        ts = pd.Timestamp(live_pred['timestamp'])

        # Find candle in buffer
        mask = df_full['Open time'] == ts
        if not mask.any():
            detail = f"{ts} Candle missing from buffer"
            logger.error(detail)
            compare_state['total_errors'] += 1
            compare_state['error_details'].append(detail)
            continue

        idx = df_full.index[mask][-1]

        # Check enough history
        start_idx = idx - buffer_size + 1
        if start_idx < 0:
            detail = (f"{ts} Insufficient history "
                      f"(need {buffer_size}, available {idx + 1})")
            logger.error(detail)
            compare_state['total_errors'] += 1
            compare_state['error_details'].append(detail)
            continue

        df_slice = df_full.iloc[start_idx:idx + 1].copy().reset_index(drop=True)

        # Re-run inference
        result = engine.predict(df_slice)
        if result is None:
            detail = f"{ts} Re-inference returned None"
            logger.error(detail)
            compare_state['total_errors'] += 1
            compare_state['error_details'].append(detail)
            continue

        # Compare everything
        class_mismatch = result['pred'] != live_pred['pred']

        conf_live = round(float(live_pred['confidence']), 4)
        conf_check = round(float(result['confidence']), 4)
        conf_mismatch = conf_live != conf_check

        proba_mismatch = False
        for p_live, p_check in zip(live_pred['probs'], result['probs']):
            if round(float(p_live), 4) != round(float(p_check), 4):
                proba_mismatch = True
                break

        if class_mismatch or conf_mismatch or proba_mismatch:
            parts = []
            if class_mismatch:
                parts.append(f"Class {class_names[live_pred['pred']]}→{class_names[result['pred']]}")
            if conf_mismatch:
                parts.append(f"Conf {conf_live:.4f}→{conf_check:.4f}")
            if proba_mismatch:
                proba_diffs = []
                for i, (p_live, p_check) in enumerate(
                        zip(live_pred['probs'], result['probs'])):
                    p_l, p_c = round(float(p_live), 4), round(float(p_check), 4)
                    if p_l != p_c:
                        proba_diffs.append(f"{class_names[i]} {p_l:.4f}→{p_c:.4f}")
                parts.append(f"Proba [{', '.join(proba_diffs)}]")

            detail = f"{ts} Error {' | '.join(parts)}"

            logger.error(detail)
            compare_state['total_errors'] += 1
            compare_state['error_details'].append(detail)
        else:
            compare_state['total_ok'] += 1

    # Log results
    total = compare_state['total_ok'] + compare_state['total_errors']
    n_err = compare_state['total_errors']
    logger.info(f"Comparison: {compare_state['total_ok']}/{total} OK, "
                f"{n_err} error(s)")

    if n_err > 0:
        print(f"\n{'!'*60}")
        print(f"  COMPARISON: {n_err} ERROR(S) / {total} predictions")
        print(f"{'!'*60}")
        for detail in compare_state['error_details']:
            print(f"  {detail}")
        print(f"{'!'*60}\n")

    # Update alerts (only on errors — OK status shown in stats card)
    ALERT_TAG = "[COMPARE]"
    alerts[:] = [a for a in alerts if not a.startswith(ALERT_TAG)]

    if n_err > 0:
        alerts.append(f"{ALERT_TAG} {n_err}/{total} predictions with errors")


if __name__ == '__main__':
    main()
