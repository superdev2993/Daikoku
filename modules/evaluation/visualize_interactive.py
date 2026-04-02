"""
Interactive input features visualization - TradingView-style with Lightweight Charts v4.

7 or 8 synchronized panels (8 when Volume Profile is enabled):
1. OHLC candlestick + ZigZag overlay (candles colored by regime trend)
2. Normalized z-score features (7 lines)
3. Cyclical time features sin/cos (4 lines)
4. Structural S/R features (5 lines)
5. Volume Profile features (conditional, 5 lines)
6. Regime Trend oscillator (colored line + area fill)
7. Regime VolTrend oscillator (grey line + area fill)
8. Triple Barrier labels (colored histogram)
"""

import argparse
import json
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from modules.data.labeling import get_labels
from modules.data.loader import get_raw_data
from modules.data.transform import transform_pipeline
from modules.data.zigzag import compute_zigzag_interpolation, compute_structural_features
from modules.data.regime import compute_regime_filter
from modules.data.volume_profile import compute_vp_features


# Feature names for panels (imported from single source of truth)
from modules.data.dataset import (
    NORM_NAMES as ZSCORE_FEATURE_NAMES,
    TIME_NAMES as TIME_FEATURE_NAMES,
    STRUCT_NAMES as STRUCT_FEATURE_NAMES,
    VP_NAMES as VP_FEATURE_NAMES,
)

# Colors for features (distinct, readable on dark bg)
ZSCORE_COLORS = [
    '#2196F3',  # log_price - blue
    '#4CAF50',  # log_open - green
    '#FF9800',  # log_wick_high - orange
    '#F44336',  # log_wick_low - red
    '#9C27B0',  # log_volume - purple
    '#00BCD4',  # log_zz - cyan
    '#FFEB3B',  # log_atr - yellow
]
TIME_COLORS = [
    '#E91E63',  # sin_hour - pink
    '#3F51B5',  # cos_hour - indigo
    '#8BC34A',  # sin_weekday - light green
    '#FF5722',  # cos_weekday - deep orange
]
STRUCT_COLORS = [
    '#FFFFFF',  # struct_rank - white
    '#FF1744',  # dist_res_1 - bright red
    '#FF8A80',  # dist_res_2 - light red
    '#00E676',  # dist_sup_1 - bright green
    '#69F0AE',  # dist_sup_2 - light green
]

VP_COLORS = [
    '#FFD700',  # vp_poc - gold
    '#1E90FF',  # vp_vah - dodger blue
    '#FF6347',  # vp_val - tomato
    '#BA55D3',  # vp_width - medium orchid
    '#00CED1',  # vp_skew - dark turquoise
]

LABEL_COLORS = {0: '#EF553B', 1: '#FFA15A', 2: '#00CC96'}
LABEL_NAMES = {0: 'Bearish', 1: 'Uncertain', 2: 'Bullish'}


def trend_to_color(score):
    """
    Map trend score [-10, +10] to color gradient (BigBeluga style).
    score < 0: red (#ef5350) -> orange (#FF9800)
    score > 0: orange (#FF9800) -> green (#26a69a)
    score == 0: orange
    """
    if np.isnan(score):
        return '#555555'
    s = max(-10.0, min(10.0, score))
    if s <= 0:
        t = (s + 10.0) / 10.0
        r = int(0xef + (0xff - 0xef) * t)
        g = int(0x53 + (0x98 - 0x53) * t)
        b = int(0x50 + (0x00 - 0x50) * t)
    else:
        t = s / 10.0
        r = int(0xff + (0x26 - 0xff) * t)
        g = int(0x98 + (0xa6 - 0x98) * t)
        b = int(0x00 + (0x9a - 0x00) * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def _apply_rolling_normalization(data, n_norm_features, window_size):
    """
    Apply rolling window normalization (median/IQR) on first n_norm_features columns.
    Approximates what the model sees in window mode.
    """
    result = data.copy()
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i + 1, :n_norm_features]
        if len(window) < 2:
            continue
        median = np.median(window, axis=0)
        q75 = np.percentile(window, 75, axis=0)
        q25 = np.percentile(window, 25, axis=0)
        iqr = q75 - q25
        result[i, :n_norm_features] = (data[i, :n_norm_features] - median) / (iqr + 1e-8)
    return result


def prepare_data(file_path, checkpoint_params=None, split=None):
    """
    Load data, compute labels, transform, and compute zigzag + regime.

    Args:
        file_path: Path to CSV data file
        checkpoint_params: Optional dict from checkpoint config.
            If provided, uses checkpoint's D, etc.
            If None, uses current config.py settings.
        split: Optional split to visualize ('train', 'test', 'all', None).
            If provided, only the corresponding portion is returned.
            If None, returns all data.

    Returns dict with all data arrays aligned by a shared timestamp array.
    """
    print(f"Loading data from {file_path}...")
    df = get_raw_data(file_path)
    n_raw = len(df)
    print(f"  {n_raw} candles loaded")

    # Determine parameters: checkpoint overrides > config defaults
    window_size_cfg = checkpoint_params['WINDOW_SIZE'] if checkpoint_params else config.WINDOW_SIZE
    atr_multiplier_tp = checkpoint_params['ATR_MULTIPLIER_TP'] if checkpoint_params else config.ATR_MULTIPLIER_TP
    atr_multiplier_sl = checkpoint_params['ATR_MULTIPLIER_SL'] if checkpoint_params else config.ATR_MULTIPLIER_SL
    atr_period = checkpoint_params['ATR_PERIOD'] if checkpoint_params else config.ATR_PERIOD
    max_horizon = checkpoint_params['MAX_HORIZON'] if checkpoint_params else config.MAX_HORIZON
    regime_length = checkpoint_params['REGIME_LENGTH'] if checkpoint_params else config.REGIME_LENGTH

    print("Computing labels...")
    labels, atr_array, _, _, _ = get_labels(
        df, window_size=window_size_cfg,
        atr_multiplier_tp=atr_multiplier_tp, atr_multiplier_sl=atr_multiplier_sl, max_horizon=max_horizon,
        atr_period=atr_period
    )
    n_labels = len(labels)

    print("Running transform pipeline...")
    result = transform_pipeline(df, atr_array, window_size=window_size_cfg)
    all_data = result['data']
    window_size = result['window_size']

    # Compute zigzag on full data (needed before normalization)
    high_full = df['High'].values.astype(np.float64)
    low_full = df['Low'].values.astype(np.float64)
    close_full = df['Close'].values.astype(np.float64)
    zz_interp, pivot_indices, pivot_prices, pivot_types = compute_zigzag_interpolation(
        high_full, low_full, close_full, length=config.ZIGZAG_LENGTH
    )
    # Slice zigzag to match all_data (starts at window_size in raw df)
    log_zz_aligned = np.log(np.maximum(zz_interp[window_size:window_size + len(all_data)], 1e-10))

    # Insert log_zz at index 5 (between OHLCV and ATR) to match model's 13-col layout:
    # [0-4] OHLCV, [5] log_zz, [6] log_atr, [7-10] time, [11-12] regime
    all_data = np.insert(all_data, 5, log_zz_aligned, axis=1)
    n_norm_features = 7  # First 7 features: 5 OHLCV + log_zz + log_atr

    # Apply rolling normalization for visualization (window mode)
    print(f"  Applying rolling normalization (window={window_size}) for visualization...")
    all_data = _apply_rolling_normalization(all_data, n_norm_features, window_size)

    # Alignment offset: transformed data starts at window_size in raw df
    raw_offset = window_size

    # Labels start at window_size in raw df, same as transformed data
    labels_aligned = labels

    # Common length
    n = min(len(all_data), len(labels_aligned))
    all_data = all_data[:n]
    labels_aligned = labels_aligned[:n]

    # Apply split filter if requested
    split_ratio = checkpoint_params['TRAIN_TEST_SPLIT'] if checkpoint_params else config.TRAIN_TEST_SPLIT
    split_index = int(n * split_ratio)

    if split == 'train':
        all_data = all_data[:split_index]
        labels_aligned = labels_aligned[:split_index]
        n = len(all_data)
        print(f"  Split '{split}': keeping first {n} points (0:{split_index})")
    elif split == 'test':
        all_data = all_data[split_index:]
        labels_aligned = labels_aligned[split_index:]
        raw_offset = raw_offset + split_index
        n = len(all_data)
        print(f"  Split '{split}': keeping last {n} points ({split_index}:)")
    else:
        print(f"  Split: all ({n} points)")

    # Extract OHLC for candlestick from raw df and apply log transformation
    ohlc_slice = df.iloc[raw_offset:raw_offset + n]
    timestamps = ohlc_slice['Open time'].values

    # Apply log transformation to OHLC (as model sees it)
    log_open = np.log(ohlc_slice['Open'].values)
    log_high = np.log(ohlc_slice['High'].values)
    log_low = np.log(ohlc_slice['Low'].values)
    log_close = np.log(ohlc_slice['Close'].values)

    # Slice zigzag for overlay display (post-split)
    zz_slice = zz_interp[raw_offset:raw_offset + n]
    log_zz_slice = np.log(np.maximum(zz_slice, 1e-10))

    # Compute structural features on full data, then slice
    struct_features_full = compute_structural_features(
        close_full, pivot_indices, pivot_prices,
        length=config.ZIGZAG_LENGTH, lookback=window_size
    )
    struct_slice = struct_features_full[raw_offset:raw_offset + n]

    # Filter pivots that fall within the displayed range and apply log to prices
    pivots_in_range = []
    for idx, price, ptype in zip(pivot_indices, pivot_prices, pivot_types):
        if raw_offset <= idx < raw_offset + n:
            pivots_in_range.append((idx - raw_offset, np.log(price), ptype))  # Log-transformed price

    # Compute raw regime filter for visualization ([-10, +10])
    print(f"Computing regime filter (length={regime_length})...")
    trend_raw, voltrend_raw = compute_regime_filter(df, length=regime_length)
    trend_slice = trend_raw[raw_offset:raw_offset + n]
    voltrend_slice = voltrend_raw[raw_offset:raw_offset + n]

    print(f"  Aligned {n} data points (raw_offset={raw_offset})")

    # Compute Volume Profile features
    vp_lookback = checkpoint_params['VP_LOOKBACK'] if checkpoint_params else config.VP_LOOKBACK
    vp_bins = checkpoint_params['VP_BINS'] if checkpoint_params else config.VP_BINS
    vp_va_pct = checkpoint_params['VP_VA_PCT'] if checkpoint_params else config.VP_VA_PCT

    vp_raw_start = max(0, raw_offset - vp_lookback + 1)
    vp_raw_end = raw_offset + n
    h_vp = df['High'].values[vp_raw_start:vp_raw_end].astype(np.float64)
    l_vp = df['Low'].values[vp_raw_start:vp_raw_end].astype(np.float64)
    c_vp = df['Close'].values[vp_raw_start:vp_raw_end].astype(np.float64)
    v_vp = df['Volume'].values[vp_raw_start:vp_raw_end].astype(np.float64)

    print(f"Computing Volume Profile features (lookback={vp_lookback}, bins={vp_bins})...")
    vp_features = compute_vp_features(h_vp, l_vp, c_vp, v_vp, vp_lookback, vp_bins, vp_va_pct)
    vp_n = len(vp_features)
    if vp_n < n:
        pad = np.zeros((n - vp_n, 5), dtype=np.float32)
        vp_features = np.concatenate([pad, vp_features], axis=0)
    elif vp_n > n:
        vp_features = vp_features[-n:]
    print(f"  VP features: {vp_features.shape}")

    # Build mode description for panel titles
    norm_desc = "rolling window median/IQR"

    return {
        'timestamps': timestamps,
        'open': log_open,
        'high': log_high,
        'low': log_low,
        'close': log_close,
        'zigzag': log_zz_slice,
        'pivots': pivots_in_range,
        'features': all_data,
        'labels': labels_aligned,
        'trend': trend_slice,
        'voltrend': voltrend_slice,
        'vp_features': vp_features,
        'struct_features': struct_slice,
        'n': n,
        'mode_desc': f"log + {norm_desc}",
        'normalize_mode': 'window',
    }


def build_json_data(data):
    """
    Serialize data to JSON for embedding in HTML.

    Uses shared timestamp array + separate value arrays for efficiency.
    """
    n = data['n']
    timestamps = data['timestamps']

    # Convert timestamps to unix seconds
    ts_unix = []
    for t in timestamps:
        ts = np.datetime64(t, 's').astype(int)
        ts_unix.append(int(ts))

    # Panel 1: OHLC candles (log-transformed, colored by regime trend)
    candles = []
    has_trend = 'trend' in data
    for i in range(n):
        c = {
            'time': ts_unix[i],
            'open': round(float(data['open'][i]), 4),
            'high': round(float(data['high'][i]), 4),
            'low': round(float(data['low'][i]), 4),
            'close': round(float(data['close'][i]), 4),
        }
        if has_trend:
            color = trend_to_color(data['trend'][i])
            c['color'] = color
            c['borderColor'] = color
            c['wickColor'] = color
        candles.append(c)

    # Panel 1: ZigZag interpolated overlay (log-transformed)
    zigzag_line = []
    for i in range(n):
        zigzag_line.append({
            'time': ts_unix[i],
            'value': round(float(data['zigzag'][i]), 4),
        })

    # Panel 1: ZigZag pivot lines (connecting confirmed pivots, log-transformed)
    zigzag_pivots = []
    for local_idx, log_price, ptype in data['pivots']:
        zigzag_pivots.append({
            'time': ts_unix[local_idx],
            'value': round(float(log_price), 4),
        })

    # Panel 2: Z-score features (indices 0-6)
    zscore_series = {}
    for j, name in enumerate(ZSCORE_FEATURE_NAMES):
        series = []
        for i in range(n):
            val = float(data['features'][i, j])
            if np.isfinite(val):
                series.append({'time': ts_unix[i], 'value': round(val, 4)})
        zscore_series[name] = series

    # Panel 3: Time features (indices 7-10)
    time_series = {}
    for j, name in enumerate(TIME_FEATURE_NAMES):
        series = []
        for i in range(n):
            val = float(data['features'][i, 7 + j])
            if np.isfinite(val):
                series.append({'time': ts_unix[i], 'value': round(val, 4)})
        time_series[name] = series

    # Panel 4: Structural features (from dedicated struct_features array)
    struct_series = {}
    if data.get('struct_features') is not None:
        sf = data['struct_features']
        for j, name in enumerate(STRUCT_FEATURE_NAMES):
            series = []
            for i in range(n):
                val = float(sf[i, j])
                if np.isfinite(val):
                    series.append({'time': ts_unix[i], 'value': round(val, 4)})
            struct_series[name] = series

    # Panel 5: Volume Profile features (if available)
    vp_series = {}
    if data.get('vp_features') is not None:
        vp = data['vp_features']
        for j, name in enumerate(VP_FEATURE_NAMES):
            series = []
            for i in range(n):
                val = float(vp[i, j])
                if np.isfinite(val):
                    series.append({'time': ts_unix[i], 'value': round(val, 6)})
            vp_series[name] = series

    # Panel 6: Regime Trend oscillator (raw [-10, +10])
    regime_trend_data = []
    regime_trend_pos = []
    regime_trend_neg = []
    if 'trend' in data:
        for i in range(n):
            val = float(data['trend'][i])
            if np.isfinite(val):
                regime_trend_data.append({
                    'time': ts_unix[i],
                    'value': round(val, 4),
                    'color': trend_to_color(val),
                })
                if val >= 0:
                    regime_trend_pos.append({'time': ts_unix[i], 'value': round(val, 4)})
                    regime_trend_neg.append({'time': ts_unix[i], 'value': 0})
                else:
                    regime_trend_pos.append({'time': ts_unix[i], 'value': 0})
                    regime_trend_neg.append({'time': ts_unix[i], 'value': round(val, 4)})

    # Panel 6: Regime VolTrend oscillator (raw [-10, +10])
    regime_voltrend_data = []
    regime_voltrend_pos = []
    regime_voltrend_neg = []
    if 'voltrend' in data:
        for i in range(n):
            val = float(data['voltrend'][i])
            if np.isfinite(val):
                regime_voltrend_data.append({
                    'time': ts_unix[i],
                    'value': round(val, 4),
                })
                if val >= 0:
                    regime_voltrend_pos.append({'time': ts_unix[i], 'value': round(val, 4)})
                    regime_voltrend_neg.append({'time': ts_unix[i], 'value': 0})
                else:
                    regime_voltrend_pos.append({'time': ts_unix[i], 'value': 0})
                    regime_voltrend_neg.append({'time': ts_unix[i], 'value': round(val, 4)})

    # Panel 7: Labels histogram
    label_hist = []
    for i in range(n):
        label = int(data['labels'][i])
        # Map label to signed value for visual separation
        val_map = {0: -1.0, 1: 0.0, 2: 1.0}
        label_hist.append({
            'time': ts_unix[i],
            'value': val_map[label],
            'color': LABEL_COLORS[label],
        })

    return {
        'candles': candles,
        'zigzag': zigzag_line,
        'zigzag_pivots': zigzag_pivots,
        'zscore': zscore_series,
        'time': time_series,
        'struct': struct_series,
        'vp': vp_series,
        'regime_trend': regime_trend_data,
        'regime_trend_pos': regime_trend_pos,
        'regime_trend_neg': regime_trend_neg,
        'regime_voltrend': regime_voltrend_data,
        'regime_voltrend_pos': regime_voltrend_pos,
        'regime_voltrend_neg': regime_voltrend_neg,
        'labels': label_hist,
        'mode_desc': data.get('mode_desc', ''),
    }


def generate_html(json_data, output_path):
    """Generate the interactive HTML file with synchronized panels (7 or 8 with VP)."""
    candle_json = json.dumps(json_data['candles'])
    zigzag_json = json.dumps(json_data['zigzag'])
    zigzag_pivots_json = json.dumps(json_data['zigzag_pivots'])
    label_json = json.dumps(json_data['labels'])
    has_struct = bool(json_data.get('struct'))
    has_vp = bool(json_data.get('vp'))
    mode_desc = json_data.get('mode_desc', '')

    # Regime oscillator data
    regime_trend_json = json.dumps(json_data.get('regime_trend', []))
    regime_trend_pos_json = json.dumps(json_data.get('regime_trend_pos', []))
    regime_trend_neg_json = json.dumps(json_data.get('regime_trend_neg', []))
    regime_voltrend_json = json.dumps(json_data.get('regime_voltrend', []))
    regime_voltrend_pos_json = json.dumps(json_data.get('regime_voltrend_pos', []))
    regime_voltrend_neg_json = json.dumps(json_data.get('regime_voltrend_neg', []))

    # Build zscore series JS assignments
    zscore_js_vars = []
    for name in ZSCORE_FEATURE_NAMES:
        zscore_js_vars.append(f'const data_{name} = {json.dumps(json_data["zscore"][name])};')
    zscore_js_block = '\n        '.join(zscore_js_vars)

    # Build time series JS assignments
    time_js_vars = []
    for name in TIME_FEATURE_NAMES:
        time_js_vars.append(f'const data_{name} = {json.dumps(json_data["time"][name])};')
    time_js_block = '\n        '.join(time_js_vars)

    # Build structural series JS assignments
    struct_js_vars = []
    if has_struct:
        for name in STRUCT_FEATURE_NAMES:
            struct_js_vars.append(f'const data_{name} = {json.dumps(json_data["struct"][name])};')
    struct_js_block = '\n        '.join(struct_js_vars) if struct_js_vars else ''

    # Build zscore series creation
    zscore_series_js = []
    for i, name in enumerate(ZSCORE_FEATURE_NAMES):
        color = ZSCORE_COLORS[i]
        zscore_series_js.append(f'''
            const series_{name} = chart2.addLineSeries({{
                color: '{color}',
                lineWidth: 1,
                title: '{name}',
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            series_{name}.setData(data_{name});
            zscoreSeries.push({{ series: series_{name}, name: '{name}', color: '{color}' }});''')
    zscore_series_block = '\n'.join(zscore_series_js)

    # Build time series creation
    time_series_js = []
    for i, name in enumerate(TIME_FEATURE_NAMES):
        color = TIME_COLORS[i]
        time_series_js.append(f'''
            const series_{name} = chart3.addLineSeries({{
                color: '{color}',
                lineWidth: 1,
                title: '{name}',
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            series_{name}.setData(data_{name});
            timeSeries.push({{ series: series_{name}, name: '{name}', color: '{color}' }});''')
    time_series_block = '\n'.join(time_series_js)

    # Build zscore checkboxes
    zscore_checkboxes = []
    for i, name in enumerate(ZSCORE_FEATURE_NAMES):
        color = ZSCORE_COLORS[i]
        zscore_checkboxes.append(
            f'<label style="color:{color};cursor:pointer;display:inline-flex;align-items:center;gap:3px;">'
            f'<input type="checkbox" checked data-series="{name}" data-panel="zscore" '
            f'style="cursor:pointer;accent-color:{color};">{name}</label>'
        )
    zscore_checkboxes_html = '\n                '.join(zscore_checkboxes)

    # Build structural series creation
    struct_series_js = []
    if has_struct:
        for i, name in enumerate(STRUCT_FEATURE_NAMES):
            color = STRUCT_COLORS[i]
            struct_series_js.append(f'''
            const series_{name} = chart4.addLineSeries({{
                color: '{color}',
                lineWidth: 1,
                title: '{name}',
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            series_{name}.setData(data_{name});
            structSeries.push({{ series: series_{name}, name: '{name}', color: '{color}' }});''')
    struct_series_block = '\n'.join(struct_series_js)

    # Build time checkboxes
    time_checkboxes = []
    for i, name in enumerate(TIME_FEATURE_NAMES):
        color = TIME_COLORS[i]
        time_checkboxes.append(
            f'<label style="color:{color};cursor:pointer;display:inline-flex;align-items:center;gap:3px;">'
            f'<input type="checkbox" checked data-series="{name}" data-panel="time" '
            f'style="cursor:pointer;accent-color:{color};">{name}</label>'
        )
    time_checkboxes_html = '\n                '.join(time_checkboxes)

    # Build structural checkboxes
    struct_checkboxes = []
    if has_struct:
        for i, name in enumerate(STRUCT_FEATURE_NAMES):
            color = STRUCT_COLORS[i]
            struct_checkboxes.append(
                f'<label style="color:{color};cursor:pointer;display:inline-flex;align-items:center;gap:3px;">'
                f'<input type="checkbox" checked data-series="{name}" data-panel="struct" '
                f'style="cursor:pointer;accent-color:{color};">{name}</label>'
            )
    struct_checkboxes_html = '\n                '.join(struct_checkboxes)

    # Build VP series JS assignments
    vp_js_vars = []
    vp_series_js = []
    vp_checkboxes = []
    if has_vp:
        for name in VP_FEATURE_NAMES:
            vp_js_vars.append(f'const data_{name} = {json.dumps(json_data["vp"][name])};')
        for i, name in enumerate(VP_FEATURE_NAMES):
            color = VP_COLORS[i]
            vp_series_js.append(f'''
            const series_{name} = chartVP.addLineSeries({{
                color: '{color}',
                lineWidth: 1,
                title: '{name}',
                priceScaleId: 'right',
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            series_{name}.setData(data_{name});
            vpSeries.push({{ series: series_{name}, name: '{name}', color: '{color}' }});''')
            vp_checkboxes.append(
                f'<label style="color:{color};cursor:pointer;display:inline-flex;align-items:center;gap:3px;">'
                f'<input type="checkbox" checked data-series="{name}" data-panel="vp" '
                f'style="cursor:pointer;accent-color:{color};">{name}</label>'
            )
    vp_js_block = '\n        '.join(vp_js_vars)
    vp_series_block = '\n'.join(vp_series_js)
    vp_checkboxes_html = '\n                '.join(vp_checkboxes)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Input Features Visualization</title>
    <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #1a1a2e;
            color: #e0e0e0;
        }}
        .header {{
            background: #16213e;
            padding: 12px 24px;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .header h1 {{
            font-size: 16px;
            color: #82b1ff;
        }}
        .panel {{
            border-bottom: 1px solid #333;
            position: relative;
        }}
        .panel-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 12px;
            background: #16213e;
            font-size: 12px;
            flex-wrap: wrap;
        }}
        .panel-title {{
            font-weight: bold;
            color: #82b1ff;
            min-width: 120px;
        }}
        .panel-header label {{
            font-size: 11px;
        }}
        .chart-container {{
            width: 100%;
        }}
        .legend {{
            display: flex;
            gap: 12px;
            font-size: 11px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}
        .gradient-bar {{
            width: 60px;
            height: 8px;
            border-radius: 3px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Input Features Explorer <span style="font-size:12px;color:#aaa;font-weight:normal;">({mode_desc})</span></h1>
        <div class="legend">
            <span class="legend-item">
                <span class="gradient-bar" style="background:linear-gradient(to right, #ef5350, #FF9800, #26a69a);"></span>
                Regime
            </span>
            <span class="legend-item"><span class="legend-dot" style="background:#EF553B"></span>Bearish</span>
            <span class="legend-item"><span class="legend-dot" style="background:#FFA15A"></span>Uncertain</span>
            <span class="legend-item"><span class="legend-dot" style="background:#00CC96"></span>Bullish</span>
        </div>
    </div>

    <!-- Panel 1: OHLC + ZigZag (colored by regime trend) -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">OHLC + ZigZag</span>
            <label style="color:#FF6D00;cursor:pointer;display:inline-flex;align-items:center;gap:3px;">
                <input type="checkbox" checked id="zigzagToggle" style="cursor:pointer;accent-color:#FF6D00;">ZigZag smooth
            </label>
            <label style="color:#00E676;cursor:pointer;display:inline-flex;align-items:center;gap:3px;margin-left:10px;">
                <input type="checkbox" checked id="zigzagPivotsToggle" style="cursor:pointer;accent-color:#00E676;">ZigZag pivots
            </label>
        </div>
        <div id="chart1" class="chart-container"></div>
    </div>

    <!-- Panel 2: Normalized features -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Normalized Features <span style="font-size:10px;color:#aaa;">({mode_desc})</span></span>
            {zscore_checkboxes_html}
        </div>
        <div id="chart2" class="chart-container"></div>
    </div>

    <!-- Panel 3: Time features -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Time Features</span>
            {time_checkboxes_html}
        </div>
        <div id="chart3" class="chart-container"></div>
    </div>

    <!-- Panel 4: Structural features -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Structural S/R</span>
            {struct_checkboxes_html}
        </div>
        <div id="chart4" class="chart-container"></div>
    </div>

    <!-- Panel VP: Volume Profile features (conditional) -->
    {'<div class="panel"><div class="panel-header"><span class="panel-title">Volume Profile</span>' + vp_checkboxes_html + '</div><div id="chartVP" class="chart-container"></div></div>' if has_vp else ''}

    <!-- Panel 5: Regime Trend oscillator -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Regime Trend</span>
            <span style="font-size:10px;color:#aaa;">[-10, +10] HMA</span>
        </div>
        <div id="chart5" class="chart-container"></div>
    </div>

    <!-- Panel 6: Regime VolTrend oscillator -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Regime VolTrend</span>
            <span style="font-size:10px;color:#aaa;">[-10, +10] HMA</span>
        </div>
        <div id="chart6" class="chart-container"></div>
    </div>

    <!-- Panel 7: Labels -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Triple Barrier Labels</span>
        </div>
        <div id="chart7" class="chart-container"></div>
    </div>

    <script>
    (function() {{
        // ===== Data =====
        const candleData = {candle_json};
        const zigzagData = {zigzag_json};
        const zigzagPivotsData = {zigzag_pivots_json};
        const labelData = {label_json};
        {zscore_js_block}
        {time_js_block}
        {struct_js_block}
        {vp_js_block}
        const regimeTrendData = {regime_trend_json};
        const regimeTrendPosData = {regime_trend_pos_json};
        const regimeTrendNegData = {regime_trend_neg_json};
        const regimeVoltrendData = {regime_voltrend_json};
        const regimeVoltrendPosData = {regime_voltrend_pos_json};
        const regimeVoltrendNegData = {regime_voltrend_neg_json};

        // ===== Chart config =====
        const darkTheme = {{
            layout: {{
                background: {{ type: 'solid', color: '#1a1a2e' }},
                textColor: '#aaa',
            }},
            grid: {{
                vertLines: {{ color: 'rgba(255,255,255,0.04)' }},
                horzLines: {{ color: 'rgba(255,255,255,0.04)' }},
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            timeScale: {{
                timeVisible: true,
                secondsVisible: false,
                borderColor: '#333',
            }},
            rightPriceScale: {{ borderColor: '#333' }},
        }};

        function createChart(containerId, height) {{
            const container = document.getElementById(containerId);
            const chart = LightweightCharts.createChart(container, {{
                ...darkTheme,
                width: container.clientWidth,
                height: height,
            }});
            return chart;
        }}

        // ===== Create charts =====
        const chart1 = createChart('chart1', 400);
        const chart2 = createChart('chart2', 250);
        const chart3 = createChart('chart3', 150);
        const chart4 = createChart('chart4', 200);
        const chartVP = {'true' if has_vp else 'false'} ? createChart('chartVP', 180) : null;
        const chart5 = createChart('chart5', 150);
        const chart6 = createChart('chart6', 150);
        const chart7 = createChart('chart7', 120);

        const charts = [chart1, chart2, chart3, chart4, chart5, chart6, chart7];
        if (chartVP) charts.splice(4, 0, chartVP);

        // ===== Panel 1: Candlestick + ZigZag (colored by regime) =====
        const candleSeries = chart1.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        candleSeries.setData(candleData);

        const zigzagSeries = chart1.addLineSeries({{
            color: '#FF6D00',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        zigzagSeries.setData(zigzagData);

        // ZigZag pivot lines (straight lines connecting confirmed pivots)
        const zigzagPivotsSeries = chart1.addLineSeries({{
            color: '#00E676',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        zigzagPivotsSeries.setData(zigzagPivotsData);

        // ZigZag toggles
        document.getElementById('zigzagToggle').addEventListener('change', (e) => {{
            zigzagSeries.applyOptions({{
                visible: e.target.checked,
            }});
        }});
        document.getElementById('zigzagPivotsToggle').addEventListener('change', (e) => {{
            zigzagPivotsSeries.applyOptions({{
                visible: e.target.checked,
            }});
        }});

        // ===== Panel 2: Z-score features =====
        const zscoreSeries = [];
        {zscore_series_block}

        // Zero line for z-score panel
        const zeroLine = chart2.addLineSeries({{
            color: 'rgba(255,255,255,0.2)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        // Build zero line data from first/last timestamps
        if (candleData.length > 0) {{
            zeroLine.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel 3: Time features =====
        const timeSeries = [];
        {time_series_block}

        // ===== Panel 4: Structural features =====
        const structSeries = [];
        {struct_series_block}

        // Zero line for structural panel
        const structZeroLine = chart4.addLineSeries({{
            color: 'rgba(255,255,255,0.2)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        if (candleData.length > 0) {{
            structZeroLine.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel VP: Volume Profile features =====
        const vpSeries = [];
        {vp_series_block}

        // Zero line for VP panel
        if (chartVP && candleData.length > 0) {{
            const vpZeroLine = chartVP.addLineSeries({{
                color: 'rgba(255,255,255,0.2)',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            vpZeroLine.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel 5: Regime Trend oscillator =====
        // Area fill positive (green)
        const regimeTrendAreaPos = chart5.addAreaSeries({{
            topColor: 'rgba(38, 166, 154, 0.4)',
            bottomColor: 'rgba(38, 166, 154, 0.0)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        regimeTrendAreaPos.setData(regimeTrendPosData);

        // Area fill negative (red)
        const regimeTrendAreaNeg = chart5.addAreaSeries({{
            topColor: 'rgba(239, 83, 80, 0.0)',
            bottomColor: 'rgba(239, 83, 80, 0.4)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        regimeTrendAreaNeg.setData(regimeTrendNegData);

        // Trend line (colored per point)
        const regimeTrendLine = chart5.addLineSeries({{
            color: '#FF9800',
            lineWidth: 2,
            priceScaleId: 'right',
            lastValueVisible: true,
            priceLineVisible: false,
        }});
        regimeTrendLine.setData(regimeTrendData);

        // Zero line
        const regimeTrendZero = chart5.addLineSeries({{
            color: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        if (candleData.length > 0) {{
            regimeTrendZero.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel 6: Regime VolTrend oscillator =====
        // Area fill positive
        const regimeVoltrendAreaPos = chart6.addAreaSeries({{
            topColor: 'rgba(150, 150, 150, 0.3)',
            bottomColor: 'rgba(150, 150, 150, 0.0)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        regimeVoltrendAreaPos.setData(regimeVoltrendPosData);

        // Area fill negative
        const regimeVoltrendAreaNeg = chart6.addAreaSeries({{
            topColor: 'rgba(150, 150, 150, 0.0)',
            bottomColor: 'rgba(150, 150, 150, 0.3)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        regimeVoltrendAreaNeg.setData(regimeVoltrendNegData);

        // VolTrend line
        const regimeVoltrendLine = chart6.addLineSeries({{
            color: '#9e9e9e',
            lineWidth: 2,
            priceScaleId: 'right',
            lastValueVisible: true,
            priceLineVisible: false,
        }});
        regimeVoltrendLine.setData(regimeVoltrendData);

        // Zero line
        const regimeVoltrendZero = chart6.addLineSeries({{
            color: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        if (candleData.length > 0) {{
            regimeVoltrendZero.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel 7: Labels histogram =====
        const labelSeries = chart7.addHistogramSeries({{
            priceFormat: {{ type: 'price', precision: 1, minMove: 0.1 }},
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        labelSeries.setData(labelData);

        // ===== Checkbox toggles =====
        document.querySelectorAll('input[data-panel="zscore"]').forEach(cb => {{
            cb.addEventListener('change', (e) => {{
                const name = e.target.dataset.series;
                const entry = zscoreSeries.find(s => s.name === name);
                if (entry) {{
                    entry.series.applyOptions({{ visible: e.target.checked }});
                }}
            }});
        }});

        document.querySelectorAll('input[data-panel="time"]').forEach(cb => {{
            cb.addEventListener('change', (e) => {{
                const name = e.target.dataset.series;
                const entry = timeSeries.find(s => s.name === name);
                if (entry) {{
                    entry.series.applyOptions({{ visible: e.target.checked }});
                }}
            }});
        }});

        document.querySelectorAll('input[data-panel="struct"]').forEach(cb => {{
            cb.addEventListener('change', (e) => {{
                const name = e.target.dataset.series;
                const entry = structSeries.find(s => s.name === name);
                if (entry) {{
                    entry.series.applyOptions({{ visible: e.target.checked }});
                }}
            }});
        }});

        document.querySelectorAll('input[data-panel="vp"]').forEach(cb => {{
            cb.addEventListener('change', (e) => {{
                const name = e.target.dataset.series;
                const entry = vpSeries.find(s => s.name === name);
                if (entry) {{
                    entry.series.applyOptions({{ visible: e.target.checked }});
                }}
            }});
        }});

        // ===== Range synchronization =====
        let isSyncing = false;

        function syncRange(sourceChart) {{
            if (isSyncing) return;
            isSyncing = true;

            const range = sourceChart.timeScale().getVisibleLogicalRange();
            if (range) {{
                charts.forEach(c => {{
                    if (c !== sourceChart) {{
                        c.timeScale().setVisibleLogicalRange(range);
                    }}
                }});
            }}

            isSyncing = false;
        }}

        charts.forEach(chart => {{
            chart.timeScale().subscribeVisibleLogicalRangeChange(() => {{
                syncRange(chart);
            }});
        }});

        // ===== Resize =====
        const resizeObserver = new ResizeObserver(() => {{
            const w = document.body.clientWidth;
            charts.forEach((chart, i) => {{
                chart.applyOptions({{ width: w }});
            }});
        }});
        resizeObserver.observe(document.body);

        // ===== Initial fit =====
        chart1.timeScale().fitContent();

    }})();
    </script>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Interactive input features visualization')
    parser.add_argument('--file', type=str, default=None,
                        help='CSV data file (default: config.INPUT_FILES[0])')
    parser.add_argument('--output', type=str, default='visualizations/interactive.html',
                        help='Output HTML file path')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'test', 'all'],
                        help='Dataset split to visualize (default: all data)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser after generation')
    args = parser.parse_args()

    file_path = args.file or config.INPUT_FILES[0]

    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    data = prepare_data(file_path, split=args.split)
    json_data = build_json_data(data)
    generate_html(json_data, args.output)

    if not args.no_browser:
        abs_path = os.path.abspath(args.output)
        print(f"Opening in browser: file://{abs_path}")
        webbrowser.open(f'file://{abs_path}')


if __name__ == '__main__':
    main()
