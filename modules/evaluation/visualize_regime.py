"""
Interactive regime filter visualization - TradingView-style with Lightweight Charts v4.

3 synchronized panels:
1. OHLC candlestick colorized by trend score (red → orange → green gradient)
2. Trend oscillator (colored line + area fill + zero line)
3. VolTrend oscillator (grey line + area fill + zero line)
"""

import argparse
import json
import os
import sys
import webbrowser

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from modules.data.loader import get_raw_data
from modules.data.regime import compute_regime_filter


def prepare_data(file_path, length):
    """Load raw data and compute regime filter."""
    print(f"Loading data from {file_path}...")
    df = get_raw_data(file_path)
    n = len(df)
    print(f"  {n} candles loaded")

    print(f"Computing regime filter (length={length})...")
    trend, voltrend = compute_regime_filter(df, length=length)

    return df, trend, voltrend


def trend_to_color(score):
    """
    Map trend score to color gradient (like BigBeluga TV script).
    score < 0: red (#ef5350) → orange (#FF9800)  (lerp on abs ratio)
    score > 0: orange (#FF9800) → green (#26a69a)  (lerp on ratio)
    score == 0: orange
    """
    if np.isnan(score):
        return '#555555'

    # Clamp to [-10, 10]
    s = max(-10.0, min(10.0, score))
    if s <= 0:
        # ratio 0 (score=-10, full red) to 1 (score=0, full orange)
        t = (s + 10.0) / 10.0
        r = int(0xef + (0xff - 0xef) * t)
        g = int(0x53 + (0x98 - 0x53) * t)
        b = int(0x50 + (0x00 - 0x50) * t)
    else:
        # ratio 0 (score=0, full orange) to 1 (score=10, full green)
        t = s / 10.0
        r = int(0xff + (0x26 - 0xff) * t)
        g = int(0x98 + (0xa6 - 0x98) * t)
        b = int(0x00 + (0x9a - 0x00) * t)

    return f'#{r:02x}{g:02x}{b:02x}'



def build_json_data(df, trend, voltrend):
    """Serialize data to JSON for embedding in HTML."""
    n = len(df)
    timestamps = df['Open time'].values

    # Convert timestamps to unix seconds
    ts_unix = []
    for t in timestamps:
        ts = np.datetime64(t, 's').astype(int)
        ts_unix.append(int(ts))

    # Panel 1: OHLC candles colorized by trend score
    candles = []
    for i in range(n):
        color = trend_to_color(trend[i])
        candles.append({
            'time': ts_unix[i],
            'open': round(float(df['Open'].values[i]), 2),
            'high': round(float(df['High'].values[i]), 2),
            'low': round(float(df['Low'].values[i]), 2),
            'close': round(float(df['Close'].values[i]), 2),
            'color': color,
            'borderColor': color,
            'wickColor': color,
        })

    # Panel 2: Trend oscillator line data (colored per point)
    trend_data = []
    for i in range(n):
        val = float(trend[i])
        if np.isfinite(val):
            trend_data.append({
                'time': ts_unix[i],
                'value': round(val, 4),
                'color': trend_to_color(val),
            })

    # Panel 2: Trend area (positive and negative separately for area fill)
    trend_pos = []
    trend_neg = []
    for i in range(n):
        val = float(trend[i])
        if np.isfinite(val):
            if val >= 0:
                trend_pos.append({'time': ts_unix[i], 'value': round(val, 4)})
                trend_neg.append({'time': ts_unix[i], 'value': 0})
            else:
                trend_pos.append({'time': ts_unix[i], 'value': 0})
                trend_neg.append({'time': ts_unix[i], 'value': round(val, 4)})

    # Panel 3: VolTrend oscillator
    voltrend_data = []
    for i in range(n):
        val = float(voltrend[i])
        if np.isfinite(val):
            voltrend_data.append({
                'time': ts_unix[i],
                'value': round(val, 4),
            })

    voltrend_pos = []
    voltrend_neg = []
    for i in range(n):
        val = float(voltrend[i])
        if np.isfinite(val):
            if val >= 0:
                voltrend_pos.append({'time': ts_unix[i], 'value': round(val, 4)})
                voltrend_neg.append({'time': ts_unix[i], 'value': 0})
            else:
                voltrend_pos.append({'time': ts_unix[i], 'value': 0})
                voltrend_neg.append({'time': ts_unix[i], 'value': round(val, 4)})

    return {
        'candles': candles,
        'trend': trend_data,
        'trend_pos': trend_pos,
        'trend_neg': trend_neg,
        'voltrend': voltrend_data,
        'voltrend_pos': voltrend_pos,
        'voltrend_neg': voltrend_neg,
    }


def generate_html(json_data, output_path, length):
    """Generate the interactive HTML file with 3 synchronized panels."""
    candle_json = json.dumps(json_data['candles'])
    trend_json = json.dumps(json_data['trend'])
    trend_pos_json = json.dumps(json_data['trend_pos'])
    trend_neg_json = json.dumps(json_data['trend_neg'])
    voltrend_json = json.dumps(json_data['voltrend'])
    voltrend_pos_json = json.dumps(json_data['voltrend_pos'])
    voltrend_neg_json = json.dumps(json_data['voltrend_neg'])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regime Filter — Length({length})</title>
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
        .header .params {{
            font-size: 12px;
            color: #aaa;
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
        }}
        .panel-title {{
            font-weight: bold;
            color: #82b1ff;
        }}
        .chart-container {{
            width: 100%;
        }}
        .legend {{
            display: flex;
            gap: 16px;
            font-size: 11px;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        .gradient-bar {{
            width: 80px;
            height: 10px;
            border-radius: 3px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Regime Filter <span style="font-size:12px;color:#aaa;font-weight:normal;">[BigBeluga style]</span></h1>
        <div class="legend">
            <span class="legend-item">
                <span class="gradient-bar" style="background:linear-gradient(to right, #ef5350, #FF9800, #26a69a);"></span>
                Trend color
            </span>
        </div>
        <div class="params">Length: {length}</div>
    </div>

    <!-- Panel 1: Candlestick colored by trend -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">OHLC — Colored by Trend Score</span>
        </div>
        <div id="chart1" class="chart-container"></div>
    </div>

    <!-- Panel 2: Trend oscillator -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Trend Oscillator</span>
            <span style="font-size:10px;color:#aaa;">[-10, +10]</span>
        </div>
        <div id="chart2" class="chart-container"></div>
    </div>

    <!-- Panel 3: VolTrend oscillator -->
    <div class="panel">
        <div class="panel-header">
            <span class="panel-title">Volume Trend Oscillator</span>
            <span style="font-size:10px;color:#aaa;">[-10, +10]</span>
        </div>
        <div id="chart3" class="chart-container"></div>
    </div>

    <script>
    (function() {{
        // ===== Data =====
        const candleData = {candle_json};
        const trendData = {trend_json};
        const trendPosData = {trend_pos_json};
        const trendNegData = {trend_neg_json};
        const voltrendData = {voltrend_json};
        const voltrendPosData = {voltrend_pos_json};
        const voltrendNegData = {voltrend_neg_json};

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
            return LightweightCharts.createChart(container, {{
                ...darkTheme,
                width: container.clientWidth,
                height: height,
            }});
        }}

        // ===== Create 3 charts =====
        const chart1 = createChart('chart1', 500);
        const chart2 = createChart('chart2', 200);
        const chart3 = createChart('chart3', 200);
        const charts = [chart1, chart2, chart3];

        // ===== Panel 1: Candlestick colored by trend =====
        const candleSeries = chart1.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        candleSeries.setData(candleData);

        // ===== Panel 2: Trend oscillator =====
        // Area fill positive (green)
        const trendAreaPos = chart2.addAreaSeries({{
            topColor: 'rgba(38, 166, 154, 0.4)',
            bottomColor: 'rgba(38, 166, 154, 0.0)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        trendAreaPos.setData(trendPosData);

        // Area fill negative (red)
        const trendAreaNeg = chart2.addAreaSeries({{
            topColor: 'rgba(239, 83, 80, 0.0)',
            bottomColor: 'rgba(239, 83, 80, 0.4)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        trendAreaNeg.setData(trendNegData);

        // Trend line (colored per point)
        const trendLine = chart2.addLineSeries({{
            color: '#FF9800',
            lineWidth: 2,
            priceScaleId: 'right',
            lastValueVisible: true,
            priceLineVisible: false,
        }});
        trendLine.setData(trendData);

        // Zero line
        const trendZero = chart2.addLineSeries({{
            color: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        if (candleData.length > 0) {{
            trendZero.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

        // ===== Panel 3: VolTrend oscillator =====
        // Area fill positive
        const voltrendAreaPos = chart3.addAreaSeries({{
            topColor: 'rgba(150, 150, 150, 0.3)',
            bottomColor: 'rgba(150, 150, 150, 0.0)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        voltrendAreaPos.setData(voltrendPosData);

        // Area fill negative
        const voltrendAreaNeg = chart3.addAreaSeries({{
            topColor: 'rgba(150, 150, 150, 0.0)',
            bottomColor: 'rgba(150, 150, 150, 0.3)',
            lineColor: 'transparent',
            lineWidth: 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        voltrendAreaNeg.setData(voltrendNegData);

        // VolTrend line
        const voltrendLine = chart3.addLineSeries({{
            color: '#9e9e9e',
            lineWidth: 2,
            priceScaleId: 'right',
            lastValueVisible: true,
            priceLineVisible: false,
        }});
        voltrendLine.setData(voltrendData);

        // Zero line
        const voltrendZero = chart3.addLineSeries({{
            color: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
        }});
        if (candleData.length > 0) {{
            voltrendZero.setData([
                {{ time: candleData[0].time, value: 0 }},
                {{ time: candleData[candleData.length - 1].time, value: 0 }},
            ]);
        }}

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
            charts.forEach(chart => {{
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
    parser = argparse.ArgumentParser(description='Interactive regime filter visualization')
    parser.add_argument('--file', type=str, default=None,
                        help='CSV data file (default: config.INPUT_FILES[0])')
    parser.add_argument('--length', type=int, default=config.REGIME_LENGTH,
                        help=f'Regime filter length (HMA period + lookback) (default: {config.REGIME_LENGTH})')
    parser.add_argument('--output', type=str, default='visualizations/regime_filter.html',
                        help='Output HTML file path')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser after generation')
    args = parser.parse_args()

    file_path = args.file or config.INPUT_FILES[0]

    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    df, trend, voltrend = prepare_data(file_path, args.length)
    json_data = build_json_data(df, trend, voltrend)
    generate_html(json_data, args.output, args.length)

    if not args.no_browser:
        abs_path = os.path.abspath(args.output)
        print(f"Opening in browser: file://{abs_path}")
        webbrowser.open(f'file://{abs_path}')


if __name__ == '__main__':
    main()
