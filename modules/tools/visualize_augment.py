"""
Side-by-side comparison of original vs augmented OHLCV data.

Two synchronized Lightweight Charts v4 candlestick charts:
- Left: Original data
- Right: Augmented variant

Usage:
    python -m modules.tools.visualize_augment
    python -m modules.tools.visualize_augment --file data/btc_futures_all_1h_aggregated.csv
    python -m modules.tools.visualize_augment --file data/ETH_1h_bin.csv --pct 10
"""

import argparse
import json
import os
import sys
import webbrowser

import pandas as pd


import config
from modules.tools.augment_data import augment_ohlcv


def build_candlestick_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to Lightweight Charts candlestick format."""
    records = []
    for _, row in df.iterrows():
        ts = row["Open time"] if "Open time" in df.columns else row.index
        # Parse timestamp to epoch seconds
        if isinstance(ts, str):
            ts = int(pd.Timestamp(ts).timestamp())
        elif hasattr(ts, 'timestamp'):
            ts = int(ts.timestamp())
        else:
            ts = int(ts)

        records.append({
            "time": ts,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
        })
    return records


def build_volume_json(df: pd.DataFrame, candles: list) -> list:
    """Build volume histogram data with color based on candle direction."""
    volumes = df["Volume"].values
    records = []
    for i, c in enumerate(candles):
        if i < len(volumes):
            color = "#26a69a80" if c["close"] >= c["open"] else "#ef535080"
            records.append({
                "time": c["time"],
                "value": round(float(volumes[i]), 2),
                "color": color,
            })
    return records


def generate_html(original_df: pd.DataFrame, augmented_df: pd.DataFrame, output_path: str, pct: float):
    """Generate side-by-side comparison HTML."""
    orig_candles = build_candlestick_json(original_df)
    aug_candles = build_candlestick_json(augmented_df)
    orig_volume = build_volume_json(original_df, orig_candles)
    aug_volume = build_volume_json(augmented_df, aug_candles)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Augmentation Comparison (PCT={pct})</title>
    <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; }}
        .header {{
            text-align: center; padding: 12px;
            background: #16213e; border-bottom: 1px solid #333;
        }}
        .header h1 {{ font-size: 16px; color: #00d4aa; }}
        .header span {{ font-size: 12px; color: #888; }}
        .container {{ display: flex; width: 100%; height: calc(100vh - 50px); }}
        .panel {{
            flex: 1; display: flex; flex-direction: column;
            border-right: 1px solid #333;
        }}
        .panel:last-child {{ border-right: none; }}
        .panel-title {{
            text-align: center; padding: 6px;
            font-size: 13px; font-weight: bold;
            background: #0f3460; color: #e94560;
        }}
        .chart-wrapper {{ flex: 1; position: relative; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Augmentation Comparison</h1>
        <span>AUGMENT_PCT={pct} | Open/Close: OU(max({pct}% body, 0.1% price)) | Wicks: &plusmn;{pct}% | Volume: &plusmn;{pct}%</span>
    </div>
    <div class="container">
        <div class="panel">
            <div class="panel-title">ORIGINAL ({len(original_df)} candles)</div>
            <div class="chart-wrapper" id="chart-orig"></div>
        </div>
        <div class="panel">
            <div class="panel-title">AUGMENTED (variant)</div>
            <div class="chart-wrapper" id="chart-aug"></div>
        </div>
    </div>

    <script>
    const chartOptions = {{
        layout: {{ background: {{ color: '#1a1a2e' }}, textColor: '#d1d4dc' }},
        grid: {{
            vertLines: {{ color: '#1f2937' }},
            horzLines: {{ color: '#1f2937' }},
        }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: '#333' }},
        timeScale: {{ borderColor: '#333', timeVisible: true }},
    }};

    // Create charts
    const origContainer = document.getElementById('chart-orig');
    const augContainer = document.getElementById('chart-aug');
    const chartOrig = LightweightCharts.createChart(origContainer, chartOptions);
    const chartAug = LightweightCharts.createChart(augContainer, chartOptions);

    // Candlestick series
    const origCandles = chartOrig.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        borderVisible: false,
    }});
    const augCandles = chartAug.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        borderVisible: false,
    }});

    // Volume series
    const origVol = chartOrig.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: 'vol',
    }});
    chartOrig.priceScale('vol').applyOptions({{
        scaleMargins: {{ top: 0.8, bottom: 0 }},
    }});
    const augVol = chartAug.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: 'vol',
    }});
    chartAug.priceScale('vol').applyOptions({{
        scaleMargins: {{ top: 0.8, bottom: 0 }},
    }});

    // Set data
    origCandles.setData({json.dumps(orig_candles)});
    augCandles.setData({json.dumps(aug_candles)});
    origVol.setData({json.dumps(orig_volume)});
    augVol.setData({json.dumps(aug_volume)});

    // Synchronize range
    let isSyncing = false;
    chartOrig.timeScale().subscribeVisibleLogicalRangeChange(range => {{
        if (isSyncing) return;
        isSyncing = true;
        chartAug.timeScale().setVisibleLogicalRange(range);
        isSyncing = false;
    }});
    chartAug.timeScale().subscribeVisibleLogicalRangeChange(range => {{
        if (isSyncing) return;
        isSyncing = true;
        chartOrig.timeScale().setVisibleLogicalRange(range);
        isSyncing = false;
    }});

    // Synchronize crosshair
    chartOrig.subscribeCrosshairMove(param => {{
        if (isSyncing) return;
        isSyncing = true;
        if (param.time) {{
            chartAug.setCrosshairPosition(NaN, NaN, augCandles);
        }}
        isSyncing = false;
    }});

    // Resize
    const resizeObserver = new ResizeObserver(() => {{
        chartOrig.applyOptions({{ width: origContainer.clientWidth, height: origContainer.clientHeight }});
        chartAug.applyOptions({{ width: augContainer.clientWidth, height: augContainer.clientHeight }});
    }});
    resizeObserver.observe(origContainer);
    resizeObserver.observe(augContainer);

    // Fit
    chartOrig.timeScale().fitContent();
    chartAug.timeScale().fitContent();
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize augmented vs original data")
    parser.add_argument("--file", type=str, default=config.INPUT_FILES[0],
                        help="Input CSV file")
    parser.add_argument("--pct", type=float, default=None,
                        help=f"Perturbation percentage (default: {config.AUGMENT_PCT})")
    parser.add_argument("--seed", type=int, default=config.SEED + 1000,
                        help="Random seed for augmentation")
    parser.add_argument("--output", type=str, default="visualizations/augment_compare.html",
                        help="Output HTML path")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    pct = args.pct if args.pct is not None else config.AUGMENT_PCT

    print(f"Loading {args.file}...")
    df = pd.read_csv(args.file)
    print(f"  {len(df)} candles loaded")

    print(f"Generating augmented variant (PCT={pct})...")
    aug_df = augment_ohlcv(df, pct, args.seed)

    generate_html(df, aug_df, args.output, pct)

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
