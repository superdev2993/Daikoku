"""
OHLCV data visualizer for CSV triage.

Generates one standalone HTML per CSV with:
- Primary TF candlestick + volume (with WINDOW_SIZE ruler overlay)
- Secondary TF candlestick (aggregated via MULTI_TF_DIVISOR), synced scroll
Only displays candles the model actually sees (after WINDOW_SIZE warmup truncation).

Usage:
    python -m modules.tools.visualize_data data/Set/BTC_15min.csv
    python -m modules.tools.visualize_data data/Set/BTC_15min.csv data/Set/ETH_15min_bin.csv
    python -m modules.tools.visualize_data data/Set/*.csv
    python -m modules.tools.visualize_data data/Set/*.csv --output-dir evaluation/
"""

import argparse
import json
import os
import sys

import config
from modules.data.loader import get_raw_data
from modules.data.aggregation import aggregate_candles


def build_candlestick_json(df):
    """Convert DataFrame to Lightweight Charts candlestick format."""
    records = []
    for _, row in df.iterrows():
        ts = row["Open time"]
        if isinstance(ts, str):
            import pandas as pd
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


def build_volume_json(df, candles):
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


def generate_html(df, df_secondary, output_path, filename, window_size, total_raw, divisor, align):
    """Generate standalone HTML with primary + secondary TF charts."""
    candles = build_candlestick_json(df)
    volume = build_volume_json(df, candles)
    candles_sec = build_candlestick_json(df_secondary)
    volume_sec = build_volume_json(df_secondary, candles_sec)

    date_start = df["Open time"].iloc[0]
    date_end = df["Open time"].iloc[-1]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{filename} — Data View</title>
    <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; }}
        .header {{
            padding: 12px 20px;
            background: #16213e; border-bottom: 1px solid #333;
        }}
        .header h1 {{ font-size: 16px; color: #00d4aa; margin-bottom: 4px; }}
        .header .stats {{
            font-size: 12px; color: #888;
            display: flex; gap: 24px; flex-wrap: wrap;
        }}
        .header .stats .val {{ color: #e0e0e0; }}
        .section-label {{
            padding: 4px 20px;
            font-size: 13px; font-weight: bold;
            background: #0f3460; color: #e94560;
            border-bottom: 1px solid #333;
        }}
        .chart-container {{ position: relative; }}
        #chart-candles {{ width: 100%; height: 500px; }}
        #chart-volume {{ width: 100%; height: 120px; }}
        #chart-secondary {{ width: 100%; height: 350px; }}
        #chart-secondary-vol {{ width: 100%; height: 100px; }}

        /* WINDOW_SIZE ruler */
        #ruler {{
            position: absolute; top: 8px; left: 60px;
            height: 18px; background: rgba(0, 212, 170, 0.15);
            border: 1px solid #00d4aa; border-radius: 2px;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; color: #00d4aa; pointer-events: none;
            z-index: 10; transition: width 0.1s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{filename}</h1>
        <div class="stats">
            <span>Candles: <span class="val">{len(df)}</span> (raw: {total_raw}, warmup: -{window_size})</span>
            <span>Range: <span class="val">{date_start}</span> → <span class="val">{date_end}</span></span>
            <span>WINDOW_SIZE: <span class="val">{window_size}</span></span>
            <span>Secondary: <span class="val">x{divisor} ({align})</span> → {len(df_secondary)} candles</span>
        </div>
    </div>

    <div class="section-label">Primary TF ({len(df)} candles)</div>
    <div class="chart-container">
        <div id="ruler">{window_size} candles</div>
        <div id="chart-candles"></div>
    </div>
    <div id="chart-volume"></div>

    <div class="section-label">Secondary TF x{divisor} ({len(df_secondary)} candles)</div>
    <div id="chart-secondary"></div>
    <div id="chart-secondary-vol"></div>

    <script>
    const WINDOW_SIZE = {window_size};

    const chartOpts = {{
        layout: {{ background: {{ color: '#1a1a2e' }}, textColor: '#d1d4dc' }},
        grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
        rightPriceScale: {{ borderColor: '#333' }},
        timeScale: {{ borderColor: '#333', timeVisible: true }},
    }};

    // ===== PRIMARY CANDLES =====
    const cC = document.getElementById('chart-candles');
    const chartC = LightweightCharts.createChart(cC, {{ ...chartOpts, width: cC.clientWidth, height: 500 }});
    const sC = chartC.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350', borderVisible: false,
    }});
    sC.setData({json.dumps(candles)});

    // ===== PRIMARY VOLUME =====
    const cV = document.getElementById('chart-volume');
    const chartV = LightweightCharts.createChart(cV, {{ ...chartOpts, width: cV.clientWidth, height: 120 }});
    const sV = chartV.addHistogramSeries({{ priceFormat: {{ type: 'volume' }} }});
    sV.setData({json.dumps(volume)});

    // ===== SECONDARY CANDLES =====
    const cS = document.getElementById('chart-secondary');
    const chartS = LightweightCharts.createChart(cS, {{ ...chartOpts, width: cS.clientWidth, height: 350 }});
    const sS = chartS.addCandlestickSeries({{
        upColor: '#26a69a', downColor: '#ef5350',
        wickUpColor: '#26a69a', wickDownColor: '#ef5350', borderVisible: false,
    }});
    sS.setData({json.dumps(candles_sec)});

    // ===== SECONDARY VOLUME =====
    const cSV = document.getElementById('chart-secondary-vol');
    const chartSV = LightweightCharts.createChart(cSV, {{ ...chartOpts, width: cSV.clientWidth, height: 100 }});
    const sSV = chartSV.addHistogramSeries({{ priceFormat: {{ type: 'volume' }} }});
    sSV.setData({json.dumps(volume_sec)});

    // ===== RULER: width = WINDOW_SIZE candles in pixels =====
    const primaryData = {json.dumps(candles)};
    const rulerEl = document.getElementById('ruler');

    function updateRuler() {{
        const ts = chartC.timeScale();
        const range = ts.getVisibleLogicalRange();
        if (!range) return;
        const totalBars = range.to - range.from;
        const chartWidth = cC.clientWidth - 60; // approx price scale width
        if (totalBars <= 0) return;
        const pxPerBar = chartWidth / totalBars;
        const rulerWidth = Math.max(20, Math.round(pxPerBar * WINDOW_SIZE));
        rulerEl.style.width = rulerWidth + 'px';
    }}
    chartC.timeScale().subscribeVisibleLogicalRangeChange(updateRuler);
    setTimeout(updateRuler, 200);

    // ===== SYNC: all 4 charts by time range =====
    const allCharts = [chartC, chartV, chartS, chartSV];
    let syncing = false;

    // Primary pair sync (candles <-> volume) by logical range
    function syncFromPrimary(range) {{
        if (syncing) return;
        syncing = true;
        chartV.timeScale().setVisibleLogicalRange(range);
        // Sync secondary by visible time range
        const vtr = chartC.timeScale().getVisibleRange();
        if (vtr) {{
            chartS.timeScale().setVisibleRange(vtr);
            chartSV.timeScale().setVisibleRange(vtr);
        }}
        syncing = false;
    }}
    function syncFromVolume(range) {{
        if (syncing) return;
        syncing = true;
        chartC.timeScale().setVisibleLogicalRange(range);
        syncing = false;
    }}
    function syncFromSecondary(range) {{
        if (syncing) return;
        syncing = true;
        chartSV.timeScale().setVisibleLogicalRange(range);
        // Sync primary by visible time range
        const vtr = chartS.timeScale().getVisibleRange();
        if (vtr) {{
            chartC.timeScale().setVisibleRange(vtr);
            chartV.timeScale().setVisibleRange(vtr);
        }}
        syncing = false;
    }}
    function syncFromSecVol(range) {{
        if (syncing) return;
        syncing = true;
        chartS.timeScale().setVisibleLogicalRange(range);
        syncing = false;
    }}

    chartC.timeScale().subscribeVisibleLogicalRangeChange(syncFromPrimary);
    chartV.timeScale().subscribeVisibleLogicalRangeChange(syncFromVolume);
    chartS.timeScale().subscribeVisibleLogicalRangeChange(syncFromSecondary);
    chartSV.timeScale().subscribeVisibleLogicalRangeChange(syncFromSecVol);

    // ===== RESIZE =====
    const ro = new ResizeObserver(() => {{
        chartC.applyOptions({{ width: cC.clientWidth }});
        chartV.applyOptions({{ width: cV.clientWidth }});
        chartS.applyOptions({{ width: cS.clientWidth }});
        chartSV.applyOptions({{ width: cSV.clientWidth }});
    }});
    ro.observe(cC); ro.observe(cV); ro.observe(cS); ro.observe(cSV);

    // ===== FIT =====
    allCharts.forEach(c => c.timeScale().fitContent());
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Visualize OHLCV data (model-visible candles only)")
    parser.add_argument("files", nargs="+", help="CSV file(s) to visualize")
    parser.add_argument("--output-dir", type=str, default="evaluation/",
                        help="Output directory for HTML files (default: evaluation/)")
    args = parser.parse_args()

    window_size = config.WINDOW_SIZE
    divisor = config.MULTI_TF_DIVISOR
    align = config.MULTI_TF_ALIGN

    for path in args.files:
        stem = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(args.output_dir, f"data_view_{stem}.html")

        print(f"\n[{stem}] Loading {path}...")
        try:
            df = get_raw_data(path)
        except Exception as e:
            print(f"  ERROR: {e} — skipping")
            continue

        total_raw = len(df)

        if total_raw <= window_size:
            print(f"  ERROR: only {total_raw} rows, need > {window_size} (WINDOW_SIZE) — skipping")
            continue

        # Aggregate secondary TF BEFORE truncation (same as pipeline)
        df_sec, _, _ = aggregate_candles(df, divisor, align)

        # Truncate warmup candles (model never sees these)
        df = df.iloc[window_size:].reset_index(drop=True)

        # Truncate secondary proportionally
        sec_skip = window_size // divisor
        df_sec = df_sec.iloc[sec_skip:].reset_index(drop=True)

        print(f"  Primary : {total_raw} → {len(df)} (truncated {window_size} warmup)")
        print(f"  Secondary: x{divisor} ({align}) → {len(df_sec)} candles (truncated {sec_skip})")
        print(f"  Range: {df['Open time'].iloc[0]} → {df['Open time'].iloc[-1]}")

        generate_html(df, df_sec, output_path, stem, window_size, total_raw, divisor, align)
        print(f"  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
