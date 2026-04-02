"""
Prediction Visualizer - Interactive HTML report generation

Candlestick + confidence oscillator: TradingView Lightweight Charts (JS)
Statistical plots (confusion, confidence, accuracy, class perf): Plotly
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.data.labeling import (
    TARGET_CLASS_NAMES, REJECT_CLASS, TRADE_DIRECTION,
    compute_atr, parse_close_horizon,
)
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _plotlyjs_version():
    """Return the Plotly.js version matching the installed plotly Python package."""
    from plotly.offline import get_plotlyjs_version
    return get_plotlyjs_version()
CORRECT_COLOR = '#2E7D32'
INCORRECT_COLOR = '#C62828'
UNCERTAIN_COLOR = '#EF6C00'


class PredictionVisualizer:
    """
    Generates interactive visualizations for model evaluation results.

    Candlestick chart uses TradingView Lightweight Charts for smooth navigation.
    Statistical plots use Plotly for rich interactivity.
    """

    def __init__(self, df_raw, predictions, labels, probabilities,
                 timestamps, metrics, config_params, num_classes=3):
        """
        Args:
            df_raw: Raw DataFrame with OHLC columns
            predictions: np.ndarray of predicted labels (n_samples,)
            labels: np.ndarray of true labels (n_samples,) or None
            probabilities: np.ndarray of probabilities (n_samples, num_classes)
            timestamps: np.ndarray of timestamps
            metrics: dict of computed metrics or None
            config_params: dict with 'raw_start_idx', 'EVAL_ACCURACY_WINDOW'
            num_classes: Number of classes (3)
        """
        self.num_classes = num_classes
        self.prediction_target = config_params['PREDICTION_TARGET']
        class_name_map = TARGET_CLASS_NAMES.get(self.prediction_target, TARGET_CLASS_NAMES['triple'])
        self.class_names = [class_name_map[i] for i in range(num_classes)]
        if num_classes == 2:
            self.class_colors = {0: '#EF553B', 1: '#00CC96'}
        else:
            self.class_colors = {0: '#EF553B', 1: '#FFA726', 2: '#00CC96'}
        self.reject_class = REJECT_CLASS.get(self.prediction_target)
        self.trade_dir = TRADE_DIRECTION.get(self.prediction_target, {})

        self.predictions = predictions
        self.labels = labels
        self.probabilities = probabilities
        self.metrics = metrics
        self.inference_only = labels is None

        self.timestamps = pd.to_datetime(timestamps)
        # Margin confidence: top1 - top2 softmax proba (more discriminant than max)
        sorted_probs = np.sort(probabilities, axis=1)
        self.confidences = sorted_probs[:, -1] - sorted_probs[:, -2]

        # Margin percentile rank for each prediction (0-100, 100=highest margin)
        n_preds = len(self.confidences)
        order = np.argsort(self.confidences)
        pct_ranks = np.empty(n_preds, dtype=float)
        pct_ranks[order] = np.arange(1, n_preds + 1)
        self.margin_pct_rank = (pct_ranks / n_preds) * 100

        # High confidence threshold: top 10% (90th percentile) of margin scores
        self.high_conf_threshold = float(np.percentile(self.confidences, 90))
        logger.info(f"High confidence threshold (P90): {self.high_conf_threshold:.4f}")

        # Extract OHLC from df_raw aligned with predictions
        raw_start = config_params['raw_start_idx']
        n = len(predictions)
        ohlc_slice = df_raw.iloc[raw_start:raw_start + n]
        self.open = ohlc_slice['Open'].values
        self.high = ohlc_slice['High'].values
        self.low = ohlc_slice['Low'].values
        self.close = ohlc_slice['Close'].values

        self.accuracy_window = config_params['EVAL_ACCURACY_WINDOW']

        # Store for PNL simulation
        self.df_raw = df_raw
        self.config_params = config_params

        # Compute ATR once (used by trade zones and PNL simulation)
        is_close_mode = parse_close_horizon(self.prediction_target) is not None
        has_barrier_config = 'ATR_MULTIPLIER_TP' in config_params and self.trade_dir
        if has_barrier_config:
            self._atr_full = compute_atr(df_raw, period=config_params['ATR_PERIOD'])
            if not is_close_mode:
                self.trades = self._compute_trades(df_raw, config_params)
                logger.info(f"Trade zones computed: {len(self.trades)} directional trades")
            else:
                self.trades = []
        else:
            self._atr_full = None
            self.trades = []

        # Pre-compute masks for normal mode
        if not self.inference_only:
            self.correct_mask = (labels == predictions)
            self.incorrect_mask = (labels != predictions)

        logger.info(f"PredictionVisualizer initialized: {n} samples, "
                    f"inference_only={self.inference_only}")

    # ------------------------------------------------------------------
    # Trade zones computation
    # ------------------------------------------------------------------

    def _compute_trades(self, df_raw, config_params):
        """
        Compute trade zones for each directional prediction.

        For each non-uncertain prediction, scan forward candle by candle
        to find TP/SL hit or timeout at max_horizon.

        Returns:
            List of trade dicts with compact keys for JSON serialization.
        """
        raw_start = config_params['raw_start_idx']
        n = len(self.predictions)
        atr_mult_tp = config_params['ATR_MULTIPLIER_TP']
        atr_mult_sl = config_params['ATR_MULTIPLIER_SL']
        max_horizon = config_params['MAX_HORIZON']

        atr_full = self._atr_full

        # Extract arrays with enough room for forward scan
        high_arr = df_raw['High'].values
        low_arr = df_raw['Low'].values
        open_arr = df_raw['Open'].values

        # Timestamps from df_raw (always 'Open time' — same source as self.timestamps)
        ts_raw = pd.to_datetime(df_raw['Open time'])

        raw_len = len(df_raw)
        trades = []

        # Candle interval in seconds (for projecting unresolved trades)
        candle_sec = int(ts_raw.iloc[1].timestamp() - ts_raw.iloc[0].timestamp())

        trade_dir = self.trade_dir

        for i in range(n):
            pred = int(self.predictions[i])
            direction = trade_dir.get(pred)
            if direction is None:
                continue  # Skip non-tradable classes (reject)

            raw_idx = raw_start + i

            atr_val = atr_full[raw_idx]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # Realistic entry: Open of next candle (signal known at Close[N], act at Open[N+1])
            exec_idx = raw_idx + 1
            if exec_idx >= raw_len:
                continue
            entry_price = float(open_arr[exec_idx])
            entry_ts = int(ts_raw.iloc[exec_idx].timestamp())
            conf = float(self.confidences[i])
            is_long = (direction == "long")

            if is_long:
                tp_price = entry_price + atr_mult_tp * atr_val
                sl_price = entry_price - atr_mult_sl * atr_val
            else:  # short
                tp_price = entry_price - atr_mult_tp * atr_val
                sl_price = entry_price + atr_mult_sl * atr_val

            # Scan forward from execution candle
            end_raw = min(exec_idx + max_horizon, raw_len)
            outcome = 'timeout'
            exit_ts = entry_ts + max_horizon * candle_sec  # default: full projection

            for j in range(exec_idx, end_raw):
                h = high_arr[j]
                l = low_arr[j]

                if is_long:
                    tp_hit = h >= tp_price
                    sl_hit = l <= sl_price
                else:
                    tp_hit = l <= tp_price
                    sl_hit = h >= sl_price

                if tp_hit and sl_hit:
                    outcome = 'both'
                    exit_ts = int(ts_raw.iloc[j].timestamp())
                    break
                elif tp_hit:
                    outcome = 'tp'
                    exit_ts = int(ts_raw.iloc[j].timestamp())
                    break
                elif sl_hit:
                    outcome = 'sl'
                    exit_ts = int(ts_raw.iloc[j].timestamp())
                    break

            trades.append({
                'et': entry_ts,
                'xt': exit_ts,
                'ep': round(entry_price, 2),
                'tp': round(float(tp_price), 2),
                'sl': round(float(sl_price), 2),
                'p': pred,
                'o': outcome,
                'c': round(conf, 3),
                'r': round(float(self.margin_pct_rank[i]), 1),
            })

        return trades

    # ------------------------------------------------------------------
    # Lightweight Charts: Candlestick + Confidence Oscillator
    # ------------------------------------------------------------------

    def _build_lightweight_charts_section(self):
        """
        Build HTML/JS section for TradingView Lightweight Charts.

        Returns candlestick + markers + histogram pane as raw HTML string.
        """
        n = len(self.predictions)

        # Build candlestick data as JSON
        candle_data = []
        for i in range(n):
            ts = int(self.timestamps[i].timestamp())
            candle_data.append({
                'time': ts,
                'open': round(float(self.open[i]), 2),
                'high': round(float(self.high[i]), 2),
                'low': round(float(self.low[i]), 2),
                'close': round(float(self.close[i]), 2),
            })

        # Dummy candle for last prediction's execution candle (N+1)
        # Flat candle at previous close, no wicks
        last_close = round(float(self.close[n - 1]), 2)
        candle_interval = int(self.timestamps[1].timestamp() - self.timestamps[0].timestamp())
        dummy_ts = int(self.timestamps[n - 1].timestamp()) + candle_interval
        candle_data.append({
            'time': dummy_ts,
            'open': last_close,
            'high': last_close,
            'low': last_close,
            'close': last_close,
        })

        # Build markers for directional predictions
        # Markers placed on execution candle (N+1) since trade opens at Open[N+1]
        markers = []
        trade_dir = self.trade_dir
        for i in range(n):
            pred = int(self.predictions[i])

            # Place marker on execution candle (next candle after signal)
            if i + 1 < n:
                ts = int(self.timestamps[i + 1].timestamp())
            else:
                # Last prediction: use dummy candle timestamp
                ts = dummy_ts
            conf = float(self.confidences[i])

            # Build probability dict for tooltip
            prob_dict = {}
            for ci in range(self.num_classes):
                prob_dict[f'p{ci}'] = round(float(self.probabilities[i, ci]), 3)

            direction = trade_dir.get(pred)
            if direction == "long":
                color = CORRECT_COLOR
                position = 'aboveBar'
            elif direction == "short":
                color = INCORRECT_COLOR
                position = 'belowBar'
            else:  # Non-tradable (reject class)
                color = UNCERTAIN_COLOR
                position = 'inBar'

            true_label_val = int(self.labels[i]) if (not self.inference_only) else -1

            marker = {
                'time': ts,
                'position': position,
                'color': color,
                'shape': 'circle',
                'text': '',
                'size': 0.5,
                'pred': pred,
                'true': true_label_val,
                'conf': round(conf, 3),
                'pctRank': round(float(self.margin_pct_rank[i]), 1),
            }
            marker.update(prob_dict)
            markers.append(marker)

        # Build confidence histogram from actual trade outcomes
        # TP = green, SL/both = red, timeout = orange
        hist_data = []
        for trade in self.trades:
            if trade['o'] == 'tp':
                color = CORRECT_COLOR
            elif trade['o'] in ('sl', 'both'):
                color = INCORRECT_COLOR
            else:  # 'timeout'
                color = UNCERTAIN_COLOR

            hist_data.append({
                'time': trade['et'],
                'value': trade['c'],
                'color': color,
            })

        candle_json = json.dumps(candle_data)
        markers_json = json.dumps(markers)
        hist_json = json.dumps(hist_data)
        trades_json = json.dumps(self.trades)

        # Legend info
        cls_first = self.class_names[0]
        cls_last = self.class_names[-1]
        if self.inference_only:
            legend_html = (
                f'<span style="color:{INCORRECT_COLOR}">&#9679; {cls_first}</span> '
                f'<span style="color:{CORRECT_COLOR}">&#9679; {cls_last}</span>'
            )
        else:
            legend_html = (
                '<b>Markers:</b> '
                f'<span style="color:{INCORRECT_COLOR}">&#9679; {cls_first}</span> '
                f'<span style="color:{CORRECT_COLOR}">&#9679; {cls_last}</span> '
                '&nbsp;&nbsp; <b>Histogram:</b> '
                f'<span style="color:{CORRECT_COLOR}">&#9679; TP</span> '
                f'<span style="color:{UNCERTAIN_COLOR}">&#9679; Timeout</span> '
                f'<span style="color:{INCORRECT_COLOR}">&#9679; SL/Both</span>'
            )

        # Trade zones checkbox (only if trades available)
        trade_checkbox_html = ''
        if self.trades:
            trade_checkbox_html = (
                '<label style="display:flex; align-items:center; gap:6px; cursor:pointer;">'
                '<input type="checkbox" id="tradeToggle" style="cursor:pointer;">'
                '<span>Show trade zones (TP/SL rectangles)</span>'
                '</label>'
            )

        return f'''
        <section id="candlestick">
            <h2>Candlestick & Margin</h2>
            <div style="margin-bottom:8px; font-size:13px; display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
                <div>{legend_html}</div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span>Margin Top</span>
                    <input type="range" id="confSlider" min="5" max="100" step="5" value="100" style="cursor:pointer;">
                    <span id="confSliderVal">100%</span>
                </div>
                {trade_checkbox_html}
            </div>
            <div id="chart-container" style="position:relative; width:100%; height:600px;"></div>
            <div id="tooltip" style="display:none; position:absolute; z-index:1000;
                 background:rgba(0,0,0,0.85); color:#fff; padding:8px 12px; border-radius:4px;
                 font-size:12px; pointer-events:none; white-space:nowrap;"></div>
            <script>
            (function() {{
                const candleData = {candle_json};
                const markersRaw = {markers_json};
                const histData = {hist_json};
                const tradesData = {trades_json};

                // Margin percentile filter helpers
                function applyPctFilter(topPct) {{
                    const minRank = 100 - topPct;
                    if (topPct >= 100) {{
                        // No filter: show all markers/hist
                        const lwAll = markersRaw.map(m => ({{
                            time: m.time, position: m.position, color: m.color,
                            shape: m.shape, text: m.text, size: m.size,
                        }}));
                        candleSeries.setMarkers(lwAll);
                        histSeries.setData(histData);
                    }} else {{
                        const filtered = markersRaw.filter(m => m.pctRank >= minRank);
                        const lwFiltered = filtered.map(m => ({{
                            time: m.time, position: m.position, color: m.color,
                            shape: m.shape, text: m.text, size: m.size,
                        }}));
                        candleSeries.setMarkers(lwFiltered);
                        const filteredTimes = new Set(filtered.map(m => m.time));
                        histSeries.setData(histData.filter(h => filteredTimes.has(h.time)));
                    }}
                    requestTradeRedraw();
                }}

                const container = document.getElementById('chart-container');
                const chart = LightweightCharts.createChart(container, {{
                    width: container.clientWidth,
                    height: 600,
                    layout: {{
                        background: {{ type: 'solid', color: '#ffffff' }},
                        textColor: '#333',
                    }},
                    grid: {{
                        vertLines: {{ color: '#f0f0f0' }},
                        horzLines: {{ color: '#f0f0f0' }},
                    }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    timeScale: {{
                        timeVisible: true,
                        secondsVisible: false,
                        borderColor: '#ddd',
                    }},
                    rightPriceScale: {{ borderColor: '#ddd' }},
                }});

                // Candlestick series
                const candleSeries = chart.addCandlestickSeries({{
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderUpColor: '#26a69a',
                    borderDownColor: '#ef5350',
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                }});
                candleSeries.setData(candleData);

                // Initial markers (all directional, no percentile filter)
                candleSeries.setMarkers(markersRaw.map(m => ({{
                    time: m.time, position: m.position, color: m.color,
                    shape: m.shape, text: m.text, size: m.size,
                }})));

                // Confidence histogram as a second pane
                const histSeries = chart.addHistogramSeries({{
                    priceScaleId: 'confidence',
                    priceFormat: {{ type: 'price', precision: 2, minMove: 0.01 }},
                }});
                histSeries.priceScale().applyOptions({{
                    scaleMargins: {{ top: 0.85, bottom: 0 }},
                }});
                histSeries.setData(histData);

                // --- Trade zones canvas overlay ---
                let showTrades = false;
                const overlay = document.createElement('canvas');
                overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;';
                container.appendChild(overlay);
                const ctx = overlay.getContext('2d');

                // Binary search: find first trade with et >= target
                function lowerBound(arr, target) {{
                    let lo = 0, hi = arr.length;
                    while (lo < hi) {{
                        const mid = (lo + hi) >>> 1;
                        if (arr[mid].et < target) lo = mid + 1;
                        else hi = mid;
                    }}
                    return lo;
                }}

                let rafId = 0;
                function requestTradeRedraw() {{
                    if (rafId) return;
                    rafId = requestAnimationFrame(() => {{
                        rafId = 0;
                        drawTradeZones();
                    }});
                }}

                function drawTradeZones() {{
                    // Resize canvas for DPR
                    const dpr = window.devicePixelRatio || 1;
                    const rect = overlay.getBoundingClientRect();
                    overlay.width = rect.width * dpr;
                    overlay.height = rect.height * dpr;
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    ctx.clearRect(0, 0, rect.width, rect.height);

                    if (!showTrades || tradesData.length === 0) return;

                    const timeScale = chart.timeScale();
                    const range = timeScale.getVisibleRange();
                    if (!range) return;

                    const rangeFrom = range.from;
                    const rangeTo = range.to;
                    const topPct = parseFloat(document.getElementById('confSlider').value);
                    const minRank = 100 - topPct;

                    // Find first trade that could be visible (exit_time >= rangeFrom)
                    // Start a bit before to catch trades that start before but end in view
                    let startIdx = lowerBound(tradesData, rangeFrom);
                    if (startIdx > 0) startIdx--;

                    for (let i = startIdx; i < tradesData.length; i++) {{
                        const t = tradesData[i];
                        // Skip if trade ends before visible range
                        if (t.xt < rangeFrom) continue;
                        // Stop if trade starts after visible range
                        if (t.et > rangeTo) break;

                        // Filter by margin percentile
                        if (t.r < minRank) continue;

                        const x1 = timeScale.timeToCoordinate(t.et);
                        if (x1 === null) continue;
                        let x2 = timeScale.timeToCoordinate(t.xt);
                        if (x2 === null) x2 = rect.width;  // extend to chart right edge

                        const yEntry = candleSeries.priceToCoordinate(t.ep);
                        const yTp = candleSeries.priceToCoordinate(t.tp);
                        const ySl = candleSeries.priceToCoordinate(t.sl);
                        if (yEntry === null || yTp === null || ySl === null) continue;

                        const w = Math.max(x2 - x1, 2);

                        // Profit zone (entry to TP)
                        const profitTop = Math.min(yEntry, yTp);
                        const profitH = Math.abs(yTp - yEntry);
                        ctx.fillStyle = 'rgba(38,166,154,0.15)';
                        ctx.fillRect(x1, profitTop, w, profitH);

                        // Loss zone (entry to SL)
                        const lossTop = Math.min(yEntry, ySl);
                        const lossH = Math.abs(ySl - yEntry);
                        ctx.fillStyle = 'rgba(239,83,80,0.15)';
                        ctx.fillRect(x1, lossTop, w, lossH);

                        // Entry line
                        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
                        ctx.lineWidth = 0.5;
                        ctx.beginPath();
                        ctx.moveTo(x1, yEntry);
                        ctx.lineTo(x1 + w, yEntry);
                        ctx.stroke();
                    }}
                }}

                // Subscribe to chart events for redraw
                chart.timeScale().subscribeVisibleLogicalRangeChange(requestTradeRedraw);

                // Resize
                const resizeObserver = new ResizeObserver(entries => {{
                    const cr = entries[0].contentRect;
                    chart.applyOptions({{ width: cr.width }});
                    requestTradeRedraw();
                }});
                resizeObserver.observe(container);

                // Tooltip on crosshair
                const tooltip = document.getElementById('tooltip');
                const markerMap = {{}};
                markersRaw.forEach(m => {{ markerMap[m.time] = m; }});

                chart.subscribeCrosshairMove(param => {{
                    if (!param || !param.time) {{
                        tooltip.style.display = 'none';
                        return;
                    }}
                    const m = markerMap[param.time];
                    if (!m) {{
                        tooltip.style.display = 'none';
                        return;
                    }}
                    const classNames = {json.dumps(self.class_names)};
                    let html = '<b>Pred:</b> ' + classNames[m.pred] + '<br>';
                    if (m.true >= 0) html += '<b>True:</b> ' + classNames[m.true] + '<br>';
                    html += '<b>Margin:</b> ' + m.conf + ' (P' + Math.round(m.pctRank) + ')<br>';
                    const probParts = [];
                    for (let ci = 0; ci < classNames.length; ci++) {{
                        if (m['p' + ci] !== undefined) probParts.push('P(' + classNames[ci] + '): ' + m['p' + ci]);
                    }}
                    html += probParts.join(' | ');
                    tooltip.innerHTML = html;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (param.point.x + 20) + 'px';
                    tooltip.style.top = (param.point.y - 20) + 'px';
                }});

                // Margin percentile slider handler
                const confSlider = document.getElementById('confSlider');
                const confSliderVal = document.getElementById('confSliderVal');
                confSlider.addEventListener('input', () => {{
                    const topPct = parseInt(confSlider.value);
                    confSliderVal.textContent = topPct + '%';
                    applyPctFilter(topPct);
                }});

                // Trade zones toggle
                const tradeToggle = document.getElementById('tradeToggle');
                if (tradeToggle) {{
                    tradeToggle.addEventListener('change', (e) => {{
                        showTrades = e.target.checked;
                        drawTradeZones();
                    }});
                }}

                chart.timeScale().fitContent();
            }})();
            </script>
        </section>
        '''

    # ------------------------------------------------------------------
    # Plotly: Statistical plots
    # ------------------------------------------------------------------

    def _build_confusion_matrix_section(self):
        """Build confusion matrix section with dynamic confidence slider."""
        if self.inference_only:
            return None

        from sklearn.metrics import confusion_matrix
        from plotly.subplots import make_subplots

        # Raw data for client-side recomputation
        plot_labels = self.labels
        plot_predictions = self.predictions
        plot_confidences = self.confidences
        plot_pct_ranks = self.margin_pct_rank
        cm_data_json = json.dumps({
            'labels': plot_labels.tolist(),
            'predictions': plot_predictions.tolist(),
            'confidences': [round(float(c), 4) for c in plot_confidences],
            'pctRanks': [round(float(r), 1) for r in plot_pct_ranks],
        })

        # Initial figure: all predictions (no updatemenus)
        nc = self.num_classes
        cm_all = confusion_matrix(plot_labels, plot_predictions, labels=list(range(nc)))

        col_sums = cm_all.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        cm_norm_col = cm_all / col_sums

        row_sums = cm_all.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm_row = cm_all / row_sums

        ann_col = []
        ann_row = []
        for i in range(nc):
            row_c = []
            row_r = []
            for j in range(nc):
                row_c.append(f"{cm_all[i, j]}<br>{cm_norm_col[i, j]:.2%}")
                row_r.append(f"{cm_all[i, j]}<br>{cm_norm_row[i, j]:.2%}")
            ann_col.append(row_c)
            ann_row.append(row_r)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Normalized by Column (Precision)', 'Normalized by Row (Recall)'),
            horizontal_spacing=0.15
        )

        fig.add_trace(
            go.Heatmap(
                z=cm_norm_col.tolist(), x=self.class_names, y=self.class_names,
                colorscale='Blues', text=ann_col, texttemplate='%{text}',
                textfont=dict(size=14),
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>',
                showscale=False,
            ), row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=cm_norm_row.tolist(), x=self.class_names, y=self.class_names,
                colorscale='Greens', text=ann_row, texttemplate='%{text}',
                textfont=dict(size=14),
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>',
                showscale=False,
            ), row=1, col=2
        )

        fig.update_xaxes(title_text='Predicted', row=1, col=1)
        fig.update_xaxes(title_text='Predicted', row=1, col=2)
        fig.update_yaxes(title_text='True', autorange='reversed', row=1, col=1)
        fig.update_yaxes(title_text='True', autorange='reversed', row=1, col=2)

        n_total = len(self.labels)
        fig.update_layout(
            title=f'Confusion Matrix — All predictions (n={n_total})',
            width=1000, height=500,
        )

        div = fig.to_html(full_html=False, include_plotlyjs=False, div_id='cm-plot')

        return f'''
        <section id="confusion">
            <h2>Confusion Matrix</h2>
            {div}
            <div style="margin-top:12px; display:flex; align-items:center; gap:10px; font-size:14px;">
                <span>Margin Top</span>
                <input type="range" id="cmConfSlider" min="5" max="100" step="5" value="100"
                       style="width:300px; cursor:pointer;">
                <span id="cmConfVal">100%</span>
                <span id="cmCount" style="color:#666;">(n={n_total})</span>
            </div>
            <script>
            (function() {{
                const cmRaw = {cm_data_json};
                const plotDiv = document.getElementById('cm-plot');

                const NC = {self.num_classes};
                function recomputeCM(topPct) {{
                    const minRank = 100 - topPct;
                    const cm = Array.from({{length: NC}}, () => new Array(NC).fill(0));
                    let count = 0;
                    for (let i = 0; i < cmRaw.labels.length; i++) {{
                        if (cmRaw.pctRanks[i] >= minRank) {{
                            cm[cmRaw.labels[i]][cmRaw.predictions[i]]++;
                            count++;
                        }}
                    }}
                    const colSums = new Array(NC).fill(0);
                    for (let j = 0; j < NC; j++) for (let i = 0; i < NC; i++) colSums[j] += cm[i][j];
                    const normCol = cm.map(row => row.map((v, j) => colSums[j] > 0 ? v / colSums[j] : 0));
                    const rowSums = cm.map(row => row.reduce((a, b) => a + b, 0));
                    const normRow = cm.map((row, i) => row.map(v => rowSums[i] > 0 ? v / rowSums[i] : 0));
                    const textCol = cm.map((row, i) => row.map((v, j) =>
                        v + '<br>' + (normCol[i][j] * 100).toFixed(1) + '%'));
                    const textRow = cm.map((row, i) => row.map((v, j) =>
                        v + '<br>' + (normRow[i][j] * 100).toFixed(1) + '%'));
                    return {{normCol, normRow, textCol, textRow, count}};
                }}

                const slider = document.getElementById('cmConfSlider');
                const valSpan = document.getElementById('cmConfVal');
                const countSpan = document.getElementById('cmCount');

                slider.addEventListener('input', function() {{
                    const topPct = parseInt(this.value);
                    valSpan.textContent = topPct + '%';
                    const r = recomputeCM(topPct);
                    countSpan.textContent = '(n=' + r.count + ')';
                    Plotly.restyle(plotDiv, {{
                        z: [r.normCol, r.normRow],
                        text: [r.textCol, r.textRow],
                    }});
                    const label = topPct < 100
                        ? 'Confusion Matrix \\u2014 Margin Top ' + topPct + '% (n=' + r.count + ')'
                        : 'Confusion Matrix \\u2014 All predictions (n=' + r.count + ')';
                    Plotly.relayout(plotDiv, {{title: label}});
                }});
            }})();
            </script>
        </section>
        '''

    def _dir_button_label(self):
        """Return the label for the 'directional only' toggle button."""
        if self.prediction_target == "bull":
            return "Bull only"
        elif self.prediction_target == "bear":
            return "Bear only"
        elif self.prediction_target == "triple":
            return "Directional only"
        else:
            return None  # uncertain, closeN: no directional filter applicable

    def _build_stacked_bar_traces(self, correct_confs, incorrect_confs, edges):
        """
        Build stacked bar traces (correct + incorrect) with accuracy annotations.

        Args:
            correct_confs: confidence values of correct predictions
            incorrect_confs: confidence values of incorrect predictions
            edges: bin edges from np.histogram

        Returns:
            Tuple of (bar_correct, bar_incorrect, scatter_accuracy) traces
        """
        counts_correct, _ = np.histogram(correct_confs, bins=edges)
        counts_incorrect, _ = np.histogram(incorrect_confs, bins=edges)
        totals = counts_correct + counts_incorrect

        # Bin centers for x axis
        centers = (edges[:-1] + edges[1:]) / 2
        bin_width = edges[1] - edges[0]

        # Accuracy text per bin
        acc_texts = []
        for c, t in zip(counts_correct, totals):
            if t > 0:
                acc_texts.append(f"{c / t:.0%}")
            else:
                acc_texts.append("")

        bar_correct = go.Bar(
            x=centers, y=counts_correct, name='Correct',
            marker_color='#00CC96', opacity=0.9,
            width=bin_width * 0.9,
        )
        bar_incorrect = go.Bar(
            x=centers, y=counts_incorrect, name='Incorrect',
            marker_color='#EF553B', opacity=0.9,
            width=bin_width * 0.9,
        )
        # Accuracy % displayed above stacked bars
        scatter_acc = go.Scatter(
            x=centers, y=totals + max(totals.max() * 0.03, 1),
            mode='text', text=acc_texts, name='Accuracy %',
            textfont=dict(size=10, color='#666'),
            showlegend=False, hoverinfo='skip',
        )
        return bar_correct, bar_incorrect, scatter_acc

    def _build_confidence_distribution_section(self):
        """Build confidence distribution section with HTML toggle buttons below chart."""
        fig = go.Figure()
        edges = np.linspace(0, 1, 21)  # 20 bins

        has_buttons = False
        if not self.inference_only:
            lbl = self.labels
            pred = self.predictions
            conf = self.confidences

            correct_mask = lbl == pred
            # All predictions
            t0, t1, t2 = self._build_stacked_bar_traces(conf[correct_mask], conf[~correct_mask], edges)
            fig.add_trace(t0)  # [0]
            fig.add_trace(t1)  # [1]
            fig.add_trace(t2)  # [2]

            # Directional only: filter on predictions only (show all trades
            # the model would take, including errors against any true label)
            if self.reject_class is not None:
                dir_mask = pred != self.reject_class
            else:
                dir_mask = np.ones(len(pred), dtype=bool)
            t3, t4, t5 = self._build_stacked_bar_traces(conf[correct_mask & dir_mask], conf[~correct_mask & dir_mask], edges)
            t3.visible = False
            t4.visible = False
            t5.visible = False
            fig.add_trace(t3)  # [3]
            fig.add_trace(t4)  # [4]
            fig.add_trace(t5)  # [5]

            fig.update_layout(barmode='stack')
            has_buttons = True
        else:
            fig.add_trace(go.Histogram(
                x=self.confidences, nbinsx=20, name='Confidence',
                marker_color='#636EFA', opacity=0.8,
                xbins=dict(start=0, end=1),
            ))

        fig.update_layout(
            title='Margin Distribution - All predictions',
            xaxis_title='Margin (top1 − top2)',
            yaxis_title='Count',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(autorange=True),
            height=400,
        )

        div = fig.to_html(full_html=False, include_plotlyjs=False, div_id='confdist-plot')

        buttons_html = ''
        dir_label = self._dir_button_label()
        if has_buttons and dir_label is not None:
            buttons_html = f'''
            <div style="margin-top:12px; display:flex; gap:10px;">
                <button id="confDistAll" style="padding:6px 16px; cursor:pointer; border:2px solid #1976d2;
                        border-radius:4px; background:#e3f2fd; font-weight:bold; color:#1976d2;">All predictions</button>
                <button id="confDistDir" style="padding:6px 16px; cursor:pointer; border:1px solid #ccc;
                        border-radius:4px; background:#fff; font-weight:normal; color:#333;">{dir_label}</button>
            </div>
            <script>
            (function() {{
                const plotDiv = document.getElementById('confdist-plot');
                const btnAll = document.getElementById('confDistAll');
                const btnDir = document.getElementById('confDistDir');
                function setActive(active, inactive) {{
                    active.style.border = '2px solid #1976d2';
                    active.style.background = '#e3f2fd';
                    active.style.fontWeight = 'bold';
                    active.style.color = '#1976d2';
                    inactive.style.border = '1px solid #ccc';
                    inactive.style.background = '#fff';
                    inactive.style.fontWeight = 'normal';
                    inactive.style.color = '#333';
                }}
                btnAll.addEventListener('click', function() {{
                    setActive(btnAll, btnDir);
                    Plotly.restyle(plotDiv, {{visible: [true, true, true, false, false, false]}});
                    Plotly.relayout(plotDiv, {{title: 'Margin Distribution - All predictions'}});
                }});
                btnDir.addEventListener('click', function() {{
                    setActive(btnDir, btnAll);
                    Plotly.restyle(plotDiv, {{visible: [false, false, false, true, true, true]}});
                    Plotly.relayout(plotDiv, {{title: 'Margin Distribution - {dir_label}'}});
                }});
            }})();
            </script>'''

        return f'''
        <section id="confidence">
            <h2>Margin Distribution</h2>
            {div}
            {buttons_html}
        </section>
        '''

    def _build_high_confidence_zoom_section(self):
        """Build high confidence zoom with dynamic threshold slider and mode toggle."""
        if not self.inference_only:
            lbl = self.labels
            pred = self.predictions
            conf = self.confidences
            pct_rank = self.margin_pct_rank

            # Raw data for JS recomputation
            hc_data_json = json.dumps({
                'labels': lbl.tolist(),
                'predictions': pred.tolist(),
                'confidences': [round(float(c), 4) for c in conf],
                'pctRanks': [round(float(r), 1) for r in pct_rank],
                'rejectClass': self.reject_class,
            })

            # Initial figure at top 10%
            top_pct_init = 10
            min_rank_init = 100 - top_pct_init
            mask = pct_rank >= min_rank_init
            correct_mask = lbl == pred

            filtered_conf = conf[mask]
            filtered_correct = correct_mask[mask]
            n_filtered = int(mask.sum())

            if n_filtered > 0 and filtered_conf.max() > filtered_conf.min():
                edges = np.linspace(float(filtered_conf.min()), float(filtered_conf.max()), 11)
                overall_acc = float(filtered_correct.mean())
            else:
                edges = np.linspace(0, 1, 11)
                overall_acc = 0.0

            t0, t1, t2 = self._build_stacked_bar_traces(
                filtered_conf[filtered_correct], filtered_conf[~filtered_correct], edges)
            fig = go.Figure(data=[t0, t1, t2])
            fig.update_layout(
                barmode='stack',
                title=f'High Margin Zoom — Top {top_pct_init}% ({n_filtered} preds, {overall_acc:.1%} acc)',
                xaxis_title='Margin',
                yaxis_title='Count',
                height=400,
            )

            div = fig.to_html(full_html=False, include_plotlyjs=False, div_id='hczoom-plot')

            hc_dir_label = self._dir_button_label()
            hc_dir_btn_html = ''
            if hc_dir_label is not None:
                hc_dir_btn_html = f'''<button id="hcDir" style="padding:6px 16px; cursor:pointer; border:1px solid #ccc;
                            border-radius:4px; background:#fff; font-weight:normal; color:#333;">{hc_dir_label}</button>'''

            return f'''
            <section id="highconf">
                <h2>High Margin Zoom</h2>
                {div}
                <div style="margin-top:12px; display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-size:14px;">
                    <button id="hcAll" style="padding:6px 16px; cursor:pointer; border:2px solid #1976d2;
                            border-radius:4px; background:#e3f2fd; font-weight:bold; color:#1976d2;">All predictions</button>
                    {hc_dir_btn_html}
                    <span style="margin-left:20px;">Top %:</span>
                    <input type="range" id="hcThreshSlider" min="5" max="100" step="5" value="10"
                           style="width:200px; cursor:pointer;">
                    <span id="hcThreshVal">10%</span>
                </div>
                <script>
                (function() {{
                    const hcRaw = {hc_data_json};
                    const plotDiv = document.getElementById('hczoom-plot');
                    let currentMode = 'all';

                    function rebuild(topPct, mode) {{
                        const minRank = 100 - topPct;
                        const rejectClass = hcRaw.rejectClass;

                        // Collect qualifying predictions
                        const margins = [];
                        const isCorrect = [];
                        for (let i = 0; i < hcRaw.confidences.length; i++) {{
                            if (hcRaw.pctRanks[i] < minRank) continue;
                            if (mode === 'dir' && rejectClass !== null && hcRaw.predictions[i] === rejectClass) continue;
                            margins.push(hcRaw.confidences[i]);
                            isCorrect.push(hcRaw.labels[i] === hcRaw.predictions[i]);
                        }}

                        if (margins.length === 0) {{
                            Plotly.react(plotDiv, [], {{
                                title: 'High Margin Zoom — Top ' + topPct + '% — No data',
                                height: 400,
                            }});
                            return;
                        }}

                        const minMargin = Math.min(...margins);
                        const maxMargin = Math.max(...margins);
                        const nBins = 10;
                        const range = maxMargin - minMargin;
                        const binW = range > 0 ? range / nBins : 0.01;

                        const centers = [];
                        const correctCounts = new Array(nBins).fill(0);
                        const incorrectCounts = new Array(nBins).fill(0);
                        for (let b = 0; b < nBins; b++) centers.push(minMargin + (b + 0.5) * binW);

                        for (let j = 0; j < margins.length; j++) {{
                            let bin = Math.floor((margins[j] - minMargin) / binW);
                            if (bin >= nBins) bin = nBins - 1;
                            if (isCorrect[j]) correctCounts[bin]++;
                            else incorrectCounts[bin]++;
                        }}

                        const totals = correctCounts.map((v, i) => v + incorrectCounts[i]);
                        const maxT = Math.max(...totals, 1);
                        const accTexts = totals.map((t, i) => t > 0 ? Math.round(correctCounts[i] / t * 100) + '%' : '');
                        const accY = totals.map(t => t + maxT * 0.03);
                        const modeLabel = mode === 'all' ? 'All predictions' : '{hc_dir_label or "Filtered"}';
                        const totalCount = margins.length;
                        const totalCorrect = isCorrect.filter(x => x).length;
                        const overallAcc = totalCount > 0 ? (totalCorrect / totalCount * 100).toFixed(1) : '0.0';

                        Plotly.react(plotDiv, [
                            {{
                                x: centers, y: correctCounts, type: 'bar',
                                name: 'Correct', marker: {{color: '#00CC96', opacity: 0.9}},
                                width: binW * 0.9,
                            }},
                            {{
                                x: centers, y: incorrectCounts, type: 'bar',
                                name: 'Incorrect', marker: {{color: '#EF553B', opacity: 0.9}},
                                width: binW * 0.9,
                            }},
                            {{
                                x: centers, y: accY, type: 'scatter', mode: 'text',
                                text: accTexts, name: 'Accuracy %',
                                textfont: {{size: 10, color: '#666'}},
                                showlegend: false, hoverinfo: 'skip',
                            }},
                        ], {{
                            barmode: 'stack',
                            title: 'High Margin Zoom — Top ' + topPct + '% (' + totalCount + ' preds, ' + overallAcc + '% acc) — ' + modeLabel,
                            xaxis: {{title: 'Margin'}},
                            yaxis: {{title: 'Count', autorange: true}},
                            height: 400,
                        }});
                    }}

                    const slider = document.getElementById('hcThreshSlider');
                    const valSpan = document.getElementById('hcThreshVal');
                    slider.addEventListener('input', function() {{
                        const topPct = parseInt(this.value);
                        valSpan.textContent = topPct + '%';
                        rebuild(topPct, currentMode);
                    }});

                    const btnAll = document.getElementById('hcAll');
                    const btnDir = document.getElementById('hcDir');
                    function setActive(active, inactive) {{
                        if (!active || !inactive) return;
                        active.style.border = '2px solid #1976d2';
                        active.style.background = '#e3f2fd';
                        active.style.fontWeight = 'bold';
                        active.style.color = '#1976d2';
                        inactive.style.border = '1px solid #ccc';
                        inactive.style.background = '#fff';
                        inactive.style.fontWeight = 'normal';
                        inactive.style.color = '#333';
                    }}

                    btnAll.addEventListener('click', function() {{
                        currentMode = 'all';
                        setActive(btnAll, btnDir);
                        rebuild(parseInt(slider.value), currentMode);
                    }});
                    if (btnDir) {{
                        btnDir.addEventListener('click', function() {{
                            currentMode = 'dir';
                            setActive(btnDir, btnAll);
                            rebuild(parseInt(slider.value), currentMode);
                        }});
                    }}
                }})();
                </script>
            </section>
            '''
        else:
            # Inference mode: static histogram of top 10% margin
            mask = self.margin_pct_rank >= 90  # top 10%
            high_conf = self.confidences[mask]
            n_hc = int(mask.sum())
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=high_conf, nbinsx=10, name='Margin',
                marker_color='#636EFA', opacity=0.8,
            ))
            fig.update_layout(
                title=f'High Margin Zoom — Top 10% ({n_hc} preds)',
                xaxis_title='Margin',
                yaxis_title='Count',
                height=400,
            )
            div = fig.to_html(full_html=False, include_plotlyjs=False, div_id='hczoom-plot')
            return f'''
            <section id="highconf">
                <h2>High Margin Zoom</h2>
                {div}
            </section>
            '''

    # ------------------------------------------------------------------
    # PNL Simulation: Equity curve + Drawdown
    # ------------------------------------------------------------------

    def _build_pnl_simulation_section(self):
        """
        Build PNL simulation section with equity curve and drawdown charts.

        Embeds extended OHLCV + ATR + predictions as JSON. Full simulation
        runs client-side in JS so sliders/inputs trigger instant recomputation.
        """
        raw_start = self.config_params['raw_start_idx']
        n = len(self.predictions)
        atr_mult_tp = self.config_params['ATR_MULTIPLIER_TP']
        atr_mult_sl = self.config_params['ATR_MULTIPLIER_SL']
        max_horizon = self.config_params['MAX_HORIZON']

        # Extended range: predictions + MAX_HORIZON extra candles for last trades
        raw_len = len(self.df_raw)
        ext_end = min(raw_start + n + max_horizon, raw_len)

        # Build JSON arrays aligned: index 0 = raw_start in df_raw
        ext_open = self.df_raw['Open'].values[raw_start:ext_end]
        ext_high = self.df_raw['High'].values[raw_start:ext_end]
        ext_low = self.df_raw['Low'].values[raw_start:ext_end]
        ext_close = self.df_raw['Close'].values[raw_start:ext_end]
        ext_atr = self._atr_full[raw_start:ext_end]
        ext_ts = pd.to_datetime(
            self.df_raw['Open time'].values[raw_start:ext_end]
        )

        # Replace NaN ATR with 0 for JSON
        ext_atr = np.where(np.isnan(ext_atr), 0, ext_atr)

        # Detect simulation mode: "barrier" (triple barrier TP/SL) or "hold" (fixed horizon, closeN)
        close_horizon = parse_close_horizon(self.prediction_target)
        sim_mode = "hold" if close_horizon is not None else "barrier"

        sim_data = {
            'open': [round(float(v), 2) for v in ext_open],
            'high': [round(float(v), 2) for v in ext_high],
            'low': [round(float(v), 2) for v in ext_low],
            'close': [round(float(v), 2) for v in ext_close],
            'atr': [round(float(v), 4) for v in ext_atr],
            'timestamps': [int(t.timestamp()) for t in ext_ts],
            'predictions': self.predictions.tolist(),
            'margins': [round(float(c), 4) for c in self.confidences],
            'pctRanks': [round(float(r), 1) for r in self.margin_pct_rank],
            'n_predictions': n,
            'atr_mult_tp': float(atr_mult_tp),
            'atr_mult_sl': float(atr_mult_sl),
            'max_horizon': int(max_horizon),
            'tradeDir': {str(k): v for k, v in self.trade_dir.items()},
            'simMode': sim_mode,
            'holdHorizon': close_horizon if close_horizon else 0,
        }
        sim_data_json = json.dumps(sim_data)

        # Initial empty Plotly charts (will be filled by JS)
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=[], y=[], mode='lines', name='Equity',
            line=dict(color='#1976d2', width=2),
        ))
        fig_equity.update_layout(
            title='Equity Curve — Simulated PNL',
            xaxis_title='Date', yaxis_title='Balance ($)',
            height=400, hovermode='x unified',
        )

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=[], y=[], mode='lines', name='Drawdown',
            fill='tozeroy', line=dict(color='#C62828', width=1),
            fillcolor='rgba(198,40,40,0.2)',
        ))
        fig_dd.update_layout(
            title='Drawdown',
            xaxis_title='Date', yaxis_title='Drawdown (%)',
            height=250, hovermode='x unified',
        )

        div_equity = fig_equity.to_html(
            full_html=False, include_plotlyjs=False,
            div_id='pnl-equity-plot')
        div_dd = fig_dd.to_html(
            full_html=False, include_plotlyjs=False,
            div_id='pnl-dd-plot')

        # Build control panel HTML — hide risk/leverage/compounding for hold modes
        hide_barrier = ' style="display:none;"' if sim_mode == 'hold' else ''
        controls_html = f'''
        <div style="margin-top:12px; display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end; font-size:13px;">
            <div>
                <label>Margin Top %</label><br>
                <input type="range" id="pnlConfSlider" min="5" max="100" step="5" value="100"
                       style="width:180px; cursor:pointer;">
                <span id="pnlConfVal">100%</span>
            </div>
            <div>
                <label>Balance ($)</label><br>
                <input type="number" id="pnlBalance" value="1000" min="1" step="100"
                       style="width:90px; padding:4px;">
            </div>
            <div{hide_barrier}>
                <label>Risk/trade (%)</label><br>
                <input type="number" id="pnlRisk" value="2" min="0.1" max="100" step="0.1"
                       style="width:70px; padding:4px;">
            </div>
            <div>
                <label>Fees/side (%)</label><br>
                <input type="number" id="pnlFees" value="0.02" min="0" max="5" step="0.01"
                       style="width:80px; padding:4px;">
            </div>
            <div>
                <label>Slippage/side (%)</label><br>
                <input type="number" id="pnlSlippage" value="0" min="0" max="5" step="0.01"
                       style="width:80px; padding:4px;">
            </div>
            <div{hide_barrier}>
                <label>Max leverage</label><br>
                <input type="number" id="pnlMaxLev" value="10" min="1" max="100" step="1"
                       style="width:60px; padding:4px;">
            </div>
            <div{hide_barrier}>
                <label>Position sizing</label><br>
                <select id="pnlSizingMode" style="padding:4px;">
                    <option value="fixed" selected>Fixed (initial balance)</option>
                    <option value="compound">Compounding</option>
                </select>
            </div>
            <div>
                <label style="display:flex; align-items:center; gap:4px; cursor:pointer;">
                    <input type="checkbox" id="pnlLogScale">
                    <span>Log scale</span>
                </label>
            </div>
            <div>
                <button id="pnlRunBtn" style="padding:6px 20px; cursor:pointer; background:#1976d2;
                        color:#fff; border:none; border-radius:4px; font-weight:bold;">Recalculate</button>
            </div>
        </div>
        <div id="pnlStats" style="margin-top:12px; font-size:13px; display:flex; flex-wrap:wrap; gap:16px;
             background:#f9f9f9; padding:12px; border-radius:6px; border:1px solid #e0e0e0;">
        </div>
        '''

        # JavaScript simulation engine
        js_engine = f'''
        <script>
        (function() {{
            const D = {sim_data_json};
            const equityDiv = document.getElementById('pnl-equity-plot');
            const ddDiv = document.getElementById('pnl-dd-plot');

            function getParams() {{
                return {{
                    topPct: parseInt(document.getElementById('pnlConfSlider').value),
                    balance: parseFloat(document.getElementById('pnlBalance').value),
                    riskPct: parseFloat(document.getElementById('pnlRisk').value) / 100,
                    feePct: parseFloat(document.getElementById('pnlFees').value) / 100,
                    slipPct: parseFloat(document.getElementById('pnlSlippage').value) / 100,
                    maxLev: parseFloat(document.getElementById('pnlMaxLev').value),
                    compound: document.getElementById('pnlSizingMode').value === 'compound',
                }};
            }}

            function runSimulation() {{
                const P = getParams();
                const n = D.n_predictions;
                const maxH = D.max_horizon;
                const multTP = D.atr_mult_tp;
                const multSL = D.atr_mult_sl;
                const breakevenThresh = Math.ceil(0.66 * maxH);
                const tradeDir = D.tradeDir;  // {{"0":"short","2":"long"}} etc.
                const simMode = D.simMode;     // "barrier" or "hold"
                const holdHorizon = D.holdHorizon;

                let balance = P.balance;
                const initBalance = P.balance;

                // Position state
                let pos = null; // {{dir, entry, tp, sl, qty, entryFee, conf, openIdx, breakevenSet}}

                // Results
                const equityTs = [];
                const equityVals = [];
                const trades = [];
                let peak = balance;
                let maxDD = 0;
                const ddTs = [];
                const ddVals = [];
                let maxLevUsed = 0;
                let signalCount = 0; // Qualifying directional signals above threshold

                function calcPosSize(isLong, entryRaw, slRaw) {{
                    // Exact per-coin calculation
                    let actualEntry, actualSl;
                    if (isLong) {{
                        actualEntry = entryRaw * (1 + P.slipPct);
                        actualSl = slRaw * (1 - P.slipPct);
                    }} else {{
                        actualEntry = entryRaw * (1 - P.slipPct);
                        actualSl = slRaw * (1 + P.slipPct);
                    }}
                    const lossPerCoin = Math.abs(actualEntry - actualSl);
                    const feeEntry = actualEntry * P.feePct;
                    const feeExit = actualSl * P.feePct;
                    const totalLossPerCoin = lossPerCoin + feeEntry + feeExit;
                    if (totalLossPerCoin <= 0) return null;

                    // Fixed mode: always size based on initial balance
                    const sizingBalance = P.compound ? balance : initBalance;
                    let qty = (sizingBalance * P.riskPct) / totalLossPerCoin;
                    let posValue = qty * actualEntry;
                    let leverage = posValue / sizingBalance;

                    if (leverage > P.maxLev) {{
                        qty = (sizingBalance * P.maxLev) / actualEntry;
                        posValue = qty * actualEntry;
                        leverage = P.maxLev;
                    }}
                    if (leverage > maxLevUsed) maxLevUsed = leverage;

                    const entryFee = qty * actualEntry * P.feePct;
                    return {{qty, actualEntry, entryFee, leverage}};
                }}

                // Simple position sizing for hold mode (no SL-based sizing)
                function calcPosSizeHold(isLong, entryRaw) {{
                    let actualEntry = isLong ? entryRaw * (1 + P.slipPct) : entryRaw * (1 - P.slipPct);
                    const sizingBalance = P.compound ? balance : initBalance;
                    const riskAmount = sizingBalance * P.riskPct;
                    let qty = riskAmount / actualEntry;
                    let posValue = qty * actualEntry;
                    let leverage = posValue / sizingBalance;
                    if (leverage > P.maxLev) {{
                        qty = (sizingBalance * P.maxLev) / actualEntry;
                        leverage = P.maxLev;
                    }}
                    if (leverage > maxLevUsed) maxLevUsed = leverage;
                    const entryFee = qty * actualEntry * P.feePct;
                    return {{qty, actualEntry, entryFee, leverage}};
                }}

                function closePosition(exitPriceRaw, candleIdx) {{
                    if (!pos) return;
                    let exitPrice;
                    if (pos.isLong) {{
                        exitPrice = exitPriceRaw * (1 - P.slipPct);
                    }} else {{
                        exitPrice = exitPriceRaw * (1 + P.slipPct);
                    }}
                    let grossPnl;
                    if (pos.isLong) {{
                        grossPnl = pos.qty * (exitPrice - pos.entry);
                    }} else {{
                        grossPnl = pos.qty * (pos.entry - exitPrice);
                    }}
                    const exitFee = pos.qty * exitPrice * P.feePct;
                    const netPnl = grossPnl - pos.entryFee - exitFee;
                    balance += netPnl;
                    trades.push({{
                        pnl: netPnl,
                        win: netPnl > 0,
                    }});
                    pos = null;
                }}

                function unrealizedPnl(currentPrice) {{
                    if (!pos) return 0;
                    if (pos.isLong) {{
                        return pos.qty * (currentPrice - pos.entry);
                    }} else {{
                        return pos.qty * (pos.entry - currentPrice);
                    }}
                }}

                // Helper: record equity/drawdown snapshot
                function recordEquity(candleIdx) {{
                    equityTs.push(D.timestamps[candleIdx]);
                    equityVals.push(balance);
                    if (balance > peak) peak = balance;
                    const dd = (balance - peak) / peak * 100;
                    ddTs.push(D.timestamps[candleIdx]);
                    ddVals.push(dd);
                    if (dd < maxDD) maxDD = dd;
                }}

                // Main simulation loop: iterate over each candle
                for (let i = 0; i < n; i++) {{
                    const candleIdx = i + 1; // entry candle (Open[i+1] in extended array)
                    if (candleIdx >= D.open.length) break;
                    if (balance <= 0) break;

                    // --- Phase 1: Check existing position ---
                    if (pos) {{
                        const candlesOpen = candleIdx - pos.openIdx;

                        if (simMode === 'hold') {{
                            // Hold mode: close at exact horizon
                            if (candlesOpen >= holdHorizon) {{
                                closePosition(D.close[candleIdx], candleIdx);
                            }}
                        }} else {{
                            // Barrier mode: breakeven + TP/SL scan
                            if (!pos.breakevenSet && candlesOpen >= breakevenThresh) {{
                                const uPnl = unrealizedPnl(D.close[candleIdx]);
                                if (uPnl < 0) {{
                                    pos.tp = pos.entry;
                                    pos.breakevenSet = true;
                                }}
                            }}

                            const h = D.high[candleIdx];
                            const l = D.low[candleIdx];
                            let tpHit = false, slHit = false;

                            if (pos.isLong) {{
                                tpHit = h >= pos.tp;
                                slHit = l <= pos.sl;
                            }} else {{
                                tpHit = l <= pos.tp;
                                slHit = h >= pos.sl;
                            }}

                            if (slHit && tpHit) {{
                                closePosition(pos.sl, candleIdx);
                            }} else if (slHit) {{
                                closePosition(pos.sl, candleIdx);
                            }} else if (tpHit) {{
                                closePosition(pos.tp, candleIdx);
                            }} else if (candlesOpen >= maxH) {{
                                closePosition(D.close[candleIdx], candleIdx);
                            }}
                        }}
                    }}

                    // --- Phase 2: Check new signal ---
                    if (balance <= 0) break;
                    const pred = D.predictions[i];
                    const direction = tradeDir[String(pred)];
                    if (!direction) {{
                        // Non-tradable class (reject): record equity and skip
                        recordEquity(candleIdx);
                        continue;
                    }}

                    const margin = D.margins[i];
                    const pctRank = D.pctRanks[i];
                    if (pctRank < (100 - P.topPct)) {{
                        recordEquity(candleIdx);
                        continue;
                    }}

                    const isLong = (direction === 'long');
                    const entryRaw = D.open[candleIdx];

                    if (simMode === 'hold') {{
                        // Hold mode: simple fixed-horizon trades, no TP/SL
                        signalCount++;
                        if (!pos) {{
                            const sizing = calcPosSizeHold(isLong, entryRaw);
                            if (sizing) {{
                                pos = {{
                                    isLong: isLong,
                                    entry: sizing.actualEntry,
                                    tp: 0, sl: 0,
                                    qty: sizing.qty,
                                    entryFee: sizing.entryFee,
                                    margin: margin,
                                    openIdx: candleIdx,
                                    breakevenSet: false,
                                }};
                            }}
                        }}
                        // In hold mode, no position switching — wait for current to expire
                    }} else {{
                        // Barrier mode: ATR-based TP/SL
                        const atrVal = D.atr[i];
                        if (atrVal <= 0) {{
                            recordEquity(candleIdx);
                            continue;
                        }}

                        signalCount++;
                        let newTp, newSl;
                        if (isLong) {{
                            newTp = entryRaw + multTP * atrVal;
                            newSl = entryRaw - multSL * atrVal;
                        }} else {{
                            newTp = entryRaw - multTP * atrVal;
                            newSl = entryRaw + multSL * atrVal;
                        }}

                        if (!pos) {{
                            const sizing = calcPosSize(isLong, entryRaw, newSl);
                            if (sizing) {{
                                pos = {{
                                    isLong: isLong,
                                    entry: sizing.actualEntry,
                                    tp: newTp,
                                    sl: newSl,
                                    qty: sizing.qty,
                                    entryFee: sizing.entryFee,
                                    margin: margin,
                                    openIdx: candleIdx,
                                    breakevenSet: false,
                                }};
                            }}
                        }} else if (pos.isLong === isLong && margin > pos.margin) {{
                            const uPnl = unrealizedPnl(D.close[i]);
                            if (uPnl > 0) {{
                                pos.tp = newTp;
                                pos.sl = newSl;
                                pos.margin = margin;
                                pos.breakevenSet = false;
                            }}
                        }} else if (pos.isLong !== isLong && margin > pos.margin) {{
                            closePosition(entryRaw, candleIdx);
                            if (balance > 0) {{
                                const sizing = calcPosSize(isLong, entryRaw, newSl);
                                if (sizing) {{
                                    pos = {{
                                        isLong: isLong,
                                        entry: sizing.actualEntry,
                                        tp: newTp,
                                        sl: newSl,
                                        qty: sizing.qty,
                                        entryFee: sizing.entryFee,
                                        margin: margin,
                                        openIdx: candleIdx,
                                        breakevenSet: false,
                                    }};
                                }}
                            }}
                        }}
                    }}

                    recordEquity(candleIdx);
                }}

                // Close any remaining position at last available close
                if (pos && D.close.length > 0) {{
                    closePosition(D.close[D.close.length - 1], D.close.length - 1);
                }}

                // Convert timestamps to date strings for Plotly
                const tsToDate = (ts) => new Date(ts * 1000).toISOString().slice(0, 19);
                const eqDates = equityTs.map(tsToDate);
                const ddDates = ddTs.map(tsToDate);

                // Log scale toggle
                const useLog = document.getElementById('pnlLogScale') && document.getElementById('pnlLogScale').checked;

                // Update equity chart
                Plotly.react(equityDiv, [{{
                    x: eqDates, y: equityVals, type: 'scatter', mode: 'lines',
                    name: 'Equity', line: {{color: '#1976d2', width: 2}},
                }}], {{
                    title: 'Equity Curve — Simulated PNL',
                    xaxis: {{title: 'Date', rangeslider: {{visible: true}}}},
                    yaxis: {{title: 'Balance ($)', type: useLog ? 'log' : 'linear'}},
                    height: 500, hovermode: 'x unified',
                    shapes: [{{
                        type: 'line', x0: eqDates[0], x1: eqDates[eqDates.length-1],
                        y0: initBalance, y1: initBalance,
                        line: {{color: 'gray', width: 1, dash: 'dash'}},
                    }}],
                }});

                // Update drawdown chart
                Plotly.react(ddDiv, [{{
                    x: ddDates, y: ddVals, type: 'scatter', mode: 'lines',
                    name: 'Drawdown', fill: 'tozeroy',
                    line: {{color: '#C62828', width: 1}},
                    fillcolor: 'rgba(198,40,40,0.2)',
                }}], {{
                    title: 'Drawdown',
                    xaxis: {{title: 'Date', rangeslider: {{visible: true}}}},
                    yaxis: {{title: 'Drawdown (%)'}},
                    height: 350, hovermode: 'x unified',
                }});

                // Stats
                const totalTrades = trades.length;
                const wins = trades.filter(t => t.win).length;
                const losses = totalTrades - wins;
                const winRate = totalTrades > 0 ? (wins / totalTrades * 100).toFixed(1) : '0.0';
                const sumWins = trades.filter(t => t.pnl > 0).reduce((a, t) => a + t.pnl, 0);
                const sumLosses = Math.abs(trades.filter(t => t.pnl <= 0).reduce((a, t) => a + t.pnl, 0));
                const profitFactor = sumLosses > 0 ? (sumWins / sumLosses).toFixed(2) : (sumWins > 0 ? '\\u221e' : '0.00');
                const avgWin = wins > 0 ? (sumWins / wins).toFixed(2) : '0.00';
                const avgLoss = losses > 0 ? (sumLosses / losses).toFixed(2) : '0.00';
                // Avg win/loss as % of sizing balance (initBalance for fixed, varies for compound)
                const avgWinPct = wins > 0 ? (sumWins / wins / initBalance * 100).toFixed(2) : '0.00';
                const avgLossPct = losses > 0 ? (sumLosses / losses / initBalance * 100).toFixed(2) : '0.00';
                const totalReturn = ((balance - initBalance) / initBalance * 100).toFixed(2);
                const sizingLabel = P.compound ? 'Compounding' : 'Fixed';

                document.getElementById('pnlStats').innerHTML =
                    '<div style="width:100%"><b>Mode:</b> ' + sizingLabel +
                    ' &nbsp;|&nbsp; <b>Signals:</b> ' + signalCount + '</div>' +
                    '<b>Trades:</b> ' + totalTrades +
                    ' &nbsp;|&nbsp; <b>Win rate:</b> ' + winRate + '%' +
                    ' (' + wins + 'W / ' + losses + 'L)' +
                    ' &nbsp;|&nbsp; <b>Profit factor:</b> ' + profitFactor +
                    ' &nbsp;|&nbsp; <b>Avg win:</b> $' + avgWin +
                    ' (' + avgWinPct + '%)' +
                    ' &nbsp;|&nbsp; <b>Avg loss:</b> $' + avgLoss +
                    ' (' + avgLossPct + '%)' +
                    ' &nbsp;|&nbsp; <b>Max DD:</b> ' + maxDD.toFixed(2) + '%' +
                    ' &nbsp;|&nbsp; <b>Max leverage:</b> ' + maxLevUsed.toFixed(1) + 'x' +
                    ' &nbsp;|&nbsp; <b>Final:</b> $' + balance.toFixed(2) +
                    ' (' + (totalReturn >= 0 ? '+' : '') + totalReturn + '%)';
            }}

            // Event listeners
            const slider = document.getElementById('pnlConfSlider');
            const sliderVal = document.getElementById('pnlConfVal');
            slider.addEventListener('input', function() {{
                sliderVal.textContent = parseInt(this.value) + '%';
            }});

            document.getElementById('pnlRunBtn').addEventListener('click', runSimulation);

            // Also run on slider change (debounced)
            let debounceTimer;
            const inputs = ['pnlConfSlider', 'pnlBalance', 'pnlRisk', 'pnlFees', 'pnlSlippage', 'pnlMaxLev', 'pnlLogScale', 'pnlSizingMode'];
            inputs.forEach(id => {{
                document.getElementById(id).addEventListener('input', function() {{
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(runSimulation, 300);
                }});
            }});

            // Initial run
            runSimulation();
        }})();
        </script>
        '''

        return f'''
        <section id="pnl-simulation">
            <h2>PNL Simulation</h2>
            {controls_html}
            {div_equity}
            {div_dd}
            {js_engine}
        </section>
        '''

    def _plot_accuracy_timeline(self):
        """Rolling accuracy over time with colored zones, confidence filter, and rolling window slider."""
        if self.inference_only:
            return None

        # Define rolling windows to test
        windows = [50, 100, 200, 500, 1000]

        fig = go.Figure()

        # Calculate rolling accuracy for each window and each mode (All / High confidence)
        for window in windows:
            # ALL PREDICTIONS
            correct_series_all = pd.Series(
                (self.labels == self.predictions).astype(float),
                index=self.timestamps
            )
            rolling_correct_all = correct_series_all.rolling(window=window, min_periods=1).sum()
            rolling_count_all = pd.Series(1.0, index=self.timestamps).rolling(window=window, min_periods=1).sum()
            rolling_acc_all = rolling_correct_all / rolling_count_all

            # HIGH CONFIDENCE (top 10%)
            high_conf_mask = self.confidences >= self.high_conf_threshold
            correct_series_high = pd.Series(
                ((self.labels == self.predictions) & high_conf_mask).astype(float),
                index=self.timestamps
            )
            count_series_high = pd.Series(
                high_conf_mask.astype(float),
                index=self.timestamps
            )
            rolling_correct_high = correct_series_high.rolling(window=window, min_periods=1).sum()
            rolling_count_high = count_series_high.rolling(window=window, min_periods=1).sum()
            rolling_acc_high = rolling_correct_high / rolling_count_high.replace(0, np.nan)

            # Add traces (initially only window=100 visible)
            visible_all = (window == self.accuracy_window)
            visible_high = False  # High confidence starts hidden

            fig.add_trace(go.Scatter(
                x=self.timestamps,
                y=rolling_acc_all.values,
                mode='lines',
                name=f'All predictions (window={window})',
                line=dict(color='#636EFA', width=2),
                visible=visible_all,
                legendgroup=f'window_{window}',
            ))

            fig.add_trace(go.Scatter(
                x=self.timestamps,
                y=rolling_acc_high.values,
                mode='lines',
                name=f'High confidence top 10% (window={window})',
                line=dict(color='#00CC96', width=2),
                visible=visible_high,
                legendgroup=f'window_{window}',
            ))

        # Add reference lines and zones (adapted to number of classes)
        random_baseline = 1.0 / self.num_classes
        fig.add_hline(y=random_baseline, line_dash='dash', line_color='gray',
                      annotation_text=f'Random ({random_baseline:.1%})')

        # Zone thresholds: bad < random+margin, medium, good
        zone_mid = random_baseline + 0.1  # ~43% for 3-class, ~60% for 2-class
        zone_high = random_baseline + 0.2  # ~53% for 3-class, ~70% for 2-class
        fig.add_hrect(y0=0, y1=zone_mid, fillcolor='#EF553B', opacity=0.08,
                      line_width=0)
        fig.add_hrect(y0=zone_mid, y1=zone_high, fillcolor='#FFA15A', opacity=0.08,
                      line_width=0)
        fig.add_hrect(y0=zone_high, y1=1.0, fillcolor='#00CC96', opacity=0.08,
                      line_width=0)

        # Single button row: "All w=N" + "High w=N" — mutually exclusive
        default_window_idx = windows.index(self.accuracy_window)

        buttons = []
        for i, window in enumerate(windows):
            # All predictions button
            visibility = [False] * (len(windows) * 2)
            visibility[2*i] = True
            buttons.append(dict(
                label=f'All {window}',
                method='update',
                args=[{"visible": visibility}],
            ))

        for i, window in enumerate(windows):
            # High confidence button
            visibility = [False] * (len(windows) * 2)
            visibility[2*i + 1] = True
            buttons.append(dict(
                label=f'High {window}',
                method='update',
                args=[{"visible": visibility}],
            ))

        fig.update_layout(
            title='Accuracy Timeline',
            xaxis_title='Date',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            height=550,
            xaxis_rangeslider_visible=True,
            updatemenus=[
                dict(
                    type='buttons',
                    direction='left',
                    x=0.0, xanchor='left',
                    y=1.15, yanchor='top',
                    buttons=buttons,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='#C0C0C0',
                    borderwidth=1,
                    active=default_window_idx,
                ),
            ],
        )
        return fig

    def _plot_class_performance(self):
        """Grouped bar chart: 3 classes x 4 metrics."""
        if self.inference_only or self.metrics is None:
            return None

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']

        fig = go.Figure()

        for m_name, m_key in zip(metric_names, metric_keys):
            values = []
            for cls_idx in range(self.num_classes):
                val = self.metrics.get(f'{m_key}_class_{cls_idx}', 0)
                values.append(float(val))

            fig.add_trace(go.Bar(
                name=m_name,
                x=self.class_names,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='outside',
            ))

        fig.update_layout(
            title='Performance by Class',
            barmode='group',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1.15]),
            height=450,
        )
        return fig

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_html_report(self, output_path):
        """
        Generate a complete interactive HTML report.

        Args:
            output_path: Path to save the HTML file
        """
        logger.info(f"Generating HTML report: {output_path}")

        sections = []

        # Summary table
        sections.append(self._build_summary_section())

        # Plot 1+2: Candlestick + Oscillator (Lightweight Charts)
        sections.append(self._build_lightweight_charts_section())

        # Plot 3: Confusion Matrix (with confidence slider)
        cm_section = self._build_confusion_matrix_section()
        if cm_section is not None:
            sections.append(cm_section)

        # Plot 4: Confidence Distribution (with toggle buttons below)
        sections.append(self._build_confidence_distribution_section())

        # Plot 4b: High Confidence Zoom (with threshold slider + toggle)
        sections.append(self._build_high_confidence_zoom_section())

        # Plot 5: PNL Simulation (Equity + Drawdown) — skip if no trade directions
        if 'ATR_MULTIPLIER_TP' in self.config_params and self.trade_dir:
            sections.append(self._build_pnl_simulation_section())

        # Plot 6: Accuracy Timeline (Plotly)
        fig_acc = self._plot_accuracy_timeline()
        if fig_acc is not None:
            sections.append(
                self._figure_section('accuracy', 'Accuracy Timeline', fig_acc)
            )

        # Plot 7: Class Performance (Plotly)
        fig_cls = self._plot_class_performance()
        if fig_cls is not None:
            sections.append(
                self._figure_section('classperf', 'Class Performance', fig_cls)
            )

        # Build navigation
        nav_links = self._build_nav_links(fig_acc, fig_cls)

        # Assemble HTML
        html = self._assemble_html(nav_links, sections)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML report saved: {output_path}")

    def _build_summary_section(self):
        """Build the summary statistics table."""
        n = len(self.predictions)
        date_start = self.timestamps[0].strftime('%Y-%m-%d %H:%M')
        date_end = self.timestamps[-1].strftime('%Y-%m-%d %H:%M')

        counts = np.bincount(self.predictions, minlength=self.num_classes)
        dist_str = ' / '.join(
            f'{self.class_names[i]}: {counts[i]} ({counts[i]/n:.1%})'
            for i in range(self.num_classes)
        )

        rows = [
            ('Prediction target', self.prediction_target.upper()),
            ('Classes', ' / '.join(self.class_names)),
            ('Total predictions', str(n)),
            ('Date range', f'{date_start} &rarr; {date_end}'),
            ('Distribution', dist_str),
            ('Mean margin', f'{self.confidences.mean():.3f}'),
        ]

        if not self.inference_only and self.metrics is not None:
            rows.extend([
                ('Accuracy', f'{self.metrics.get("accuracy", 0):.4f}'),
                ('Balanced Accuracy', f'{self.metrics.get("balanced_accuracy", 0):.4f}'),
                ('F1 Macro', f'{self.metrics.get("f1_macro", 0):.4f}'),
            ])

        table_rows = ''.join(
            f'<tr><td><strong>{k}</strong></td><td>{v}</td></tr>' for k, v in rows
        )

        return f'''
        <section id="summary">
            <h2>Summary</h2>
            <table class="summary-table">
                {table_rows}
            </table>
        </section>
        '''

    def _figure_section(self, anchor_id, title, fig):
        """Wrap a Plotly figure in an HTML section."""
        div = fig.to_html(full_html=False, include_plotlyjs=False)
        return f'''
        <section id="{anchor_id}">
            <h2>{title}</h2>
            {div}
        </section>
        '''

    def _build_nav_links(self, fig_acc, fig_cls):
        """Build navigation link list based on available sections."""
        links = [
            ('summary', 'Summary'),
            ('candlestick', 'Candlestick'),
        ]
        if not self.inference_only:
            links.append(('confusion', 'Confusion Matrix'))
        links.append(('confidence', 'Margin Distribution'))
        links.append(('highconf', 'High Margin Zoom'))
        if 'ATR_MULTIPLIER_TP' in self.config_params and self.trade_dir:
            links.append(('pnl-simulation', 'PNL Simulation'))
        if fig_acc is not None:
            links.append(('accuracy', 'Accuracy Timeline'))
        if fig_cls is not None:
            links.append(('classperf', 'Class Performance'))

        items = ' | '.join(
            f'<a href="#{aid}">{label}</a>' for aid, label in links
        )
        return items

    def _assemble_html(self, nav_links, sections):
        """Assemble the final HTML document."""
        mode_label = 'Inference Only' if self.inference_only else 'Evaluation'
        body = '\n'.join(sections)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Report - {mode_label}</title>
    <script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
    <script src="https://cdn.plot.ly/plotly-{_plotlyjs_version()}.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f5f5; color: #333; }}
        nav {{ position: sticky; top: 0; z-index: 100; background: #1a1a2e; color: #eee;
               padding: 12px 24px; font-size: 14px; }}
        nav a {{ color: #82b1ff; text-decoration: none; }}
        nav a:hover {{ text-decoration: underline; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ margin: 20px 0; font-size: 24px; }}
        h2 {{ margin: 16px 0 8px; font-size: 20px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
        section {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .summary-table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
        .summary-table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
        .summary-table tr:last-child td {{ border-bottom: none; }}
    </style>
</head>
<body>
    <nav>{nav_links}</nav>
    <div class="container">
        <h1>Prediction Report &mdash; {mode_label}</h1>
        {body}
    </div>
</body>
</html>'''
