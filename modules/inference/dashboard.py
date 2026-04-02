"""
Live dashboard for inference monitoring.

Serves an HTML page with:
- Candlestick chart (Lightweight Charts v4)
- Prediction markers (Bull/Bear with confidence)
- Trade zones (TP/SL areas)
- Stats panel (winrate, counts, last prediction)
- Alert banner (divergence warnings)
- Confidence slider filter
- Auto-refresh every 10 seconds via /data.json
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd

from modules.data.labeling import TRADE_DIRECTION
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _to_epoch(ts):
    """Convert any timestamp type to int epoch seconds."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp())
    return int(pd.Timestamp(ts).timestamp())


class _DashboardState:
    """Shared state between the HTTP handler and the main loop."""

    def __init__(self, pair, timeframe, num_classes=3, prediction_target='triple'):
        self.pair = pair
        self.timeframe = timeframe
        self.prediction_target = prediction_target
        self.trade_direction_map = TRADE_DIRECTION.get(prediction_target, {})
        self.candles = []       # list of {time, open, high, low, close}
        self.markers = []       # list of {time, position, color, shape, text, size, conf}
        self.trades = []        # list of closed trade dicts
        self.open_trades = []   # list of open trade dicts
        self.stats = {}
        self.alerts = []
        self.compare_details = []
        self.compare_ok = 0
        self.compare_errors = 0
        self._lock = threading.Lock()

    def update(self, candle_df, predictions, closed_trades, open_trades, stats, alerts=None, live_candle=None, compare_details=None, compare_state=None):
        """
        Update dashboard state from main loop.

        Args:
            candle_df: DataFrame with Open time, Open, High, Low, Close
            predictions: list of dicts with keys: timestamp, pred, confidence, class_name
            closed_trades: list of trade dicts from tracker
            open_trades: list of open trade dicts from tracker
            stats: dict from tracker.stats
            alerts: list of alert strings
            live_candle: dict with current open candle (display only, not in buffer)
            compare_details: list of error detail strings from comparison
            compare_state: dict with total_ok, total_errors counts
        """
        with self._lock:
            # Convert candles to Lightweight Charts format
            self.candles = []
            for _, row in candle_df.iterrows():
                epoch = _to_epoch(row['Open time'])
                self.candles.append({
                    'time': epoch,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                })

            # Append live (unclosed) candle if provided
            if live_candle:
                epoch = _to_epoch(live_candle['timestamp'])
                self.candles.append({
                    'time': epoch,
                    'open': live_candle['open'],
                    'high': live_candle['high'],
                    'low': live_candle['low'],
                    'close': live_candle['close'],
                })

            # Convert predictions to markers
            self.markers = []
            for p in predictions:
                epoch = _to_epoch(p['timestamp'])
                pred_class = p['pred']
                direction = self.trade_direction_map.get(pred_class)
                class_name = p.get('class_name', f'Class_{pred_class}')

                if direction == 'long':
                    self.markers.append({
                        'time': epoch,
                        'position': 'belowBar',
                        'color': '#26a69a',
                        'shape': 'arrowUp',
                        'text': f"L {p['confidence']:.0%}",
                        'size': 1 if p['confidence'] < 0.6 else 2,
                        'conf': p['confidence'],
                        'directional': True,
                        'class_name': class_name,
                    })
                elif direction == 'short':
                    self.markers.append({
                        'time': epoch,
                        'position': 'aboveBar',
                        'color': '#ef5350',
                        'shape': 'arrowDown',
                        'text': f"S {p['confidence']:.0%}",
                        'size': 1 if p['confidence'] < 0.6 else 2,
                        'conf': p['confidence'],
                        'directional': True,
                        'class_name': class_name,
                    })
                else:
                    # Non-directional prediction (rejected class)
                    self.markers.append({
                        'time': epoch,
                        'position': 'aboveBar',
                        'color': '#ffa15a',
                        'shape': 'circle',
                        'text': f"- {p['confidence']:.0%}",
                        'size': 1,
                        'conf': p['confidence'],
                        'directional': False,
                        'class_name': class_name,
                    })

            # Trade zones
            self.trades = []
            for t in closed_trades:
                entry_ts = t.get('entry_ts')
                close_ts = t.get('close_ts')
                if entry_ts and close_ts:
                    e_epoch = _to_epoch(entry_ts)
                    c_epoch = _to_epoch(close_ts)
                    self.trades.append({
                        'et': e_epoch,
                        'ct': c_epoch,
                        'ep': t['entry_price'],
                        'tp': t['tp_level'],
                        'sl': t['sl_level'],
                        'dir': t['direction'],
                        'out': t['outcome'],
                        'conf': t.get('confidence', 0),
                    })

            self.open_trades = []
            for t in open_trades:
                entry_ts = t.get('entry_ts')
                if entry_ts:
                    e_epoch = _to_epoch(entry_ts)
                    self.open_trades.append({
                        'et': e_epoch,
                        'ep': t['entry_price'],
                        'tp': t['tp_level'],
                        'sl': t['sl_level'],
                        'dir': t['direction'],
                        'conf': t.get('confidence', 0),
                    })

            self.stats = stats
            self.alerts = alerts or []
            if compare_details is not None:
                self.compare_details = compare_details
            if compare_state is not None:
                self.compare_ok = compare_state.get('total_ok', 0)
                self.compare_errors = compare_state.get('total_errors', 0)

    def get_json(self):
        """Return serializable state dict."""
        with self._lock:
            return {
                'candles': self.candles,
                'markers': self.markers,
                'trades': self.trades,
                'open_trades': self.open_trades,
                'stats': self.stats,
                'alerts': self.alerts,
                'compare_details': self.compare_details,
                'compare_ok': self.compare_ok,
                'compare_errors': self.compare_errors,
                'pair': self.pair,
                'timeframe': self.timeframe,
                'prediction_target': self.prediction_target,
            }


# Global reference to state (set by LiveDashboard)
_state = None


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        # Suppress default access logs
        pass

    def do_GET(self):
        if self.path == '/data.json':
            self._serve_json()
        elif self.path == '/' or self.path == '/index.html':
            self._serve_html()
        else:
            self.send_error(404)

    def _serve_json(self):
        data = _state.get_json() if _state else {}
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = _build_html()
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


class LiveDashboard:
    """
    HTTP server serving a live TradingView-style dashboard.

    Usage:
        dashboard = LiveDashboard(port=7777, pair='BTC/USDT', timeframe='15m')
        dashboard.start()
        # ... in main loop:
        dashboard.update(candles_df, predictions, trades, stats, alerts)
        # ... on shutdown:
        dashboard.stop()
    """

    def __init__(self, port, pair, timeframe, num_classes=3, prediction_target='triple'):
        global _state
        self.port = port
        _state = _DashboardState(pair, timeframe, num_classes=num_classes,
                                 prediction_target=prediction_target)
        self.state = _state
        self.server = None
        self.thread = None

    def start(self):
        """Start HTTP server in a daemon thread."""
        self.server = HTTPServer(('127.0.0.1', self.port), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Dashboard started: http://localhost:{self.port}")

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            logger.info("Dashboard stopped")

    def update(self, candle_df, predictions, closed_trades, open_trades, stats, alerts=None, live_candle=None, compare_details=None, compare_state=None):
        """Update dashboard data (thread-safe)."""
        self.state.update(candle_df, predictions, closed_trades, open_trades, stats, alerts, live_candle, compare_details, compare_state)


def _build_html():
    """Generate the full HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Inference Dashboard</title>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { display: flex; justify-content: space-between; align-items: center;
            padding: 12px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }
  .header h1 { font-size: 18px; color: #e94560; }
  .header .info { font-size: 13px; color: #999; }
  .alert-banner { display: none; padding: 8px 20px; background: #e94560;
                  color: #fff; font-size: 13px; text-align: center; cursor: pointer; }
  .alert-banner.visible { display: block; }
  .compare-details { display: none; max-height: 200px; overflow-y: auto;
                     padding: 8px 20px; background: #1e1e2e; border-bottom: 1px solid #e94560;
                     font-family: monospace; font-size: 12px; color: #ef5350; }
  .compare-details.visible { display: block; }
  .compare-details div { padding: 2px 0; }
  #chart-container { width: 100%; height: 500px; position: relative; }
  .controls { display: flex; gap: 16px; align-items: center; padding: 10px 20px;
              background: #16213e; border-top: 1px solid #0f3460; }
  .controls label { font-size: 13px; color: #999; }
  .controls input[type=range] { width: 200px; }
  .controls .slider-val { font-size: 13px; color: #e94560; min-width: 40px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 12px; padding: 16px 20px; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 14px;
               border: 1px solid #0f3460; }
  .stat-card .label { font-size: 11px; color: #999; text-transform: uppercase; margin-bottom: 4px; }
  .stat-card .value { font-size: 22px; font-weight: 600; }
  .stat-card .value.green { color: #26a69a; }
  .stat-card .value.red { color: #ef5350; }
  .stat-card .value.neutral { color: #e0e0e0; }
  .stat-card .sub { font-size: 11px; color: #666; margin-top: 4px; }
  .last-pred { padding: 12px 20px; background: #16213e; border-top: 1px solid #0f3460;
               font-size: 14px; }
  .last-pred .long { color: #26a69a; }
  .last-pred .short { color: #ef5350; }
  .last-pred .neutral { color: #ffa15a; }
  .footer { padding: 8px 20px; text-align: center; font-size: 11px; color: #555; }
</style>
</head>
<body>

<div class="header">
  <h1 id="title">Live Inference</h1>
  <div class="info" id="status">Loading...</div>
</div>

<div class="alert-banner" id="alert-banner" onclick="toggleCompareDetails()"></div>
<div class="compare-details" id="compare-details"></div>

<div id="chart-container"></div>

<div class="controls">
  <label>Min confidence:</label>
  <input type="range" id="conf-slider" min="0" max="100" value="0" step="5">
  <span class="slider-val" id="conf-val">0%</span>
  <label style="margin-left:20px">
    <input type="checkbox" id="show-all-preds"> Show all predictions
  </label>
  <label style="margin-left:20px">
    <input type="checkbox" id="show-trades" checked> Show trades
  </label>
</div>

<div class="stats-grid" id="stats-grid"></div>

<div class="last-pred" id="last-pred"></div>

<div class="footer">Auto-refresh every 10s | Daikoku Live Inference</div>

<script>
const container = document.getElementById('chart-container');
const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 500,
    layout: {
        background: { type: 'solid', color: '#1a1a2e' },
        textColor: '#e0e0e0',
    },
    grid: {
        vertLines: { color: '#1e2d4a' },
        horzLines: { color: '#1e2d4a' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#0f3460',
    },
    rightPriceScale: { borderColor: '#0f3460' },
});

const candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderUpColor: '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
});

// Trade overlay canvas
const overlay = document.createElement('canvas');
overlay.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;';
container.appendChild(overlay);
const ctx = overlay.getContext('2d');

let allMarkers = [];
let allTrades = [];
let allOpenTrades = [];
let minConf = 0;
let showTrades = true;
let showAllPreds = false;
let _predictionTarget = 'triple';

const slider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
const showTradesBox = document.getElementById('show-trades');
const showAllPredsBox = document.getElementById('show-all-preds');

slider.addEventListener('input', () => {
    minConf = parseInt(slider.value) / 100;
    confVal.textContent = slider.value + '%';
    applyFilter();
});

showTradesBox.addEventListener('change', () => {
    showTrades = showTradesBox.checked;
    requestTradeRedraw();
});

showAllPredsBox.addEventListener('change', () => {
    showAllPreds = showAllPredsBox.checked;
    applyFilter();
});

let _lastCompareOk = 0, _lastCompareErrors = 0;

function applyFilter(cmpOk, cmpErrors) {
    if (cmpOk !== undefined) { _lastCompareOk = cmpOk; _lastCompareErrors = cmpErrors; }
    const filtered = allMarkers.filter(m => m.conf >= minConf && (showAllPreds || m.directional));
    candleSeries.setMarkers(filtered.map(m => ({
        time: m.time, position: m.position, color: m.color,
        shape: m.shape, text: m.text, size: m.size,
    })));
    const stats = computeFilteredStats(allTrades, allOpenTrades, minConf);
    updateStats(stats, _lastCompareOk, _lastCompareErrors);
    requestTradeRedraw();
}

let rafId = 0;
function requestTradeRedraw() {
    if (rafId) return;
    rafId = requestAnimationFrame(() => { rafId = 0; drawTradeZones(); });
}

function drawTradeZones() {
    const dpr = window.devicePixelRatio || 1;
    overlay.width = container.clientWidth * dpr;
    overlay.height = container.clientHeight * dpr;
    overlay.style.width = container.clientWidth + 'px';
    overlay.style.height = container.clientHeight + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, container.clientWidth, container.clientHeight);

    if (!showTrades) return;

    const ts = chart.timeScale();

    const filteredTrades = allTrades.filter(t => t.conf >= minConf);
    const filteredOpen = allOpenTrades.filter(t => t.conf >= minConf);

    // Draw closed trades
    filteredTrades.forEach(t => {
        const x1 = ts.timeToCoordinate(t.et);
        const x2 = ts.timeToCoordinate(t.ct);
        if (x1 === null || x2 === null) return;

        const yEp = candleSeries.priceToCoordinate(t.ep);
        if (yEp === null) return;

        // closeN mode: tp=0 and sl=0 → draw entry line only (no TP/SL zones)
        const hasBarriers = t.tp !== 0 || t.sl !== 0;

        if (hasBarriers) {
            const yTp = candleSeries.priceToCoordinate(t.tp);
            const ySl = candleSeries.priceToCoordinate(t.sl);
            if (yTp === null || ySl === null) return;

            // TP zone (green)
            ctx.fillStyle = t.out === 'tp' ? 'rgba(38,166,154,0.15)' : 'rgba(38,166,154,0.08)';
            const tpTop = Math.min(yTp, yEp);
            const tpH = Math.abs(yTp - yEp);
            ctx.fillRect(x1, tpTop, x2 - x1, tpH);

            // SL zone (red)
            ctx.fillStyle = t.out === 'sl' ? 'rgba(239,83,80,0.15)' : 'rgba(239,83,80,0.08)';
            const slTop = Math.min(ySl, yEp);
            const slH = Math.abs(ySl - yEp);
            ctx.fillRect(x1, slTop, x2 - x1, slH);
        }

        // Entry line (+ outcome color for closeN)
        const lineColor = hasBarriers ? '#ffffff44'
            : (t.out === 'tp' ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)');
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = hasBarriers ? 1 : 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x1, yEp);
        ctx.lineTo(x2, yEp);
        ctx.stroke();
        ctx.setLineDash([]);
    });

    // Draw open trades (ongoing)
    const rightEdge = container.clientWidth;
    filteredOpen.forEach(t => {
        const x1 = ts.timeToCoordinate(t.et);
        if (x1 === null) return;

        const yEp = candleSeries.priceToCoordinate(t.ep);
        if (yEp === null) return;

        const hasBarriers = t.tp !== 0 || t.sl !== 0;

        if (hasBarriers) {
            const yTp = candleSeries.priceToCoordinate(t.tp);
            const ySl = candleSeries.priceToCoordinate(t.sl);
            if (yTp === null || ySl === null) return;

            // TP zone
            ctx.fillStyle = 'rgba(38,166,154,0.12)';
            const tpTop = Math.min(yTp, yEp);
            ctx.fillRect(x1, tpTop, rightEdge - x1, Math.abs(yTp - yEp));

            // SL zone
            ctx.fillStyle = 'rgba(239,83,80,0.12)';
            const slTop = Math.min(ySl, yEp);
            ctx.fillRect(x1, slTop, rightEdge - x1, Math.abs(ySl - yEp));
        }

        // Entry dashed
        ctx.strokeStyle = '#ffffff66';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, yEp);
        ctx.lineTo(rightEdge, yEp);
        ctx.stroke();
        ctx.setLineDash([]);
    });
}

chart.timeScale().subscribeVisibleLogicalRangeChange(requestTradeRedraw);

function computeFilteredStats(trades, openTrades, minConf) {
    const filtered = trades.filter(t => t.conf >= minConf);
    const filteredOpen = openTrades.filter(t => t.conf >= minConf);

    const wins = filtered.filter(t => t.out === 'tp').length;
    const losses = filtered.filter(t => t.out === 'sl').length;
    const timeouts = filtered.filter(t => t.out === 'timeout').length;
    const total = filtered.length;
    const resolved = wins + losses;

    const longTrades = filtered.filter(t => t.dir === 'long');
    const shortTrades = filtered.filter(t => t.dir === 'short');

    return {
        total, wins, losses, timeouts,
        winrate: resolved > 0 ? (wins / resolved * 100) : 0,
        open: filteredOpen.length,
        bull_total: longTrades.length,
        bull_wins: longTrades.filter(t => t.out === 'tp').length,
        bear_total: shortTrades.length,
        bear_wins: shortTrades.filter(t => t.out === 'tp').length,
    };
}

function updateStats(stats, compareOk, compareErrors) {
    const grid = document.getElementById('stats-grid');
    if (!stats || !stats.total && stats.total !== 0) { grid.innerHTML = ''; return; }

    const wr = stats.winrate || 0;
    const wrClass = wr >= 55 ? 'green' : (wr < 45 ? 'red' : 'neutral');

    const cmpTotal = (compareOk || 0) + (compareErrors || 0);
    const cmpClass = cmpTotal === 0 ? 'neutral' : (compareErrors > 0 ? 'red' : 'green');
    const cmpText = cmpTotal === 0 ? '-' : `${compareOk}/${cmpTotal}`;
    const cmpSub = cmpTotal === 0 ? 'waiting' : (compareErrors > 0 ? `${compareErrors} error(s)` : 'all match');

    // Adapt trade card labels based on prediction_target
    const target = _predictionTarget || 'triple';
    const isCloseN = target.startsWith('close');
    const longLabel = isCloseN ? 'Up Trades' : 'Long Trades';
    const shortLabel = isCloseN ? 'Down Trades' : 'Short Trades';

    let tradeCards = '';
    if (target === 'uncertain') {
        tradeCards = `
        <div class="stat-card">
            <div class="label">Trades</div>
            <div class="value neutral">-</div>
            <div class="sub">no trades in uncertain mode</div>
        </div>`;
    } else {
        tradeCards = `
        <div class="stat-card">
            <div class="label">${longLabel}</div>
            <div class="value green">${stats.bull_wins || 0}/${stats.bull_total || 0}</div>
            <div class="sub">wins/total</div>
        </div>
        <div class="stat-card">
            <div class="label">${shortLabel}</div>
            <div class="value red">${stats.bear_wins || 0}/${stats.bear_total || 0}</div>
            <div class="sub">wins/total</div>
        </div>`;
    }

    grid.innerHTML = `
        <div class="stat-card">
            <div class="label">Total Trades</div>
            <div class="value neutral">${stats.total}</div>
            <div class="sub">${stats.open || 0} open</div>
        </div>
        <div class="stat-card">
            <div class="label">Win Rate (excl timeout)</div>
            <div class="value ${wrClass}">${wr.toFixed(1)}%</div>
            <div class="sub">${stats.wins}W / ${stats.losses}L / ${stats.timeouts}T</div>
        </div>
        ${tradeCards}
        <div class="stat-card">
            <div class="label">Comparison</div>
            <div class="value ${cmpClass}">${cmpText}</div>
            <div class="sub">${cmpSub}</div>
        </div>
    `;
}

function updateLastPred(markers) {
    const el = document.getElementById('last-pred');
    if (!markers || markers.length === 0) {
        el.innerHTML = 'No predictions yet';
        return;
    }
    const last = markers[markers.length - 1];
    const dt = new Date(last.time * 1000).toISOString().replace('T', ' ').slice(0, 19);
    const cls = last.shape === 'arrowUp' ? 'long' : (last.shape === 'arrowDown' ? 'short' : 'neutral');
    const name = last.class_name || (last.shape === 'arrowUp' ? 'LONG' : (last.shape === 'arrowDown' ? 'SHORT' : 'REJECTED'));
    el.innerHTML = `Last prediction: <span class="${cls}">${name}</span> ` +
                   `(conf: ${(last.conf * 100).toFixed(1)}%) at ${dt} UTC`;
}

function updateAlerts(alerts, compareDetails) {
    const banner = document.getElementById('alert-banner');
    const detailsPanel = document.getElementById('compare-details');
    if (alerts && alerts.length > 0) {
        banner.textContent = alerts.join(' | ');
        banner.classList.add('visible');
    } else {
        banner.classList.remove('visible');
        detailsPanel.classList.remove('visible');
    }
    // Update compare details content
    if (compareDetails && compareDetails.length > 0) {
        detailsPanel.innerHTML = compareDetails.map(d => '<div>' + d + '</div>').join('');
    } else {
        detailsPanel.innerHTML = '';
        detailsPanel.classList.remove('visible');
    }
}

function toggleCompareDetails() {
    const panel = document.getElementById('compare-details');
    if (panel.innerHTML) {
        panel.classList.toggle('visible');
    }
}

async function refresh() {
    try {
        const resp = await fetch('/data.json');
        const data = await resp.json();

        _predictionTarget = data.prediction_target || 'triple';
        const targetLabel = data.prediction_target ? ` [${data.prediction_target}]` : '';
        document.getElementById('title').textContent =
            `Live Inference${targetLabel} - ${data.pair} ${data.timeframe}`;
        document.getElementById('status').textContent =
            `${data.candles.length} candles | ${data.markers.length} predictions | ` +
            new Date().toLocaleTimeString();

        if (data.candles.length > 0) {
            candleSeries.setData(data.candles);
        }

        allMarkers = data.markers || [];
        allTrades = data.trades || [];
        allOpenTrades = data.open_trades || [];

        applyFilter(data.compare_ok || 0, data.compare_errors || 0);
        updateLastPred(allMarkers);
        updateAlerts(data.alerts, data.compare_details);

    } catch (e) {
        document.getElementById('status').textContent = 'Fetch error: ' + e.message;
    }
}

// Initial load + auto-refresh
refresh();
setInterval(refresh, 10000);

// Resize handler
window.addEventListener('resize', () => {
    chart.applyOptions({ width: container.clientWidth });
    requestTradeRedraw();
});
</script>
</body>
</html>'''
