# Standalone Tools

Reusable tools for the Daikoku project. All Python tools are invoked via `python -m modules.tools.<name>`.

## Data

`download_candles.py`
Multi-exchange OHLCV downloader. Supports Binance, Kraken, Bybit.
Usage: `python -m modules.tools.download_candles [--exchange binance --timeframe 1h --start 2024-01-01]`

`split_test_periods.py`
Split downloaded CSV files into TestA/TestB/TestC date ranges (staggered ends, shared start). Reads from data/download/, writes to data/TestA/, data/TestB/, data/TestC/.
Usage: `python -m modules.tools.split_test_periods`

`augment_data.py`
Data augmentation via Ornstein-Uhlenbeck process. Generates N variant CSVs from original OHLCV.
Usage: `python -m modules.tools.augment_data [--file data/ETH_1h_bin.csv] [--variants 4] [--pct 5]`

`analyze_labels.py`
Label distribution + both-hit analysis for any dataset with configurable labeling parameters.
Usage: `python -m modules.tools.analyze_labels data/BTC_1h_agg.csv [--atr-mult 1.5]`

## Training

`run_training.sh`
Bash wrapper to launch main.py with safety checks (prevent duplicate runs, auto-logging). Determines next run number automatically.
Usage: `bash modules/tools/run_training.sh`

`extract_metrics.py`
Extract TensorBoard metrics in compact tabular format. Samples epochs linearly for trend visibility. Two-phase workflow: essential metrics first, then drill-down on demand.
Usage: `python -m modules.tools.extract_metrics run_003 [--metrics Loss/train,Loss/test] [--list] [--all] [--epochs 0,5,10]`

## Evaluation

`eval_per_file.py`
Evaluate model on each INPUT_FILE separately. Reports winrate, profit factor, edge/trade, and trade count on test split. Supports fee deduction in R-multiples. Uses TRADE_DIRECTION for all prediction targets.
Usage: `python -m modules.tools.eval_per_file [--fee 0.001]`

## Visualization

`visualize_data.py`
OHLCV candlestick + volume viewer for CSV triage. Shows primary + secondary TF with synced scroll. Only displays candles the model sees (after WINDOW_SIZE warmup).
Usage: `python -m modules.tools.visualize_data data/Set/*.csv [--output-dir evaluation/]`

`visualize_augment.py`
Side-by-side original vs augmented OHLCV charts using Lightweight Charts v4.
Usage: `python -m modules.tools.visualize_augment [--file data/ETH_1h_bin.csv]`

## Analysis

`tsne_analysis.py`
t-SNE visualization of raw features and model embeddings. Projects high-dimensional data to 2D to assess class separability. Supports raw-only mode (no checkpoint) and embedding mode (with checkpoint).
Usage: `python -m modules.tools.tsne_analysis --split test [--checkpoint models/latest.pt] [--perplexity 30] [--workers 6]`

## Baselines

`xgboost_baseline.py`
XGBoost baseline for feature informativeness validation. Tests whether features contain predictive signal using gradient boosting (architecture-agnostic).
Usage: `python -m modules.tools.xgboost_baseline`

`mlp_cascade_baseline.py`
MLP Cascade-Correlation baseline. Tests whether a growing MLP rivals Mamba on the same data.
Usage: `python -m modules.tools.mlp_cascade_baseline`

`build_etf.py`
Build a synthetic ETF OHLCV from multiple altcoin CSVs (equal-weight portfolio).
Usage: `python -m modules.tools.build_etf`
