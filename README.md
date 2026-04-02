# Daikoku

**Deep learning cryptocurrency market direction prediction using Mamba (Selective State Space Model)**

![Tests](https://github.com/yannpointud/Daikoku/actions/workflows/tests.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.1-red)
![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-brightgreen)
![License MIT](https://img.shields.io/badge/License-MIT-green)

*[Version francaise](README_FR.md)*

| TensorBoard | Features Explorer | Evaluation Report | Live Dashboard |
|:-:|:-:|:-:|:-:|
| ![TensorBoard](docs/images/tensorboard.png) | ![Features](docs/images/features.png) | ![Evaluation](docs/images/evaluate1.png) | ![Live](docs/images/live1.png) |

<details>
<summary>More evaluation screenshots</summary>

| Confusion Matrix | Margin Distribution |
|:-:|:-:|
| ![Confusion Matrix](docs/images/evaluate2.png) | ![Margin Distribution](docs/images/evaluate3.png) |

| PNL Simulation | Accuracy Timeline |
|:-:|:-:|
| ![PNL Simulation](docs/images/evaluate4.png) | ![Accuracy Timeline](docs/images/evaluate5.png) |

</details>

---

## Overview

Daikoku is an end-to-end framework for cryptocurrency price direction prediction. It covers the complete lifecycle: **data preparation**, **model training**, **evaluation**, and **live inference** — all from a single centralized configuration.

The core model is built on the [Mamba](https://arxiv.org/abs/2312.00752) Selective State Space architecture, achieving linear-time sequence processing as an alternative to Transformers.

### Prediction Modes

Controlled by `PREDICTION_TARGET` in `config.py`:

| Mode          | Classes | Description                                                       |
|---------------|---------|-------------------------------------------------------------------|
| `triple`      | 3       | **Bull / Bear / Uncertain** — Triple Barrier with ATR-based TP/SL |
| `bull`        | 2       | Binary: Bull vs. not-Bull                                         |
| `bear`        | 2       | Binary: Bear vs. not-Bear                                         |
| `uncertain`   | 2       | Binary: Uncertain vs. directional                                 |
| `close1`      | 2       | Binary: next candle closes higher or lower                        |
| `close2`      | 2       | Binary: price higher or lower after 2 candles                     |
| `close3`      | 2       | Binary: price higher or lower after 3 candles                     |

The default `triple` mode uses Triple Barrier labeling with ATR-based dynamic take-profit and stop-loss levels (configurable multipliers). Each prediction defines a fixed risk per trade (1R = distance to stop-loss), and performance is evaluated in R-multiples, enabling consistent allocation regardless of volatility. The `bull`, `bear` and `uncertain` modes are binary variants of the same Triple Barrier labeling.

The `close*` modes provide simpler directional targets without explicit TP/SL levels: the number indicates how many candles ahead (1, 2 or 3) to determine whether the price closes higher or lower.

## Key Features

- **Mamba SSM architecture** — causal state-space model with linear complexity, optional dual-path input projection (Linear + CNN)
- **Multi-timeframe fusion** — dual-branch encoder processing two timeframes simultaneously, with 4 attention/fusion modes (off, pre_gate, post_gate, aligned)
- **Triple Barrier labeling** — ATR-based dynamic TP/SL levels computed on raw data before any transformation (no leakage)
- **23 engineered features** — log OHLCV, zigzag interpolation, structural S/R distances, volume profile, HMA regime filters, cyclical time encoding, with per-window median/IQR normalization
- **Optuna multi-objective optimization** — automated hyperparameter search maximizing directional accuracy while minimizing overfitting
- **Live inference** — real-time prediction connected to exchanges via ccxt, with virtual trade tracking, web dashboard, and deterministic reproducibility verification
- **Comprehensive evaluation** — interactive HTML reports, per-class metrics, high-confidence filtering, edge analysis in R-multiples
- **No look-ahead bias** — labels computed on raw data before any transformation, per-window features computed independently, causal architecture, verified by 6 dedicated tests (full-pipeline truncation, future-data mutation, multi-TF isolation, causal scan)
- **CPU fallback** — pure PyTorch implementation (`mamba_cpu.py`) with JIT-compiled selective scan when CUDA is not available

## Training Pipeline

```
╔════════════════════════════════════════════════════════════════════╗
║                        1. DATA PREPARATION                         ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  CSV (OHLCV)                                                       ║
║      │                                                             ║
║      ▼                                                             ║
║  Load & Validate                                                   ║
║      │                                                             ║
║      ├──────────────────────────────┐                              ║
║      ▼                              ▼                              ║
║  Triple Barrier Labels         Log Transform                       ║
║  (on RAW data)                 12 global features                  ║
║  Bear=0 / Uncertain=1 / Bull=2                                     ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                          2. WINDOWING                              ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Sliding Windows (WINDOW_SIZE candles)                             ║
║      │                                                             ║
║      ▼                                                             ║
║  Per-Window Features (zigzag + 5 structural)                       ║
║      │                                                             ║
║      ▼                                                             ║
║  Median/IQR Normalization (first 7 features)                       ║
║      │                                                             ║
║      ▼                                                             ║
║  23 features per window                                            ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                   3. MULTI-TIMEFRAME (optional)                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ┌─────────────────────┐         ┌──────────────────────────┐      ║
║  │  Primary Branch     │         │  Secondary Branch        │      ║
║  │  23 feat (with time)│         │  19 feat (no time)       │      ║
║  │  original TF (1h)   │         │  aggregated TF (4h)      │      ║
║  └────────┬────────────┘         └────────────┬─────────────┘      ║
║           │                                   │                    ║
║           ▼                                   ▼                    ║
╠════════════════════════════════════════════════════════════════════╣
║                           4. MODEL                                 ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ┌─────────────────────┐         ┌──────────────────────────┐      ║
║  │  Input Projection   │         │  Input Projection        │      ║
║  │  Linear (+CNN opt.) │         │  Linear                  │      ║
║  └────────┬────────────┘         └────────────┬─────────────┘      ║
║           │                                   │                    ║
║           ▼                                   ▼                    ║
║  ┌─────────────────────┐         ┌──────────────────────────┐      ║
║  │  Mamba Encoder      │         │  Mamba Encoder           │      ║
║  │  N_LAYERS blocks    │         │  MTF_N_LAYERS blocks     │      ║
║  │  (causal SSM)       │         │  (causal SSM)            │      ║
║  └────────┬────────────┘         └────────────┬─────────────┘      ║
║           │                                   │                    ║
║           └──────────────┬────────────────────┘                    ║
║                          │                                         ║
║                          ▼                                         ║
╠════════════════════════════════════════════════════════════════════╣
║                     5. FUSION & HEAD                               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║                    Attention Mode ?                                ║
║                          │                                         ║
║       ┌──────┬───────────┼───────────┐                             ║
║       ▼      ▼           ▼           ▼                             ║
║     off   pre_gate   post_gate   aligned                           ║
║       │      │           │           │                             ║
║       ▼      ▼           ▼           ▼                             ║
║    Unified  Unified   Unified   AlignedFusion                      ║
║    Head     Head      Head      (gated residual)                   ║
║    2 paths  4 paths   3 paths   → MLP head                         ║
║             +SelfAttn +CrossAttn                                   ║
║       │      │           │           │                             ║
║       └──────┴───────────┴───────────┘                             ║
║                          │                                         ║
║                          ▼                                         ║
║                  Logits (batch, 3)                                 ║
║                  Bear / Uncertain / Bull                           ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                        6. TRAINING                                 ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Logits ──► UnifiedLoss (Focal + Class Weights) ◄── Labels         ║
║                          │                                         ║
║                          ▼                                         ║
║              AdamW + LR Scheduler (cosine/wsd)                     ║
║                          │                                         ║
║                          ▼                                         ║
║              Checkpoint (best balanced_accuracy)                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yannpointud/Daikoku.git
cd Daikoku

# 2. Install (CUDA 12.4 required)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install mamba-ssm==2.3.0 causal-conv1d==1.5.2

# 3. Download data
python modules/tools/download_candles.py --exchange binance --pair BTC/USDT --timeframe 1h --start 2020-01-01 --output data/BTC_1h_bin.csv

# 4. Train
python main.py

# 5. Evaluate
python evaluate.py

# 6. Monitor
tensorboard --logdir runs/
```

All parameters are configured in a single file: `config.py`. For detailed installation instructions, configuration options, and all operational modes, see the **[Guide](docs/Guide.md)**.

## Operational Modes

- **`main.py`** — Trains the model on configured historical data. Produces a checkpoint (`models/best_model.pt`) and TensorBoard metrics.
- **`optimize.py`** — Automated hyperparameter search via Optuna (multi-objective). Each trial trains a full model and results are compared to find the best configuration.
- **`evaluate.py`** — Evaluates a checkpoint on any dataset (including data never seen during training) and generates an interactive HTML report with per-class metrics, confidence analysis and edge in R-multiples.
- **`inference.py`** — Connects the model to a live exchange via ccxt to verify predictions on real-time data, with virtual trade tracking and web dashboard.

## Project Structure

```
Daikoku/
├── main.py          # Training pipeline
├── evaluate.py      # Offline evaluation with HTML reports
├── inference.py     # Live inference with exchange feed and web dashboard
├── optimize.py      # Multi-objective hyperparameter search (Optuna)
├── config.py        # Single source of truth for all parameters
│
├── modules/
│   ├── model/       # Mamba architecture (GPU + CPU fallback)
│   ├── data/        # Loading, labeling, transform, dataset, multi-TF
│   ├── training/    # Trainer, loss, metrics, monitoring
│   ├── inference/   # Prediction engine, exchange feed, trade tracker, dashboard
│   └── evaluation/  # Model evaluator, HTML report generation
│
└── tests/           # Test suite (pytest)
    └── data/        # Test dataset (included)
```

For the complete architecture breakdown (model internals, data pipeline, training system, metrics, live inference), see **[Architecture.md](docs/Architecture.md)**.

## Documentation

| Document                                        | Description                                        |
|-------------------------------------------------|----------------------------------------------------|
| [Architecture](docs/Architecture.md)            | Technical architecture — model, pipeline, training  |
| [Guide](docs/Guide.md)                          | Installation, configuration, all operational modes  |
| [Metrics TB](docs/Metrics_TB.md)                | TensorBoard metrics reference (100+)                |
| [Contributing](CONTRIBUTING.md)                 | How to contribute                                   |

## Tech Stack

Python 3.11 · PyTorch 2.4.1 · Mamba SSM · CUDA 12.4 · Optuna · ccxt · TensorBoard · Plotly

## Disclaimer

**This software is for educational and research purposes only. It is not financial advice.**

- Cryptocurrency trading involves **substantial risk of loss** — you may lose part or all of your capital
- Past performance does **not** guarantee future results
- Machine learning models can and will produce incorrect predictions
- The authors assume **no responsibility** for any trading decisions or financial losses
- **Use at your own risk**

For the full disclaimer, see **[DISCLAIMER.md](docs/DISCLAIMER.md)**.

## License

[MIT](LICENSE)
