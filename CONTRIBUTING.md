# Contributing to Daikoku

*[Version francaise](CONTRIBUTING_FR.md)*

Thank you for your interest in contributing to Daikoku! This document explains how to get involved.

---

## Reporting Bugs

Open a [GitHub Issue](../../issues) with:

- **Python version**, **PyTorch version**, **GPU model** (or CPU-only)
- **Steps to reproduce** the problem
- **Full error traceback**
- **Config changes** (if any) from the defaults in `config.py`

## Suggesting Features

Open a [GitHub Issue](../../issues) labeled "feature request" **before writing code**. Describe:

- What problem does it solve?
- How does it fit with the existing architecture?

This avoids wasted effort on features that don't align with the project direction.

## Submitting Code

### Workflow

1. **Fork** the repository
2. **Create a feature branch** from `master`: `git checkout -b feature/my-feature`
3. **Make your changes** — keep commits focused and descriptive
4. **Run the tests**: `python -m pytest tests/ -v`
5. **Open a Pull Request** against `master` with a clear description

### Code Conventions

- **Comments in English**, always
- **Configuration**: all parameters live in `config.py` — never use `getattr(config, 'X', default)` fallbacks
- **No backward compatibility shims**: breaking changes to checkpoints or config are acceptable
- **Parity**: `main.py` and `optimize.py` must produce identical results for the same config — any change to one must be reflected in the other
- **Labels on raw data**: Triple Barrier labeling is always computed before any data transformation (no leakage)
- **Tests must pass**: never modify a test just to make it pass — fix the source code instead

### What We Accept

- Bug fixes with clear reproduction steps
- Performance improvements backed by benchmarks
- New features that fit the Mamba-based crypto prediction scope
- Documentation improvements

### What We Don't Accept

- Features outside the project scope (e.g., non-crypto assets, non-Mamba architectures)
- Changes that break existing tests without justification

## Development Setup

Follow the full installation instructions in the [Guide](docs/Guide.md#installation). In short:

1. **CUDA 12.4+** — required for GPU training. Install from [NVIDIA](https://developer.nvidia.com/cuda-toolkit)
2. **Python 3.11**
3. **PyTorch 2.4.1** with CUDA support:
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Mamba CUDA kernels** (strongly recommended):
   ```bash
   pip install mamba-ssm==2.3.0 causal-conv1d==1.5.2
   ```
6. **Run tests**:
   ```bash
   python -m pytest tests/ -v
   ```

The test suite requires `tests/data/test_Dataset.csv` (included in the repository). See [Guide](docs/Guide.md#tests) for details.

## Project Structure

- `config.py` — single source of truth for all parameters
- `main.py` — training pipeline
- `optimize.py` — hyperparameter optimization (must stay in sync with `main.py`)
- `evaluate.py` — offline evaluation
- `inference.py` — live inference with exchange connection
- `modules/` — all implementation code
- `tests/` — test suite (pytest)

For full technical details, see [Architecture.md](docs/Architecture.md).
