"""
Shared test fixtures.

Session-scoped fixture that trains a mini-model (2 epochs) on test_Dataset.csv
via main(), shared by all tests that need a valid checkpoint.
"""

import os
import sys
import pytest
from unittest.mock import patch as mock_patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from main import main as run_main

# Canonical path to the test dataset — import this in test files instead of hardcoding
TEST_CSV = os.path.join(os.path.dirname(__file__), "data", "test_Dataset.csv")


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory):
    """
    Train a 2-epoch model on test_Dataset.csv via main().
    Yields (checkpoint_path, csv_path).

    Using main() ensures the checkpoint contains the real config
    (no desync risk with hardcoded params).
    """
    if os.environ.get('CI'):
        pytest.skip("Skipped on CI (requires model training)")
    if not os.path.exists(TEST_CSV):
        pytest.skip(f"Test dataset not found: {TEST_CSV}")

    temp_dir = str(tmp_path_factory.mktemp("trained_ckpt"))
    temp_models = os.path.join(temp_dir, "models")
    temp_runs = os.path.join(temp_dir, "runs")
    temp_logs = os.path.join(temp_dir, "logs")
    os.makedirs(temp_models, exist_ok=True)
    os.makedirs(temp_runs, exist_ok=True)
    os.makedirs(temp_logs, exist_ok=True)

    # Save originals
    originals = {
        attr: getattr(config, attr)
        for attr in [
            "INPUT_FILES",
            "EPOCHS",
            "CHECKPOINT_DIR",
            "TENSORBOARD_DIR",
            "LOG_DIR",
            "SAVE_EVERY_N_EPOCHS",
            "NUM_WORKERS",
            "PERSISTENT_WORKERS",
        ]
    }

    try:
        config.INPUT_FILES = [TEST_CSV]
        config.EPOCHS = 1
        config.CHECKPOINT_DIR = temp_models
        config.TENSORBOARD_DIR = temp_runs
        config.LOG_DIR = temp_logs
        config.SAVE_EVERY_N_EPOCHS = 1
        config.NUM_WORKERS = 0
        config.PERSISTENT_WORKERS = False

        with mock_patch("sys.argv", ["main.py"]):
            run_main()

        ckpt = os.path.join(temp_models, "latest.pt")
        assert os.path.exists(ckpt), f"Checkpoint not created at {ckpt}"
        yield ckpt, TEST_CSV

    finally:
        for k, v in originals.items():
            setattr(config, k, v)
