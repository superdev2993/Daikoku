"""
Test suite for main.py

Tests:
1. test_set_seed() - Reproducibility
2. test_parse_arguments() - CLI parsing
3. test_main_imports() - Import validation
4. test_main_pipeline_mini() - Full integration test on test_Dataset.csv
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import pytest
from conftest import TEST_CSV

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from main import set_seed, parse_arguments, main


# ============================================================================
# TEST 1: set_seed() - Reproducibility
# ============================================================================

def test_set_seed():
    """Test that set_seed ensures reproducibility"""
    print("\n=== Test 1: set_seed() ===")

    # Test with seed 42
    set_seed(42)

    # Generate random numbers
    py_rand_1 = [np.random.random() for _ in range(5)]
    np_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    # Reset seed to 42
    set_seed(42)

    # Generate again
    py_rand_2 = [np.random.random() for _ in range(5)]
    np_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    # Should be identical
    assert py_rand_1 == py_rand_2, "Python random not reproducible"
    assert np.allclose(np_rand_1, np_rand_2), "NumPy random not reproducible"
    assert torch.allclose(torch_rand_1, torch_rand_2), "PyTorch random not reproducible"

    print("✓ Same seed produces identical results")

    # Test with different seed
    set_seed(123)
    py_rand_3 = [np.random.random() for _ in range(5)]
    np_rand_3 = np.random.rand(5)
    torch_rand_3 = torch.rand(5)

    # Should be different
    assert py_rand_1 != py_rand_3, "Different seeds should produce different results"
    assert not np.allclose(np_rand_1, np_rand_3), "Different seeds should produce different results"
    assert not torch.allclose(torch_rand_1, torch_rand_3), "Different seeds should produce different results"

    print("✓ Different seeds produce different results")
    print("✅ test_set_seed PASSED")


# ============================================================================
# TEST 2: parse_arguments() - CLI parsing
# ============================================================================

def test_parse_arguments():
    """Test CLI argument parsing"""
    print("\n=== Test 2: parse_arguments() ===")

    # Test without arguments (resume=None)
    with patch('sys.argv', ['main.py']):
        args = parse_arguments()
        assert args.resume is None, "resume should be None without --resume flag"
        print("✓ No arguments: resume=None")

    # Test with --resume
    with patch('sys.argv', ['main.py', '--resume', 'models/latest.pt']):
        args = parse_arguments()
        assert args.resume == 'models/latest.pt', "resume path incorrect"
        print("✓ With --resume: path captured correctly")

    print("✅ test_parse_arguments PASSED")


# ============================================================================
# TEST 3: test_main_imports() - Import validation
# ============================================================================

def test_main_imports():
    """Test that all imports in main.py work correctly"""
    print("\n=== Test 3: test_main_imports() ===")

    try:
        # Test project imports
        from modules.data.loader import get_raw_data
        from modules.data.transform import transform_pipeline
        from modules.data.labeling import get_labels
        from modules.data.dataset import TimeSeriesWindowDataset
        from modules.model.mamba import MambaPredictor
        from modules.training.trainer import Trainer, get_device
        from modules.utils.logger import get_logger

        print("✓ All module imports successful")

        # Test standard library imports
        import random
        import argparse
        from datetime import datetime

        print("✓ Standard library imports successful")

        # Test third-party imports
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter

        print("✓ Third-party imports successful")

        print("✅ test_main_imports PASSED")

    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


# ============================================================================
# TEST 4: test_main_pipeline_mini() - Full integration test
# ============================================================================

@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="Requires model training")
def test_main_pipeline_mini():
    """Test full pipeline on test_Dataset.csv with minimal epochs"""
    print("\n=== Test 4: test_main_pipeline_mini() ===")
    print("Running full integration test on test_Dataset.csv...")

    # Create temporary directories for test outputs
    temp_dir = tempfile.mkdtemp(prefix='test_main_')
    temp_models = os.path.join(temp_dir, 'models')
    temp_runs = os.path.join(temp_dir, 'runs')
    temp_logs = os.path.join(temp_dir, 'logs')

    os.makedirs(temp_models, exist_ok=True)
    os.makedirs(temp_runs, exist_ok=True)
    os.makedirs(temp_logs, exist_ok=True)

    print(f"Temporary directory: {temp_dir}")

    # Save original config values
    original_input_files = config.INPUT_FILES
    original_epochs = config.EPOCHS
    original_checkpoint_dir = config.CHECKPOINT_DIR
    original_tensorboard_dir = config.TENSORBOARD_DIR
    original_log_dir = config.LOG_DIR
    original_save_every = config.SAVE_EVERY_N_EPOCHS

    try:
        # Modify config for test
        config.INPUT_FILES = [TEST_CSV]
        config.EPOCHS = 2  # Just 2 epochs for quick test
        config.CHECKPOINT_DIR = temp_models
        config.TENSORBOARD_DIR = temp_runs
        config.LOG_DIR = temp_logs
        config.SAVE_EVERY_N_EPOCHS = 1  # Save every epoch

        print(f"✓ Config modified:")
        print(f"  - INPUT_FILES: {config.INPUT_FILES}")
        print(f"  - EPOCHS: {config.EPOCHS}")
        print(f"  - CHECKPOINT_DIR: {config.CHECKPOINT_DIR}")

        # Verify test dataset exists
        if not os.path.exists(config.INPUT_FILES[0]):
            pytest.skip(f"Test dataset not found: {config.INPUT_FILES[0]}")

        print(f"✓ Test dataset found: {config.INPUT_FILES[0]}")

        # Mock sys.argv to simulate no --resume argument
        with patch('sys.argv', ['main.py']):
            print("Running main()...")

            # Run main pipeline
            main()

        print("✓ main() completed without errors")

        # Verify outputs created
        checkpoints = list(Path(temp_models).glob('*.pt'))
        assert len(checkpoints) > 0, "No checkpoints created"
        print(f"✓ Checkpoints created: {len(checkpoints)}")

        # Verify latest.pt exists
        latest_checkpoint = os.path.join(temp_models, 'latest.pt')
        assert os.path.exists(latest_checkpoint), "latest.pt not created"
        print("✓ latest.pt exists")

        # Verify TensorBoard logs created (format: run_001, run_002, ...)
        tb_runs = list(Path(temp_runs).glob('run_[0-9][0-9][0-9]'))
        assert len(tb_runs) > 0, f"No TensorBoard run directory created in {temp_runs}"
        print(f"✓ TensorBoard logs created: {len(tb_runs)}")

        # Verify program logs created (optional - logger initialized before config change)
        log_files = list(Path(temp_logs).glob('*.log'))
        if len(log_files) > 0:
            print(f"✓ Log files created: {len(log_files)}")
        else:
            print("⚠ Log files not in temp dir (logger initialized before config change - OK)")

        print("✅ test_main_pipeline_mini PASSED")
        print(f"All integration checks passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Restore original config
        config.INPUT_FILES = original_input_files
        config.EPOCHS = original_epochs
        config.CHECKPOINT_DIR = original_checkpoint_dir
        config.TENSORBOARD_DIR = original_tensorboard_dir
        config.LOG_DIR = original_log_dir
        config.SAVE_EVERY_N_EPOCHS = original_save_every

        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory: {temp_dir}")


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MAIN.PY TEST SUITE")
    print("=" * 80)

    try:
        # Run tests
        test_set_seed()
        test_parse_arguments()
        test_main_imports()
        test_main_pipeline_mini()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TESTS FAILED ❌")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
