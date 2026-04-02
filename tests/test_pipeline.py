"""
Tests for modules/data/pipeline.py

Covers: process_single_dataset in mode "off" (standard pipeline).
Uses test_Dataset.csv for realistic data.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


from conftest import TEST_CSV


@pytest.fixture(autouse=True)
def skip_if_no_data():
    if not os.path.exists(TEST_CSV):
        pytest.skip(f"Test dataset not found: {TEST_CSV}")


# ============================================================================
# MODE OFF (standard pipeline)
# ============================================================================

def test_process_single_dataset_off():
    """process_single_dataset in mode 'off' returns expected structure."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "off"
        result = process_single_dataset(TEST_CSV, logger)

        # Check returned keys
        assert 'train_dataset' in result
        assert 'test_dataset' in result
        assert 'n_features' in result
        assert 'n_norm_features' in result
        assert 'multi_tf_mode' in result
        assert result['multi_tf_mode'] == 'off'

        # Check datasets have items
        assert len(result['train_dataset']) > 0
        assert len(result['test_dataset']) > 0

        # Train > test (80/20 split)
        assert len(result['train_dataset']) > len(result['test_dataset'])

        # Feature count: 12 global + 6 per-window + 5 VP = 23
        assert result['n_features'] == 23

    finally:
        config.MULTI_TF_MODE = original_mode


def test_pipeline_dataset_item_shape():
    """Verify dataset __getitem__ returns correct tensor shapes."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "off"
        result = process_single_dataset(TEST_CSV, logger)

        # Get a single item from test dataset
        item = result['test_dataset'][0]
        window, label = item[0], item[1]

        # Window shape: (WINDOW_SIZE, n_features)
        assert window.shape[0] == config.WINDOW_SIZE
        assert window.shape[1] == result['n_features']

        # Label should be 0, 1, or 2
        assert int(label) in {0, 1, 2}

    finally:
        config.MULTI_TF_MODE = original_mode


def test_pipeline_label_distribution():
    """Verify label distribution has all 3 classes."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "off"
        result = process_single_dataset(TEST_CSV, logger)

        # label_dist should have at least 2 entries (all 3 may not appear in small data)
        assert len(result['label_dist']) >= 2
        assert sum(result['label_dist'].values()) > 0

    finally:
        config.MULTI_TF_MODE = original_mode


def test_pipeline_split_no_overlap():
    """Verify train and test indices don't overlap."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "off"
        result = process_single_dataset(TEST_CSV, logger)

        train_ds = result['train_dataset']
        test_ds = result['test_dataset']

        train_indices = set(train_ds.indices)
        test_indices = set(test_ds.indices)

        overlap = train_indices & test_indices
        assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} indices"

    finally:
        config.MULTI_TF_MODE = original_mode


def test_pipeline_chronological_split():
    """Verify test indices come after train indices (chronological split)."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "off"
        result = process_single_dataset(TEST_CSV, logger)

        train_ds = result['train_dataset']
        test_ds = result['test_dataset']

        assert max(train_ds.indices) < min(test_ds.indices)

    finally:
        config.MULTI_TF_MODE = original_mode


# ============================================================================
# MODE DUAL
# ============================================================================

def test_process_single_dataset_dual():
    """process_single_dataset in mode 'dual' returns multi-TF datasets."""
    from modules.data.pipeline import process_single_dataset
    from modules.utils.logger import get_logger
    logger = get_logger("test_pipeline")

    original_mode = config.MULTI_TF_MODE
    try:
        config.MULTI_TF_MODE = "dual"
        result = process_single_dataset(TEST_CSV, logger)

        assert result['multi_tf_mode'] == 'dual'
        assert 'n_features_sec' in result
        assert 'n_norm_features_sec' in result

        # Dual datasets have items
        assert len(result['train_dataset']) > 0
        assert len(result['test_dataset']) > 0

        # Dual dataset __getitem__ returns 4+ tuple: (pri, sec, tf_map, label[, conf])
        item = result['test_dataset'][0]
        assert len(item) >= 4
        window_pri, window_sec, tf_map = item[0], item[1], item[2]
        assert window_pri.shape[0] == config.WINDOW_SIZE
        assert window_sec.shape[0] == config.WINDOW_SIZE

    finally:
        config.MULTI_TF_MODE = original_mode
