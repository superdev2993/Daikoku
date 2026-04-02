"""
Unit tests for ModelEvaluator
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.evaluation.evaluator import ModelEvaluator
from modules.model.mamba import MambaPredictor
from conftest import TEST_CSV


class TestModelEvaluator:
    """Test ModelEvaluator class — uses session-scoped trained_checkpoint fixture."""

    def test_init(self, trained_checkpoint):
        """Test ModelEvaluator initialization"""
        ckpt, _ = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')

        assert evaluator.checkpoint_path == ckpt
        assert evaluator.device is not None
        assert evaluator.checkpoint is None
        assert evaluator.model is None

    def test_init_invalid_checkpoint(self):
        """Test initialization with non-existent checkpoint"""
        with pytest.raises(Exception):
            evaluator = ModelEvaluator('/nonexistent/checkpoint.pt', device='auto')
            evaluator.load_checkpoint()

    def test_load_checkpoint(self, trained_checkpoint):
        """Test checkpoint loading"""
        ckpt, _ = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')
        checkpoint = evaluator.load_checkpoint()

        assert checkpoint is not None
        assert evaluator.model is not None
        assert evaluator.config_params is not None

        # Verify config parameters match current config
        assert evaluator.config_params['WINDOW_SIZE'] == config.WINDOW_SIZE
        assert evaluator.config_params['D_MODEL'] == config.D_MODEL

        # Verify model is in eval mode
        assert not evaluator.model.training

    def test_load_checkpoint_missing_keys(self, tmp_path):
        """Test loading checkpoint with missing required keys"""
        invalid_checkpoint_path = str(tmp_path / 'invalid_checkpoint.pt')

        # Temporarily set config for minimal model creation
        saved_attn = config.ATTENTION_POSITION
        saved_cnn = config.CNN_ENABLED
        config.ATTENTION_POSITION = "off"
        config.CNN_ENABLED = False
        try:
            model = MambaPredictor(input_dim=5, d_model=16, d_state=8, d_conv=2, expand=2, n_layers=2)
        finally:
            config.ATTENTION_POSITION = saved_attn
            config.CNN_ENABLED = saved_cnn

        invalid_checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict()
            # Missing 'config' key
        }
        torch.save(invalid_checkpoint, invalid_checkpoint_path)

        evaluator = ModelEvaluator(invalid_checkpoint_path, device='auto')
        with pytest.raises(ValueError):
            evaluator.load_checkpoint()

    def test_predict_shape(self, trained_checkpoint):
        """Test predict returns correct shapes (3-class) using real data."""
        ckpt, csv_path = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')
        evaluator.load_checkpoint()

        dataset, y_true, timestamps, split_info = evaluator.prepare_data(csv_path, split='test')
        n = len(dataset)
        assert n > 0

        predictions, probabilities = evaluator.predict(dataset)

        # Verify shapes (num_classes from checkpoint)
        nc = evaluator.num_classes
        assert predictions.shape == (n,)
        assert probabilities.shape == (n, nc)

        # Verify predictions are valid classes
        assert np.all(predictions >= 0)
        assert np.all(predictions < nc)

        # Verify probabilities sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(n), decimal=5
        )

    def test_compute_metrics(self, trained_checkpoint):
        """Test metrics computation (3-class)"""
        ckpt, _ = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')
        evaluator.load_checkpoint()

        # Create dummy 3-class predictions
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0, 2, 0, 1])
        y_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.5, 0.3, 0.2],
            [0.1, 0.1, 0.8],
            [0.85, 0.1, 0.05],
            [0.1, 0.8, 0.1]
        ])

        metrics = evaluator.compute_metrics(y_true, y_pred, y_probs)

        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'f1_macro' in metrics

        # 7 out of 8 correct
        expected_accuracy = 7 / 8
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-4

    def test_save_results_creates_files(self, trained_checkpoint, tmp_path):
        """Test that save_results creates expected files"""
        ckpt, _ = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')
        evaluator.load_checkpoint()

        predictions = np.array([0, 1, 2, 1, 0])
        probabilities = np.random.rand(5, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1h')
        labels = np.array([0, 1, 2, 0, 1])

        metrics = {
            'accuracy': 0.6,
            'balanced_accuracy': 0.55,
            'f1_macro': 0.58,
            'precision_class_0': 0.7,
            'recall_class_0': 0.6,
            'f1_class_0': 0.65
        }

        split_info = {
            'split': 'test',
            'total_samples': 5,
            'train_split_index': 100,
            'window_size': 50,
            'nan_offset': 0,
            'data_path': 'test_data.csv'
        }

        df_raw = pd.read_csv(TEST_CSV)

        output_dir = str(tmp_path / 'test_results')

        evaluator.save_results(
            output_dir=output_dir,
            predictions=predictions,
            probabilities=probabilities,
            timestamps=timestamps,
            labels=labels,
            metrics=metrics,
            split_info=split_info,
            df_raw=df_raw
        )

        assert os.path.exists(os.path.join(output_dir, 'predictions.csv'))
        assert os.path.exists(os.path.join(output_dir, 'metrics.json'))
        assert os.path.exists(os.path.join(output_dir, 'summary.txt'))

        df_pred = pd.read_csv(os.path.join(output_dir, 'predictions.csv'))
        assert len(df_pred) == 5
        assert 'label_pred' in df_pred.columns
        assert 'label_true' in df_pred.columns
        assert 'correct' in df_pred.columns

    def test_save_results_inference_only(self, trained_checkpoint, tmp_path):
        """Test save_results with inference_only mode (no labels)"""
        ckpt, _ = trained_checkpoint
        evaluator = ModelEvaluator(ckpt, device='auto')
        evaluator.load_checkpoint()

        predictions = np.array([0, 1, 2])
        probabilities = np.random.rand(3, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h')

        split_info = {
            'split': 'all',
            'total_samples': 3,
            'window_size': 50,
            'nan_offset': 0
        }

        output_dir = str(tmp_path / 'test_inference')

        evaluator.save_results(
            output_dir=output_dir,
            predictions=predictions,
            probabilities=probabilities,
            timestamps=timestamps,
            labels=None,
            metrics=None,
            split_info=split_info,
            df_raw=None
        )

        assert os.path.exists(os.path.join(output_dir, 'predictions.csv'))
        assert not os.path.exists(os.path.join(output_dir, 'metrics.json'))
        assert not os.path.exists(os.path.join(output_dir, 'summary.txt'))

        df_pred = pd.read_csv(os.path.join(output_dir, 'predictions.csv'))
        assert 'label_true' not in df_pred.columns


# ============================================================================
# Integration tests with real data (use session-scoped trained_checkpoint)
# ============================================================================


def test_evaluator_prepare_data_real(trained_checkpoint):
    """prepare_data on real checkpoint returns correct shapes."""
    ckpt, csv_path = trained_checkpoint
    evaluator = ModelEvaluator(ckpt)
    evaluator.load_checkpoint()

    dataset, y_true, timestamps, split_info = evaluator.prepare_data(csv_path, split='test')

    assert len(dataset) > 0
    assert len(y_true) == len(dataset)
    assert len(timestamps) == len(dataset)
    assert set(np.unique(y_true)).issubset({0, 1, 2})


def test_evaluator_all_split_larger(trained_checkpoint):
    """split='all' returns more samples than split='test'."""
    ckpt, csv_path = trained_checkpoint
    evaluator = ModelEvaluator(ckpt)
    evaluator.load_checkpoint()

    _, y_test, _, _ = evaluator.prepare_data(csv_path, split='test')
    _, y_all, _, _ = evaluator.prepare_data(csv_path, split='all')

    assert len(y_all) > len(y_test)


def test_evaluator_inference_only(trained_checkpoint):
    """inference_only=True returns None for labels."""
    ckpt, csv_path = trained_checkpoint
    evaluator = ModelEvaluator(ckpt)
    evaluator.load_checkpoint()

    dataset, y_true, timestamps, _ = evaluator.prepare_data(csv_path, split='all', inference_only=True)

    assert len(dataset) > 0
    assert y_true is None
    assert len(timestamps) == len(dataset)


def test_evaluator_predict_real(trained_checkpoint):
    """predict on real data returns valid shapes and probabilities."""
    ckpt, csv_path = trained_checkpoint
    evaluator = ModelEvaluator(ckpt)
    evaluator.load_checkpoint()

    dataset, y_true, _, _ = evaluator.prepare_data(csv_path, split='test')
    y_pred, y_probs = evaluator.predict(dataset)

    n = len(dataset)
    assert y_pred.shape == (n,)
    assert y_probs.shape == (n, evaluator.num_classes)
    np.testing.assert_allclose(y_probs.sum(axis=1), 1.0, atol=1e-4)


def test_evaluator_full_pipeline(trained_checkpoint):
    """Full pipeline: prepare_data → predict → compute_metrics → save_results."""
    ckpt, csv_path = trained_checkpoint
    evaluator = ModelEvaluator(ckpt)
    evaluator.load_checkpoint()

    dataset, y_true, timestamps, split_info = evaluator.prepare_data(csv_path, split='test')
    y_pred, y_probs = evaluator.predict(dataset)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_probs)

    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

    df_raw = pd.read_csv(csv_path)
    temp_out = tempfile.mkdtemp(prefix='eval_full_')
    try:
        evaluator.save_results(
            output_dir=temp_out,
            predictions=y_pred,
            probabilities=y_probs,
            timestamps=timestamps,
            labels=y_true,
            metrics=metrics,
            split_info=split_info,
            df_raw=df_raw,
        )

        df_pred = pd.read_csv(os.path.join(temp_out, 'predictions.csv'))
        assert 'open' in df_pred.columns
        assert 'close' in df_pred.columns
        assert len(df_pred) == len(y_pred)

        with open(os.path.join(temp_out, 'metrics.json')) as f:
            loaded = json.load(f)
        assert 'accuracy' in loaded

    finally:
        shutil.rmtree(temp_out)
