"""
Tests for modules/training/metrics.py

Validates:
- Accuracy computation (global and per class)
- Balanced accuracy
- Precision, recall, F1-score
- Confusion matrix
- Confidence metrics
- Diagnostic metrics
- Gap computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from modules.training.metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix,
    compute_confidence_metrics,
    compute_diagnostic_metrics,
    compute_gaps,
    compute_highconf_metrics,
    METRIC_REGISTRY,
    compute_all_metrics,
    filter_metrics,
)
from modules.model.mamba import MambaPredictor


class TestAccuracyMetrics:
    """Tests for accuracy computations."""

    def test_perfect_predictions(self):
        """Perfect predictions should give 100% accuracy."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_accuracy(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == 1.0

    def test_all_wrong_predictions(self):
        """All wrong predictions should give 0% accuracy."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        metrics = compute_accuracy(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_mixed_predictions(self):
        """Mixed predictions should give correct accuracy."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])  # 3/6 correct

        metrics = compute_accuracy(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == 0.5

    def test_balanced_accuracy(self):
        """Balanced accuracy should handle imbalanced classes."""
        # Imbalanced: many class 0, few class 1
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 0])  # All class 0 correct, 1/2 class 1 correct

        bal_acc = compute_balanced_accuracy(y_true, y_pred)

        # Balanced accuracy: (4/4 + 1/2) / 2 = 0.75
        assert abs(bal_acc - 0.75) < 0.01


class TestPrecisionRecallF1:
    """Tests for precision, recall, F1-score."""

    def test_perfect_predictions(self):
        """Perfect predictions should give perfect scores."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_precision_recall_f1(y_true, y_pred, num_classes=3)

        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_class_specific_metrics(self):
        """Should compute correct per-class metrics."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 0, 0])

        metrics = compute_precision_recall_f1(y_true, y_pred, num_classes=3)

        # Class 0: 2 TP (idx 0,1), 3 FP (idx 3,4,5), 0 FN
        # Precision = 2/(2+3) = 0.4, Recall = 2/(2+0) = 1.0
        assert abs(metrics["precision_class_0"] - 0.4) < 0.01
        assert metrics["recall_class_0"] == 1.0

        # Class 1: 1 TP (idx 2), 0 FP, 1 FN (idx 3)
        # Precision = 1/(1+0) = 1.0, Recall = 1/(1+1) = 0.5
        assert metrics["precision_class_1"] == 1.0
        assert metrics["recall_class_1"] == 0.5


class TestConfusionMatrix:
    """Tests for confusion matrix."""

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 3x3."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Perfect predictions should give diagonal matrix."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

        # Diagonal should be [2, 2, 2]
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2

        # Off-diagonal should be 0
        assert cm.sum() - cm.trace() == 0


class TestConfidenceMetrics:
    """Tests for confidence metrics."""

    def test_high_confidence_predictions(self):
        """High confidence predictions should have low entropy."""
        y_true = np.array([0, 1, 2])
        # Very confident predictions
        y_pred_probs = np.array([
            [0.99, 0.005, 0.005],  # Class 0
            [0.005, 0.99, 0.005],  # Class 1
            [0.005, 0.005, 0.99],  # Class 2
        ])

        metrics = compute_confidence_metrics(y_true, y_pred_probs)

        # Low entropy (predictions are confident)
        assert metrics["entropy_mean"] < 0.1

        # High confidence on correct predictions
        assert metrics["confidence_correct_mean"] > 0.9

    def test_low_confidence_predictions(self):
        """Uncertain predictions should have high entropy."""
        y_true = np.array([0, 1, 2])
        # Uncertain predictions (uniform distribution)
        y_pred_probs = np.array([
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34],
        ])

        metrics = compute_confidence_metrics(y_true, y_pred_probs)

        # High entropy (predictions are uncertain)
        assert metrics["entropy_mean"] > 1.0

        # Low confidence
        assert metrics["confidence_correct_mean"] < 0.5


class TestDiagnosticMetrics:
    """Tests for diagnostic metrics."""

    def test_diagnostic_metrics_structure(self):
        """Should return all expected diagnostic metrics."""
        model = MambaPredictor().to(device)

        # Simulate forward + backward to create gradients
        x = torch.randn(4, 400, 5, requires_grad=True).to(device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        metrics = compute_diagnostic_metrics(model)

        # Check required keys
        assert "grad_norm_mean" in metrics
        assert "grad_norm_max" in metrics
        assert "weight_norm_mean" in metrics
        assert "weight_norm_max" in metrics
        assert "grad_norm_per_layer" in metrics
        assert "weight_norm_per_layer" in metrics

    def test_gradient_norms_positive(self):
        """Gradient norms should be positive after backward pass."""
        model = MambaPredictor().to(device)

        x = torch.randn(4, 400, 5, requires_grad=True).to(device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        metrics = compute_diagnostic_metrics(model)

        assert metrics["grad_norm_mean"] > 0
        assert metrics["grad_norm_max"] > 0

    def test_weight_norms_positive(self):
        """Weight norms should always be positive."""
        model = MambaPredictor().to(device)

        metrics = compute_diagnostic_metrics(model)

        assert metrics["weight_norm_mean"] > 0
        assert metrics["weight_norm_max"] > 0


class TestGapMetrics:
    """Tests for overfitting gap computation."""

    def test_no_overfitting(self):
        """Equal train/test metrics should give zero gaps."""
        gaps = compute_gaps(
            train_loss=0.5,
            test_loss=0.5,
            train_acc=0.8,
            test_acc=0.8,
        )

        assert gaps["gap_loss"] == 0.0
        assert gaps["gap_accuracy"] == 0.0

    def test_overfitting_detection(self):
        """Train >> Test should give positive gaps."""
        gaps = compute_gaps(
            train_loss=0.1,  # Low train loss
            test_loss=0.5,   # High test loss
            train_acc=0.9,   # High train accuracy
            test_acc=0.6,    # Low test accuracy
        )

        # Positive gaps indicate overfitting
        assert gaps["gap_loss"] > 0  # Test loss higher
        assert gaps["gap_accuracy"] > 0  # Train accuracy higher

    def test_underfitting_detection(self):
        """Train << Test should give negative gaps."""
        gaps = compute_gaps(
            train_loss=0.8,  # High train loss
            test_loss=0.5,   # Low test loss
            train_acc=0.5,   # Low train accuracy
            test_acc=0.7,    # High test accuracy
        )

        # Negative gaps indicate underfitting (unusual but possible)
        assert gaps["gap_loss"] < 0
        assert gaps["gap_accuracy"] < 0


class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation with all metrics."""
        # Simulate predictions
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1, 0, 1, 2])
        y_pred_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.7, 0.1],
            [0.4, 0.4, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])

        # Compute all metrics
        acc_metrics = compute_accuracy(y_true, y_pred, num_classes=3)
        bal_acc = compute_balanced_accuracy(y_true, y_pred)
        prf_metrics = compute_precision_recall_f1(y_true, y_pred, num_classes=3)
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        conf_metrics = compute_confidence_metrics(y_true, y_pred_probs)

        # All should return valid values
        assert 0 <= acc_metrics["accuracy"] <= 1
        assert 0 <= bal_acc <= 1
        assert 0 <= prf_metrics["f1_macro"] <= 1
        assert cm.shape == (3, 3)
        assert conf_metrics["entropy_mean"] >= 0


class TestMetricRegistry:
    """Tests for centralized metric registry and filtering."""

    def test_registry_contains_all_metrics(self):
        """METRIC_REGISTRY should contain all expected metric keys."""
        required_metrics = [
            # Performance metrics (3-class: class_0, class_1, class_2)
            "accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro",
            "precision_class_0", "precision_class_1", "precision_class_2",
            "recall_class_0", "recall_class_1", "recall_class_2",
            "f1_class_0", "f1_class_1", "f1_class_2",
            # Confidence metrics
            "entropy_mean", "confidence_correct_mean", "confidence_incorrect_mean",
            # Diagnostic metrics
            "grad_norm_mean", "grad_norm_max", "weight_norm_mean", "weight_norm_max",
            "grad_norm_per_layer", "weight_norm_per_layer",
            # Gap metrics
            "gap_loss", "gap_accuracy",
        ]

        for metric in required_metrics:
            assert metric in METRIC_REGISTRY, f"Missing metric in registry: {metric}"

        # class_2 metrics SHOULD exist (3-class mode)
        for suffix in ["precision_class_2", "recall_class_2", "f1_class_2"]:
            assert suffix in METRIC_REGISTRY, f"class_2 metric should exist: {suffix}"

    def test_registry_destinations_valid(self):
        """All registry destinations should be valid (console, tensorboard, log)."""
        valid_destinations = {"console", "tensorboard", "log"}

        for metric, destinations in METRIC_REGISTRY.items():
            assert isinstance(destinations, set), f"{metric} destinations should be a set"
            assert destinations.issubset(valid_destinations), f"{metric} has invalid destinations"
            assert len(destinations) > 0, f"{metric} has no destinations"

    def test_console_metrics_are_essential(self):
        """Console should only show essential metrics (not too verbose)."""
        console_metrics = [k for k, v in METRIC_REGISTRY.items() if "console" in v]

        # Essential metrics should be in console
        assert "accuracy" in console_metrics
        assert "f1_macro" in console_metrics
        assert "balanced_accuracy" in console_metrics

        # Per-class metrics should NOT be in console (too verbose)
        assert "precision_class_0" not in console_metrics

        # Per-layer metrics should NOT be in console
        assert "grad_norm_per_layer" not in console_metrics


class TestComputeAllMetrics:
    """Tests for centralized compute_all_metrics() function."""

    def test_compute_all_metrics_basic(self):
        """Should compute all basic metrics without model or gaps."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_pred_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])

        metrics = compute_all_metrics(y_true, y_pred, y_pred_probs, num_classes=3)

        # Should contain performance metrics
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_class_0" in metrics

        # Should contain confidence metrics
        assert "entropy_mean" in metrics
        assert "confidence_correct_mean" in metrics

        # Should NOT contain diagnostic metrics (no model)
        assert "grad_norm_mean" not in metrics

        # Should NOT contain gap metrics (no train/test data)
        assert "gap_loss" not in metrics

    def test_compute_all_metrics_with_model(self):
        """Should include diagnostic metrics when model is provided."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_pred_probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

        model = MambaPredictor().to(device)

        # Create gradients
        x = torch.randn(2, 400, 5, requires_grad=True).to(device)
        logits = model(x)
        logits.sum().backward()

        metrics = compute_all_metrics(y_true, y_pred, y_pred_probs, num_classes=3, model=model)

        # Should contain diagnostic metrics
        assert "grad_norm_mean" in metrics
        assert "grad_norm_max" in metrics
        assert "weight_norm_mean" in metrics

    def test_compute_all_metrics_with_gaps(self):
        """Should include gap metrics when train/test data is provided."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_pred_probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

        metrics = compute_all_metrics(
            y_true, y_pred, y_pred_probs,
            num_classes=3,
            train_loss=0.1, test_loss=0.3,
            train_acc=0.9, test_acc=0.7
        )

        # Should contain gap metrics
        assert "gap_loss" in metrics
        assert "gap_accuracy" in metrics
        assert abs(metrics["gap_loss"] - 0.2) < 0.001
        assert abs(metrics["gap_accuracy"] - 0.2) < 0.001


class TestFilterMetrics:
    """Tests for filter_metrics() function."""

    def test_filter_console_metrics(self):
        """Should filter only console-designated metrics."""
        all_metrics = {
            "accuracy": 0.85,
            "precision_class_0": 0.90,
            "f1_macro": 0.80,
            "grad_norm_mean": 0.05,
        }

        console_metrics = filter_metrics(all_metrics, "console")

        # Should include console metrics
        assert "accuracy" in console_metrics
        assert "f1_macro" in console_metrics

        # Should NOT include non-console metrics
        assert "precision_class_0" not in console_metrics
        assert "grad_norm_mean" not in console_metrics

    def test_filter_tensorboard_metrics(self):
        """Should filter tensorboard-designated metrics."""
        all_metrics = {
            "accuracy": 0.85,
            "precision_class_0": 0.90,
            "grad_norm_per_layer": {"layer1": 0.01},
        }

        tb_metrics = filter_metrics(all_metrics, "tensorboard")

        # Should include tensorboard metrics
        assert "accuracy" in tb_metrics
        assert "precision_class_0" in tb_metrics

        # Should NOT include log-only metrics
        assert "grad_norm_per_layer" not in tb_metrics

    def test_filter_log_metrics(self):
        """Should filter log-designated metrics (should include everything)."""
        all_metrics = {
            "accuracy": 0.85,
            "precision_class_0": 0.90,
            "grad_norm_per_layer": {"layer1": 0.01},
        }

        log_metrics = filter_metrics(all_metrics, "log")

        # Log should have all metrics
        assert "accuracy" in log_metrics
        assert "precision_class_0" in log_metrics
        assert "grad_norm_per_layer" in log_metrics

    def test_filter_unknown_metrics(self):
        """Should exclude metrics not in registry."""
        all_metrics = {
            "accuracy": 0.85,
            "unknown_metric": 0.99,
        }

        console_metrics = filter_metrics(all_metrics, "console")

        # Should include registered metric
        assert "accuracy" in console_metrics

        # Should exclude unregistered metric
        assert "unknown_metric" not in console_metrics

    def test_filter_empty_metrics(self):
        """Should handle empty metrics dict."""
        empty_metrics = {}

        result = filter_metrics(empty_metrics, "console")

        assert result == {}

    def test_filter_preserves_values(self):
        """Should preserve original metric values."""
        all_metrics = {"accuracy": 0.8765}

        filtered = filter_metrics(all_metrics, "console")

        assert filtered["accuracy"] == 0.8765


class TestBinaryMetrics:
    """Tests for metrics with num_classes=2 (binary classification)."""

    def test_accuracy_binary(self):
        """Accuracy should work with binary labels."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_accuracy(y_true, y_pred, num_classes=2)
        assert metrics["accuracy"] == pytest.approx(4/6)

    def test_balanced_accuracy_binary(self):
        """Balanced accuracy should work with binary labels."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        result = compute_balanced_accuracy(y_true, y_pred)
        # Class 0: 2/3 correct, Class 1: 2/3 correct → balanced = 2/3
        assert result == pytest.approx(2/3)

    def test_confusion_matrix_binary(self):
        """Confusion matrix should be 2x2 for binary."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_highconf_metrics_binary(self):
        """Highconf metrics should work with num_classes=2."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        # Probabilities: (N, 2)
        y_probs = np.array([
            [0.8, 0.2],  # pred=0, conf=0.8
            [0.3, 0.7],  # pred=1, conf=0.7
            [0.2, 0.8],  # pred=1, conf=0.8
            [0.6, 0.4],  # pred=0, conf=0.6
            [0.9, 0.1],  # pred=0, conf=0.9
            [0.1, 0.9],  # pred=1, conf=0.9
        ])
        metrics = compute_highconf_metrics(y_true, y_pred, y_probs, thresholds=(0.5,), num_classes=2)
        # bull_label = 1, bear predictions (pred==0) with conf>=0.5: indices 0,3,4
        # bear correct: index 0 (true=0), index 3 (true=1, wrong), index 4 (true=0)
        assert metrics["highconf_05_bear_acc"] == pytest.approx(2/3)
        # bull predictions (pred==1) with conf>=0.5: indices 1,2,5
        # bull correct: index 1 (true=0, wrong), index 2 (true=1), index 5 (true=1)
        assert metrics["highconf_05_bull_acc"] == pytest.approx(2/3)
        assert metrics["highconf_05_count"] == 6.0

    def test_compute_all_metrics_binary(self, monkeypatch):
        """compute_all_metrics should work with num_classes=2."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_probs = np.random.rand(6, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize

        model = MambaPredictor(d_model=32, d_state=8, n_layers=1, num_classes=2).to(device)
        metrics = compute_all_metrics(
            y_true=y_true, y_pred=y_pred, y_pred_probs=y_probs,
            model=model, num_classes=2,
        )
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_macro" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
