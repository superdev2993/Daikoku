"""
Metrics computation for model evaluation and diagnostics.

Provides:
- Performance metrics: accuracy, precision, recall, F1
- Confidence metrics: entropy, prediction confidence
- Diagnostic metrics: gradients, weights, activations
- Overfitting detection: train/test gaps
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix,
)
from scipy.stats import entropy


def _get_directional_masks(y_pred: np.ndarray, prediction_target: str, num_classes: int = 3):
    """
    Return (bull_mask, bear_mask) boolean arrays for directional predictions.

    For triple: bull=pred==2, bear=pred==0
    For bull: bull=pred==1, bear=empty
    For bear: bull=empty, bear=pred==1
    Otherwise: both empty
    """
    if prediction_target == "triple":
        bull_mask = y_pred == (num_classes - 1)
        bear_mask = y_pred == 0
    elif prediction_target == "bull":
        bull_mask = y_pred == 1
        bear_mask = np.zeros(len(y_pred), dtype=bool)
    elif prediction_target == "bear":
        bull_mask = np.zeros(len(y_pred), dtype=bool)
        bear_mask = y_pred == 1
    else:
        bull_mask = np.zeros(len(y_pred), dtype=bool)
        bear_mask = np.zeros(len(y_pred), dtype=bool)
    return bull_mask, bear_mask


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """
    Compute accuracy metrics (global and per class).

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        num_classes: Number of classes (2 or 3)

    Returns:
        Dict with accuracy and per-class accuracy
    """
    # Global accuracy
    return {"accuracy": accuracy_score(y_true, y_pred)}


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy (handles class imbalance).

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)

    Returns:
        Balanced accuracy score
    """
    return balanced_accuracy_score(y_true, y_pred)


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict:
    """
    Compute precision, recall, and F1-score (per class and macro).

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        num_classes: Number of classes (2 or 3)

    Returns:
        Dict with per-class and macro precision/recall/f1
    """
    class_labels = list(range(num_classes))

    # Compute per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    # Compute macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average="macro", zero_division=0
    )

    metrics = {}

    # Per-class metrics
    for class_idx in class_labels:
        metrics[f"precision_class_{class_idx}"] = precision[class_idx]
        metrics[f"recall_class_{class_idx}"] = recall[class_idx]
        metrics[f"f1_class_{class_idx}"] = f1[class_idx]

    # Macro averages
    metrics["precision_macro"] = precision_macro
    metrics["recall_macro"] = recall_macro
    metrics["f1_macro"] = f1_macro

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        num_classes: Number of classes (2 or 3)

    Returns:
        Confusion matrix, shape (num_classes, num_classes)
        Rows: true labels, Columns: predicted labels
    """
    return sklearn_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_confidence_metrics(
    y_true: np.ndarray, y_pred_probs: np.ndarray
) -> dict:
    """
    Compute confidence-related metrics.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred_probs: Predicted probabilities, shape (n_samples, 3)

    Returns:
        Dict with:
            - entropy_mean: Mean entropy of predictions
            - confidence_correct_mean: Mean max probability on correct predictions
            - confidence_incorrect_mean: Mean max probability on incorrect predictions
    """
    # Predicted class
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Entropy of predictions (higher = more uncertain)
    entropies = entropy(y_pred_probs.T)  # scipy expects (n_classes, n_samples)
    entropy_mean = entropies.mean()

    # Max probability (confidence)
    confidences = y_pred_probs.max(axis=1)

    # Split by correctness
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    confidence_correct = confidences[correct_mask]
    confidence_incorrect = confidences[incorrect_mask]

    confidence_correct_mean = confidence_correct.mean() if len(confidence_correct) > 0 else 0.0
    confidence_incorrect_mean = confidence_incorrect.mean() if len(confidence_incorrect) > 0 else 0.0

    return {
        "entropy_mean": entropy_mean,
        "confidence_correct_mean": confidence_correct_mean,
        "confidence_incorrect_mean": confidence_incorrect_mean,
    }


def compute_diagnostic_metrics(model: torch.nn.Module) -> dict:
    """
    Compute diagnostic metrics for model health.

    Args:
        model: PyTorch model

    Returns:
        Dict with:
            - grad_norm_mean: Mean gradient norm across all parameters
            - grad_norm_max: Max gradient norm across all parameters
            - weight_norm_mean: Mean weight norm across all parameters
            - weight_norm_max: Max weight norm across all parameters
            - grad_norm_per_layer: Dict of gradient norms per layer
            - weight_norm_per_layer: Dict of weight norms per layer
    """
    metrics = {}

    grad_norms = []
    weight_norms = []
    grad_norms_per_layer = {}
    weight_norms_per_layer = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Weight norm
            weight_norm = param.data.norm(2).item()
            weight_norms.append(weight_norm)
            weight_norms_per_layer[name] = weight_norm

            # Gradient norm (if available)
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norms.append(grad_norm)
                grad_norms_per_layer[name] = grad_norm

    # Global statistics
    if len(grad_norms) > 0:
        metrics["grad_norm_mean"] = np.mean(grad_norms)
        metrics["grad_norm_max"] = np.max(grad_norms)
    else:
        metrics["grad_norm_mean"] = 0.0
        metrics["grad_norm_max"] = 0.0

    if len(weight_norms) > 0:
        metrics["weight_norm_mean"] = np.mean(weight_norms)
        metrics["weight_norm_max"] = np.max(weight_norms)
    else:
        metrics["weight_norm_mean"] = 0.0
        metrics["weight_norm_max"] = 0.0

    # Per-layer statistics
    metrics["grad_norm_per_layer"] = grad_norms_per_layer
    metrics["weight_norm_per_layer"] = weight_norms_per_layer

    return metrics


def compute_highconf_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: tuple = (0.5, 0.6, 0.7),
    num_classes: int = 3,
) -> dict:
    """
    Compute accuracy for directional predictions (Bear/Bull) at confidence thresholds.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_probs: Predicted probabilities (N, num_classes)
        thresholds: Confidence thresholds to evaluate
        num_classes: Number of classes (2 or 3)

    Returns:
        Dict with highconf_T_bear_acc, highconf_T_bull_acc, highconf_T_count per threshold
    """
    metrics = {}
    confidence = y_pred_probs.max(axis=1)
    bull_label = num_classes - 1
    base_bull, base_bear = _get_directional_masks(y_pred, "triple", num_classes)

    for t in thresholds:
        tag = str(t).replace(".", "")

        # Bear predictions above threshold
        bear_mask = base_bear & (confidence >= t)
        n_bear = bear_mask.sum()
        if n_bear > 0:
            metrics[f"highconf_{tag}_bear_acc"] = float((y_true[bear_mask] == 0).mean())
        else:
            metrics[f"highconf_{tag}_bear_acc"] = 0.0

        # Bull predictions above threshold
        bull_mask = base_bull & (confidence >= t)
        n_bull = bull_mask.sum()
        if n_bull > 0:
            metrics[f"highconf_{tag}_bull_acc"] = float((y_true[bull_mask] == bull_label).mean())
        else:
            metrics[f"highconf_{tag}_bull_acc"] = 0.0

        # Total directional count
        n_dir = n_bear + n_bull
        metrics[f"highconf_{tag}_count"] = float(n_dir)

        # Combined directional accuracy (bear correct + bull correct) / total directional
        if n_dir > 0:
            correct_bear = int((y_true[bear_mask] == 0).sum()) if n_bear > 0 else 0
            correct_bull = int((y_true[bull_mask] == bull_label).sum()) if n_bull > 0 else 0
            metrics[f"highconf_{tag}_dir_acc"] = float((correct_bear + correct_bull) / n_dir)
        else:
            metrics[f"highconf_{tag}_dir_acc"] = 0.0

    return metrics


def compute_margin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    percentiles: tuple = (5, 10, 25, 50),
    prediction_target: str = "triple",
) -> dict:
    """
    Compute Final_Accuracy on top-N% most confident actionable predictions,
    ranked by margin (top1 - top2 softmax proba).

    Actionable predictions:
    - triple: Bear(0) or Bull(2)
    - bull/bear/uncertain: class 1 (positive)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_probs: Predicted probabilities (N, num_classes)
        percentiles: Top-N% slices to evaluate (e.g. 5 = top 5% most confident)
        prediction_target: One of "triple", "bull", "bear", "uncertain"

    Returns:
        Dict with confmargin_topNN_acc and confmargin_topNN_count per percentile
    """
    metrics = {}

    # Compute margin: top1 - top2 probability
    sorted_probs = np.sort(y_pred_probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Select actionable predictions
    if prediction_target == "triple":
        action_mask = (y_pred == 0) | (y_pred == 2)
    else:
        action_mask = y_pred == 1

    action_indices = np.where(action_mask)[0]
    n_actionable = len(action_indices)

    if n_actionable == 0:
        for p in percentiles:
            tag = f"{p:02d}"
            metrics[f"confmargin_top{tag}_acc"] = 0.0
            metrics[f"confmargin_top{tag}_count"] = 0.0
        return metrics

    # Sort actionable predictions by margin (descending)
    action_margins = margin[action_indices]
    sorted_order = np.argsort(-action_margins)
    sorted_indices = action_indices[sorted_order]

    for p in percentiles:
        tag = f"{p:02d}"
        n_top = max(1, int(n_actionable * p / 100))
        top_indices = sorted_indices[:n_top]

        # Compute accuracy on this slice
        if prediction_target == "triple":
            acc = float((y_pred[top_indices] == y_true[top_indices]).mean())
        else:
            acc = float((y_true[top_indices] == 1).mean())

        metrics[f"confmargin_top{tag}_acc"] = acc
        metrics[f"confmargin_top{tag}_count"] = float(n_top)

    return metrics


def compute_final_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_target: str = "triple",
) -> dict:
    """
    Compute Final_Accuracy: precision of the actionable prediction class.

    - triple: precision of directional predictions (Bear + Bull correct / total Bear + Bull preds)
    - bull/bear/uncertain: precision of class 1 (positive class)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        prediction_target: One of "triple", "bull", "bear", "uncertain"

    Returns:
        Dict with final_accuracy and final_accuracy_count
    """
    if prediction_target == "triple":
        # Directional: among preds that are Bear(0) or Bull(2), how many correct?
        dir_mask = (y_pred == 0) | (y_pred == 2)
        count = int(dir_mask.sum())
        acc = float((y_pred[dir_mask] == y_true[dir_mask]).mean()) if count > 0 else 0.0
    else:
        # Binary: precision of class 1 (positive = what we detect)
        pos_mask = y_pred == 1
        count = int(pos_mask.sum())
        acc = float((y_true[pos_mask] == 1).mean()) if count > 0 else 0.0

    return {
        "final_accuracy": acc,
        "final_accuracy_count": float(count),
    }


def compute_edge_metrics(
    y_pred: np.ndarray,
    prediction_target: str,
    long_pnl_R: np.ndarray = None,
    short_pnl_R: np.ndarray = None,
    y_true: np.ndarray = None,
) -> dict:
    """
    Compute edge metrics (expected R per trade) using exact pre-calculated P&L.

    For triple/bull/bear modes (long_pnl_R/short_pnl_R available):
        - Index directly into the P&L array matching predicted direction
        - Bull prediction → long_pnl_R[i], Bear prediction → short_pnl_R[i]
        - edge_per_trade = total_pnl / n_trades
        - edge_total_R = sum of all trade P&Ls

    For uncertain/closeN modes (P&L arrays not available):
        - Convention RR=1:1 → edge_per_trade = 2 * accuracy - 1
        - edge_total_R = edge_per_trade * count

    Args:
        y_pred: Predicted labels
        prediction_target: One of "triple", "bull", "bear", "uncertain", "closeN"
        long_pnl_R: Exact P&L in R-multiples for long trades (from labeling)
        short_pnl_R: Exact P&L in R-multiples for short trades (from labeling)
        y_true: Remapped labels (needed only for uncertain/closeN fallback)

    Returns:
        Dict with edge_per_trade and edge_total_R
    """
    # Exact P&L path: triple/bull/bear with pre-calculated arrays
    if long_pnl_R is not None and prediction_target not in ("uncertain", "close1", "close2", "close3"):
        bull_mask, bear_mask = _get_directional_masks(y_pred, prediction_target, num_classes=3)

        n_trades = int(bull_mask.sum() + bear_mask.sum())
        if n_trades == 0:
            return {"edge_per_trade": 0.0, "edge_total_R": 0.0}

        edge_total = float(long_pnl_R[bull_mask].sum() + short_pnl_R[bear_mask].sum())
        edge_per = edge_total / n_trades
    else:
        # Uncertain or closeN: RR=1, edge = 2*acc - 1
        if prediction_target == "uncertain":
            action_mask = y_pred == 1
        else:
            # closeN: all predictions are actionable
            action_mask = np.ones(len(y_pred), dtype=bool)

        n_trades = int(action_mask.sum())
        if n_trades == 0:
            return {"edge_per_trade": 0.0, "edge_total_R": 0.0}

        acc = float((y_pred[action_mask] == y_true[action_mask]).mean())
        edge_per = 2.0 * acc - 1.0
        edge_total = edge_per * n_trades

    return {
        "edge_per_trade": float(edge_per),
        "edge_total_R": float(edge_total),
    }


def compute_gaps(
    train_loss: float,
    test_loss: float,
    train_acc: float,
    test_acc: float,
) -> dict:
    """
    Compute gaps between train and test metrics (overfitting detection).

    Args:
        train_loss: Training loss
        test_loss: Test loss
        train_acc: Training accuracy
        test_acc: Test accuracy

    Returns:
        Dict with:
            - gap_loss: test_loss - train_loss (positive = overfitting)
            - gap_accuracy: train_acc - test_acc (positive = overfitting)
    """
    return {
        "gap_loss": test_loss - train_loss,
        "gap_accuracy": train_acc - test_acc,
    }


# ============================================================================
# METRIC REGISTRY - Centralized metric destinations
# ============================================================================
# Each metric key defines where it should be displayed:
# - "console": Printed to console during training
# - "tensorboard": Logged to TensorBoard
# - "log": Written to log files/checkpoints

METRIC_REGISTRY = {
    # === Performance metrics ===
    # Global metrics - show everywhere
    "accuracy": {"console", "tensorboard", "log"},
    "balanced_accuracy": {"console", "tensorboard", "log"},
    "f1_macro": {"console", "tensorboard", "log"},
    "precision_macro": {"console", "tensorboard", "log"},
    "recall_macro": {"console", "tensorboard", "log"},

    # Per-class precision/recall/F1 - skip console (too verbose)
    "precision_class_0": {"tensorboard", "log"},
    "precision_class_1": {"tensorboard", "log"},
    "precision_class_2": {"tensorboard", "log"},
    "recall_class_0": {"tensorboard", "log"},
    "recall_class_1": {"tensorboard", "log"},
    "recall_class_2": {"tensorboard", "log"},
    "f1_class_0": {"tensorboard", "log"},
    "f1_class_1": {"tensorboard", "log"},
    "f1_class_2": {"tensorboard", "log"},

    # === Confidence metrics ===
    "entropy_mean": {"console", "tensorboard", "log"},
    "confidence_correct_mean": {"console", "tensorboard", "log"},
    "confidence_incorrect_mean": {"console", "tensorboard", "log"},

    # === Diagnostic metrics ===
    "grad_norm_mean": {"tensorboard", "log"},
    "grad_norm_max": {"tensorboard", "log"},
    "weight_norm_mean": {"tensorboard", "log"},
    "weight_norm_max": {"tensorboard", "log"},
    # Per-layer stats - only in logs (too many for TensorBoard scalar view)
    "grad_norm_per_layer": {"log"},
    "weight_norm_per_layer": {"log"},

    # === Overfitting metrics ===
    "gap_loss": {"console", "tensorboard", "log"},
    "gap_accuracy": {"console", "tensorboard", "log"},

    # === Final Accuracy (precision of actionable class) ===
    "final_accuracy": {"console", "tensorboard", "log"},
    "final_accuracy_count": {"tensorboard", "log"},

    # === High confidence directional metrics ===
    "highconf_05_bear_acc": {"tensorboard", "log"},
    "highconf_05_bull_acc": {"tensorboard", "log"},
    "highconf_05_count": {"tensorboard", "log"},
    "highconf_06_bear_acc": {"tensorboard", "log"},
    "highconf_06_bull_acc": {"tensorboard", "log"},
    "highconf_06_count": {"tensorboard", "log"},
    "highconf_05_dir_acc": {"tensorboard", "log"},
    "highconf_06_dir_acc": {"tensorboard", "log"},
    "highconf_07_bear_acc": {"tensorboard", "log"},
    "highconf_07_bull_acc": {"tensorboard", "log"},
    "highconf_07_count": {"tensorboard", "log"},
    "highconf_07_dir_acc": {"tensorboard", "log"},

    # === Margin confidence metrics (top-N% by margin = top1 - top2 proba) ===
    "confmargin_top05_acc": {"tensorboard", "log"},
    "confmargin_top05_count": {"tensorboard", "log"},
    "confmargin_top10_acc": {"tensorboard", "log"},
    "confmargin_top10_count": {"tensorboard", "log"},
    "confmargin_top25_acc": {"tensorboard", "log"},
    "confmargin_top25_count": {"tensorboard", "log"},
    "confmargin_top50_acc": {"tensorboard", "log"},
    "confmargin_top50_count": {"tensorboard", "log"},

    # === Edge metrics (expected R per trade) ===
    "edge_per_trade": {"console", "tensorboard", "log"},
    "edge_total_R": {"console", "tensorboard", "log"},

}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    model: torch.nn.Module = None,
    train_loss: float = None,
    test_loss: float = None,
    train_acc: float = None,
    test_acc: float = None,
    num_classes: int = 3,
    prediction_target: str = "triple",
    long_pnl_R: np.ndarray = None,
    short_pnl_R: np.ndarray = None,
) -> dict:
    """
    Centralized metric computation. Call all metric functions and return flat dict.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_probs: Predicted probabilities
        model: PyTorch model (optional, for diagnostic metrics)
        train_loss: Training loss (optional, for gap metrics)
        test_loss: Test loss (optional, for gap metrics)
        train_acc: Training accuracy (optional, for gap metrics)
        test_acc: Test accuracy (optional, for gap metrics)
        num_classes: Number of classes
        prediction_target: Target mode (triple, bull, bear, uncertain, closeN)
        long_pnl_R: Exact P&L in R-multiples for long trades (from labeling)
        short_pnl_R: Exact P&L in R-multiples for short trades (from labeling)

    Returns:
        Flat dict of all computed metrics
    """
    metrics = {}

    # Performance metrics
    metrics.update(compute_accuracy(y_true, y_pred, num_classes=num_classes))
    metrics["balanced_accuracy"] = compute_balanced_accuracy(y_true, y_pred)

    metrics.update(compute_precision_recall_f1(y_true, y_pred, num_classes=num_classes))

    # Confidence metrics
    metrics.update(compute_confidence_metrics(y_true, y_pred_probs))

    # High confidence directional metrics
    metrics.update(compute_highconf_metrics(y_true, y_pred, y_pred_probs, num_classes=num_classes))

    # Final Accuracy (precision of actionable predictions)
    metrics.update(compute_final_accuracy(y_true, y_pred, prediction_target=prediction_target))

    # Margin confidence metrics (top1 - top2 proba filtering)
    metrics.update(compute_margin_metrics(y_true, y_pred, y_pred_probs, prediction_target=prediction_target))

    # Edge metrics (exact P&L or fallback for uncertain/closeN)
    if long_pnl_R is not None or prediction_target in ("uncertain", "close1", "close2", "close3"):
        metrics.update(compute_edge_metrics(
            y_pred, prediction_target,
            long_pnl_R=long_pnl_R,
            short_pnl_R=short_pnl_R,
            y_true=y_true,
        ))

    # Diagnostic metrics (if model provided)
    if model is not None:
        metrics.update(compute_diagnostic_metrics(model))

    # Gap metrics (if train/test data provided)
    if all(x is not None for x in [train_loss, test_loss, train_acc, test_acc]):
        metrics.update(compute_gaps(train_loss, test_loss, train_acc, test_acc))

    return metrics


def filter_metrics(metrics: dict, destination: str) -> dict:
    """
    Filter metrics by destination (console, tensorboard, log).

    Args:
        metrics: Dict of all computed metrics
        destination: Target destination ("console", "tensorboard", "log")

    Returns:
        Filtered dict containing only metrics for the specified destination
    """
    return {
        key: value
        for key, value in metrics.items()
        if destination in METRIC_REGISTRY.get(key, set())
    }
