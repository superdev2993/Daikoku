"""
Loss functions for training pipeline.

UnifiedLoss: Focal Loss + dynamic class weighting in a single module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLoss(nn.Module):
    """
    Unified loss combining focal modulation and class weighting.

    Formula: loss = w_c × (1 - p_t)^gamma × (-log(p_t))

    Components (orthogonal, controlled by their value):
      - gamma=0 → standard cross-entropy (no focal)
      - gamma>0 → down-weight easy examples
      - class_weights=None → no class weighting
      - class_weights=tensor → compensate class imbalance

    When both are 0/None: equivalent to nn.CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 0.0, class_weights: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        if not (0.0 <= gamma <= 5.0):
            raise ValueError(f"gamma must be in [0.0, 5.0], got {gamma}")
        self.gamma = gamma
        self.reduction = reduction
        # Store class_weights as buffer (moves with .to(device) automatically)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                sample_confidence: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw logits
            labels: (batch,) integer class indices
            sample_confidence: (batch,) optional per-sample confidence in [0,1].
                Multiplies loss before reduction. None = no weighting.
        Returns:
            Loss scalar (reduction='mean') or (batch,) tensor
        """
        log_probs = F.log_softmax(logits, dim=1)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        true_log_probs = log_probs[batch_idx, labels]

        # Base CE: -log(p_t)
        loss = -true_log_probs

        # Focal modulation: (1 - p_t)^gamma
        if self.gamma > 0:
            true_probs = torch.exp(true_log_probs)
            focal_weight = (1.0 - true_probs) ** self.gamma
            loss = focal_weight * loss

        # Class weighting: w_c per sample
        if self.class_weights is not None:
            sample_weights = self.class_weights.to(logits.device)[labels]
            loss = sample_weights * loss

        # Label confidence weighting: down-weight uncertain/slow-hit samples
        if sample_confidence is not None:
            loss = sample_confidence * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights(label_dist: dict, alpha: float, num_classes: int = 3) -> torch.Tensor:
    """
    Compute dynamic class weights from label distribution.

    Formula: w_c = (N / (K × n_c))^alpha
      - alpha=0 → all weights = 1 (disabled)
      - alpha=0.5 → sqrt inverse frequency (soft)
      - alpha=1.0 → full inverse frequency

    Args:
        label_dist: {class_id: count} dict
        alpha: weighting strength in [0, 1]
        num_classes: number of classes

    Returns:
        Tensor of shape (num_classes,) with weights
    """
    if alpha <= 0:
        return None

    missing = [c for c in range(num_classes) if c not in label_dist]
    if missing:
        raise ValueError(
            f"Classes missing from label_dist: {missing}. "
            f"All {num_classes} classes must be present in training data."
        )

    total = sum(label_dist.values())
    weights = []
    for c in range(num_classes):
        count = label_dist[c]
        w = (total / (num_classes * count)) ** alpha
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def create_criterion(config: dict, class_weights: torch.Tensor = None) -> nn.Module:
    """
    Create loss criterion based on config.

    Returns UnifiedLoss with focal modulation and class weights.

    Args:
        config: Configuration dict
        class_weights: Optional pre-computed class weight tensor

    Returns:
        Loss criterion module
    """
    gamma = config['FOCAL_GAMMA']

    return UnifiedLoss(
        gamma=gamma,
        class_weights=class_weights,
        reduction='mean'
    )
