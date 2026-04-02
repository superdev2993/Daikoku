"""
Tests for loss functions: UnifiedLoss, create_criterion.
"""

import pytest
import torch
import torch.nn as nn

from modules.training.loss import (
    UnifiedLoss,
    create_criterion
)


# ============================================================================
# UnifiedLoss Tests (Focal + Class Weighting)
# ============================================================================

def test_unified_loss_basic():
    """Test basic functionality of UnifiedLoss"""
    criterion = UnifiedLoss(gamma=2.0)

    # Create dummy logits and labels
    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.tensor([0, 1, 2, 1])  # Bear, Uncertain, Bull, Uncertain

    # Compute loss
    loss = criterion(logits, labels)

    # Check output
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"
    assert loss.item() > 0, "Loss should be positive"


def test_unified_loss_gamma_zero_equals_ce():
    """Test that UnifiedLoss with gamma=0 equals CrossEntropyLoss"""
    unified_criterion = UnifiedLoss(gamma=0.0)
    ce_criterion = nn.CrossEntropyLoss()

    # Create test data
    batch_size = 8
    logits = torch.randn(batch_size, 3)
    labels = torch.randint(0, 3, (batch_size,))

    # Compute losses
    unified_loss = unified_criterion(logits, labels)
    ce_loss = ce_criterion(logits, labels)

    # They should be equal (within numerical precision)
    assert torch.allclose(unified_loss, ce_loss, rtol=1e-5, atol=1e-7), \
        f"UnifiedLoss(gamma=0) should equal CrossEntropyLoss, got {unified_loss.item():.6f} vs {ce_loss.item():.6f}"


def test_unified_loss_easy_vs_hard_examples():
    """Test that focal loss is lower for easy examples than hard examples"""
    criterion = UnifiedLoss(gamma=2.0, reduction='none')

    # Easy example: correct prediction with high confidence
    easy_logits = torch.tensor([[10.0, 0.0, 0.0]])  # Predict Bear correctly
    easy_labels = torch.tensor([0])  # True label: Bear

    # Hard example: wrong prediction with high confidence
    hard_logits = torch.tensor([[0.0, 0.0, 10.0]])  # Predict Bull (wrong)
    hard_labels = torch.tensor([0])  # True label: Bear

    # Compute losses
    easy_loss = criterion(easy_logits, easy_labels)
    hard_loss = criterion(hard_logits, hard_labels)

    # Easy example should have much lower loss
    assert easy_loss.item() < hard_loss.item(), \
        f"Easy example loss ({easy_loss.item():.4f}) should be < hard example loss ({hard_loss.item():.4f})"

    # With gamma=2 and high confidence (0.999), focal weight should be very small
    assert easy_loss.item() < 0.01, \
        f"Easy example with high confidence should have very low loss, got {easy_loss.item():.6f}"


def test_unified_loss_validation():
    """Test parameter validation for UnifiedLoss"""
    # Test negative gamma
    with pytest.raises(ValueError):
        UnifiedLoss(gamma=-1.0)

    # Test gamma > 5.0
    with pytest.raises(ValueError):
        UnifiedLoss(gamma=6.0)

    # Test valid parameters (should not raise)
    UnifiedLoss(gamma=0.0)  # Minimum
    UnifiedLoss(gamma=2.0)  # Recommended
    UnifiedLoss(gamma=5.0)  # Maximum


def test_unified_loss_reduction():
    """Test different reduction modes for UnifiedLoss"""
    criterion_mean = UnifiedLoss(gamma=2.0, reduction='mean')
    criterion_sum = UnifiedLoss(gamma=2.0, reduction='sum')
    criterion_none = UnifiedLoss(gamma=2.0, reduction='none')

    batch_size = 4
    logits = torch.randn(batch_size, 3)
    labels = torch.tensor([0, 1, 2, 1])

    loss_mean = criterion_mean(logits, labels)
    loss_sum = criterion_sum(logits, labels)
    loss_none = criterion_none(logits, labels)

    # Check dimensions
    assert loss_mean.dim() == 0, "Mean reduction should return scalar"
    assert loss_sum.dim() == 0, "Sum reduction should return scalar"
    assert loss_none.shape == (batch_size,), "None reduction should return (batch,) tensor"

    # Check relationship between reductions
    assert torch.allclose(loss_mean, loss_none.mean(), rtol=1e-5), \
        "Mean reduction should equal mean of none reduction"
    assert torch.allclose(loss_sum, loss_none.sum(), rtol=1e-5), \
        "Sum reduction should equal sum of none reduction"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unified_loss_cuda():
    """Test that UnifiedLoss works on CUDA"""
    criterion = UnifiedLoss(gamma=2.0)

    logits = torch.randn(4, 3).cuda()
    labels = torch.tensor([0, 1, 2, 1]).cuda()

    loss = criterion(logits, labels)

    assert loss.device.type == 'cuda', "Loss should be on CUDA"
    assert loss.item() > 0, "Loss should be positive"


# ============================================================================
# Factory Tests
# ============================================================================

def test_create_criterion_default():
    """Test create_criterion function with default config"""
    config = {'FOCAL_GAMMA': 0.0}

    criterion = create_criterion(config)

    assert isinstance(criterion, UnifiedLoss), \
        "Should return UnifiedLoss by default"


def test_create_criterion_focal_only():
    """Test create_criterion with focal loss"""
    config = {
        'FOCAL_GAMMA': 2.0,
    }

    criterion = create_criterion(config)

    assert isinstance(criterion, UnifiedLoss), \
        "Should return UnifiedLoss"
    assert criterion.gamma == 2.0, "Gamma should be set from config"


def test_create_criterion_without_focal():
    """Test create_criterion without focal (gamma=0)"""
    config = {
        'FOCAL_GAMMA': 0.0,
    }

    criterion = create_criterion(config)

    assert isinstance(criterion, UnifiedLoss), \
        "Should return UnifiedLoss"
    assert criterion.gamma == 0.0, "Gamma should be 0"
