"""
Tests for modules/training/monitoring.py

Validates:
- ActivationMonitor returns correct stats structure
- Dead neuron detection works
- Forward hooks are cleaned up after collect()
- Works with both mono and dual models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from modules.model.mamba import MambaPredictor
from modules.training.monitoring import ActivationMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestActivationMonitor:
    """Tests for ActivationMonitor."""

    def _make_model(self, monkeypatch):
        """Create a small mono model on the available device."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        return MambaPredictor(
            input_dim=5, d_model=32, d_state=8, d_conv=2, expand=2, n_layers=2,
        ).to(device)

    def test_stats_structure(self, monkeypatch):
        """collect() should return dict with one key per layer + _global."""
        model = self._make_model(monkeypatch)
        monitor = ActivationMonitor(model)

        batch = (torch.randn(2, 50, 5),)
        stats = monitor.collect(batch, device)

        # Must have _global key
        assert '_global' in stats
        assert 'total_neurons' in stats['_global']
        assert 'total_dead' in stats['_global']
        assert 'dead_pct_total' in stats['_global']

        # Must have linear_proj + 2 mamba_blocks
        assert 'linear_proj' in stats
        assert 'mamba_block_0' in stats
        assert 'mamba_block_1' in stats

        # Each layer has required fields
        for key in ['linear_proj', 'mamba_block_0', 'mamba_block_1']:
            assert 'mean' in stats[key]
            assert 'std' in stats[key]
            assert 'dead_pct' in stats[key]
            assert 'n_dead' in stats[key]
            assert 'n_total' in stats[key]

    def test_hook_cleanup(self, monkeypatch):
        """After collect(), no forward hooks should remain on the model."""
        model = self._make_model(monkeypatch)
        monitor = ActivationMonitor(model)

        # Count hooks before
        hooks_before = sum(len(m._forward_hooks) for m in model.modules())

        batch = (torch.randn(2, 50, 5),)
        monitor.collect(batch, device)

        # Count hooks after — should be same as before (all removed)
        hooks_after = sum(len(m._forward_hooks) for m in model.modules())
        assert hooks_after == hooks_before, \
            f"Hooks leaked: {hooks_after - hooks_before} hooks remain after collect()"

    def test_model_restored_to_training(self, monkeypatch):
        """Model should be restored to training mode after collect()."""
        model = self._make_model(monkeypatch)
        model.train()
        monitor = ActivationMonitor(model)

        batch = (torch.randn(2, 50, 5),)
        monitor.collect(batch, device)

        assert model.training, "Model should be back in training mode after collect()"

    def test_dead_neuron_detection(self, monkeypatch):
        """Monitor should detect dead neurons (features with near-zero std)."""
        model = self._make_model(monkeypatch)

        # Zero out linear_proj weights to create dead activations
        with torch.no_grad():
            model.linear_proj.weight.zero_()
            model.linear_proj.bias.zero_()

        monitor = ActivationMonitor(model, dead_threshold=1e-6)
        batch = (torch.randn(2, 50, 5),)
        stats = monitor.collect(batch, device)

        # linear_proj output should be all zeros -> 100% dead
        assert stats['linear_proj']['dead_pct'] == 100.0
        assert stats['_global']['total_dead'] > 0

    def test_healthy_model_low_dead_pct(self, monkeypatch):
        """A freshly initialized model should have very few dead neurons."""
        model = self._make_model(monkeypatch)
        monitor = ActivationMonitor(model, dead_threshold=1e-6)

        batch = (torch.randn(4, 50, 5),)
        stats = monitor.collect(batch, device)

        # Fresh model should not have massive dead neuron percentage
        assert stats['_global']['dead_pct_total'] < 50.0, \
            f"Too many dead neurons in fresh model: {stats['_global']['dead_pct_total']:.1f}%"

    def test_dual_model(self, monkeypatch):
        """Monitor should work with MultiTFMambaPredictor (dual branches)."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        from modules.model.mamba import MultiTFMambaPredictor

        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=2, n_layers_secondary=1,
        ).to(device)
        monitor = ActivationMonitor(model)

        batch = (torch.randn(2, 50, 5), torch.randn(2, 50, 5))
        stats = monitor.collect(batch, device)

        # Should have primary_ and secondary_ prefixed keys
        assert 'primary_linear_proj' in stats
        assert 'primary_mamba_block_0' in stats
        assert 'primary_mamba_block_1' in stats
        assert 'secondary_linear_proj' in stats
        assert 'secondary_mamba_block_0' in stats
        assert '_global' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
