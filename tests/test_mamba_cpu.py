"""
Tests for modules/model/mamba_cpu.py

Direct import of the CPU Mamba implementation (bypasses auto-detection)
to guarantee coverage even when CUDA is available.

Validates:
- Instantiation and forward pass shapes
- State dict key compatibility with mamba_ssm.Mamba
- Causal property (future input does not affect past output)
- Gradient flow through JIT-compiled selective scan
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

# Direct import — NOT via mamba.py auto-detection
from modules.model.mamba_cpu import Mamba, _selective_scan_jit


class TestMambaCPU:
    """Tests for the pure PyTorch CPU Mamba implementation."""

    def test_instantiation(self):
        """Mamba CPU should instantiate with default params."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        assert m.d_model == 32
        assert m.d_state == 16
        assert m.d_inner == 64  # d_model * expand

    def test_forward_shape(self):
        """Forward pass should preserve (batch, seq, d_model) shape."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 50, 32)
        y = m(x)
        assert y.shape == (2, 50, 32)

    def test_forward_no_nan(self):
        """Output should not contain NaN."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        x = torch.randn(4, 100, 32)
        y = m(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_variable_sequence_length(self):
        """Should work with different sequence lengths."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        for seq_len in [10, 50, 200]:
            x = torch.randn(2, seq_len, 32)
            y = m(x)
            assert y.shape == (2, seq_len, 32)

    def test_state_dict_keys(self):
        """State dict keys must match mamba_ssm.Mamba for checkpoint compatibility."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        keys = set(m.state_dict().keys())
        expected = {
            'in_proj.weight',
            'conv1d.weight', 'conv1d.bias',
            'x_proj.weight',
            'dt_proj.weight', 'dt_proj.bias',
            'A_log',
            'D',
            'out_proj.weight',
        }
        assert keys == expected, f"Key mismatch: extra={keys - expected}, missing={expected - keys}"

    def test_causal_property(self):
        """Modifying future timesteps must NOT change past output (causal scan)."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        m.eval()

        x = torch.randn(1, 50, 32)
        y_original = m(x).detach().clone()

        # Modify last 10 timesteps
        x_modified = x.clone()
        x_modified[:, 40:, :] = torch.randn(1, 10, 32)
        y_modified = m(x_modified).detach()

        # First 36 timesteps should be identical (40 - d_conv = 36 safe zone)
        assert torch.allclose(y_original[:, :36, :], y_modified[:, :36, :], atol=1e-5), \
            "Causal violation: modifying future changed past output"

    def test_gradient_flow(self):
        """Gradients should flow through the JIT selective scan."""
        m = Mamba(d_model=32, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 50, 32, requires_grad=True)
        y = m(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSelectiveScanJIT:
    """Tests for the JIT-compiled selective scan kernel."""

    def test_output_shape(self):
        """Selective scan should return same shape as input x."""
        batch, seq, d_inner, d_state = 2, 20, 16, 8
        x = torch.randn(batch, seq, d_inner)
        dt = torch.rand(batch, seq, d_inner).abs() + 0.001
        A = -torch.rand(d_inner, d_state).abs()
        B = torch.randn(batch, seq, d_state)
        C = torch.randn(batch, seq, d_state)

        y = _selective_scan_jit(x, dt, A, B, C)
        assert y.shape == (batch, seq, d_inner)

    def test_zero_input_zero_output(self):
        """Zero input with zero initial state should produce zero output."""
        batch, seq, d_inner, d_state = 2, 20, 16, 8
        x = torch.zeros(batch, seq, d_inner)
        dt = torch.ones(batch, seq, d_inner) * 0.01
        A = -torch.ones(d_inner, d_state)
        B = torch.randn(batch, seq, d_state)
        C = torch.randn(batch, seq, d_state)

        y = _selective_scan_jit(x, dt, A, B, C)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
