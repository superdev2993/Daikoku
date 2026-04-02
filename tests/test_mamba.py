"""
Tests for modules/model/mamba.py

Validates:
- Model instantiation
- Forward pass with correct shapes
- Output is logits (raw values)
- No NaN in outputs
- Predict and predict_proba methods
- SelfAttentionBlock (bidirectional attention)
- AttentionPooling (learned query pooling)
- UnifiedHead (v2/v3 gate mechanism)
- Config combinations (mono/dual x attention on/off)
- 3-class classification (Bear=0, Uncertain=1, Bull=2)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from modules.model.mamba import (
    MambaPredictor, MambaBlock,
    SelfAttentionBlock, AttentionPooling, UnifiedHead,
)


class TestMambaBlock:
    """Tests for MambaBlock component."""

    def test_instantiation(self):
        """MambaBlock should instantiate without error."""
        block = MambaBlock(d_model=64, d_state=16, d_conv=2, expand=2)
        assert block is not None

    def test_forward_shape(self):
        """MambaBlock should preserve input shape."""
        block = MambaBlock(d_model=64, d_state=16, d_conv=2, expand=2).to(device)
        x = torch.randn(4, 100, 64).to(device)  # (batch, seq, d_model)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """MambaBlock should have residual connection (output != mamba output)."""
        block = MambaBlock(d_model=64, d_state=16, d_conv=2, expand=2).to(device)
        x = torch.randn(4, 100, 64).to(device)
        output = block(x)
        # With residual, output should be different from just mamba(norm(x))
        mamba_only = block.mamba(block.norm(x))
        assert not torch.allclose(output, mamba_only)


class TestSelfAttentionBlock:
    """Tests for SelfAttentionBlock component."""

    def test_instantiation(self):
        """SelfAttentionBlock should instantiate without error."""
        block = SelfAttentionBlock(d_model=64, num_heads=2)
        assert block is not None

    def test_forward_shape(self):
        """SelfAttentionBlock: (batch, seq, d_model) -> (batch, seq, d_model)."""
        block = SelfAttentionBlock(d_model=64, num_heads=2).to(device)
        x = torch.randn(4, 100, 64).to(device)
        output = block(x)
        assert output.shape == (4, 100, 64)

    def test_bidirectional(self):
        """SelfAttentionBlock should be sensitive to changes at any position."""
        block = SelfAttentionBlock(d_model=32, num_heads=2).to(device)
        block.eval()
        x = torch.randn(2, 50, 32).to(device)
        out1 = block(x).detach()
        # Change a token in the middle
        x_mod = x.clone()
        x_mod[:, 25, :] = 10.0
        out2 = block(x_mod).detach()
        # Output at position 0 should change (bidirectional can see position 25)
        assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4)

    def test_num_heads_validation(self):
        """SelfAttentionBlock should raise error if num_heads doesn't divide d_model."""
        with pytest.raises(AssertionError):
            SelfAttentionBlock(d_model=64, num_heads=5)


class TestAttentionPooling:
    """Tests for AttentionPooling component."""

    def test_instantiation(self):
        """AttentionPooling should instantiate without error."""
        pool = AttentionPooling(d_model=64)
        assert pool is not None

    def test_forward_shape(self):
        """AttentionPooling: (batch, seq, d_model) -> (batch, d_model)."""
        pool = AttentionPooling(d_model=64).to(device)
        x = torch.randn(4, 100, 64).to(device)
        output = pool(x)
        assert output.shape == (4, 64)

    def test_variable_sequence_length(self):
        """AttentionPooling should work with different sequence lengths."""
        pool = AttentionPooling(d_model=32).to(device)
        for seq_len in [10, 50, 200]:
            x = torch.randn(2, seq_len, 32).to(device)
            output = pool(x)
            assert output.shape == (2, 32)

    def test_gradient_flow(self):
        """Gradients should flow through AttentionPooling."""
        pool = AttentionPooling(d_model=32).to(device)
        x = torch.randn(2, 50, 32, requires_grad=True, device=device)
        output = pool(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestUnifiedHead:
    """Tests for UnifiedHead component."""

    @pytest.mark.parametrize("gate_version", ["v2", "v3"])
    @pytest.mark.parametrize("n_paths", [2, 4])
    def test_forward_shape(self, gate_version, n_paths):
        """UnifiedHead should produce logits (batch, 3)."""
        head = UnifiedHead(d_model=48, n_paths=n_paths, gate_version=gate_version, num_classes=3).to(device)
        batch = 4
        paths = [(torch.randn(batch, 48).to(device), torch.randn(batch, 48).to(device))
                 for _ in range(n_paths)]
        output = head(paths)
        assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"
        assert output.shape == (batch, 3)

    def test_v2_softmax_sum_one(self):
        """V2 gate weights should sum to 1."""
        head = UnifiedHead(d_model=32, n_paths=2, gate_version="v2").to(device)
        batch = 4
        paths = [(torch.randn(batch, 32).to(device), torch.randn(batch, 32).to(device))
                 for _ in range(2)]
        # Extract gate weights
        context_parts = []
        for last, mx in paths:
            context_parts.extend([last, mx])
        context = torch.cat(context_parts, dim=-1)
        weights = head.gate(context)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch).to(device), atol=1e-5)

    def test_invalid_version_raises(self):
        """UnifiedHead should raise ValueError for unknown gate_version."""
        with pytest.raises(ValueError, match="Unknown gate_version"):
            UnifiedHead(d_model=32, n_paths=2, gate_version="v99")

    @pytest.mark.parametrize("gate_version", ["v2", "v3"])
    def test_gradient_flow(self, gate_version):
        """Gradients should flow through UnifiedHead."""
        head = UnifiedHead(d_model=32, n_paths=2, gate_version=gate_version, num_classes=3).to(device)
        paths = [(torch.randn(4, 32, requires_grad=True, device=device),
                  torch.randn(4, 32, requires_grad=True, device=device))
                 for _ in range(2)]
        logits = head(paths)
        loss = logits.sum()
        loss.backward()
        for last, mx in paths:
            assert last.grad is not None
            assert mx.grad is not None


class TestMambaPredictor:
    """Tests for MambaPredictor model (3-class: Bear=0, Uncertain=1, Bull=2)."""

    def test_instantiation_default_config(self):
        """Model should instantiate with default config values from config.py."""
        import config
        model = MambaPredictor()
        assert model is not None
        assert model.d_model == config.D_MODEL
        assert model.d_state == config.D_STATE
        assert model.d_conv == config.D_CONV
        assert model.expand == config.EXPAND
        assert model.n_layers == config.N_LAYERS
        # post_gate and aligned fall back to pre_gate for mono models
        if config.ATTENTION_POSITION in ("post_gate", "aligned"):
            assert model.attention_position == "pre_gate"
        else:
            assert model.attention_position == config.ATTENTION_POSITION

    def test_instantiation_custom_params(self):
        """Model should accept custom parameters."""
        model = MambaPredictor(
            input_dim=5,
            d_model=32,
            d_state=8,
            d_conv=4,
            expand=1,
            n_layers=1,
        )
        assert model.d_model == 32
        assert model.d_state == 8
        assert model.d_conv == 4
        assert model.expand == 1
        assert model.n_layers == 1

    def test_forward_returns_tensor(self):
        """Forward pass should return logits tensor (batch, 3)."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        output = model(x)
        assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"
        assert output.shape == (8, 3), f"Expected (8, 3), got {output.shape}"

    def test_forward_variable_batch(self):
        """Forward pass should work with different batch sizes."""
        model = MambaPredictor().to(device)
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 400, 5).to(device)
            logits = model(x)
            assert logits.shape == (batch_size, 3)

    def test_forward_variable_sequence(self):
        """Forward pass should work with different sequence lengths."""
        model = MambaPredictor().to(device)
        for seq_len in [100, 200, 400, 800]:
            x = torch.randn(4, seq_len, 5).to(device)
            logits = model(x)
            assert logits.shape == (4, 3)

    def test_output_is_logits(self):
        """Output should be raw logits (can be any real value)."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        logits = model(x)
        # Logits can be negative, positive, and don't sum to 1
        has_negative = (logits < 0).any()
        has_large = (logits > 1).any()
        assert has_negative or has_large, "Output seems constrained, should be raw logits"

    def test_no_nan_in_output(self):
        """Output should not contain NaN values."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        logits = model(x)
        assert not torch.isnan(logits).any(), "logits contains NaN"

    def test_no_inf_in_output(self):
        """Output should not contain Inf values."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        logits = model(x)
        assert not torch.isinf(logits).any(), "logits contains Inf"

    def test_predict_proba_shape(self):
        """predict_proba should return probabilities of shape (batch, 3)."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        probs = model.predict_proba(x)
        assert probs.shape == (8, 3)

    def test_predict_proba_sums_to_one(self):
        """predict_proba output should sum to 1 per sample."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8).to(device), atol=1e-5)

    def test_predict_proba_range(self):
        """predict_proba values should be in [0, 1]."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        probs = model.predict_proba(x)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_shape(self):
        """predict should return class indices of shape (batch,)."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        preds = model.predict(x)
        assert preds.shape == (8,)

    def test_predict_values(self):
        """predict should return values in {0, 1, 2}."""
        model = MambaPredictor().to(device)
        x = torch.randn(8, 400, 5).to(device)
        preds = model.predict(x)
        assert (preds >= 0).all() and (preds <= 2).all()

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = MambaPredictor().to(device)
        x = torch.randn(4, 400, 5, requires_grad=True, device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_crossentropy_compatibility(self):
        """Model logits should be compatible with CrossEntropyLoss."""
        model = MambaPredictor().to(device)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(8, 400, 5).to(device)
        labels = torch.randint(0, 3, (8,)).to(device)

        logits = model(x)
        loss = criterion(logits, labels)

        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestConfigCombinations:
    """Tests for different config combinations via monkeypatch."""

    def test_mono_no_attention(self, monkeypatch):
        """Mono mode (off) without attention: MLP head."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        monkeypatch.setattr('config.MULTI_TF_MODE', 'off')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        assert model.attention_position == "off"
        assert hasattr(model, 'head')
        x = torch.randn(2, 100, 5).to(device)
        logits = model(x)
        assert logits.shape == (2, 3)

    def test_mono_with_attention(self, monkeypatch):
        """Mono mode with attention: UnifiedHead (2 paths)."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'pre_gate')
        monkeypatch.setattr('config.ATTENTION_NUM_HEADS', 2)
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        assert model.attention_position == "pre_gate"
        assert isinstance(model.head, UnifiedHead)
        assert model.head.n_paths == 2
        x = torch.randn(2, 100, 5).to(device)
        logits = model(x)
        assert logits.shape == (2, 3)

    def test_dual_no_attention(self, monkeypatch):
        """Dual mode without attention: UnifiedHead (2 paths)."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        from modules.model.mamba import MultiTFMambaPredictor
        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=1, n_layers_secondary=1,
        ).to(device)
        assert model.attention_position == "off"
        assert isinstance(model.head, UnifiedHead)
        assert model.head.n_paths == 2
        x_pri = torch.randn(2, 100, 5).to(device)
        x_sec = torch.randn(2, 100, 5).to(device)
        logits = model(x_pri, x_sec)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 3)

    def test_dual_with_attention(self, monkeypatch):
        """Dual mode with attention: UnifiedHead (4 paths)."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'pre_gate')
        monkeypatch.setattr('config.ATTENTION_NUM_HEADS', 2)
        monkeypatch.setattr('config.GATE_VERSION', 'v3')
        from modules.model.mamba import MultiTFMambaPredictor
        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=1, n_layers_secondary=1,
        ).to(device)
        assert model.attention_position == "pre_gate"
        assert isinstance(model.head, UnifiedHead)
        assert model.head.n_paths == 4
        x_pri = torch.randn(2, 100, 5).to(device)
        x_sec = torch.randn(2, 100, 5).to(device)
        logits = model(x_pri, x_sec)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 3)

    def test_dual_post_gate(self, monkeypatch):
        """Dual mode with post_gate: UnifiedHead (3 paths) with cross-TF attention."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'post_gate')
        monkeypatch.setattr('config.ATTENTION_NUM_HEADS', 2)
        monkeypatch.setattr('config.GATE_VERSION', 'v3')
        from modules.model.mamba import MultiTFMambaPredictor
        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=1, n_layers_secondary=1,
        ).to(device)
        assert model.attention_position == "post_gate"
        assert hasattr(model, 'cross_attn')
        assert hasattr(model, 'cross_attn_query')
        assert isinstance(model.head, UnifiedHead)
        assert model.head.n_paths == 3
        x_pri = torch.randn(2, 100, 5).to(device)
        x_sec = torch.randn(2, 100, 5).to(device)
        logits = model(x_pri, x_sec)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 3)


class TestMambaIntegration:
    """Integration tests for the Mamba model."""

    def test_training_step(self):
        """Simulate a single training step."""
        model = MambaPredictor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(8, 400, 5).to(device)
        labels = torch.randint(0, 3, (8,)).to(device)

        # Forward
        logits = model(x)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = MambaPredictor().to(device)
        model.eval()

        with torch.no_grad():
            x = torch.randn(8, 400, 5).to(device)
            logits = model(x)
            assert logits.shape == (8, 3)
            assert not torch.isnan(logits).any()


class TestThreeClassHead:
    """Tests for 3-class single head (Bear=0, Uncertain=1, Bull=2)."""

    def test_mono_forward_returns_tensor(self):
        """MambaPredictor.forward() returns logits tensor."""
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        x = torch.randn(4, 100, 5).to(device)
        output = model(x)
        assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"
        assert output.shape == (4, 3)

    def test_mono_forward_no_nan(self):
        """Outputs should not contain NaN."""
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        x = torch.randn(4, 100, 5).to(device)
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_mono_with_attention(self, monkeypatch):
        """With attention: UnifiedHead returns logits tensor."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'pre_gate')
        monkeypatch.setattr('config.ATTENTION_NUM_HEADS', 2)
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        assert model.attention_position == "pre_gate"
        x = torch.randn(4, 100, 5).to(device)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_predict_proba_returns_3class(self):
        """predict_proba should return probabilities for 3 classes."""
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        x = torch.randn(4, 100, 5).to(device)
        probs = model.predict_proba(x)
        assert probs.shape == (4, 3)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4).to(device), atol=1e-5)

    def test_predict_returns_3class_indices(self):
        """predict should return class indices in {0, 1, 2}."""
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        x = torch.randn(4, 100, 5).to(device)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert (preds >= 0).all() and (preds <= 2).all()

    def test_gradient_flow(self):
        """Gradients should flow through the head."""
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        x = torch.randn(4, 100, 5, requires_grad=True, device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_multi_tf_3class(self, monkeypatch):
        """MultiTFMambaPredictor returns logits tensor."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        from modules.model.mamba import MultiTFMambaPredictor
        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=1, n_layers_secondary=1,
        ).to(device)
        x_pri = torch.randn(4, 100, 5).to(device)
        x_sec = torch.randn(4, 100, 5).to(device)
        logits = model(x_pri, x_sec)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (4, 3)


class TestBinaryClassification:
    """Tests for num_classes=2 (binary mode)."""

    def test_mono_binary_forward(self, monkeypatch):
        """MambaPredictor with num_classes=2 returns (batch, 2)."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1, num_classes=2).to(device)
        x = torch.randn(4, 100, 5).to(device)
        logits = model(x)
        assert logits.shape == (4, 2)

    def test_mono_binary_predict_proba(self, monkeypatch):
        """predict_proba with num_classes=2 returns (batch, 2) summing to 1."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1, num_classes=2).to(device)
        x = torch.randn(4, 100, 5).to(device)
        probs = model.predict_proba(x)
        assert probs.shape == (4, 2)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4).to(device), atol=1e-5)

    def test_mono_binary_predict(self, monkeypatch):
        """predict with num_classes=2 returns values in {0, 1}."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1, num_classes=2).to(device)
        x = torch.randn(4, 100, 5).to(device)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_mono_binary_crossentropy(self, monkeypatch):
        """Binary model logits should work with CrossEntropyLoss."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1, num_classes=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        x = torch.randn(4, 100, 5).to(device)
        labels = torch.randint(0, 2, (4,)).to(device)
        logits = model(x)
        loss = criterion(logits, labels)
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_unified_head_binary(self):
        """UnifiedHead with num_classes=2 produces (batch, 2)."""
        head = UnifiedHead(d_model=32, n_paths=2, gate_version="v2", num_classes=2).to(device)
        paths = [(torch.randn(4, 32).to(device), torch.randn(4, 32).to(device)) for _ in range(2)]
        logits = head(paths)
        assert logits.shape == (4, 2)

    def test_dual_binary(self, monkeypatch):
        """MultiTFMambaPredictor with num_classes=2."""
        monkeypatch.setattr('config.ATTENTION_POSITION', 'off')
        monkeypatch.setattr('config.GATE_VERSION', 'v2')
        from modules.model.mamba import MultiTFMambaPredictor
        model = MultiTFMambaPredictor(
            input_dim_primary=5, input_dim_secondary=5,
            d_model=32, d_state_primary=8, d_state_secondary=8,
            d_conv=2, expand=2, n_layers_primary=1, n_layers_secondary=1,
            num_classes=2,
        ).to(device)
        x_pri = torch.randn(4, 100, 5).to(device)
        x_sec = torch.randn(4, 100, 5).to(device)
        logits = model(x_pri, x_sec)
        assert logits.shape == (4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
