"""
Mamba architecture for crypto trading prediction.

Architecture:
    Input (batch, seq_len, n_features)
        |
    Dual-path projection (Linear + optional causal CNN)
        |
    Mamba Block x n_layers (causal SSM)
        |
    ATTENTION_POSITION controls attention mode:
        "off":       No attention
        "pre_gate":  Intra-TF SelfAttention parallel to Mamba
        "post_gate": Cross-TF attention as 3rd path in gate (Multi-TF only)
        "aligned":   Temporally aligned cross-TF fusion (Multi-TF only)
        |
    Head selection:
        Mono + off:       MLP head (concat → Linear → GELU → Dropout → Linear → 3)
        Mono + pre_gate:  UnifiedHead(2 paths)
        Multi-TF + off:       UnifiedHead(2 paths)
        Multi-TF + pre_gate:  UnifiedHead(4 paths)
        Multi-TF + post_gate: UnifiedHead(3 paths) with cross-TF attention
        Multi-TF + aligned:   MLP head on enriched primary (single path, no Gate/UnifiedHead)
                              AlignedFusion already injects secondary into primary via gated residual,
                              so a separate secondary path would be redundant and causes gate collapse.
        |
    Output: logits [batch, 3]
"""

import logging

import torch
import torch.nn as nn
from typing import List

import config

# Mamba import: CUDA-accelerated if available, pure PyTorch CPU fallback otherwise
_logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba as _MambaGPU
    _HAS_CUDA_MAMBA = True
except ImportError:
    _HAS_CUDA_MAMBA = False

_force_cpu = getattr(config, 'DEVICE', 'auto').lower() == 'cpu'

if not _HAS_CUDA_MAMBA:
    from modules.model.mamba_cpu import Mamba
    _logger.warning(
        "mamba_ssm package not installed. Using pure PyTorch CPU implementation "
        "(extremely slower for training). "
        "To enable GPU acceleration: pip install mamba-ssm causal-conv1d"
    )
elif not torch.cuda.is_available():
    from modules.model.mamba_cpu import Mamba
    _logger.warning(
        "mamba_ssm is installed but no CUDA GPU detected. Using pure PyTorch CPU "
        "implementation. Inference and evaluation will work normally. "
        "Training will be extremely slower."
    )
elif _force_cpu:
    from modules.model.mamba_cpu import Mamba
    _logger.warning(
        "DEVICE='cpu' set in config.py. Using pure PyTorch CPU implementation "
        "instead of mamba_ssm CUDA kernels. Training will be extremely slower. "
        "Set DEVICE='auto' or DEVICE='cuda' to use GPU acceleration."
    )
else:
    Mamba = _MambaGPU

# Architectural keys read from config for model construction
ARCH_KEYS = (
    'CNN_ENABLED', 'CNN_FUSION', 'CNN_KERNEL_SIZE', 'CNN_DROPOUT', 'CNN_LAYERNORM',
    'MAMBA_LAYERNORM',
    'DROPOUT', 'DROPOUT_LAST_N_LAYERS', 'DROPOUT_HEAD',
    'ATTENTION_POSITION', 'ATTENTION_NUM_HEADS',
    'GATE_VERSION', 'GATE_BOTTLENECK',
)


def build_arch_params():
    """Build arch_params dict from current config.py values."""
    return {key: getattr(config, key) for key in ARCH_KEYS}


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection, optional layer norm, and optional dropout."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dropout: float = 0.0, use_layernorm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm + Mamba + dropout + residual
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class SelfAttentionBlock(nn.Module):
    """
    Bidirectional self-attention block (parallel branch to Mamba).

    Uses nn.MultiheadAttention (FlashAttention auto on PyTorch 2.0+).
    No causal mask = bidirectional. No residual (standalone parallel branch).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        attn_out, _ = self.attn(x, x, x)  # Self-attention, no mask = bidirectional
        return self.dropout(attn_out)


class AttentionPooling(nn.Module):
    """
    Learned query-based attention pooling.

    Projects sequence to a single vector via learned query attention.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, d_model)
        """
        # scores: (batch, seq_len, 1)
        scores = torch.softmax(x @ self.query.T, dim=1)
        # Weighted sum: (batch, d_model)
        return (scores * x).sum(dim=1)


class AlignedFusion(nn.Module):
    """
    Temporally aligned cross-TF fusion.

    For each primary token, gathers the corresponding secondary token
    using the temporal mapping, then applies a gated residual fusion.
    The gate learns per-position and per-feature how much secondary
    context to inject into the primary representation.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_pri: torch.Tensor, h_sec: torch.Tensor,
                tf_mapping: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pri: (batch, seq_pri, d_model) — primary encoded sequence
            h_sec: (batch, seq_sec, d_model) — secondary encoded sequence
            tf_mapping: (batch, seq_pri) LongTensor — for each primary position,
                        the corresponding secondary position in the window

        Returns:
            enriched: (batch, seq_pri, d_model) — primary enriched with aligned secondary
        """
        # Gather aligned secondary tokens: h_sec_aligned[b,i] = h_sec[b, tf_mapping[b,i]]
        idx = tf_mapping.unsqueeze(-1).expand(-1, -1, h_sec.size(-1))  # (B, seq_pri, d_model)
        h_sec_aligned = torch.gather(h_sec, 1, idx)  # (B, seq_pri, d_model)

        # Gated residual fusion
        combined = torch.cat([h_pri, h_sec_aligned], dim=-1)  # (B, seq_pri, 2*d_model)
        projected = self.proj(combined)   # (B, seq_pri, d_model)
        gate = self.gate(combined)        # (B, seq_pri, d_model) values in [0, 1]

        enriched = h_pri + self.dropout(gate * projected)
        return self.norm(enriched)


class UnifiedHead(nn.Module):
    """
    Unified classification head with configurable gate mechanism.

    Receives a list of (last_token, max_pool) tuples, one per active path.
    For attention paths: (attn_pool, attn_pool).

    V2 (scalar gate):
        Aggregation: agg_i = last_i + max_i (addition -> d_model)
        Context: concat(last_0, max_0, last_1, max_1, ...) -> (n_paths x 2 x d_model)
        Gate: Linear(n*2*d -> d) -> Tanh -> Linear(d -> n) -> Softmax -> 1 scalar/path
        Fusion: concat(agg_i * w_i) -> (n_paths x d_model)
        Classifier: Linear(n*d -> d) -> GELU -> Dropout -> Linear(d -> num_classes)

    V3 (MLP per-feature gate):
        Projection: concat(last_i, max_i) -> Linear(2*d -> d) per path
        Gate MLP: concat(proj_all) -> Linear(n*d -> bottleneck) -> ReLU -> Linear(bottleneck -> n*d) -> Sigmoid
        Fusion: sum(proj_i * gate_i) -> (d_model)
        Classifier: Linear(d -> d) -> GELU -> Dropout -> Linear(d -> num_classes)
    """

    def __init__(self, d_model: int, n_paths: int, gate_version: str,
                 dropout: float = 0.1, gate_bottleneck: int = 64,
                 num_classes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_paths = n_paths
        self.gate_version = gate_version

        if gate_version == "v2":
            # Gate: sees full context (all last+max pairs)
            self.gate = nn.Sequential(
                nn.Linear(n_paths * 2 * d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, n_paths),
                nn.Softmax(dim=-1),
            )
            # Classifier on weighted fusion
            self.classifier = nn.Sequential(
                nn.Linear(n_paths * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        elif gate_version == "v3":
            # Per-path projection: (2*d -> d)
            self.projections = nn.ModuleList([
                nn.Linear(2 * d_model, d_model)
                for _ in range(n_paths)
            ])
            # Gate MLP: concat(proj_all) -> per-feature gate
            bottleneck = gate_bottleneck
            self.gate_mlp = nn.Sequential(
                nn.Linear(n_paths * d_model, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, n_paths * d_model),
                nn.Sigmoid(),
            )
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        else:
            raise ValueError(f"Unknown gate_version: {gate_version}. Must be 'v2' or 'v3'.")

    def forward(self, paths: List[tuple]) -> torch.Tensor:
        """
        Args:
            paths: List of (last_token, max_pool) tuples, each (batch, d_model)
        Returns:
            logits: (batch, num_classes)
        """
        if self.gate_version == "v2":
            return self._forward_v2(paths)
        else:
            return self._forward_v3(paths)

    def _forward_v2(self, paths: List[tuple]) -> torch.Tensor:
        """V2: scalar gate per path."""
        # Aggregate per path: last + max
        aggs = [last + mx for last, mx in paths]  # list of (batch, d_model)

        # Context: concat all last+max pairs
        context_parts = []
        for last, mx in paths:
            context_parts.extend([last, mx])
        context = torch.cat(context_parts, dim=-1)  # (batch, n_paths * 2 * d_model)

        # Gate weights
        weights = self.gate(context)  # (batch, n_paths)

        # Weighted fusion: concat(agg_i * w_i)
        fused_parts = [
            aggs[i] * weights[:, i:i+1]  # (batch, d_model) * (batch, 1)
            for i in range(self.n_paths)
        ]
        fused = torch.cat(fused_parts, dim=-1)  # (batch, n_paths * d_model)

        return self.classifier(fused)

    def _forward_v3(self, paths: List[tuple]) -> torch.Tensor:
        """V3: MLP per-feature gate."""
        # Project each path: concat(last, max) -> d_model
        projections = [
            self.projections[i](torch.cat([last, mx], dim=-1))
            for i, (last, mx) in enumerate(paths)
        ]  # list of (batch, d_model)

        # Concat all projections for gate input
        proj_cat = torch.cat(projections, dim=-1)  # (batch, n_paths * d_model)

        # Gate: per-feature gating
        gate_values = self.gate_mlp(proj_cat)  # (batch, n_paths * d_model)
        gate_values = gate_values.view(-1, self.n_paths, self.d_model)  # (batch, n_paths, d_model)

        # Stack projections and apply gate
        proj_stack = torch.stack(projections, dim=1)  # (batch, n_paths, d_model)
        gated = (proj_stack * gate_values).sum(dim=1)  # (batch, d_model)

        return self.classifier(gated)


class MambaPredictor(nn.Module):
    """
    Mamba-based model for predicting market direction.

    3-class classification: Bear=0, Uncertain=1, Bull=2.
    Output is raw logits (batch, num_classes).

    Note: CrossEntropyLoss applies softmax internally.
    For inference, use softmax(logits) to get probabilities.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = None,
        d_state: int = None,
        d_conv: int = None,
        expand: int = None,
        n_layers: int = None,
        encoder_only: bool = False,
        arch_params: dict = None,
        num_classes: int = 3,
    ):
        super().__init__()

        if arch_params is None:
            arch_params = build_arch_params()

        self.d_model = d_model if d_model is not None else config.D_MODEL
        self.d_state = d_state if d_state is not None else config.D_STATE
        self.d_conv = d_conv if d_conv is not None else config.D_CONV
        self.expand = expand if expand is not None else config.EXPAND
        self.n_layers = n_layers if n_layers is not None else config.N_LAYERS
        self.input_dim = input_dim
        self.attention_position = "off"
        self.cnn_enabled = arch_params['CNN_ENABLED']
        use_ln = arch_params['MAMBA_LAYERNORM']

        # Input projection: linear path (raw features -> d_model)
        self.linear_proj = nn.Linear(input_dim, self.d_model)

        # CNN path: local pattern extraction (causal convolution)
        self.cnn_fusion = arch_params['CNN_FUSION'] if self.cnn_enabled else "add"
        if self.cnn_enabled:
            kernel_size = arch_params['CNN_KERNEL_SIZE']
            cnn_dropout = arch_params['CNN_DROPOUT']
            self.cnn_pad = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)  # Causal: left-pad only
            self.cnn_conv = nn.Conv1d(input_dim, self.d_model, kernel_size=kernel_size)
            self.cnn_act = nn.GELU()
            self.cnn_drop = nn.Dropout(cnn_dropout)
            self.cnn_norm = nn.LayerNorm(self.d_model) if arch_params['CNN_LAYERNORM'] else nn.Identity()

            # Gated residual: learned gate filters CNN contribution
            if self.cnn_fusion == "gated_residual":
                self.cnn_gate_proj = nn.Linear(self.d_model * 2, self.d_model)

        # Stack of Mamba blocks with partial dropout (only last N layers)
        dropout_threshold = max(0, self.n_layers - arch_params['DROPOUT_LAST_N_LAYERS'])
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=arch_params['DROPOUT'] if i >= dropout_threshold else 0.0,
                use_layernorm=use_ln,
            )
            for i in range(self.n_layers)
        ])

        # Final layer norm before classification
        self.final_norm = nn.LayerNorm(self.d_model) if use_ln else nn.Identity()

        # Cascade training: active depth (None = all layers, set by Trainer)
        self.active_depth = None

        # Classification head (only if not encoder_only)
        if not encoder_only:
            attn_pos = arch_params['ATTENTION_POSITION']
            if attn_pos in ("post_gate", "aligned"):
                # post_gate/aligned only valid for Multi-TF — fallback to pre_gate for mono
                logging.getLogger(__name__).warning(
                    f"ATTENTION_POSITION='{attn_pos}' not supported for mono model, falling back to 'pre_gate'"
                )
                attn_pos = "pre_gate"
            if attn_pos == "pre_gate":
                # Intra-TF attention branch + UnifiedHead
                self.attention_position = "pre_gate"
                self.attn_block = SelfAttentionBlock(
                    self.d_model, arch_params['ATTENTION_NUM_HEADS'],
                    dropout=arch_params['DROPOUT_HEAD']
                )
                self.attn_pool = AttentionPooling(self.d_model)
                self.head = UnifiedHead(
                    self.d_model, n_paths=2,
                    gate_version=arch_params['GATE_VERSION'],
                    dropout=arch_params['DROPOUT_HEAD'],
                    gate_bottleneck=arch_params['GATE_BOTTLENECK'],
                    num_classes=num_classes,
                )
            else:
                # MLP head: concat(last_token, max_pool) -> classification
                self.head = nn.Sequential(
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.GELU(),
                    nn.Dropout(arch_params['DROPOUT_HEAD']),
                    nn.Linear(self.d_model, num_classes),
                )

    def encode(self, x: torch.Tensor, active_depth: int = None) -> torch.Tensor:
        """
        Dual-path input projection + Mamba encoding.

        Path A (Linear): raw features projected to d_model
        Path B (CNN): causal Conv1d extracts local candle patterns
        Fusion: "add" (simple sum) or "gated_residual" (learned gate on CNN)

        Args:
            active_depth: Number of Mamba blocks to process (None = all blocks).
                          Used by cascade training to progressively grow depth.
        """
        # x shape: (Batch, Seq_Len, Features)

        # Path A: linear projection
        out = self.linear_proj(x)

        # Path B: causal CNN (only if enabled)
        if self.cnn_enabled:
            x_cnn = x.transpose(1, 2)               # (B, Features, Seq_Len)
            x_cnn = self.cnn_pad(x_cnn)              # Causal left-padding
            out_cnn = self.cnn_conv(x_cnn)           # (B, d_model, Seq_Len)
            out_cnn = self.cnn_act(out_cnn)
            out_cnn = out_cnn.transpose(1, 2)        # (B, Seq_Len, d_model)
            out_cnn = self.cnn_drop(out_cnn)
            out_cnn = self.cnn_norm(out_cnn)

            if self.cnn_fusion == "gated_residual":
                # Gate decides how much CNN to inject per dimension/timestep
                gate = torch.sigmoid(self.cnn_gate_proj(torch.cat([out, out_cnn], dim=-1)))
                out = out + gate * out_cnn
            else:
                out = out + out_cnn

        # Mamba sequential processing
        depth = active_depth if active_depth is not None else len(self.mamba_blocks)
        for i in range(depth):
            out = self.mamba_blocks[i](out)

        return self.final_norm(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        h = self.encode(x, active_depth=self.active_depth)

        # Extract pooled representations
        last_token = h[:, -1, :]       # (batch, d_model)
        max_pool, _ = h.max(dim=1)     # (batch, d_model)

        if self.attention_position == "pre_gate":
            # Attention parallel branch
            attn_out = self.attn_block(h)           # (batch, seq, d_model)
            attn_pooled = self.attn_pool(attn_out)  # (batch, d_model)
            paths = [(last_token, max_pool), (attn_pooled, attn_pooled)]
            return self.head(paths)
        else:
            # MLP head: concat(last_token, max_pool) -> logits
            combined = torch.cat([last_token, max_pool], dim=-1)  # (batch, 2*d_model)
            return self.head(combined)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities (for inference)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices (for inference)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class MultiTFMambaPredictor(nn.Module):
    """
    Multi-timeframe Mamba predictor with dual encoder branches.

    Two parallel MambaPredictor encoders (primary and secondary) feed into
    a UnifiedHead for fusion and classification.
    """

    def __init__(
        self,
        input_dim_primary: int,
        input_dim_secondary: int,
        d_model: int = None,
        d_state_primary: int = None,
        d_state_secondary: int = None,
        d_conv: int = None,
        expand: int = None,
        n_layers_primary: int = None,
        n_layers_secondary: int = None,
        arch_params: dict = None,
        num_classes: int = 3,
    ):
        super().__init__()

        if arch_params is None:
            arch_params = build_arch_params()

        d_model = d_model if d_model is not None else config.D_MODEL
        d_state_primary = d_state_primary if d_state_primary is not None else config.D_STATE
        d_state_secondary = d_state_secondary if d_state_secondary is not None else config.MULTI_TF_D_STATE
        d_conv = d_conv if d_conv is not None else config.D_CONV
        expand = expand if expand is not None else config.EXPAND
        n_layers_primary = n_layers_primary if n_layers_primary is not None else config.N_LAYERS
        n_layers_secondary = n_layers_secondary if n_layers_secondary is not None else config.MULTI_TF_N_LAYERS

        self.d_model = d_model
        self.input_dim_primary = input_dim_primary
        self.input_dim_secondary = input_dim_secondary
        self.attention_position = arch_params['ATTENTION_POSITION']

        # Primary encoder (main branch) - encoder only, no head
        self.encoder_primary = MambaPredictor(
            input_dim=input_dim_primary,
            d_model=d_model,
            d_state=d_state_primary,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers_primary,
            encoder_only=True,
            arch_params=arch_params,
        )

        # Secondary encoder - encoder only, no head
        self.encoder_secondary = MambaPredictor(
            input_dim=input_dim_secondary,
            d_model=d_model,
            d_state=d_state_secondary,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers_secondary,
            encoder_only=True,
            arch_params=arch_params,
        )

        # Attention branches based on ATTENTION_POSITION
        dropout_head = arch_params['DROPOUT_HEAD']
        if self.attention_position == "pre_gate":
            # Intra-TF self-attention per branch (legacy 4-path mode)
            self.attn_block_primary = SelfAttentionBlock(
                d_model, arch_params['ATTENTION_NUM_HEADS'], dropout=dropout_head
            )
            self.attn_pool_primary = AttentionPooling(d_model)
            self.attn_block_secondary = SelfAttentionBlock(
                d_model, arch_params['ATTENTION_NUM_HEADS'], dropout=dropout_head
            )
            self.attn_pool_secondary = AttentionPooling(d_model)
            n_paths = 4  # std_pri, attn_pri, std_sec, attn_sec
        elif self.attention_position == "post_gate":
            # Cross-TF attention: learned query attends to combined sequences
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=arch_params['ATTENTION_NUM_HEADS'],
                dropout=dropout_head,
                batch_first=True,
            )
            self.cross_attn_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.cross_attn_dropout = nn.Dropout(dropout_head)
            n_paths = 3  # std_pri, std_sec, cross_tf_attn
        elif self.attention_position == "aligned":
            # Aligned cross-TF fusion: token-level temporal alignment.
            # AlignedFusion injects secondary into primary via gated residual
            # (enriched = h_pri + gate * proj(h_pri, h_sec_aligned)).
            # The enriched primary already contains cross-TF info, so the head
            # only needs a single path — no Gate/UnifiedHead required.
            self.aligned_fusion = AlignedFusion(d_model, dropout=dropout_head)
            n_paths = 1
        else:
            n_paths = 2  # std_pri, std_sec

        # Cascade training: active depth per branch (None = all layers, set by Trainer)
        self.active_depth_primary = None
        self.active_depth_secondary = None

        # Head: single-path MLP for aligned (secondary already fused),
        # multi-path UnifiedHead with Gate for all other modes
        if n_paths == 1:
            self.head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(d_model, num_classes),
            )
        else:
            self.head = UnifiedHead(
                d_model, n_paths, arch_params['GATE_VERSION'],
                dropout=dropout_head, gate_bottleneck=arch_params['GATE_BOTTLENECK'],
                num_classes=num_classes,
            )

    def forward(self, x_primary: torch.Tensor, x_secondary: torch.Tensor,
                tf_mapping: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with dual inputs.

        Args:
            x_primary: Primary branch input (batch, seq_len, input_dim_primary)
            x_secondary: Secondary branch input (batch, seq_len, input_dim_secondary)
            tf_mapping: (batch, seq_len) LongTensor — temporal alignment mapping
                        (required for attention_position="aligned", ignored otherwise)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        h_primary = self.encoder_primary.encode(x_primary, active_depth=self.active_depth_primary)      # (batch, seq, d_model)
        h_secondary = self.encoder_secondary.encode(x_secondary, active_depth=self.active_depth_secondary)  # (batch, seq, d_model)

        if self.attention_position == "aligned":
            # Single-path: AlignedFusion injects secondary into primary via gated residual,
            # then MLP head on the enriched primary only (no Gate/UnifiedHead).
            h_enriched = self.aligned_fusion(h_primary, h_secondary, tf_mapping)
            last_enriched = h_enriched[:, -1, :]
            max_enriched, _ = h_enriched.max(dim=1)
            return self.head(torch.cat([last_enriched, max_enriched], dim=-1))

        # Multi-path modes: pooled representations → UnifiedHead with Gate
        last_pri = h_primary[:, -1, :]
        max_pri, _ = h_primary.max(dim=1)
        last_sec = h_secondary[:, -1, :]
        max_sec, _ = h_secondary.max(dim=1)

        if self.attention_position == "pre_gate":
            # Intra-TF attention per branch: 4 paths
            attn_pri = self.attn_block_primary(h_primary)
            attn_pooled_pri = self.attn_pool_primary(attn_pri)
            attn_sec = self.attn_block_secondary(h_secondary)
            attn_pooled_sec = self.attn_pool_secondary(attn_sec)
            paths = [
                (last_pri, max_pri),
                (attn_pooled_pri, attn_pooled_pri),
                (last_sec, max_sec),
                (attn_pooled_sec, attn_pooled_sec),
            ]
        elif self.attention_position == "post_gate":
            # Cross-TF attention: combined sequences → learned query → 3 paths
            h_combined = torch.cat([h_primary, h_secondary], dim=1)  # (B, 2*seq, d_model)
            B = h_combined.size(0)
            query = self.cross_attn_query.expand(B, -1, -1)  # (B, 1, d_model)
            attn_out, _ = self.cross_attn(query, h_combined, h_combined)  # (B, 1, d_model)
            attn_out = self.cross_attn_dropout(attn_out.squeeze(1))  # (B, d_model)
            paths = [
                (last_pri, max_pri),
                (last_sec, max_sec),
                (attn_out, attn_out),
            ]
        else:
            # No attention: 2 paths
            paths = [(last_pri, max_pri), (last_sec, max_sec)]

        return self.head(paths)

    def predict_proba(self, x_primary: torch.Tensor, x_secondary: torch.Tensor,
                      tf_mapping: torch.Tensor = None) -> torch.Tensor:
        """Get class probabilities (for inference)."""
        logits = self.forward(x_primary, x_secondary, tf_mapping)
        return torch.softmax(logits, dim=-1)

    def predict(self, x_primary: torch.Tensor, x_secondary: torch.Tensor,
                tf_mapping: torch.Tensor = None) -> torch.Tensor:
        """Get predicted class indices (for inference)."""
        logits = self.forward(x_primary, x_secondary, tf_mapping)
        return torch.argmax(logits, dim=-1)
