"""
Pure PyTorch implementation of the Mamba SSM block for CPU execution.

Drop-in replacement for mamba_ssm.Mamba when CUDA is not available.
Implements the selective state space model using standard PyTorch operations
with a JIT-compiled recurrent scan (slower than CUDA kernels but functionally
equivalent).

State dict keys are identical to mamba_ssm.Mamba, so checkpoints trained on
GPU can be loaded directly for CPU inference/evaluation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _selective_scan_jit(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    JIT-compiled selective scan (recurrent formulation).

    Compiles the Python loop to C++ via TorchScript, eliminating interpreter
    overhead and enabling fused tensor operations within each timestep.

    Args:
        x:  (batch, seq_len, d_inner) — input after conv + activation
        dt: (batch, seq_len, d_inner) — discretization step sizes
        A:  (d_inner, d_state)        — state transition matrix (negative)
        B:  (batch, seq_len, d_state) — input-dependent projection
        C:  (batch, seq_len, d_state) — output-dependent projection

    Returns:
        y:  (batch, seq_len, d_inner)
    """
    batch = x.shape[0]
    seq_len = x.shape[1]
    d_inner = x.shape[2]
    d_state = A.shape[1]

    y = torch.zeros_like(x)
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        dt_t = dt[:, t, :].unsqueeze(-1)                     # (B, d_inner, 1)
        dA = torch.exp(dt_t * A)                              # (B, d_inner, d_state)
        dB = dt_t * B[:, t, :].unsqueeze(1)                   # (B, d_inner, d_state)
        h = dA * h + dB * x[:, t, :].unsqueeze(-1)            # (B, d_inner, d_state)
        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1)    # (B, d_inner)

    return y


class Mamba(nn.Module):
    """
    Pure PyTorch Mamba (Selective State Space Model).

    Same interface and state_dict as mamba_ssm.Mamba:
        Mamba(d_model, d_state, d_conv, expand)
        input:  (batch, seq_len, d_model)
        output: (batch, seq_len, d_model)

    Initialization mirrors mamba_ssm exactly (S4D/HiPPO for A, inverse-softplus
    for dt bias) so models trained from scratch on CPU produce comparable results
    to GPU-initialized models.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        # Accepted but ignored (CUDA-specific)
        use_fast_path: bool = True,
        layer_idx: int = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        factory_kwargs = {"device": device, "dtype": dtype}

        # Input projection: x and z (gate) branches
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias, **factory_kwargs)

        # Depthwise causal convolution on x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )

        # SSM parameter projections: dt (low-rank), B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs
        )

        # dt: low-rank -> full inner dimension
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # --- Initialization matching mamba_ssm exactly ---

        # dt_proj.weight: uniform(-dt_init_std, dt_init_std)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt_proj.bias: inverse softplus so that softplus(bias) ~ Uniform[dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A matrix: S4D real initialization (log-space HiPPO)
        A = torch.arange(
            1, d_state + 1, dtype=torch.float32, device=device
        ).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection scalar per inner dimension
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
            inference_params: ignored (CUDA step-mode compatibility)

        Returns:
            (batch, seq_len, d_model)
        """
        B, L, _ = hidden_states.shape

        # Project to x and z (gate) branches
        xz = self.in_proj(hidden_states)                # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)               # each (B, L, d_inner)

        # Causal depthwise conv1d on x branch
        x_conv = x_branch.transpose(1, 2)               # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]          # trim to causal length
        x_branch = F.silu(x_conv.transpose(1, 2))       # (B, L, d_inner)

        # SSM parameters from x
        x_ssm = self.x_proj(x_branch)                   # (B, L, dt_rank + 2*d_state)
        dt_raw, B_param, C_param = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt: low-rank projection + softplus
        dt = F.softplus(self.dt_proj(dt_raw))            # (B, L, d_inner)

        # A: negative exponential for stability
        A = -torch.exp(self.A_log.float())               # (d_inner, d_state)

        # Selective scan (JIT-compiled recurrence)
        y = _selective_scan_jit(x_branch, dt, A, B_param, C_param)

        # Skip connection + output gating
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)
