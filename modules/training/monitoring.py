"""
Activation monitoring for training health diagnostics.

Collects activation statistics via temporary forward hooks.
Designed for minimal overhead: hooks are registered and removed per collect() call.
"""

import torch
import torch.nn as nn


class ActivationMonitor:
    """
    Monitors activation health during training.

    Registers temporary forward hooks on linear_proj and mamba_blocks,
    runs a single forward pass, computes stats, then removes all hooks.
    """

    def __init__(self, model: nn.Module, dead_threshold: float = 1e-6):
        """
        Args:
            model: MambaPredictor instance
            dead_threshold: Features with std < threshold are considered dead
        """
        self.model = model
        self.dead_threshold = dead_threshold

    def collect(self, batch: tuple, device: torch.device) -> dict:
        """
        Collect activation statistics for one batch.

        Cycle:
        1. Register forward hooks on linear_proj + mamba_blocks
        2. Run forward pass in eval mode with no_grad
        3. Compute stats per layer
        4. Remove all hooks
        5. Return stats dict

        Args:
            batch: Tuple of window tensors (primary,) or (primary, secondary)
            device: torch.device to use

        Returns:
            Dict with stats per layer + '_global' summary.
            Each layer key maps to:
                mean, std, dead_pct, n_dead, n_total
        """
        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                activations[name] = output.detach()
            return hook_fn

        # Detect model type and register hooks accordingly
        # Lazy import to avoid circular dependency (monitoring -> mamba -> monitoring)
        from modules.model.mamba import MultiTFMambaPredictor
        is_dual = isinstance(self.model, MultiTFMambaPredictor)

        if is_dual:
            encoders = [
                (self.model.encoder_primary, "primary_"),
                (self.model.encoder_secondary, "secondary_"),
            ]
        else:
            encoders = [(self.model, "")]

        for encoder, prefix in encoders:
            hooks.append(
                encoder.linear_proj.register_forward_hook(
                    make_hook(f"{prefix}linear_proj")
                )
            )
            for i, block in enumerate(encoder.mamba_blocks):
                hooks.append(
                    block.register_forward_hook(make_hook(f"{prefix}mamba_block_{i}"))
                )

        # Forward pass (eval, no grad) — hooks must be removed even on failure
        windows = batch[0].to(device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                if is_dual:
                    windows_sec = batch[1].to(device)
                    self.model.encoder_primary.encode(windows)
                    self.model.encoder_secondary.encode(windows_sec)
                else:
                    self.model(windows)
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                self.model.train()

        # Compute stats per layer
        stats = {}
        total_neurons = 0
        total_dead = 0

        for name, act in activations.items():
            # act shape: (batch, seq_len, d_model) or (batch, d_model)
            # Compute per-feature std across batch and sequence dims
            if act.dim() == 3:
                # (batch, seq, features) -> std over batch and seq
                feature_std = act.std(dim=(0, 1))
                feature_mean = act.mean(dim=(0, 1))
            elif act.dim() == 2:
                # (batch, features) -> std over batch
                feature_std = act.std(dim=0)
                feature_mean = act.mean(dim=0)
            else:
                continue

            n_total = feature_std.shape[0]
            n_dead = int((feature_std < self.dead_threshold).sum().item())
            dead_pct = (n_dead / n_total * 100.0) if n_total > 0 else 0.0

            stats[name] = {
                "mean": float(feature_mean.mean().item()),
                "std": float(feature_std.mean().item()),
                "dead_pct": dead_pct,
                "n_dead": n_dead,
                "n_total": n_total,
            }

            total_neurons += n_total
            total_dead += n_dead

        # Global summary
        stats["_global"] = {
            "dead_pct_total": (total_dead / total_neurons * 100.0) if total_neurons > 0 else 0.0,
            "total_neurons": total_neurons,
            "total_dead": total_dead,
        }

        # Free activation tensors
        activations.clear()

        return stats
