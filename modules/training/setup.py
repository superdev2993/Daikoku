"""
Training setup utilities shared between main.py and optimize.py.

Provides factory functions for model, optimizer, dataloaders, and trainer config
to eliminate duplication and ensure consistency between entry points.
"""

import logging
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset

import config
from modules.data.dataset import ChunkSampler
from modules.model.mamba import MambaPredictor, MultiTFMambaPredictor

_logger = logging.getLogger(__name__)


def resolve_num_workers(num_workers=None, use_cuda=None):
    """
    Resolve NUM_WORKERS: "auto" adapts to CPU cores and device, int is passed through.

    Auto logic:
        GPU mode:  min(cpu_count // 2, 8) — workers load data while GPU computes
        CPU mode:  0 — CPU is already busy with tensor ops, workers would cause contention

    Args:
        num_workers: "auto" or int. None defaults to config.NUM_WORKERS.
        use_cuda: whether CUDA is used. None auto-detects.

    Returns:
        int: resolved number of workers
    """
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if use_cuda is None:
        use_cuda = torch.cuda.is_available() and getattr(config, 'DEVICE', 'auto').lower() != 'cpu'

    if isinstance(num_workers, int):
        return num_workers

    # Auto mode
    cpu_count = os.cpu_count() or 4
    if use_cuda:
        resolved = min(cpu_count // 2, 8)
    else:
        resolved = 0

    _logger.info(f"NUM_WORKERS='auto' resolved to {resolved} "
                 f"(cpu_count={cpu_count}, cuda={'yes' if use_cuda else 'no'})")
    return resolved


def merge_datasets(train_datasets, test_datasets):
    """
    Merge multiple train/test datasets into single datasets.

    Args:
        train_datasets: list of train Dataset objects
        test_datasets: list of test Dataset objects

    Returns:
        (train_dataset, test_dataset)
    """
    if len(train_datasets) == 1:
        return train_datasets[0], test_datasets[0]
    return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


def create_dataloaders(train_dataset, test_dataset, batch_size,
                       shuffle_train, shuffle_chunks, seed,
                       num_workers=0, persistent_workers=False, use_cuda=True):
    """
    Create train and test DataLoaders with optional ChunkSampler.

    Args:
        train_dataset: training Dataset
        test_dataset: test Dataset
        batch_size: batch size
        shuffle_train: whether to shuffle training data
        shuffle_chunks: number of chunks for ChunkSampler (>1 enables it)
        seed: random seed for ChunkSampler
        num_workers: DataLoader workers (0 for Optuna compatibility)
        persistent_workers: keep workers alive between batches
        use_cuda: enable pin_memory for GPU

    Returns:
        (train_loader, test_loader, sampler_info) where sampler_info is None or chunk count
    """
    pw = persistent_workers and num_workers > 0

    # Chunk shuffle: SHUFFLE_CHUNKS > 1 uses ChunkSampler (mutually exclusive with shuffle=True)
    if shuffle_chunks > 1 and shuffle_train:
        train_sampler = ChunkSampler(train_dataset, n_chunks=shuffle_chunks, seed=seed)
        train_shuffle = False
        sampler_info = shuffle_chunks
    else:
        train_sampler = None
        train_shuffle = shuffle_train
        sampler_info = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=pw,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=pw,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, test_loader, sampler_info


def create_model(mtf_mode, n_features, device, d_model, d_state, d_conv,
                 expand, n_layers, arch_params, num_classes,
                 n_features_secondary=None, mtf_d_state=None, mtf_n_layers=None):
    """
    Create and initialize model based on multi-timeframe mode.

    Args:
        mtf_mode: "off", "calibration", or "dual"
        n_features: primary input feature count
        device: torch device
        d_model: model dimension
        d_state: SSM state dimension (primary)
        d_conv: convolution kernel size
        expand: expansion factor
        n_layers: number of layers (primary)
        arch_params: architecture parameters from build_arch_params()
        num_classes: number of output classes
        n_features_secondary: secondary input features (dual mode)
        mtf_d_state: SSM state dimension for secondary/calibration
        mtf_n_layers: number of layers for secondary/calibration

    Returns:
        model on device
    """
    if mtf_mode == "dual":
        model = MultiTFMambaPredictor(
            input_dim_primary=n_features,
            input_dim_secondary=n_features_secondary,
            d_model=d_model,
            d_state_primary=d_state,
            d_state_secondary=mtf_d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers_primary=n_layers,
            n_layers_secondary=mtf_n_layers,
            arch_params=arch_params,
            num_classes=num_classes,
        )
    elif mtf_mode == "calibration":
        model = MambaPredictor(
            input_dim=n_features,
            d_model=d_model,
            d_state=mtf_d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=mtf_n_layers,
            arch_params=arch_params,
            num_classes=num_classes,
        )
    else:
        model = MambaPredictor(
            input_dim=n_features,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            arch_params=arch_params,
            num_classes=num_classes,
        )

    return model.to(device)


def create_optimizer(model, learning_rate, weight_decay):
    """
    Create AdamW optimizer with decay/no-decay parameter groups.

    SSM state params (A_log, D), biases, and norms get zero weight decay.
    All other params get the specified weight decay.

    Args:
        model: the model
        learning_rate: learning rate
        weight_decay: weight decay for projection weights

    Returns:
        (optimizer, n_decay, n_no_decay) — optimizer and param counts
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # SSM state matrix & skip connection, all biases, all layer norms
        if name.endswith(('.A_log', '.D')) or 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=learning_rate)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)

    return optimizer, n_decay, n_no_decay


def build_trainer_config(normalization_params, n_norm_features, overrides=None):
    """
    Build trainer config dict from config.py with dynamic parameters.

    Auto-extracts all UPPERCASE constants from config module,
    adds dynamic parameters, and applies optional overrides.

    Args:
        normalization_params: normalization parameters from preprocessing
        n_norm_features: number of features to normalize per-window
        overrides: optional dict of values to override (e.g. trial params)

    Returns:
        Complete trainer config dict
    """
    # Auto-extract ALL parameters from config.py (UPPERCASE constants)
    trainer_config = {
        key: value
        for key, value in vars(config).items()
        if key.isupper() and not key.startswith('_')
    }

    # Add dynamic parameters not in config.py (needed for inference)
    trainer_config['normalization_params'] = normalization_params
    trainer_config['n_norm_features'] = n_norm_features

    # Apply overrides (e.g. trial-specific parameters from Optuna)
    if overrides:
        trainer_config.update(overrides)

    return trainer_config
