"""
Test trainer.py - Validation of training components
"""

import sys
import os
import pytest

if os.environ.get('CI'):
    pytest.skip("Skipped on CI (requires model training)", allow_module_level=True)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from conftest import TEST_CSV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.dataset import TimeSeriesWindowDataset
from modules.training.trainer import get_device, Trainer
from modules.model.mamba import MambaPredictor
from modules.data.loader import get_raw_data
from modules.data.transform import transform_pipeline
from modules.data.labeling import get_labels
from modules.utils.logger import get_logger
import config
from modules.utils.seed import set_seed

logger = get_logger(__name__)


def test_dataset():
    """Test TimeSeriesWindowDataset creation and access"""
    print("\n=== Test: TimeSeriesWindowDataset ===")

    # Create test data
    # With new design: label = labels[data_idx + window_size - 1]
    # So labels must have at least max(indices) + window_size elements
    window_size = 400
    n_indices = 50
    data = np.random.randn(1000, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)

    # Create indices (50 valid indices)
    indices = list(range(n_indices))

    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)

    # Check length
    assert len(dataset) == 50, f"Expected 50 samples, got {len(dataset)}"

    # Check sample
    window, label = dataset[0]
    assert window.shape == (400, 5), f"Expected shape (400, 5), got {window.shape}"
    assert label.item() in [0, 1, 2], f"Expected label in {{0,1,2}}, got {label.item()}"
    assert window.dtype == torch.float32, f"Expected float32, got {window.dtype}"
    assert label.dtype == torch.long, f"Expected long, got {label.dtype}"

    print(f"✓ Dataset works correctly")
    print(f"  - Length: {len(dataset)}")
    print(f"  - Window shape: {window.shape}")
    print(f"  - Label: {label.item()} (type: {label.dtype})")


def test_get_device():
    """Test device auto-detection"""
    print("\n=== Test: get_device ===")

    # Test auto mode
    device = get_device("auto")
    assert device.type in ["cuda", "cpu"], f"Invalid device type: {device.type}"
    print(f"✓ Auto-detected device: {device}")

    # Test cpu mode
    device_cpu = get_device("cpu")
    assert device_cpu.type == "cpu", f"Expected cpu, got {device_cpu.type}"
    print(f"✓ CPU device: {device_cpu}")


def test_trainer_one_epoch():
    """Test one complete training epoch on small dataset"""
    print("\n=== Test: Trainer (one epoch) ===")

    # Create small dataset
    # With new design: label = labels[data_idx + window_size - 1]
    # So labels must have at least max(indices) + window_size elements
    window_size = 400
    n_indices = 100
    data = np.random.randn(1000, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)
    indices = list(range(n_indices))

    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create model and trainer components
    device = get_device("cuda")
    model = MambaPredictor(
        input_dim=5,
        d_model=32,  # Smaller for testing
        d_state=8,
        d_conv=2,
        expand=2,
        n_layers=1   # Single layer for speed
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create temporary TensorBoard writer
    writer = SummaryWriter(log_dir='runs/tests/test')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': True,
        'KEEP_LATEST': False
    }

    trainer = Trainer(
        model=model,
        train_loader=loader,
        test_loader=loader,  # Use same loader for test (just for testing)
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=trainer_config,
        tensorboard_writer=writer,
        logger=logger,
        setup_signal_handler=False  # Disable for pytest
    )

    # Run one epoch
    print("  Running training epoch...")
    results = trainer.train_epoch()

    # Validate results
    assert 'train_loss' in results, "Missing train_loss"
    assert 'train_acc' in results, "Missing train_acc"
    assert 'y_true' in results, "Missing y_true"
    assert 'y_pred' in results, "Missing y_pred"
    assert 'y_pred_probs' in results, "Missing y_pred_probs"

    assert 0 <= results['train_acc'] <= 1, f"Invalid accuracy: {results['train_acc']}"
    assert results['train_loss'] > 0, f"Invalid loss: {results['train_loss']}"

    print(f"✓ Training epoch completed")
    print(f"  - Loss: {results['train_loss']:.4f}")
    print(f"  - Accuracy: {results['train_acc']:.4f}")

    # Test evaluation
    print("  Running evaluation...")
    eval_results = trainer.evaluate()

    assert 'test_loss' in eval_results, "Missing test_loss"
    assert 'test_acc' in eval_results, "Missing test_acc"

    print(f"✓ Evaluation completed")
    print(f"  - Test Loss: {eval_results['test_loss']:.4f}")
    print(f"  - Test Accuracy: {eval_results['test_acc']:.4f}")

    # Cleanup
    writer.close()


def test_checkpoint_save_load():
    """Test checkpoint saving and loading"""
    print("\n=== Test: Checkpoint Save/Load ===")

    # Create minimal setup
    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy data for loader
    # With new design: labels must have at least max(indices) + window_size elements
    window_size = 400
    n_indices = 50
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)
    indices = list(range(n_indices))
    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    writer = SummaryWriter(log_dir='runs/tests/test_checkpoint')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': True
    }

    trainer = Trainer(
        model=model,
        train_loader=loader,
        test_loader=loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config=trainer_config,
        tensorboard_writer=writer,
        logger=logger,
        setup_signal_handler=False
    )

    # Save checkpoint
    print("  Saving checkpoint...")
    metrics = {'train_loss': 1.5, 'test_loss': 1.6, 'train_acc': 0.5, 'test_acc': 0.48}
    trainer.save_checkpoint(epoch=5, metrics=metrics, is_best=False)

    checkpoint_path = 'models/test/checkpoint_epoch_6.pt'
    assert os.path.exists(checkpoint_path), f"Checkpoint not created: {checkpoint_path}"
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Load checkpoint
    print("  Loading checkpoint...")
    start_epoch = trainer.load_checkpoint(checkpoint_path)

    assert start_epoch == 6, f"Expected start_epoch=6, got {start_epoch}"
    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Start epoch: {start_epoch}")

    # Cleanup
    writer.close()
    os.remove(checkpoint_path)


def test_full_training_pipeline():
    """Test complete training pipeline with real data (2 epochs)"""
    print("\n=== Test: Full Training Pipeline (2 epochs) ===")

    # Load real data
    print("  Loading test dataset...")
    df = get_raw_data(TEST_CSV)
    print(f"  Loaded {len(df)} rows")

    # Transform data
    print("  Transforming data...")
    result = transform_pipeline(df)
    all_data = result['data']
    normalization_params = result['normalization_params']
    window_size = result['window_size']
    split_ratio = result['split_ratio']

    print(f"  Transformed data: {len(all_data)} rows")

    # Get labels
    print("  Computing labels...")
    all_labels, _, _, _, _ = get_labels(df, window_size=window_size)
    print(f"  Total labels: {len(all_labels)}")

    # Calculate valid indices (where we have both window and label)
    # With new design: label = labels[data_idx + window_size - 1]
    # Constraints:
    #   1. data_idx + window_size <= len(all_data) (for window extraction)
    #   2. data_idx + window_size - 1 < len(all_labels) (for label extraction)
    # So: data_idx < min(len(all_data) - window_size, len(all_labels) - window_size + 1)
    max_data_idx = len(all_data) - window_size
    max_label_idx = len(all_labels) - window_size + 1
    n_valid = min(max_data_idx, max_label_idx)
    valid_indices = list(range(n_valid))

    print(f"  Valid indices: {len(valid_indices)}")

    # Split indices (80/20)
    split_idx = int(len(valid_indices) * split_ratio)
    train_indices = valid_indices[:split_idx]
    test_indices = valid_indices[split_idx:]

    print(f"  Train indices: {len(train_indices)}")
    print(f"  Test indices: {len(test_indices)}")

    # Create datasets with indices
    train_dataset = TimeSeriesWindowDataset(all_data, all_labels, train_indices, window_size=window_size)
    test_dataset = TimeSeriesWindowDataset(all_data, all_labels, test_indices, window_size=window_size)

    print(f"  Train dataset: {len(train_dataset)} windows")
    print(f"  Test dataset: {len(test_dataset)} windows")

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create model
    n_features = all_data.shape[1]
    device = get_device("cuda")
    model = MambaPredictor(
        input_dim=n_features,
        d_model=32,  # Small for fast testing
        d_state=8,
        n_layers=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir='runs/tests/test_full_training')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'EPOCHS': 2,
        'CHECKPOINT_DIR': 'models/test_full/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': True,
        'KEEP_LATEST': True,
        'normalization_params': normalization_params,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=trainer_config,
        tensorboard_writer=writer,
        logger=logger,
        setup_signal_handler=False  # Disable for pytest
    )

    # Train for 2 epochs
    print("  Training for 2 epochs...")
    trainer.train(start_epoch=0)

    # Check checkpoints created
    checkpoint_dir = Path('models/test_full/')
    assert (checkpoint_dir / 'checkpoint_epoch_1.pt').exists(), "Missing epoch 1 checkpoint"
    assert (checkpoint_dir / 'checkpoint_epoch_2.pt').exists(), "Missing epoch 2 checkpoint"
    assert (checkpoint_dir / 'latest.pt').exists(), "Missing latest checkpoint"
    assert (checkpoint_dir / 'best_model.pt').exists(), "Missing best model checkpoint"

    print(f"✓ Training completed successfully")
    print(f"  - Checkpoints created: epoch_1, epoch_2, latest, best")

    # Cleanup
    writer.close()
    for f in checkpoint_dir.glob('*.pt'):
        f.unlink()

    print(f"✓ Full training pipeline validated")


def test_checkpoint_content():
    """Test checkpoint contains all required fields"""
    print("\n=== Test: Checkpoint Content ===")

    # Create minimal training setup
    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # With new design: labels must have at least max(indices) + window_size elements
    window_size = 400
    n_indices = 50
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)
    indices = list(range(n_indices))
    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    writer = SummaryWriter(log_dir='runs/tests/test_checkpoint_content')

    # Prepare normalization params (5 features: OHLCV)
    normalization_params = {
        'mean': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'std': np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        'columns': ['log_price', 'log_open', 'log_wick_high', 'log_wick_low', 'log_volume']
    }

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test_content/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': True,
        'normalization_params': normalization_params,
    }

    trainer = Trainer(
        model=model,
        train_loader=loader,
        test_loader=loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config=trainer_config,
        tensorboard_writer=writer,
        logger=logger
    )

    # Save checkpoint
    metrics = {'train_loss': 1.5, 'test_loss': 1.6, 'train_acc': 0.5, 'test_acc': 0.48}
    trainer.save_checkpoint(epoch=0, metrics=metrics, is_best=True)

    # Load and verify checkpoint content (saved as best_model.pt)
    checkpoint_path = 'models/test_content/best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Verify required fields
    required_fields = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                      'train_loss', 'test_loss', 'train_acc', 'test_acc',
                      'normalization_params', 'config']

    for field in required_fields:
        assert field in checkpoint, f"Missing field in checkpoint: {field}"
        print(f"  ✓ {field}: present")

    # Verify normalization params
    assert checkpoint['normalization_params'] is not None, "normalization_params is None"
    assert 'mean' in checkpoint['normalization_params'], "Missing mean in normalization_params"
    assert 'std' in checkpoint['normalization_params'], "Missing std in normalization_params"
    print(f"  ✓ Normalization params: {checkpoint['normalization_params']['columns']}")

    # Verify best_model.pt also exists
    best_path = Path('models/test_content/best_model.pt')
    assert best_path.exists(), "best_model.pt not created"
    print(f"  ✓ best_model.pt created")

    print(f"✓ Checkpoint content validated")

    # Cleanup
    writer.close()
    for f in Path('models/test_content/').glob('*.pt'):
        f.unlink()


def test_logging():
    """Test console and TensorBoard logging"""
    print("\n=== Test: Logging (Console + TensorBoard) ===")

    # Create small training setup
    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # With new design: labels must have at least max(indices) + window_size elements
    window_size = 400
    n_indices = 100
    data = np.random.randn(1000, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)
    indices = list(range(n_indices))
    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # TensorBoard writer
    tb_log_dir = 'runs/tests/test_logging'
    writer = SummaryWriter(log_dir=tb_log_dir)

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test_logging/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': True
    }

    trainer = Trainer(
        model=model,
        train_loader=loader,
        test_loader=loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config=trainer_config,
        tensorboard_writer=writer,
        logger=logger,
        setup_signal_handler=False
    )

    # Train one epoch
    print("  Training one epoch to generate logs...")
    trainer.train(start_epoch=0)

    # Check TensorBoard events file was created
    tb_events = list(Path(tb_log_dir).glob('events.out.tfevents.*'))
    assert len(tb_events) > 0, "No TensorBoard events file created"
    print(f"✓ TensorBoard events file created: {tb_events[0].name}")

    # Verify event file is not empty
    event_size = tb_events[0].stat().st_size
    assert event_size > 0, "TensorBoard events file is empty"
    print(f"  - Event file size: {event_size} bytes")

    print(f"✓ Logging validated")

    # Cleanup
    writer.close()
    for f in Path('models/test_logging/').glob('*.pt'):
        f.unlink()

    print(f"✓ Logging test completed")


# ============================================================================
# Group 1: LR Scheduler creation
# ============================================================================

def test_scheduler_cosine():
    """Test cosine LR scheduler creation and step behavior."""
    print("\n=== Test: scheduler_cosine ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_sched_cosine')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'KEEP_LATEST': False,
        'EPOCHS': 10,
        'CHECKPOINT_DIR': 'models/test_sched/',
        'SAVE_EVERY_N_EPOCHS': 100,
        'SAVE_ON_INTERRUPT': False,
        'LR_SCHEDULER': 'cosine',
        'LR_MIN': 0.00001,
        'LEARNING_RATE': 0.001,
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    assert trainer.scheduler is not None, "Cosine scheduler should be created"

    # Step and verify LR decreases
    lr_start = optimizer.param_groups[0]['lr']
    trainer.scheduler.step()
    trainer.scheduler.step()
    lr_mid = optimizer.param_groups[0]['lr']
    assert lr_mid < lr_start, f"LR should decrease: {lr_start} -> {lr_mid}"

    # Step to end
    for _ in range(8):
        trainer.scheduler.step()
    lr_end = optimizer.param_groups[0]['lr']
    assert lr_end < lr_mid, f"LR should keep decreasing: {lr_mid} -> {lr_end}"
    assert abs(lr_end - 0.00001) < 1e-5, f"LR should reach LR_MIN, got {lr_end}"

    writer.close()
    print(f"  LR: {lr_start} -> {lr_mid} -> {lr_end}")
    print("  PASS")


def test_scheduler_wsd():
    """Test WSD (warmup-stable-decay) scheduler creation and phases."""
    print("\n=== Test: scheduler_wsd ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_sched_wsd')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'KEEP_LATEST': False,
        'EPOCHS': 10,
        'CHECKPOINT_DIR': 'models/test_sched/',
        'SAVE_EVERY_N_EPOCHS': 100,
        'SAVE_ON_INTERRUPT': False,
        'LR_SCHEDULER': 'wsd',
        'LR_MIN': 0.00005,
        'LEARNING_RATE': 0.001,
        'LR_WARMUP_PCT': 0.3,   # 3 epochs warmup
        'LR_STABLE_PCT': 0.3,   # 3 epochs stable
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    assert trainer.scheduler is not None, "WSD scheduler should be created"

    # Collect LR at each epoch
    lrs = [optimizer.param_groups[0]['lr']]
    for _ in range(10):
        trainer.scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Warmup: LR should increase (epochs 0-2)
    assert lrs[2] > lrs[0], f"Warmup phase: LR should increase {lrs[0]} -> {lrs[2]}"
    # Stable: LR ~= LR_MAX (epoch 3)
    assert abs(lrs[3] - 0.001) < 1e-5, f"Stable phase: LR should be ~0.001, got {lrs[3]}"
    # Decay: LR should decrease after stable (epoch 6+)
    assert lrs[9] < lrs[5], f"Decay phase: LR should decrease {lrs[5]} -> {lrs[9]}"

    writer.close()
    print(f"  LR progression: {[f'{lr:.6f}' for lr in lrs[:6]]}...")
    print("  PASS")


def test_scheduler_unknown():
    """Test unknown scheduler type falls back to None."""
    print("\n=== Test: scheduler_unknown ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_sched_unknown')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'KEEP_LATEST': False,
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test_sched/',
        'SAVE_EVERY_N_EPOCHS': 100,
        'SAVE_ON_INTERRUPT': False,
        'LR_SCHEDULER': 'nonexistent_scheduler',
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    assert trainer.scheduler is None, "Unknown scheduler should return None"

    writer.close()
    print("  PASS")


# ============================================================================
# Group 2: Health alerts
# ============================================================================

def test_health_alerts_dead_neurons():
    """Test _check_health_alerts fires dead neuron alert above threshold."""
    print("\n=== Test: health_alerts_dead_neurons ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_health_dead')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1, 'CHECKPOINT_DIR': 'models/test_health/',
        'SAVE_EVERY_N_EPOCHS': 100, 'SAVE_ON_INTERRUPT': False,
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    # Simulate stats with high dead neuron percentage
    stats = {
        '_global': {'dead_pct_total': 50.0, 'total_dead': 100, 'total_neurons': 200},
        'mamba_block_0': {'mean': 0.01, 'std': 0.005, 'dead_pct': 40.0},
    }
    metrics = {'grad_norm_mean': 0.1}

    # Should not raise — just log warnings
    trainer._check_health_alerts(0, stats, metrics)

    writer.close()
    print("  Dead neuron alert triggered (50% > threshold 30%)")
    print("  PASS")


def test_health_alerts_grad_vanish_explode():
    """Test _check_health_alerts fires gradient vanish/explode alerts."""
    print("\n=== Test: health_alerts_grad_vanish_explode ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_health_grad')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1, 'CHECKPOINT_DIR': 'models/test_health/',
        'SAVE_EVERY_N_EPOCHS': 100, 'SAVE_ON_INTERRUPT': False,
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    stats_clean = {'_global': {'dead_pct_total': 0.0}}

    # Test gradient vanishing
    trainer._check_health_alerts(0, stats_clean, {'grad_norm_mean': 1e-10})

    # Test gradient exploding
    trainer._check_health_alerts(0, stats_clean, {'grad_norm_mean': 5000.0})

    # Test no grad_norm (should not crash)
    trainer._check_health_alerts(0, stats_clean, {})

    writer.close()
    print("  Gradient vanish + explode alerts triggered")
    print("  PASS")


# ============================================================================
# Group 3: Feature names
# ============================================================================

def test_get_feature_names_with_norm_params():
    """Test _get_feature_names with normalization_params (norm_params branch)."""
    print("\n=== Test: get_feature_names_with_norm_params ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_feat_names')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1, 'CHECKPOINT_DIR': 'models/test_fn/',
        'SAVE_EVERY_N_EPOCHS': 100, 'SAVE_ON_INTERRUPT': False,
        'normalization_params': {
            'columns': ['log_price', 'log_open', 'log_wick_high', 'log_wick_low',
                        'log_volume', 'log_zz', 'log_atr'],
        },
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    # 18 features: 7 norm + 4 time + 5 struct + 2 regime (has_time=True)
    names_18 = trainer._get_feature_names(18)
    assert len(names_18) == 18, f"Expected 18 names, got {len(names_18)}"
    assert names_18[0] == 'log_price'
    assert 'sin_hour' in names_18
    assert 'regime_trend' in names_18

    # 14 features: 7 norm + 5 struct + 2 regime (no time, calibration branch)
    names_14 = trainer._get_feature_names(14)
    assert len(names_14) == 14, f"Expected 14 names, got {len(names_14)}"
    assert 'sin_hour' not in names_14
    assert 'struct_rank' in names_14

    writer.close()
    print(f"  18 features: {names_18[:4]}...{names_18[-2:]}")
    print(f"  14 features: {names_14[:4]}...{names_14[-2:]}")
    print("  PASS")


def test_get_feature_names_fallback():
    """Test _get_feature_names without normalization_params (fallback branch)."""
    print("\n=== Test: get_feature_names_fallback ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_feat_names_fb')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1, 'CHECKPOINT_DIR': 'models/test_fn/',
        'SAVE_EVERY_N_EPOCHS': 100, 'SAVE_ON_INTERRUPT': False,
        # No normalization_params → fallback branch
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    # 18 features (with time)
    names = trainer._get_feature_names(18)
    assert len(names) == 18
    assert names[0] == 'log_price'
    assert 'sin_hour' in names

    # 14 features (no time)
    names_14 = trainer._get_feature_names(14)
    assert len(names_14) == 14
    assert 'sin_hour' not in names_14

    writer.close()
    print(f"  Fallback 18: {names[:4]}...{names[-2:]}")
    print(f"  Fallback 14: {names_14[:3]}...{names_14[-2:]}")
    print("  PASS")


# ============================================================================
# Group 4: Load checkpoint with config changes (WD, EPOCHS)
# ============================================================================

def test_load_checkpoint_config_changes():
    """Test load_checkpoint detects and applies WD/EPOCHS changes."""
    print("\n=== Test: load_checkpoint_config_changes ===")

    device = get_device("cuda")
    model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    window_size = 400
    data = np.random.randn(600, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
    dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
    loader = DataLoader(dataset, batch_size=8)

    writer = SummaryWriter(log_dir='runs/tests/test_ckpt_changes')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 5,
        'CHECKPOINT_DIR': 'models/test_changes/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': False,
        'LEARNING_RATE': 0.001,
        'WEIGHT_DECAY': 0.01,
        'DROPOUT': 0.1,
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    # Save checkpoint with current config
    metrics = {'train_loss': 1.0, 'test_loss': 1.1, 'train_acc': 0.4, 'test_acc': 0.38}
    trainer.save_checkpoint(epoch=2, metrics=metrics, is_best=False)

    checkpoint_path = 'models/test_changes/checkpoint_epoch_3.pt'
    assert os.path.exists(checkpoint_path)

    # Now change config values BEFORE loading
    orig_lr = config.LEARNING_RATE
    orig_wd = config.WEIGHT_DECAY
    orig_dropout = config.DROPOUT

    try:
        config.LEARNING_RATE = 0.0005   # Changed
        config.WEIGHT_DECAY = 0.05      # Changed
        config.DROPOUT = 0.2            # Changed

        # Also update trainer_config EPOCHS (like real resume scenario)
        trainer_config['EPOCHS'] = 10

        start_epoch = trainer.load_checkpoint(checkpoint_path)

        assert start_epoch == 3, f"Expected start_epoch=3, got {start_epoch}"

        # LR should be updated to new config value
        actual_lr = optimizer.param_groups[0]['lr']
        assert abs(actual_lr - 0.0005) < 1e-8, f"LR should be 0.0005, got {actual_lr}"

        # WD should be updated for groups that had WD > 0
        for pg in optimizer.param_groups:
            if pg['weight_decay'] > 0:
                assert abs(pg['weight_decay'] - 0.05) < 1e-8, \
                    f"WD should be 0.05, got {pg['weight_decay']}"

        print(f"  LR: 0.001 -> {actual_lr}")
        print(f"  WD: 0.01 -> {optimizer.param_groups[0]['weight_decay']}")
        print(f"  EPOCHS: 5 -> 10")
        print("  PASS")

    finally:
        config.LEARNING_RATE = orig_lr
        config.WEIGHT_DECAY = orig_wd
        config.DROPOUT = orig_dropout

    writer.close()
    os.remove(checkpoint_path)


def test_load_checkpoint_scheduler_restore():
    """Test load_checkpoint restores scheduler state."""
    print("\n=== Test: load_checkpoint_scheduler_restore ===")

    # Save global config values that load_checkpoint reads
    orig_lr = config.LEARNING_RATE
    orig_wd = config.WEIGHT_DECAY
    orig_dropout = config.DROPOUT

    try:
        # Set config to match what we'll use in the test
        config.LEARNING_RATE = 0.001
        config.WEIGHT_DECAY = 0.0
        config.DROPOUT = 0.1

        device = get_device("cuda")
        model = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        window_size = 400
        data = np.random.randn(600, 5).astype(np.float32)
        labels = np.random.choice([0, 1, 2], size=50 + window_size).astype(np.int8)
        dataset = TimeSeriesWindowDataset(data, labels, list(range(50)), window_size=window_size)
        loader = DataLoader(dataset, batch_size=8)

        writer = SummaryWriter(log_dir='runs/tests/test_sched_restore')

        trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'KEEP_LATEST': False,
            'EPOCHS': 10,
            'CHECKPOINT_DIR': 'models/test_sched_restore/',
            'SAVE_EVERY_N_EPOCHS': 1,
            'SAVE_ON_INTERRUPT': False,
            'LR_SCHEDULER': 'cosine',
            'LR_MIN': 0.00001,
            'LEARNING_RATE': 0.001,
            'WEIGHT_DECAY': 0.0,
            'DROPOUT': 0.1,
        }

        trainer = Trainer(
            model=model, train_loader=loader, test_loader=loader,
            optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
            device=device, config=trainer_config,
            tensorboard_writer=writer, logger=logger,
            setup_signal_handler=False,
        )

        # Step scheduler a few times
        trainer.scheduler.step()
        trainer.scheduler.step()
        trainer.scheduler.step()

        # Save checkpoint
        metrics = {'train_loss': 1.0, 'test_loss': 1.1, 'train_acc': 0.4, 'test_acc': 0.38}
        trainer.save_checkpoint(epoch=3, metrics=metrics, is_best=False)

        checkpoint_path = 'models/test_sched_restore/checkpoint_epoch_4.pt'
        assert os.path.exists(checkpoint_path)

        # Create fresh trainer and load
        model2 = MambaPredictor(d_model=32, d_state=8, n_layers=1).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        trainer2 = Trainer(
            model=model2, train_loader=loader, test_loader=loader,
            optimizer=optimizer2, criterion=nn.CrossEntropyLoss(),
            device=device, config=trainer_config,
            tensorboard_writer=writer, logger=logger,
            setup_signal_handler=False,
        )

        start_epoch = trainer2.load_checkpoint(checkpoint_path)
        assert start_epoch == 4

        # After restore, stepping scheduler once more should give same result
        # as stepping the original 4 times
        trainer2.scheduler.step()
        lr_restored = optimizer2.param_groups[0]['lr']

        # Step original one more time for comparison
        trainer.scheduler.step()
        lr_original = optimizer.param_groups[0]['lr']

        assert abs(lr_restored - lr_original) < 1e-8, \
            f"Scheduler state not restored: {lr_restored} vs {lr_original}"

        writer.close()
        os.remove(checkpoint_path)
        print(f"  After restore+step: {lr_restored:.8f} == original+step: {lr_original:.8f}")
        print("  PASS")

    finally:
        config.LEARNING_RATE = orig_lr
        config.WEIGHT_DECAY = orig_wd
        config.DROPOUT = orig_dropout


# ============================================================================
# Group 5: Dual (MultiTF) training branches
# ============================================================================

def test_dual_train_eval():
    """Test train_epoch and evaluate with dual-branch MultiTF model."""
    print("\n=== Test: dual_train_eval ===")

    from modules.data.dataset import MultiTFWindowDataset
    from modules.model.mamba import MultiTFMambaPredictor

    device = get_device("cuda")

    # Build aligned primary/secondary data
    ws = 50  # Small window for speed
    divisor = 4
    n_samples_pri = 1200
    n_samples_sec = n_samples_pri // divisor  # 300
    n_feat_pri = 12  # 6 norm + 4 time + 2 regime (global)
    n_feat_sec = 8   # 6 norm + 2 regime (no time for secondary)

    set_seed(config.SEED)
    data_pri = np.random.randn(n_samples_pri, n_feat_pri).astype(np.float32)
    data_sec = np.random.randn(n_samples_sec, n_feat_sec).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_samples_pri).astype(np.int8)

    raw_hlc_pri = np.random.uniform(90, 110, (n_samples_pri, 3)).astype(np.float64)
    raw_hlc_sec = np.random.uniform(90, 110, (n_samples_sec, 3)).astype(np.float64)

    # Mapping: primary index -> secondary index (// divisor)
    mapping = np.minimum(np.arange(n_samples_pri) // divisor, n_samples_sec - 1)

    # Indices must ensure secondary window has ws rows:
    # j_sec = mapping[i_pri + ws - 1] >= ws - 1
    # i.e. (i_pri + ws - 1) // divisor >= ws - 1 → i_pri >= (ws-1)*divisor - ws + 1
    min_pri = (ws - 1) * divisor
    max_pri = n_samples_pri - ws
    indices = list(range(min_pri, max_pri))

    half = len(indices) // 2
    train_dataset = MultiTFWindowDataset(
        data_primary=data_pri, data_secondary=[data_sec],
        labels=labels, indices_primary=indices[:half],
        mapping_primary_to_secondary=[mapping],
        window_size=ws,
        raw_hlc_primary=raw_hlc_pri, raw_hlc_secondary=[raw_hlc_sec],
        n_norm_features_primary=7, n_norm_features_secondary=7,
    )
    test_dataset = MultiTFWindowDataset(
        data_primary=data_pri, data_secondary=[data_sec],
        labels=labels, indices_primary=indices[half:],
        mapping_primary_to_secondary=[mapping],
        window_size=ws,
        raw_hlc_primary=raw_hlc_pri, raw_hlc_secondary=[raw_hlc_sec],
        n_norm_features_primary=7, n_norm_features_secondary=7,
    )

    train_loader = DataLoader(train_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Determine actual feature count from dataset
    sample = train_dataset[0]
    n_feat_pri_actual = sample[0].shape[1]
    n_feat_sec_actual = sample[1].shape[1]

    model = MultiTFMambaPredictor(
        input_dim_primary=n_feat_pri_actual,
        input_dim_secondary=n_feat_sec_actual,
        d_model=32, d_state_primary=8, d_conv=2, expand=2,
        n_layers_primary=1, n_layers_secondary=1,
        d_state_secondary=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/tests/test_dual')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'EPOCHS': 1, 'CHECKPOINT_DIR': 'models/test_dual/',
        'SAVE_EVERY_N_EPOCHS': 100, 'SAVE_ON_INTERRUPT': False,
        'KEEP_LATEST': False,
    }

    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, criterion=criterion,
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    assert trainer.is_dual, "Trainer should detect dual-branch model"

    # Train one epoch
    train_results = trainer.train_epoch()
    assert 'train_loss' in train_results
    assert 'train_acc' in train_results
    assert train_results['train_loss'] > 0

    # Evaluate
    eval_results = trainer.evaluate()
    assert 'test_loss' in eval_results
    assert 'test_acc' in eval_results

    # Feature importance (dual branch)
    importance = trainer.compute_feature_importance()
    assert len(importance) > 0, "Feature importance should have entries"
    total_pct = sum(importance.values())
    assert abs(total_pct - 100.0) < 0.1, f"Importance should sum to ~100%, got {total_pct}"

    writer.close()
    print(f"  Dual train loss: {train_results['train_loss']:.4f}, acc: {train_results['train_acc']:.4f}")
    print(f"  Dual eval loss: {eval_results['test_loss']:.4f}, acc: {eval_results['test_acc']:.4f}")
    print(f"  Feature importance: {len(importance)} features, sum={total_pct:.1f}%")
    print("  PASS")


# ============================================================================
# Group 6: 3-class training verification
# ============================================================================

def test_three_class_train_epoch():
    """Test train_epoch with 3-class: standard CE loss on all samples."""
    print("\n=== Test: three_class_train_epoch ===")

    device = get_device("cuda")

    # Create data with 3-class labels (0=Bear, 1=Uncertain, 2=Bull)
    window_size = 400
    n_indices = 100
    data = np.random.randn(1000, 5).astype(np.float32)
    labels = np.random.choice([0, 1, 2], size=n_indices + window_size).astype(np.int8)
    indices = list(range(n_indices))

    dataset = TimeSeriesWindowDataset(data, labels, indices, window_size=window_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Model with num_classes=3 (3-class classification)
    model = MambaPredictor(
        input_dim=5, d_model=32, d_state=8, d_conv=2, expand=2, n_layers=1,
        num_classes=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='runs/tests/test_three_class')

    trainer_config = {
        'PREDICTION_TARGET': 'triple',
        'CASCADE_TRAINING': False,
        'LR_SCHEDULER': 'none',
        'KEEP_LATEST': False,
        'EPOCHS': 1,
        'CHECKPOINT_DIR': 'models/test_3class/',
        'SAVE_EVERY_N_EPOCHS': 1,
        'SAVE_ON_INTERRUPT': False,
    }

    trainer = Trainer(
        model=model, train_loader=loader, test_loader=loader,
        optimizer=optimizer, criterion=criterion,
        device=device, config=trainer_config,
        tensorboard_writer=writer, logger=logger,
        setup_signal_handler=False,
    )

    assert trainer.num_classes == 3

    # Train one epoch
    results = trainer.train_epoch()

    # Standard keys
    assert 'train_loss' in results
    assert 'train_acc' in results
    assert results['train_loss'] > 0

    # y_true should contain all 3 classes (no masking/remapping)
    y_true = results['y_true']
    unique_labels = set(np.unique(y_true))
    assert unique_labels.issubset({0, 1, 2}), f"Expected labels in {{0,1,2}}, got {unique_labels}"

    # y_pred_probs should have 3 columns
    y_pred_probs = results['y_pred_probs']
    assert y_pred_probs.shape[1] == 3, f"Expected 3 prob columns, got {y_pred_probs.shape[1]}"

    # Evaluate
    eval_results = trainer.evaluate()
    assert 'test_loss' in eval_results
    assert 'test_acc' in eval_results

    writer.close()
    print(f"  loss={results['train_loss']:.4f}, acc={results['train_acc']:.4f}")
    print(f"  y_true unique: {unique_labels}")
    print("  PASS")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing trainer.py")
    print("=" * 60)

    try:
        # Basic tests
        print("\n" + "=" * 60)
        print("BASIC TESTS")
        print("=" * 60)
        test_dataset()
        test_get_device()
        test_trainer_one_epoch()
        test_checkpoint_save_load()

        # Advanced tests
        print("\n" + "=" * 60)
        print("ADVANCED TESTS")
        print("=" * 60)
        test_full_training_pipeline()
        test_checkpoint_content()
        test_logging()

        print("\n" + "=" * 60)
        print("✅ All trainer tests passed!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
