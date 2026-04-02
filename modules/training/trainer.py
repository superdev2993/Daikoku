"""
Training module - Complete training loop with checkpointing and logging.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from pathlib import Path
import signal
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config as cfg
from modules.training.metrics import (
    compute_all_metrics,
    filter_metrics,
)
from modules.training.monitoring import ActivationMonitor
from modules.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Device Selection
# ============================================================================

def get_device(device_config: str) -> torch.device:
    """
    Auto-detect or specify device.

    Args:
        device_config: "auto", "cuda", or "cpu"

    Returns:
        torch.device instance
    """
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Auto-detected device: {device}")
        if device.type == "cuda":
            logger.debug(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.debug(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            logger.debug(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.debug(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Training orchestration for MambaPredictor model.

    Handles:
    - Training loop with epoch management
    - Evaluation on test set
    - Checkpoint saving/loading
    - TensorBoard logging
    - Graceful Ctrl+C handling
    - Training resumption
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict,
        tensorboard_writer: SummaryWriter,
        logger,
        setup_signal_handler: bool = True
    ):
        """
        Args:
            model: MambaPredictor instance
            train_loader: DataLoader for training data (shuffle=False!)
            test_loader: DataLoader for test data (shuffle=False!)
            optimizer: Adam optimizer
            criterion: CrossEntropyLoss
            device: torch.device (cuda/cpu)
            config: Dict with training config (epochs, checkpoint_dir, etc.)
            tensorboard_writer: TensorBoard SummaryWriter
            logger: Python logger instance
            setup_signal_handler: If True, setup Ctrl+C handler (default: True, disable in tests)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.writer = tensorboard_writer
        self.logger = logger

        self.current_epoch = 0
        self.best_balanced_acc = -1.0  # Always save first epoch as best
        self.interrupted = False
        self.last_train_loss = 0.0
        self.last_train_acc = 0.0

        self.prediction_target = config['PREDICTION_TARGET']
        self.num_classes = 3 if self.prediction_target == 'triple' else 2

        # Dual-branch mode flag (based on model type, not config — avoids ambiguous len(batch_data))
        self.is_dual = hasattr(model, 'encoder_primary')

        # Performance: AMP mixed precision
        self.use_amp = device.type == "cuda" and cfg.MIXED_PRECISION
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.non_blocking = cfg.NON_BLOCKING_TRANSFER and device.type == "cuda"
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled")
        if self.non_blocking:
            logger.debug("Non-blocking GPU transfers enabled")

        # Setup signal handler for Ctrl+C (can be disabled for tests)
        if setup_signal_handler:
            self._setup_signal_handler()

        # Activation monitoring
        if cfg.MONITOR_ACTIVATIONS:
            self.activation_monitor = ActivationMonitor(
                model, dead_threshold=cfg.MONITOR_DEAD_THRESHOLD
            )
        else:
            self.activation_monitor = None

        # Cascade training (progressive layer growing + freezing)
        self.cascade = config['CASCADE_TRAINING']
        if self.cascade:
            if self.is_dual:
                self.n_layers_primary = len(self.model.encoder_primary.mamba_blocks)
                self.n_layers_secondary = len(self.model.encoder_secondary.mamba_blocks)
            else:
                self.n_layers_primary = len(self.model.mamba_blocks)
                self.n_layers_secondary = 0
            logger.info(f"CASCADE training enabled: {self.n_layers_primary} primary layers"
                        + (f", {self.n_layers_secondary} secondary layers" if self.is_dual else ""))

        # LR scheduler (created from config, None if LR_SCHEDULER='none')
        self.scheduler = self._create_scheduler()

    def _cascade_setup_epoch(self, epoch):
        """Configure active depth and frozen layers for cascade training."""
        depth = epoch + 1  # epoch 0 → depth 1, epoch 1 → depth 2, etc.

        # --- Set active depth (growing) ---
        if self.is_dual:
            self.model.active_depth_primary = min(depth, self.n_layers_primary)
            self.model.active_depth_secondary = min(depth, self.n_layers_secondary)
        else:
            self.model.active_depth = min(depth, self.n_layers_primary)

        # --- Freeze completed layers (layers from previous epochs) ---
        if epoch > 0:
            layer_to_freeze = epoch - 1  # Freeze layer trained last epoch

            # Freeze Mamba block in primary
            if layer_to_freeze < self.n_layers_primary:
                block = (self.model.encoder_primary.mamba_blocks[layer_to_freeze]
                         if self.is_dual else self.model.mamba_blocks[layer_to_freeze])
                block.requires_grad_(False)
                block.eval()

            # Freeze Mamba block in secondary (if dual)
            if self.is_dual and layer_to_freeze < self.n_layers_secondary:
                self.model.encoder_secondary.mamba_blocks[layer_to_freeze].requires_grad_(False)
                self.model.encoder_secondary.mamba_blocks[layer_to_freeze].eval()

            # Freeze linear_proj with layer 0
            if layer_to_freeze == 0:
                if self.is_dual:
                    self.model.encoder_primary.linear_proj.requires_grad_(False)
                    self.model.encoder_secondary.linear_proj.requires_grad_(False)
                else:
                    self.model.linear_proj.requires_grad_(False)

        # --- Re-apply eval() on ALL previously frozen blocks ---
        # (model.train() resets everything to train mode, so we must re-eval frozen blocks)
        self._reapply_frozen_eval()

        # Log
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"  [CASCADE] Depth={depth}, frozen layers=0..{epoch-1}, "
                         f"trainable={trainable:,}/{total:,}")

    def _reapply_frozen_eval(self):
        """Re-apply eval mode on frozen Mamba blocks (after model.train() resets them)."""
        for i in range(self.current_epoch):  # Layers 0..current_epoch-1 are frozen
            if i < self.n_layers_primary:
                block = (self.model.encoder_primary.mamba_blocks[i]
                         if self.is_dual else self.model.mamba_blocks[i])
                block.eval()
            if self.is_dual and i < self.n_layers_secondary:
                self.model.encoder_secondary.mamba_blocks[i].eval()
        # Also input proj if epoch > 0
        if self.current_epoch > 0:
            if self.is_dual:
                self.model.encoder_primary.linear_proj.eval()
                self.model.encoder_secondary.linear_proj.eval()
            else:
                self.model.linear_proj.eval()

    def _create_scheduler(self):
        """Create LR scheduler from config. Returns None if LR_SCHEDULER='none'."""
        sched_type = self.config['LR_SCHEDULER']

        if sched_type == 'none':
            return None

        elif sched_type == 'cosine':
            lr_min = self.config['LR_MIN']
            total_epochs = self.config['EPOCHS']
            return CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=lr_min)

        elif sched_type == 'wsd':
            lr_min = self.config['LR_MIN']
            lr_max = self.config['LEARNING_RATE']
            total_epochs = self.config['EPOCHS']
            warmup_epochs = max(1, int(total_epochs * self.config['LR_WARMUP_PCT']))
            stable_epochs = int(total_epochs * self.config['LR_STABLE_PCT'])
            decay_start = warmup_epochs + stable_epochs

            def wsd_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup: LR_MIN → LR_MAX
                    return lr_min / lr_max + (1 - lr_min / lr_max) * epoch / warmup_epochs
                elif epoch < decay_start:
                    # Stable phase: LR_MAX
                    return 1.0
                else:
                    # Linear decay: LR_MAX → LR_MIN
                    decay_epochs = total_epochs - decay_start
                    progress = (epoch - decay_start + 1) / max(decay_epochs, 1)
                    return lr_min / lr_max + (1 - lr_min / lr_max) * (1 - progress)

            return LambdaLR(self.optimizer, lr_lambda=wsd_lambda)

        else:
            self.logger.warning(f"Unknown LR_SCHEDULER '{sched_type}', using none")
            return None

    def train_epoch(self) -> dict:
        """
        Execute one training epoch.

        Returns:
            Dict with:
                - train_loss: Average loss
                - train_acc: Training accuracy
                - y_true: All true labels (for metrics)
                - y_pred: All predictions (for metrics)
                - y_pred_probs: All probabilities (for metrics)
        """
        self.model.train()

        # Cascade: re-apply eval() on frozen blocks after model.train() resets them
        if self.cascade:
            self._reapply_frozen_eval()

        total_loss = 0.0
        batch_count = 0
        all_labels = []
        all_preds = []
        all_probs = []
        _total_batches = len(self.train_loader)
        _total_epochs = self.config['EPOCHS']
        _epoch_str = f"E{self.current_epoch+1}/{_total_epochs}"
        _epoch_start = time.time()
        _last_print = _epoch_start
        _is_tty = sys.stdout.isatty()

        for batch_idx, batch_data in enumerate(self.train_loader):
            batch_count += 1
            nb = self.non_blocking

            # Unpack batch: dual = (pri, sec, tf_map, label, [conf]) / mono = (window, label, [conf])
            confidence = None
            if self.is_dual:
                windows_pri = batch_data[0].to(self.device, non_blocking=nb)
                windows_sec = batch_data[1].to(self.device, non_blocking=nb)
                tf_mapping = batch_data[2].to(self.device, non_blocking=nb)
                labels = batch_data[3].to(self.device, non_blocking=nb)
                if len(batch_data) > 4:
                    confidence = batch_data[4].to(self.device, non_blocking=nb)
            else:
                windows = batch_data[0].to(self.device, non_blocking=nb)
                labels = batch_data[1].to(self.device, non_blocking=nb)
                if len(batch_data) > 2:
                    confidence = batch_data[2].to(self.device, non_blocking=nb)

            # Apply confidence only if enabled in config
            if not cfg.LABEL_CONFIDENCE_ENABLED:
                confidence = None

            # Forward + loss (with optional AMP)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                if self.is_dual:
                    logits = self.model(windows_pri, windows_sec, tf_mapping)
                else:
                    logits = self.model(windows)

                if confidence is not None:
                    loss = self.criterion(logits, labels, confidence)
                else:
                    loss = self.criterion(logits, labels)

                # Record metrics
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())

            # Backward pass (with optional GradScaler)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Record total loss
            total_loss += loss.item()

            # Intra-epoch progress (stdout only, every 5s)
            if _is_tty and time.time() - _last_print >= 5:
                avg = total_loss / batch_count
                pct = (batch_idx + 1) / _total_batches
                filled = int(30 * pct)
                bar = '█' * filled + '░' * (30 - filled)
                # ETA: extrapolate from current epoch speed
                elapsed = time.time() - _epoch_start
                est_epoch = elapsed / pct if pct > 0 else 0
                epochs_left = _total_epochs - self.current_epoch - 1
                total_eta = (est_epoch - elapsed) + epochs_left * est_epoch
                eta_h, eta_remainder = divmod(int(total_eta), 3600)
                eta_m = eta_remainder // 60
                print(f"\r  [{_epoch_str}] {bar} {batch_idx+1}/{_total_batches} ({pct*100:.0f}%) loss={avg:.4f} ETA={eta_h}h{eta_m:02d}", end='', flush=True)
                _last_print = time.time()

            # Check for interruption
            if self.interrupted:
                break

        if _is_tty:
            avg = total_loss / max(batch_count, 1)
            print(f"\r  [{_epoch_str}] {'█' * 30} {batch_count}/{_total_batches} (100%) loss={avg:.4f}              ")

        # Aggregate
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_pred_probs = np.concatenate(all_probs)

        train_acc = accuracy_score(y_true, y_pred)

        result = {
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }

        return result

    def evaluate(self) -> dict:
        """
        Evaluate model on test set.

        Returns:
            Dict with:
                - test_loss: Average loss
                - test_acc: Test accuracy
                - y_true: All true labels
                - y_pred: All predictions
                - y_pred_probs: All probabilities
        """
        self.model.eval()

        total_loss = 0.0
        batch_count = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_data in self.test_loader:
                batch_count += 1
                nb = self.non_blocking

                # Unpack batch — ignore confidence for eval
                if self.is_dual:
                    windows_pri = batch_data[0].to(self.device, non_blocking=nb)
                    windows_sec = batch_data[1].to(self.device, non_blocking=nb)
                    tf_mapping = batch_data[2].to(self.device, non_blocking=nb)
                    labels = batch_data[3].to(self.device, non_blocking=nb)
                else:
                    windows = batch_data[0].to(self.device, non_blocking=nb)
                    labels = batch_data[1].to(self.device, non_blocking=nb)

                with autocast('cuda', enabled=self.use_amp):
                    if self.is_dual:
                        logits = self.model(windows_pri, windows_sec, tf_mapping)
                    else:
                        logits = self.model(windows)

                    loss = self.criterion(logits, labels)

                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(probs.detach().cpu().numpy())

                total_loss += loss.item()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_pred_probs = np.concatenate(all_probs)

        test_acc = accuracy_score(y_true, y_pred)

        result = {
            'test_loss': avg_loss,
            'test_acc': test_acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }

        return result

    def train(self, start_epoch: int = 0):
        """
        Full training loop.

        Args:
            start_epoch: Starting epoch (for resumption)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting training")
        self.logger.debug(f"Epochs: {self.config['EPOCHS']}")
        self.logger.debug(f"Device: {self.device}")
        self.logger.debug("=" * 60)

        for epoch in range(start_epoch, self.config['EPOCHS']):
            self.current_epoch = epoch

            # Cascade: configure depth + freeze layers
            if self.cascade:
                self._cascade_setup_epoch(epoch)

            # Training epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config['EPOCHS']}")
            train_results = self.train_epoch()

            # Store for potential interrupt
            self.last_train_loss = train_results['train_loss']
            self.last_train_acc = train_results['train_acc']

            # Compute train metrics (per-class)
            train_balanced_acc = balanced_accuracy_score(
                train_results['y_true'],
                train_results['y_pred']
            )
            train_class_report = classification_report(
                train_results['y_true'],
                train_results['y_pred'],
                output_dict=True,
                zero_division=0
            )

            # Store in train_results
            train_results['balanced_accuracy'] = train_balanced_acc
            train_results['class_report'] = train_class_report

            if self.interrupted:
                self.logger.warning("Training interrupted by user")
                self._save_on_interrupt()
                break

            # Evaluation
            test_results = self.evaluate()

            # Compute all metrics
            long_pnl, short_pnl = self._get_test_pnl_arrays()
            metrics = compute_all_metrics(
                y_true=test_results['y_true'],
                y_pred=test_results['y_pred'],
                y_pred_probs=test_results['y_pred_probs'],
                model=self.model,
                train_loss=train_results['train_loss'],
                test_loss=test_results['test_loss'],
                train_acc=train_results['train_acc'],
                test_acc=test_results['test_acc'],
                num_classes=self.num_classes,
                prediction_target=self.prediction_target,
                long_pnl_R=long_pnl,
                short_pnl_R=short_pnl,
            )

            # Log to console (filtered metrics)
            self._log_console(epoch, train_results, test_results, metrics)

            # Log to TensorBoard (filtered metrics)
            self._log_tensorboard(epoch, train_results, test_results, metrics)

            # Log-level independent progress (first epoch + every ~25% + last)
            total_epochs = self.config['EPOCHS']
            quarter = max(1, total_epochs // 4)
            if epoch == start_epoch or (epoch + 1) % quarter == 0 or epoch + 1 == total_epochs:
                gap = test_results['test_loss'] - train_results['train_loss']
                bal_acc = metrics.get('balanced_accuracy', 0.0)
                final_acc = metrics.get('final_accuracy', 0.0)
                print(f"  [Epoch {epoch+1}/{total_epochs}] "
                      f"trn_loss={train_results['train_loss']:.4f} "
                      f"tst_loss={test_results['test_loss']:.4f} "
                      f"bal_acc={bal_acc:.4f} "
                      f"final_acc={final_acc:.4f} "
                      f"gap={gap:.4f}")

            # Extract one test batch for all diagnostics (avoid 3× next(iter()))
            diag_batch = next(iter(self.test_loader))

            # Multi-TF branch diagnostics (every epoch)
            if cfg.MONITOR_BRANCH_BALANCE:
                self._log_branch_diagnostics(epoch, diag_batch)

            # SSM delta + input gate diagnostics (mono-TF: log here since _log_branch_diagnostics skips)
            if not hasattr(self.model, 'encoder_primary'):
                self.model.eval()
                with torch.no_grad():
                    windows = diag_batch[0].to(self.device)
                    self._log_delta_diagnostics(epoch, self.model, windows, 'main')
                    self._log_input_gate_diagnostics(epoch, self.model, windows, 'main')

            # Activation monitoring
            if self.activation_monitor is not None:
                monitor_freq = cfg.MONITOR_EVERY_N_EPOCHS
                if (epoch + 1) % monitor_freq == 0:
                    if self.is_dual:
                        batch = (diag_batch[0], diag_batch[1])
                    else:
                        batch = (diag_batch[0],)
                    activation_stats = self.activation_monitor.collect(batch, self.device)
                    self._log_activation_tensorboard(epoch, activation_stats)
                    self._check_health_alerts(epoch, activation_stats, metrics)

            # Save checkpoints
            if (epoch + 1) % self.config['SAVE_EVERY_N_EPOCHS'] == 0:
                self.save_checkpoint(epoch, metrics, is_best=False)

            # Save best model (on balanced_accuracy = directional performance)
            current_bal_acc = metrics.get('balanced_accuracy', 0.0)
            if current_bal_acc > self.best_balanced_acc:
                self.best_balanced_acc = current_bal_acc
                self.save_checkpoint(epoch, metrics, is_best=True)

            # Save latest checkpoint
            if self.config['KEEP_LATEST']:
                self.save_checkpoint(epoch, metrics, is_latest=True)

            # LR scheduler step (per-epoch, after logging and checkpoints)
            if self.scheduler is not None:
                self.scheduler.step()

        # Feature importance analysis (after last epoch)
        if not self.interrupted:
            importance = self.compute_feature_importance()
            self._log_feature_importance(importance)

        self.logger.info("=" * 60)
        self.logger.info("Training complete")
        self.logger.debug("=" * 60)

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        is_latest: bool = False,
        is_interrupt: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Dict of all metrics
            is_best: Save as best_model.pt
            is_latest: Save as latest.pt
            is_interrupt: Save as interrupted.pt
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': filter_metrics(metrics, 'log'),
            'train_loss': metrics.get('train_loss'),
            'test_loss': metrics.get('test_loss'),
            'train_acc': metrics.get('train_acc'),
            'test_acc': metrics.get('test_acc'),
            'balanced_accuracy': metrics.get('balanced_accuracy'),
            'normalization_params': self.config.get('normalization_params'),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'config': self.config
        }

        checkpoint_dir = Path(self.config['CHECKPOINT_DIR'])
        checkpoint_dir.mkdir(exist_ok=True)

        if is_best:
            path = checkpoint_dir / 'best_model.pt'
            self.logger.info(f"Saving best model to {path} (balanced_accuracy={metrics.get('balanced_accuracy', 0):.4f})")
        elif is_latest:
            path = checkpoint_dir / 'latest.pt'
        elif is_interrupt:
            path = checkpoint_dir / 'interrupted.pt'
            self.logger.warning(f"Saving interrupted checkpoint to {path}")
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            self.logger.debug(f"Saving checkpoint to {path}")

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint for resumption.

        Handles config parameter changes (LEARNING_RATE, WEIGHT_DECAY, DROPOUT, EPOCHS):
        - If changed, applies new values and logs warnings
        - EPOCHS is always taken from current config

        Args:
            path: Path to checkpoint file

        Returns:
            Starting epoch number
        """
        self.logger.info(f"Loading checkpoint from {path}")

        # weights_only=False required: checkpoint contains config dict, metrics, scalars (not just tensors)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state if available
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
            self.logger.debug("✓ Scheduler state restored")

        start_epoch = checkpoint['epoch'] + 1
        # Restore best balanced_accuracy from checkpoint metrics
        saved_metrics = checkpoint.get('metrics', {})
        self.best_balanced_acc = saved_metrics.get('balanced_accuracy', 0.0)

        self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        self.logger.debug(f"Best balanced_accuracy: {self.best_balanced_acc:.4f}")

        # ====================================================================
        # Handle config parameter changes
        # ====================================================================
        saved_config = checkpoint.get('config', {})

        # Get current learning rate and weight decay from config module
        current_lr = cfg.LEARNING_RATE
        current_wd = cfg.WEIGHT_DECAY
        current_dropout = cfg.DROPOUT

        # Get saved values (from optimizer state if config not saved)
        saved_lr = saved_config.get('LEARNING_RATE', self.optimizer.param_groups[0]['lr'])
        saved_wd = saved_config.get('WEIGHT_DECAY', self.optimizer.param_groups[0]['weight_decay'])
        saved_dropout = saved_config.get('DROPOUT', 0)

        # Check and apply LEARNING_RATE changes
        if abs(current_lr - saved_lr) > 1e-10:
            self.logger.warning(f"⚠️  LEARNING_RATE changed: {saved_lr} → {current_lr}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            self.logger.debug(f"✓ Applied new learning rate: {current_lr}")

        # Check and apply WEIGHT_DECAY changes (only to groups that already have WD > 0)
        if abs(current_wd - saved_wd) > 1e-10:
            self.logger.warning(f"⚠️  WEIGHT_DECAY changed: {saved_wd} → {current_wd}")
            for param_group in self.optimizer.param_groups:
                if param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = current_wd
            self.logger.debug(f"✓ Applied new weight decay: {current_wd}")

        # Log DROPOUT changes (model already constructed with current config values)
        if abs(current_dropout - saved_dropout) > 1e-10:
            self.logger.warning(f"⚠️  DROPOUT changed: {saved_dropout} → {current_dropout}")

        # EPOCHS is always taken from current config (no need to check)
        if saved_config.get('EPOCHS') and saved_config['EPOCHS'] != self.config['EPOCHS']:
            self.logger.debug(f"EPOCHS changed: {saved_config['EPOCHS']} → {self.config['EPOCHS']}")

        return start_epoch

    def _get_test_pnl_arrays(self):
        """
        Extract long/short P&L arrays for test set, aligned with evaluate() output.

        Walks through the test_loader's dataset (ConcatDataset or single dataset),
        extracts P&L arrays at the correct window-end indices.

        Returns:
            Tuple: (long_pnl, short_pnl) as numpy arrays, or (None, None).
        """
        dataset = self.test_loader.dataset

        def _extract(ds):
            if ds.long_pnl_R is None:
                return None, None
            ws = ds.window_size
            idx_arr = np.array(ds.indices) + ws - 1
            return ds.long_pnl_R[idx_arr].numpy(), ds.short_pnl_R[idx_arr].numpy()

        if isinstance(dataset, ConcatDataset):
            long_parts, short_parts = [], []
            for sub_ds in dataset.datasets:
                lp, sp = _extract(sub_ds)
                if lp is None:
                    return None, None
                long_parts.append(lp)
                short_parts.append(sp)
            return np.concatenate(long_parts), np.concatenate(short_parts)
        else:
            return _extract(dataset)

    def _log_console(self, epoch: int, train_results: dict, test_results: dict, metrics: dict):
        """Log metrics to console (filtered by METRIC_REGISTRY)."""
        console_metrics = filter_metrics(metrics, 'console')

        self.logger.debug(f"Train Loss: {train_results['train_loss']:.4f} | Train Acc: {train_results['train_acc']:.4f}")

        # Balanced accuracy
        if 'balanced_accuracy' in train_results:
            self.logger.debug(f"Train Balanced Acc: {train_results['balanced_accuracy']:.4f}")

        # Log F1 per class for train
        if 'class_report' in train_results:
            for cls in [str(c) for c in range(self.num_classes)]:
                if cls in train_results['class_report']:
                    f1 = train_results['class_report'][cls]['f1-score']
                    self.logger.debug(f"Train F1 Class {cls}: {f1:.4f}")

        self.logger.debug(f"Test Loss: {test_results['test_loss']:.4f} | Test Acc: {test_results['test_acc']:.4f}")

        # Log selected metrics (test)
        for key, value in console_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.debug(f"{key}: {value:.4f}")

    def _log_tensorboard(self, epoch: int, train_results: dict, test_results: dict, metrics: dict):
        """Log metrics to TensorBoard (filtered by METRIC_REGISTRY)."""
        tb_metrics = filter_metrics(metrics, 'tensorboard')

        # Loss and accuracy (using tags for single file structure)
        self.writer.add_scalar('Loss/train', train_results['train_loss'], epoch)
        self.writer.add_scalar('Loss/test', test_results['test_loss'], epoch)
        self.writer.add_scalar('Accuracy/train', train_results['train_acc'], epoch)
        self.writer.add_scalar('Accuracy/test', test_results['test_acc'], epoch)

        # Balanced accuracy
        if 'balanced_accuracy' in train_results:
            self.writer.add_scalar('Train/Balanced_Accuracy', train_results['balanced_accuracy'], epoch)

        # F1 per class for train
        if 'class_report' in train_results:
            for cls in [str(c) for c in range(self.num_classes)]:
                if cls in train_results['class_report']:
                    f1 = train_results['class_report'][cls]['f1-score']
                    precision = train_results['class_report'][cls]['precision']
                    recall = train_results['class_report'][cls]['recall']

                    self.writer.add_scalar(f'Train/F1_Class_{cls}', f1, epoch)
                    self.writer.add_scalar(f'Train/Precision_Class_{cls}', precision, epoch)
                    self.writer.add_scalar(f'Train/Recall_Class_{cls}', recall, epoch)

        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/learning_rate', current_lr, epoch)

        # Cascade training metrics
        if self.cascade:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            depth = min(epoch + 1, self.n_layers_primary)
            self.writer.add_scalar('Cascade/trainable_params', trainable, epoch)
            self.writer.add_scalar('Cascade/active_depth', depth, epoch)

        # All other metrics (test)
        for key, value in tb_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)

    def _log_activation_tensorboard(self, epoch: int, stats: dict):
        """Log activation stats to TensorBoard."""
        for layer_name, layer_stats in stats.items():
            if layer_name == "_global":
                self.writer.add_scalar(
                    'Health/dead_neurons_total_pct',
                    layer_stats['dead_pct_total'],
                    epoch
                )
            else:
                self.writer.add_scalar(
                    f'Activations/{layer_name}/mean',
                    layer_stats['mean'],
                    epoch
                )
                self.writer.add_scalar(
                    f'Activations/{layer_name}/std',
                    layer_stats['std'],
                    epoch
                )
                self.writer.add_scalar(
                    f'Activations/{layer_name}/dead_pct',
                    layer_stats['dead_pct'],
                    epoch
                )

    def _check_health_alerts(self, epoch: int, stats: dict, metrics: dict):
        """Check activation health and emit console warnings if thresholds exceeded."""
        dead_alert_pct = cfg.MONITOR_DEAD_ALERT_PCT
        grad_vanish = cfg.MONITOR_GRAD_VANISH_THRESHOLD
        grad_explode = cfg.MONITOR_GRAD_EXPLODE_THRESHOLD

        # Log per-layer stats
        for layer_name, layer_stats in stats.items():
            if layer_name == "_global":
                continue
            if layer_stats['dead_pct'] > 0:
                self.logger.warning(
                    f"  {layer_name}: dead={layer_stats['dead_pct']:.1f}% "
                    f"mean={layer_stats['mean']:.6f} std={layer_stats['std']:.6f}"
                )

        # Dead neuron alert
        global_stats = stats.get("_global", {})
        global_dead_pct = global_stats.get("dead_pct_total", 0.0)
        if global_dead_pct > dead_alert_pct:
            self.logger.warning(
                f"[Epoch {epoch+1}] ALERT: {global_dead_pct:.1f}% dead neurons "
                f"({global_stats.get('total_dead', 0)}/{global_stats.get('total_neurons', 0)}) "
                f"exceeds threshold {dead_alert_pct}%"
            )

        # Gradient health (from existing metrics)
        grad_norm = metrics.get('grad_norm_mean')
        if grad_norm is not None:
            if grad_norm < grad_vanish:
                self.logger.warning(
                    f"[Epoch {epoch+1}] ALERT: Gradient vanishing "
                    f"(norm={grad_norm:.2e} < {grad_vanish:.2e})"
                )
            elif grad_norm > grad_explode:
                self.logger.warning(
                    f"[Epoch {epoch+1}] ALERT: Gradient exploding "
                    f"(norm={grad_norm:.2e} > {grad_explode:.2e})"
                )

    def compute_feature_importance(self):
        """
        Compute gradient-based feature importance on the test set.

        For each batch, computes |grad(loss, input)| averaged over the sequence dimension.
        Results are normalized to percentages across all features.

        Returns:
            Dict mapping feature names to importance percentages.
        """
        self.logger.debug("=" * 60)
        self.logger.info("Computing feature importance (gradient-based)...")

        self.model.eval()

        accumulated_importance = None
        total_samples = 0

        for batch_data in self.test_loader:
            nb = self.non_blocking
            # Unpack batch — ignore confidence
            if self.is_dual:
                windows_pri = batch_data[0].to(self.device, non_blocking=nb)
                windows_sec = batch_data[1].to(self.device, non_blocking=nb)
                tf_mapping_fi = batch_data[2].to(self.device, non_blocking=nb)
                labels = batch_data[3].to(self.device, non_blocking=nb)
                windows_pri.requires_grad_(True)
                logits = self.model(windows_pri, windows_sec, tf_mapping_fi)
                loss = self.criterion(logits, labels)
                grad = torch.autograd.grad(loss, windows_pri, create_graph=False)[0]
                batch_importance = grad.abs().mean(dim=1).sum(dim=0).detach().cpu().numpy()
                total_samples += windows_pri.shape[0]
            else:
                windows = batch_data[0].to(self.device, non_blocking=nb)
                labels = batch_data[1].to(self.device, non_blocking=nb)
                windows.requires_grad_(True)
                logits = self.model(windows)
                loss = self.criterion(logits, labels)
                grad = torch.autograd.grad(loss, windows, create_graph=False)[0]
                batch_importance = grad.abs().mean(dim=1).sum(dim=0).detach().cpu().numpy()
                total_samples += windows.shape[0]

            if accumulated_importance is None:
                accumulated_importance = np.zeros_like(batch_importance)

            accumulated_importance += batch_importance

        # Average per sample
        avg_importance = accumulated_importance / total_samples

        # Normalize to percentages
        total = avg_importance.sum()
        if total > 0:
            importance_pct = (avg_importance / total) * 100
        else:
            importance_pct = np.zeros_like(avg_importance)

        # Build feature names
        feature_names = self._get_feature_names(len(importance_pct))

        return dict(zip(feature_names, importance_pct))

    def _get_feature_names(self, n_features):
        """Build feature name list matching assembly order in _assemble_window."""
        from modules.data.dataset import get_feature_names
        has_time = n_features in (18, 23)
        has_vp = n_features in (19, 23)
        return get_feature_names(has_time=has_time, has_vp=has_vp)

    def _log_feature_importance(self, importance):
        """Log feature importance to TensorBoard (bar chart) and console."""
        # Sort descending
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Console log
        self.logger.info("Feature Importance (gradient-based, test set):")
        for name, pct in sorted_items:
            bar = '#' * int(pct / 2)
            self.logger.debug(f"  {name:20s}: {pct:5.1f}% {bar}")

        # TensorBoard bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1976d2' if not n.startswith('sin_') and not n.startswith('cos_')
                  else '#ff9800' for n in names]
        bars = ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance (%)')
        ax.set_title('Feature Importance (gradient-based, last epoch)')
        ax.invert_yaxis()

        # Add percentage labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        self.writer.add_figure('FeatureImportance/bar_chart', fig, self.config['EPOCHS'] - 1)
        plt.close(fig)

        self.logger.debug("Feature importance logged to TensorBoard")

    def _log_input_gate_diagnostics(self, epoch: int, encoder, windows, branch_name: str):
        """
        Log CNN gated residual gate statistics for monitoring CNN contribution.
        Only runs if encoder uses gated_residual fusion.
        """
        if not getattr(encoder, 'cnn_enabled', False) or getattr(encoder, 'cnn_fusion', 'add') != 'gated_residual':
            return

        # Hook on cnn_gate_proj to capture gate input
        gate_output = {}

        def gate_hook(module, input, output):
            gate_output['raw'] = output.detach()

        hook = encoder.cnn_gate_proj.register_forward_hook(gate_hook)
        encoder.encode(windows)
        hook.remove()

        if 'raw' not in gate_output:
            return

        gate = torch.sigmoid(gate_output['raw'])  # (B, Seq, d_model)

        g_mean = gate.mean().item()
        g_std = gate.std().item()
        seq_len = gate.shape[1]
        half = seq_len // 2
        g_early = gate[:, :half, :].mean().item()
        g_late = gate[:, half:, :].mean().item()

        tag = f'InputGate/{branch_name}'
        self.writer.add_scalar(f'{tag}/mean', g_mean, epoch)
        self.writer.add_scalar(f'{tag}/std', g_std, epoch)
        self.writer.add_scalar(f'{tag}/early_mean', g_early, epoch)
        self.writer.add_scalar(f'{tag}/late_mean', g_late, epoch)

        self.logger.debug(f"  InputGate ({branch_name}): mean={g_mean:.4f}, std={g_std:.4f}, "
                          f"early={g_early:.4f}, late={g_late:.4f}")

    def _log_delta_diagnostics(self, epoch: int, encoder, windows, branch_name: str):
        """
        Log SSM delta (selectivity) statistics per Mamba block via hooks on x_proj.

        Delta controls how much each timestep updates the SSM state:
        large delta = strong update (important candle), small = ignored (noise).
        Reconstructed from x_proj output without modifying mamba_ssm library.
        """
        import torch.nn.functional as F

        x_proj_outputs = {}
        hooks = []

        def make_hook(name):
            def fn(module, input, output):
                x_proj_outputs[name] = output.detach()
            return fn

        # Register hooks on x_proj inside each Mamba block
        for i, block in enumerate(encoder.mamba_blocks):
            hook = block.mamba.x_proj.register_forward_hook(make_hook(i))
            hooks.append(hook)

        # Forward pass (reuses eval mode + no_grad from caller)
        encoder.encode(windows)

        # Remove hooks immediately
        for h in hooks:
            h.remove()

        # Reconstruct delta and compute stats per block
        self.logger.debug(f"\n  SSM Delta Diagnostics ({branch_name}):")
        for i, block in enumerate(encoder.mamba_blocks):
            x_dbl = x_proj_outputs.get(i)
            if x_dbl is None:
                continue

            # CPU Mamba produces 3D (B, L, feat), GPU produces 2D (B*L, feat)
            if x_dbl.dim() == 3:
                x_dbl = x_dbl.reshape(-1, x_dbl.shape[-1])

            dt_rank = block.mamba.dt_rank
            dt_raw = x_dbl[:, :dt_rank]  # (batch*seqlen, dt_rank)

            # Reconstruct: delta = softplus(dt_proj(dt_raw) + bias)
            dt_proj = block.mamba.dt_proj
            dt_projected = F.linear(dt_raw, dt_proj.weight, dt_proj.bias)  # (batch*seqlen, d_inner)
            delta = F.softplus(dt_projected)  # (batch*seqlen, d_inner)

            d_mean = delta.mean().item()
            d_std = delta.std().item()
            d_max = delta.max().item()
            d_min = delta.min().item()

            tag = f'Delta/{branch_name}_block_{i}'
            self.writer.add_scalar(f'{tag}/mean', d_mean, epoch)
            self.writer.add_scalar(f'{tag}/std', d_std, epoch)
            self.writer.add_scalar(f'{tag}/max', d_max, epoch)

            self.logger.debug(f"    block_{i}: mean={d_mean:.4f}, std={d_std:.4f}, max={d_max:.2f}, min={d_min:.4f}")

    def _log_branch_diagnostics(self, epoch: int, batch_data):
        """Log multi-TF branch diagnostics: norms, attention paths, gate weights."""
        if not hasattr(self.model, 'encoder_primary') or not hasattr(self.model, 'encoder_secondary'):
            return

        self.logger.debug("=" * 60)
        self.logger.info(f"Multi-TF Branch Diagnostics (Epoch {epoch + 1})")
        self.logger.debug("=" * 60)

        self.model.eval()
        with torch.no_grad():
            windows_pri = batch_data[0].to(self.device)
            windows_sec = batch_data[1].to(self.device)
            tf_mapping = batch_data[2].to(self.device)

            # SSM delta + input gate diagnostics per branch
            self._log_delta_diagnostics(epoch, self.model.encoder_primary, windows_pri, 'primary')
            self._log_delta_diagnostics(epoch, self.model.encoder_secondary, windows_sec, 'secondary')
            self._log_input_gate_diagnostics(epoch, self.model.encoder_primary, windows_pri, 'primary')
            self._log_input_gate_diagnostics(epoch, self.model.encoder_secondary, windows_sec, 'secondary')

            # Encode both branches
            h_pri = self.model.encoder_primary.encode(windows_pri)
            h_sec = self.model.encoder_secondary.encode(windows_sec)

            # Representation norms
            self._diag_representation_norms(epoch, h_pri, h_sec)

            # Pooled representations
            last_pri = h_pri[:, -1, :]
            max_pri = h_pri.max(dim=1)[0]
            last_sec = h_sec[:, -1, :]
            max_sec = h_sec.max(dim=1)[0]
            paths = [(last_pri, max_pri), (last_sec, max_sec)]

            # Attention path construction (modifies paths)
            paths = self._diag_attention_paths(epoch, h_pri, h_sec, tf_mapping,
                                               last_pri, max_pri, last_sec, max_sec, paths)

            # Gate/fusion diagnostics
            self._diag_gate_weights(epoch, paths)

        self.logger.debug("=" * 60)

    def _diag_representation_norms(self, epoch, h_pri, h_sec):
        """Log representation norms for both branches."""
        norm_pri_last = h_pri[:, -1, :].norm(dim=-1).mean().item()
        norm_pri_max = h_pri.max(dim=1)[0].norm(dim=-1).mean().item()
        norm_sec_last = h_sec[:, -1, :].norm(dim=-1).mean().item()
        norm_sec_max = h_sec.max(dim=1)[0].norm(dim=-1).mean().item()

        self.logger.debug("Branch Representation Norms (averaged over batch):")
        self.logger.debug(f"  primary  - last_token: {norm_pri_last:.4f}, max_pool: {norm_pri_max:.4f}")
        self.logger.debug(f"  secondary - last_token: {norm_sec_last:.4f}, max_pool: {norm_sec_max:.4f}")

        self.writer.add_scalar('BranchDiagnostics/norm_primary_last', norm_pri_last, epoch)
        self.writer.add_scalar('BranchDiagnostics/norm_primary_max', norm_pri_max, epoch)
        self.writer.add_scalar('BranchDiagnostics/norm_secondary_last', norm_sec_last, epoch)
        self.writer.add_scalar('BranchDiagnostics/norm_secondary_max', norm_sec_max, epoch)

    def _diag_attention_paths(self, epoch, h_pri, h_sec, tf_mapping,
                              last_pri, max_pri, last_sec, max_sec, paths):
        """Build attention-dependent paths and log attention norms. Returns updated paths."""
        attn_pos = getattr(self.model, 'attention_position', 'off')

        if attn_pos == "pre_gate":
            attn_pri = self.model.attn_block_primary(h_pri)
            attn_pooled_pri = self.model.attn_pool_primary(attn_pri)
            attn_sec = self.model.attn_block_secondary(h_sec)
            attn_pooled_sec = self.model.attn_pool_secondary(attn_sec)
            paths = [
                (last_pri, max_pri), (attn_pooled_pri, attn_pooled_pri),
                (last_sec, max_sec), (attn_pooled_sec, attn_pooled_sec),
            ]
            norm_attn_pri = attn_pooled_pri.norm(dim=-1).mean().item()
            norm_attn_sec = attn_pooled_sec.norm(dim=-1).mean().item()
            self.writer.add_scalar('BranchDiagnostics/norm_attn_primary', norm_attn_pri, epoch)
            self.writer.add_scalar('BranchDiagnostics/norm_attn_secondary', norm_attn_sec, epoch)
            self.logger.debug(f"  attn_primary: {norm_attn_pri:.4f}, attn_secondary: {norm_attn_sec:.4f}")

        elif attn_pos == "post_gate":
            h_combined = torch.cat([h_pri, h_sec], dim=1)
            B = h_combined.size(0)
            query = self.model.cross_attn_query.expand(B, -1, -1)
            attn_out, _ = self.model.cross_attn(query, h_combined, h_combined)
            attn_out = self.model.cross_attn_dropout(attn_out.squeeze(1))
            paths = [(last_pri, max_pri), (last_sec, max_sec), (attn_out, attn_out)]
            norm_cross = attn_out.norm(dim=-1).mean().item()
            self.writer.add_scalar('BranchDiagnostics/norm_cross_tf_attn', norm_cross, epoch)
            self.logger.debug(f"  cross_tf_attn: {norm_cross:.4f}")

        elif attn_pos == "aligned":
            h_enriched = self.model.aligned_fusion(h_pri, h_sec, tf_mapping)
            norm_enriched = h_enriched[:, -1, :].norm(dim=-1).mean().item()
            self.writer.add_scalar('BranchDiagnostics/norm_enriched_last', norm_enriched, epoch)
            self.logger.debug(f"  enriched_primary: {norm_enriched:.4f}")

            # AlignedFusion internal gate diagnostics
            idx = tf_mapping.unsqueeze(-1).expand(-1, -1, h_sec.size(-1))
            h_sec_aligned = torch.gather(h_sec, 1, idx)
            combined_af = torch.cat([h_pri, h_sec_aligned], dim=-1)
            gate_vals = self.model.aligned_fusion.gate(combined_af)
            gate_mean = gate_vals.mean().item()
            gate_std = gate_vals.std().item()
            self.writer.add_scalar('AlignedFusion/gate_mean', gate_mean, epoch)
            self.writer.add_scalar('AlignedFusion/gate_std', gate_std, epoch)
            half = gate_vals.shape[1] // 2
            self.writer.add_scalar('AlignedFusion/gate_early', gate_vals[:, :half].mean().item(), epoch)
            self.writer.add_scalar('AlignedFusion/gate_late', gate_vals[:, half:].mean().item(), epoch)
            self.logger.debug(f"  AlignedFusion gate: mean={gate_mean:.4f}, std={gate_std:.4f}")

        return paths

    def _diag_gate_weights(self, epoch, paths):
        """Log gate/fusion weight diagnostics (v2 scalar, v3 MLP, or concat norms)."""
        from modules.model.mamba import UnifiedHead

        n_paths = self.model.head.n_paths if isinstance(self.model.head, UnifiedHead) else 2
        path_names = ['std_primary', 'std_secondary']
        if n_paths == 4:
            path_names = ['std_primary', 'attn_primary', 'std_secondary', 'attn_secondary']
        elif n_paths == 3:
            path_names = ['std_primary', 'std_secondary', 'cross_tf_attn']

        if isinstance(self.model.head, UnifiedHead) and self.model.head.gate_version == "v2":
            context_parts = []
            for last, mx in paths:
                context_parts.extend([last, mx])
            context = torch.cat(context_parts, dim=-1)
            gate_weights = self.model.head.gate(context)

            self.logger.debug(f"\nGated Fusion Weights (averaged over batch):")
            for i, name in enumerate(path_names):
                avg_w = gate_weights[:, i].mean().item()
                std_w = gate_weights[:, i].std().item()
                self.logger.debug(f"  w_{name}: {avg_w:.3f} ± {std_w:.3f}")
                self.writer.add_scalar(f'BranchDiagnostics/gate_w_{name}_mean', avg_w, epoch)

            avg_w_pri = gate_weights[:, 0].mean().item()
            avg_w_sec = gate_weights[:, n_paths - 2 if n_paths == 4 else 1].mean().item()
            std_w_pri = gate_weights[:, 0].std().item()
            self.logger.info(f"  → std_primary: {avg_w_pri*100:.1f}%, std_secondary: {avg_w_sec*100:.1f}%")

            self.writer.add_scalar('BranchDiagnostics/gate_w_primary_mean', avg_w_pri, epoch)
            self.writer.add_scalar('BranchDiagnostics/gate_w_secondary_mean', avg_w_sec, epoch)
            self.writer.add_scalar('BranchDiagnostics/gate_w_primary_std', std_w_pri, epoch)

            max_w = gate_weights.mean(dim=0).max().item()
            max_idx = gate_weights.mean(dim=0).argmax().item()
            if max_w > 0.50:
                self.logger.warning(f"  ⚠️  Gate favors {path_names[max_idx]} ({max_w*100:.1f}%)")

        elif isinstance(self.model.head, UnifiedHead) and self.model.head.gate_version == "v3":
            head = self.model.head
            projections = [
                head.projections[i](torch.cat([last, mx], dim=-1))
                for i, (last, mx) in enumerate(paths)
            ]
            proj_cat = torch.cat(projections, dim=-1)
            gate_values = head.gate_mlp(proj_cat)
            gate_values = gate_values.view(-1, head.n_paths, head.d_model)
            proj_stack = torch.stack(projections, dim=1)

            self.logger.debug(f"\nGate V3 Per-Feature Diagnostics:")
            for i, name in enumerate(path_names):
                g_mean = gate_values[:, i, :].mean().item()
                g_std = gate_values[:, i, :].std().item()
                contrib_norm = (proj_stack[:, i, :] * gate_values[:, i, :]).norm(dim=-1).mean().item()

                self.writer.add_scalar(f'GateV3/gate_mean_{name}', g_mean, epoch)
                self.writer.add_scalar(f'GateV3/gate_std_{name}', g_std, epoch)
                self.writer.add_scalar(f'GateV3/contribution_norm_{name}', contrib_norm, epoch)

                self.logger.debug(f"  {name}: gate_mean={g_mean:.3f}, gate_std={g_std:.3f}, contrib_norm={contrib_norm:.4f}")

                if g_mean < 0.1:
                    self.logger.warning(f"  ⚠️  Gate V3: path '{name}' nearly dead (mean={g_mean:.3f})")

        elif hasattr(self.model.head, 'net'):
            fusion_layer = self.model.head.net[0]
            if isinstance(fusion_layer, nn.Linear):
                weights = fusion_layer.weight.data
                d_model = self.model.d_model

                w_last_pri = weights[:, :d_model].norm().item()
                w_max_pri = weights[:, d_model:2*d_model].norm().item()
                w_last_sec = weights[:, 2*d_model:3*d_model].norm().item()
                w_max_sec = weights[:, 3*d_model:].norm().item()

                total_norm = w_last_pri + w_max_pri + w_last_sec + w_max_sec
                pct_pri = (w_last_pri + w_max_pri) / total_norm * 100
                pct_sec = (w_last_sec + w_max_sec) / total_norm * 100

                self.logger.debug("\nFusion Head Weight Norms (learned importance):")
                self.logger.debug(f"  primary   - last: {w_last_pri:.4f}, max: {w_max_pri:.4f}")
                self.logger.debug(f"  secondary - last: {w_last_sec:.4f}, max: {w_max_sec:.4f}")
                self.logger.info(f"  → primary total: {pct_pri:.1f}%, secondary total: {pct_sec:.1f}%")

                self.writer.add_scalar('BranchDiagnostics/fusion_weight_primary_pct', pct_pri, epoch)
                self.writer.add_scalar('BranchDiagnostics/fusion_weight_secondary_pct', pct_sec, epoch)

                if pct_sec > 70:
                    self.logger.warning(f"  ⚠️  ALERT: secondary branch dominates fusion ({pct_sec:.1f}% vs {pct_pri:.1f}%)")
                elif pct_pri > 70:
                    self.logger.warning(f"  ⚠️  ALERT: primary branch dominates fusion ({pct_pri:.1f}% vs {pct_sec:.1f}%)")

    def _setup_signal_handler(self):
        """Setup signal handler for graceful Ctrl+C."""
        def signal_handler(signum, frame):
            self.logger.warning("\nReceived interrupt signal (Ctrl+C)")
            self.logger.warning("Finishing current batch and saving...")
            self.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)

    def _save_on_interrupt(self):
        """Save checkpoint when interrupted."""
        if self.config['SAVE_ON_INTERRUPT']:
            # Evaluate current state
            test_results = self.evaluate()
            long_pnl, short_pnl = self._get_test_pnl_arrays()
            metrics = compute_all_metrics(
                y_true=test_results['y_true'],
                y_pred=test_results['y_pred'],
                y_pred_probs=test_results['y_pred_probs'],
                model=self.model,
                train_loss=self.last_train_loss,
                test_loss=test_results['test_loss'],
                train_acc=self.last_train_acc,
                test_acc=test_results['test_acc'],
                num_classes=self.num_classes,
                prediction_target=self.prediction_target,
                long_pnl_R=long_pnl,
                short_pnl_R=short_pnl,
            )
            self.save_checkpoint(self.current_epoch, metrics, is_interrupt=True)
