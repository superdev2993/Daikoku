"""
Main entry point for Daikoku - Crypto Trading Mamba Architecture

Orchestrates the complete training pipeline:
1. Load and validate raw data (per dataset)
2. Compute labels on raw data (Triple Barrier)
3. Transform data (log + normalize)
4. Align data and labels
5. Create train/test datasets by indices
6. Merge datasets via ConcatDataset
7. Initialize model and trainer
8. Train model with checkpointing and TensorBoard logging
"""

import signal
import time
import traceback

import sys
import os
import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.data.labeling import TARGET_CLASS_NAMES
from modules.data.pipeline import process_single_dataset
from modules.model.mamba import build_arch_params
from modules.training.trainer import Trainer, get_device
from modules.training.loss import create_criterion, compute_class_weights
from modules.training.setup import merge_datasets, create_dataloaders, create_model, create_optimizer, build_trainer_config, resolve_num_workers
from modules.utils.logger import get_logger
from modules.utils.seed import set_seed


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Daikoku - Crypto Trading Mamba Architecture"
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., models/latest.pt)'
    )
    return parser.parse_args()


def _build_config_summary(run_number, num_classes, class_names_map, label_dist, label_pcts,
                          total_n_valid, total_n_train, total_n_test, best_balanced_acc):
    """Build complete configuration summary text for TensorBoard."""
    label_lines = "\n".join(
        f"Class {c} ({class_names_map.get(c, '?')})     : {label_dist.get(c, 0)} ({label_pcts.get(c, 0):.2f}%)"
        for c in range(num_classes)
    )
    return f"""RUN {run_number:03d} - CONFIGURATION COMPLETE

=== MULTI-TIMEFRAME ===
MULTI_TF_MODE         : {config.MULTI_TF_MODE}
MULTI_TF_DIVISOR      : {config.MULTI_TF_DIVISOR}
MULTI_TF_ALIGN        : {config.MULTI_TF_ALIGN}
MULTI_TF_N_LAYERS     : {config.MULTI_TF_N_LAYERS}
MULTI_TF_D_STATE      : {config.MULTI_TF_D_STATE}

=== ATTENTION & GATE ===
ATTENTION_POSITION    : {config.ATTENTION_POSITION}
ATTENTION_NUM_HEADS   : {config.ATTENTION_NUM_HEADS}
GATE_VERSION          : {config.GATE_VERSION}

=== DATA PREPROCESSING ===
WINDOW_SIZE           : {config.WINDOW_SIZE}
ATR_PERIOD            : {config.ATR_PERIOD}
ATR_MULTIPLIER_TP     : {config.ATR_MULTIPLIER_TP}
ATR_MULTIPLIER_SL     : {config.ATR_MULTIPLIER_SL} (R:R = {config.ATR_MULTIPLIER_TP/config.ATR_MULTIPLIER_SL:.1f}:1)
MAX_HORIZON           : {config.MAX_HORIZON}
TRAIN_TEST_SPLIT      : {config.TRAIN_TEST_SPLIT}
NORMALIZE             : {config.NORMALIZE}

=== LABEL DISTRIBUTION (num_classes={num_classes}) ===
{label_lines}

=== MODEL ARCHITECTURE ===
D_MODEL               : {config.D_MODEL}
N_LAYERS              : {config.N_LAYERS}
D_STATE               : {config.D_STATE}
D_CONV                : {config.D_CONV}
EXPAND                : {config.EXPAND}

=== REGULARIZATION ===
DROPOUT               : {config.DROPOUT}
DROPOUT_LAST_N_LAYERS : {config.DROPOUT_LAST_N_LAYERS}
WEIGHT_DECAY          : {config.WEIGHT_DECAY}

=== TRAINING ===
LEARNING_RATE         : {config.LEARNING_RATE}
BATCH_SIZE            : {config.BATCH_SIZE}
EPOCHS                : {config.EPOCHS}
SHUFFLE_TRAIN         : {config.SHUFFLE_TRAIN}

=== SYSTEM ===
DEVICE                : {config.DEVICE}
SEED                  : {config.SEED}
INPUT_FILES           : {config.INPUT_FILES}

=== DATASET STATS ===
Total samples         : {total_n_valid}
Train samples         : {total_n_train}
Test samples          : {total_n_test}

=== FINAL METRICS ===
Best balanced_accuracy: {best_balanced_acc:.4f}
"""


def main():
    """
    Main orchestration function for the complete training pipeline.
    """
    # ========================================================================
    # STEP 1: INITIALIZATION
    # ========================================================================

    _run_start = time.time()

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Parse CLI arguments
    args = parse_arguments()

    # Initialize logger
    logger = get_logger(__name__)
    # Log configuration
    logger.debug("Configuration:")
    logger.debug(f"  - Data files: {config.INPUT_FILES}")
    logger.debug(f"  - Window size: {config.WINDOW_SIZE}")
    logger.debug(f"  - Train/test split: {config.TRAIN_TEST_SPLIT}")
    logger.debug(f"  - Epochs: {config.EPOCHS}")
    logger.debug(f"  - Batch size: {config.BATCH_SIZE}")
    logger.debug(f"  - Learning rate: {config.LEARNING_RATE}")
    logger.debug(f"  - Device: {config.DEVICE}")
    logger.debug(f"  - Seed: {config.SEED}")

    # ========================================================================
    # STEP 2-8: PROCESS EACH DATASET INDEPENDENTLY
    # ========================================================================

    logger.info(f"Processing {len(config.INPUT_FILES)} dataset(s)...")

    train_datasets = []
    test_datasets = []
    all_label_dist = {}
    all_label_pcts = {}
    total_n_valid = 0
    total_n_train = 0
    total_n_test = 0

    for i, csv_path in enumerate(config.INPUT_FILES):
        logger.info(f"--- Dataset {i+1}/{len(config.INPUT_FILES)}: {csv_path} ---")

        try:
            result = process_single_dataset(csv_path, logger)
        except FileNotFoundError:
            logger.error(f"Data file not found: {csv_path}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Data validation failed for {csv_path}: {e}")
            sys.exit(1)

        train_datasets.append(result['train_dataset'])
        test_datasets.append(result['test_dataset'])

        # Accumulate label distribution
        for label, count in result['label_dist'].items():
            all_label_dist[label] = all_label_dist.get(label, 0) + count
        total_n_valid += result['n_valid']
        total_n_train += result['n_train']
        total_n_test += result['n_test']

        # Keep last result for normalization_params, n_features
        # (all datasets use the same config so these should be consistent)
        last_result = result

    # Compute global label percentages
    total_labels = sum(all_label_dist.values())
    for label in all_label_dist:
        all_label_pcts[label] = all_label_dist[label] / total_labels * 100

    label_dist = all_label_dist
    label_pcts = all_label_pcts
    normalization_params = last_result['normalization_params']
    n_features = last_result['n_features']

    # ========================================================================
    # STEP 9: MERGE DATASETS + CREATE DATALOADERS
    # ========================================================================

    train_dataset, test_dataset = merge_datasets(train_datasets, test_datasets)

    logger.info(f"✓ Total dataset size: {len(train_dataset)} train / {len(test_dataset)} test")

    # Classification: num_classes derived from PREDICTION_TARGET
    num_classes = 3 if config.PREDICTION_TARGET == "triple" else 2

    class_names_map = TARGET_CLASS_NAMES[config.PREDICTION_TARGET]
    logger.debug("Global label distribution:")
    for label in sorted(label_dist.keys()):
        label_name = class_names_map.get(label, f"Class_{label}")
        logger.debug(f"  Class {label} ({label_name}): {label_dist[label]} ({label_pcts[label]:.2f}%)")

    logger.debug(f"Train shuffle: {config.SHUFFLE_TRAIN}, Test shuffle: False (always)")

    train_loader, test_loader, sampler_info = create_dataloaders(
        train_dataset, test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle_train=config.SHUFFLE_TRAIN,
        shuffle_chunks=config.SHUFFLE_CHUNKS,
        seed=config.SEED,
        num_workers=resolve_num_workers(use_cuda=torch.cuda.is_available()),
        persistent_workers=config.PERSISTENT_WORKERS,
        use_cuda=torch.cuda.is_available(),
    )
    if sampler_info:
        logger.info(f"Using ChunkSampler with {sampler_info} chunks")

    logger.debug(f"✓ Train batches: {len(train_loader)}")
    logger.debug(f"✓ Test batches: {len(test_loader)}")

    # ========================================================================
    # STEP 10: DEVICE SELECTION
    # ========================================================================

    device = get_device(config.DEVICE)

    # GPU optimizations
    if device.type == "cuda" and config.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
        logger.debug("cuDNN benchmark enabled")

    # ========================================================================
    # STEP 11: MODEL INITIALIZATION
    # ========================================================================

    logger.info("Initializing model...")

    mtf_mode = config.MULTI_TF_MODE
    arch_params = build_arch_params()
    n_features_secondary = last_result.get('n_features_sec') if mtf_mode == "dual" else None

    model = create_model(
        mtf_mode=mtf_mode,
        n_features=n_features,
        device=device,
        d_model=config.D_MODEL,
        d_state=config.D_STATE,
        d_conv=config.D_CONV,
        expand=config.EXPAND,
        n_layers=config.N_LAYERS,
        arch_params=arch_params,
        num_classes=num_classes,
        n_features_secondary=n_features_secondary,
        mtf_d_state=config.MULTI_TF_D_STATE,
        mtf_n_layers=config.MULTI_TF_N_LAYERS,
    )

    if mtf_mode == "dual":
        logger.debug(f"  Multi-timeframe DUAL model: primary({n_features}feat, {config.N_LAYERS}L) + secondary({n_features_secondary}feat, {config.MULTI_TF_N_LAYERS}L)")
    elif mtf_mode == "calibration":
        logger.debug(f"  Calibration model: {n_features}feat, {config.MULTI_TF_N_LAYERS}L, d_state={config.MULTI_TF_D_STATE}")

    # Log parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✓ Model initialized (mode={mtf_mode}, {trainable_params:,} params)")
    logger.debug(f"  Total parameters: {total_params:,}")
    logger.debug(f"  Trainable parameters: {trainable_params:,}")

    # ========================================================================
    # STEP 12: OPTIMIZER AND CRITERION
    # ========================================================================

    optimizer, n_decay, n_no_decay = create_optimizer(model, config.LEARNING_RATE, config.WEIGHT_DECAY)

    # Compute dynamic class weights from label distribution
    class_weights = compute_class_weights(label_dist, config.CLASS_WEIGHT_ALPHA, num_classes=num_classes)

    # Create criterion (UnifiedLoss)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    criterion = create_criterion(config_dict, class_weights=class_weights)

    logger.debug(f"✓ Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    logger.debug(f"  Decay params: {n_decay:,} | No-decay params: {n_no_decay:,}")
    if config.LR_SCHEDULER != "none":
        logger.info(f"✓ LR Scheduler: {config.LR_SCHEDULER} (LR_MIN={config.LR_MIN})")

    logger.info(f"✓ Volume Profile: lookback={config.VP_LOOKBACK}, bins={config.VP_BINS}, VA={config.VP_VA_PCT:.0%}")

    # Log criterion details
    parts = []
    if config.FOCAL_GAMMA > 0:
        parts.append(f"focal(γ={config.FOCAL_GAMMA})")
    if class_weights is not None:
        w_str = ", ".join(f"{w:.3f}" for w in class_weights)
        parts.append(f"class_weight(α={config.CLASS_WEIGHT_ALPHA}, w=[{w_str}])")
    if parts:
        logger.info(f"✓ Criterion: {criterion.__class__.__name__} [{' + '.join(parts)}]")
    else:
        logger.info(f"✓ Criterion: {criterion.__class__.__name__} [standard CE]")

    # ========================================================================
    # STEP 13: TENSORBOARD WRITER
    # ========================================================================

    logger.debug("Creating TensorBoard writer...")

    # Auto-increment run number
    runs_dir = Path(config.TENSORBOARD_DIR)
    runs_dir.mkdir(exist_ok=True)

    existing_runs = list(runs_dir.glob('run_[0-9][0-9][0-9]'))
    if existing_runs:
        numbers = [int(run.name.split('_')[1]) for run in existing_runs]
        run_number = max(numbers) + 1
    else:
        run_number = 1

    log_dir = os.path.join(config.TENSORBOARD_DIR, f'run_{run_number:03d}')
    writer = SummaryWriter(log_dir=log_dir)

    logger.info(f"✓ TensorBoard log dir: {log_dir} (Run #{run_number})")
    # Log-level independent progress messages
    print(f"[RUN {run_number:03d}] START | epochs={config.EPOCHS} lr={config.LEARNING_RATE} "
          f"d_model={config.D_MODEL} n_layers={config.N_LAYERS} dropout={config.DROPOUT} wd={config.WEIGHT_DECAY} "
          f"sched={config.LR_SCHEDULER}")

    # ========================================================================
    # STEP 14: CRITICAL - TRAINER CONFIG
    # ========================================================================

    logger.debug("Creating trainer configuration...")

    trainer_config = build_trainer_config(
        normalization_params=normalization_params,
        n_norm_features=last_result['n_norm_features'],
    )

    logger.debug(f"✓ Trainer config created ({len(trainer_config)} parameters auto-captured)")

    # ========================================================================
    # STEP 15: CREATE TRAINER
    # ========================================================================

    logger.debug("Creating Trainer...")

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
        setup_signal_handler=True
    )

    logger.debug("✓ Trainer initialized")

    # ========================================================================
    # STEP 16: TRAINING OR RESUMPTION
    # ========================================================================

    start_epoch = 0

    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    try:
        logger.info("STARTING TRAINING")

        trainer.train(start_epoch=start_epoch)

        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        print(f"[RUN {run_number:03d}] DONE | best_balanced_acc={trainer.best_balanced_acc:.4f}")

        # Copy final checkpoint (latest.pt) into the run's TensorBoard directory
        latest_pt = Path(config.CHECKPOINT_DIR) / 'latest.pt'
        if latest_pt.exists():
            shutil.copy2(latest_pt, Path(log_dir) / 'latest.pt')
            logger.info(f"✓ Final checkpoint copied to {log_dir}/latest.pt")

        # ====================================================================
        # LOG CONFIGURATION TO TENSORBOARD
        # ====================================================================

        logger.debug("Logging configuration to TensorBoard...")

        config_summary = _build_config_summary(
            run_number, num_classes, class_names_map, label_dist, label_pcts,
            total_n_valid, total_n_train, total_n_test, trainer.best_balanced_acc,
        )

        # Log complete configuration as single text block
        writer.add_text('config/complete', config_summary, 0)

        logger.debug("✓ Configuration logged to TensorBoard")

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Training interrupted by user (Ctrl+C)")
        logger.warning("Checkpoint saved via Trainer's signal handler")

    except Exception as e:
        logger.error("")
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        raise

    finally:
        # Always close TensorBoard writer
        writer.close()
        logger.debug("✓ TensorBoard writer closed")
        elapsed = (time.time() - _run_start) / 60
        logger.info(f"Total run duration: {elapsed:.1f} min")

        # Restore default SIGINT so process exits immediately on Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    main()
