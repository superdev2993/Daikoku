"""
Multi-Objective Hyperparameter Optimization with Optuna

Optimizes 3 objectives:
1. Maximize edge_total_R (total profitability in R-multiples)
2. Minimize gap_loss (robustness/overfitting prevention)
3. Maximize edge_per_trade (per-trade quality in R-multiples)

Search space is defined in config.OPTUNA_SEARCH_SPACE.
All params not in the search space use their config.py defaults.

Usage:
  python optimize.py --trials 50           # New study
  python optimize.py --trials 50 --resume  # Resume existing study
"""

import traceback

import optuna
import argparse
import numpy as np
import torch
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.utils.seed import set_seed


# ============================================================================
# SEARCH SPACE ENGINE
# ============================================================================

def suggest_params(trial):
    """
    Override config with Optuna trial suggestions from OPTUNA_SEARCH_SPACE.

    Params in search space are suggested by Optuna and written into config module.
    Params not in search space keep their config.py value (no trial suggestion).

    Formats in OPTUNA_SEARCH_SPACE:
        [v1, v2, ...]       → suggest_categorical
        (min, max) int       → suggest_int
        (min, max) float     → suggest_float (log scale)

    Returns:
        searched: dict {name: value} of params actually searched
        saved_config: dict {ATTR: original} for restoration in finally block
    """
    search_space = config.OPTUNA_SEARCH_SPACE
    searched = {}
    saved_config = {}

    for name, spec in search_space.items():
        config_attr = name.upper()
        if not hasattr(config, config_attr):
            raise ValueError(f"Unknown param in OPTUNA_SEARCH_SPACE: '{name}' (no config.{config_attr})")

        # Save original for restoration
        saved_config[config_attr] = getattr(config, config_attr)

        # Suggest value
        if isinstance(spec, list):
            value = trial.suggest_categorical(name, spec)
        elif isinstance(spec, tuple) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                value = trial.suggest_int(name, low, high)
            else:
                value = trial.suggest_float(name, float(low), float(high), log=True)
        else:
            raise ValueError(f"Invalid spec for '{name}': use [v1,v2,...] or (min,max)")

        setattr(config, config_attr, value)
        searched[name] = value

    return searched, saved_config


def restore_config(saved_config):
    """Restore config values modified by suggest_params."""
    for attr, value in saved_config.items():
        setattr(config, attr, value)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """
    Multi-objective function for Optuna optimization.

    Returns:
        tuple: (edge_total_R, gap_loss, edge_per_trade)
    """
    # Suggest params from search space, override config
    searched, saved_config = suggest_params(trial)

    # Imports (inside function to pick up mutated config)
    from modules.data.pipeline import process_single_dataset
    from modules.data.labeling import TARGET_CLASS_NAMES
    from modules.model.mamba import build_arch_params
    from modules.training.trainer import Trainer, get_device
    from modules.training.loss import create_criterion, compute_class_weights
    from modules.training.setup import (
        merge_datasets, create_dataloaders, create_model,
        create_optimizer, build_trainer_config, resolve_num_workers,
    )
    from modules.utils.logger import get_logger

    logger = get_logger(f"optuna.trial_{trial.number}")

    try:
        # Log trial info
        logger.info("=" * 80)
        logger.info(f"Trial {trial.number}")
        if searched:
            logger.info(f"  Searched: {searched}")
        else:
            logger.info("  No search space defined — using config.py defaults")
        logger.info("=" * 80)

        # ====================================================================
        # PREPROCESSING
        # ====================================================================

        train_datasets = []
        test_datasets = []
        all_label_dist = {}
        total_n_valid = 0
        total_n_train = 0
        total_n_test = 0

        for csv_path in config.INPUT_FILES:
            result = process_single_dataset(csv_path, logger)
            train_datasets.append(result['train_dataset'])
            test_datasets.append(result['test_dataset'])
            for label, count in result['label_dist'].items():
                all_label_dist[label] = all_label_dist.get(label, 0) + count
            total_n_valid += result['n_valid']
            total_n_train += result['n_train']
            total_n_test += result['n_test']

        last_result = result
        normalization_params = last_result['normalization_params']

        # Merge datasets
        train_dataset, test_dataset = merge_datasets(train_datasets, test_datasets)

        # Label distribution
        total_labels = sum(all_label_dist.values())
        label_dist = all_label_dist
        num_classes = 3 if config.PREDICTION_TARGET == "triple" else 2

        logger.info(f"✓ {total_n_train} train / {total_n_test} test samples")

        # DataLoaders
        train_loader, test_loader, _ = create_dataloaders(
            train_dataset, test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle_train=config.SHUFFLE_TRAIN,
            shuffle_chunks=config.SHUFFLE_CHUNKS,
            seed=config.SEED,
            num_workers=resolve_num_workers(use_cuda=torch.cuda.is_available()),
            persistent_workers=config.PERSISTENT_WORKERS,
            use_cuda=torch.cuda.is_available(),
        )

        # ====================================================================
        # MODEL + OPTIMIZER + CRITERION
        # ====================================================================

        # Same seed for all trials: differences come ONLY from hyperparameters
        set_seed(config.SEED)

        device = get_device(config.DEVICE)

        # GPU optimizations
        if device.type == "cuda" and config.CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
        n_features = last_result['n_features']
        mtf_mode = config.MULTI_TF_MODE
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
            arch_params=build_arch_params(),
            num_classes=num_classes,
            n_features_secondary=n_features_secondary,
            mtf_d_state=config.MULTI_TF_D_STATE,
            mtf_n_layers=config.MULTI_TF_N_LAYERS,
        )

        optimizer, _, _ = create_optimizer(model, config.LEARNING_RATE, config.WEIGHT_DECAY)

        class_weights = compute_class_weights(label_dist, config.CLASS_WEIGHT_ALPHA, num_classes=num_classes)
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        criterion = create_criterion(config_dict, class_weights=class_weights)

        # TensorBoard
        trial_name = f"optuna_{trial.number:03d}"
        log_dir = os.path.join(config.TENSORBOARD_DIR, trial_name)
        writer = SummaryWriter(log_dir=log_dir)

        # Trainer config (reads config.* which has been mutated by suggest_params)
        trainer_config = build_trainer_config(
            normalization_params=normalization_params,
            n_norm_features=last_result['n_norm_features'],
            overrides={
                'CHECKPOINT_DIR': os.path.join(config.CHECKPOINT_DIR, trial_name),
                'SAVE_EVERY_N_EPOCHS': config.EPOCHS,
                'KEEP_LATEST': False,
                'SAVE_ON_INTERRUPT': False,
            },
        )

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
            setup_signal_handler=False,
        )

        # ====================================================================
        # TRAINING
        # ====================================================================

        trainer.train(start_epoch=0)

        # Log config to TensorBoard
        class_names_map = TARGET_CLASS_NAMES[config.PREDICTION_TARGET]
        label_pcts = {l: c / total_labels * 100 for l, c in label_dist.items()}
        config_lines = [
            f"TRIAL {trial.number} - CONFIGURATION",
            "",
            "=== SEARCHED PARAMS ===" if searched else "=== NO SEARCH (config defaults) ===",
        ]
        for name, value in searched.items():
            config_lines.append(f"  {name.upper():<25}: {value}")
        config_lines += [
            "",
            f"=== KEY CONFIG VALUES ===",
            f"  D_MODEL               : {config.D_MODEL}",
            f"  N_LAYERS              : {config.N_LAYERS}",
            f"  D_STATE               : {config.D_STATE}",
            f"  D_CONV                : {config.D_CONV}",
            f"  EXPAND                : {config.EXPAND}",
            f"  LEARNING_RATE         : {config.LEARNING_RATE}",
            f"  BATCH_SIZE            : {config.BATCH_SIZE}",
            f"  WEIGHT_DECAY          : {config.WEIGHT_DECAY}",
            f"  DROPOUT               : {config.DROPOUT}",
            f"  WINDOW_SIZE           : {config.WINDOW_SIZE}",
            f"  EPOCHS                : {config.EPOCHS}",
            f"  MULTI_TF_MODE         : {config.MULTI_TF_MODE}",
            f"  ATTENTION_POSITION    : {config.ATTENTION_POSITION}",
            f"  GATE_VERSION          : {config.GATE_VERSION}",
            "",
            f"=== LABEL DISTRIBUTION (num_classes={num_classes}) ===",
        ]
        for c in range(num_classes):
            name = class_names_map.get(c, '?')
            config_lines.append(f"  Class {c} ({name}): {label_dist.get(c, 0)} ({label_pcts.get(c, 0):.2f}%)")
        config_lines += [
            "",
            f"=== DATASET ===",
            f"  Total: {total_n_valid} | Train: {total_n_train} | Test: {total_n_test}",
            f"  Files: {config.INPUT_FILES}",
        ]
        writer.add_text('config/complete', "\n".join(config_lines), 0)
        writer.close()

        # ====================================================================
        # EXTRACT METRICS
        # ====================================================================

        test_results = trainer.evaluate()

        from modules.training.metrics import compute_all_metrics

        long_pnl, short_pnl = trainer._get_test_pnl_arrays()
        final_metrics = compute_all_metrics(
            y_true=test_results['y_true'],
            y_pred=test_results['y_pred'],
            y_pred_probs=test_results['y_pred_probs'],
            model=model,
            train_loss=trainer.last_train_loss,
            test_loss=test_results['test_loss'],
            train_acc=trainer.last_train_acc,
            test_acc=test_results['test_acc'],
            num_classes=num_classes,
            prediction_target=config.PREDICTION_TARGET,
            long_pnl_R=long_pnl,
            short_pnl_R=short_pnl,
        )

        # Compute objectives
        gap_loss = final_metrics['gap_loss']
        edge_total = final_metrics.get('edge_total_R', 0.0)
        edge_per = final_metrics.get('edge_per_trade', 0.0)

        # NaN protection
        values = [edge_total, gap_loss, edge_per]
        if any(np.isnan(v) or np.isinf(v) for v in values):
            logger.warning(f"Trial {trial.number} - NaN/Inf detected! Returning (0.0, 999.0, 0.0)")
            return 0.0, 999.0, 0.0

        logger.info(f"Trial {trial.number} Results: EdgeTotal={edge_total:.4f}, Gap={gap_loss:.4f}, EdgePer={edge_per:.4f}")

        return edge_total, gap_loss, edge_per

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        return 0.0, 999.0, 0.0

    finally:
        restore_config(saved_config)


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_pareto_front(study):
    """Print Pareto front with searched params."""
    print(f"\n{'=' * 80}")
    print("PARETO FRONT (Best Trade-offs)")
    print(f"{'=' * 80}")

    if not study.best_trials:
        print("  No completed trials.")
        return

    for trial in study.best_trials[:10]:
        edge_total = trial.values[0]
        gap = trial.values[1]
        edge_per = trial.values[2]
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        print(f"  Trial {trial.number:<4} | EdgeTotal={edge_total:.4f} | Gap={gap:.4f} | EdgePer={edge_per:.4f} | {params_str}")


def save_results(study, study_name):
    """Save optimization results to text files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    study_dir = os.path.join('optimize', study_name)

    # Summary
    results_file = os.path.join(study_dir, f"results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Optuna Multi-Objective Optimization Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Pareto-optimal: {len(study.best_trials)}\n\n")
        f.write(f"Search space: {config.OPTUNA_SEARCH_SPACE}\n\n")

        f.write("Pareto Front:\n")
        f.write(f"{'-' * 80}\n")
        for trial in study.best_trials:
            edge_total = trial.values[0]
            gap = trial.values[1]
            edge_per = trial.values[2]
            params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
            f.write(f"  Trial {trial.number:<4} | EdgeTotal={edge_total:.4f} | Gap={gap:.4f} | EdgePer={edge_per:.4f} | {params_str}\n")

    print(f"✓ Results: {results_file}")

    # Detailed report
    detailed_file = os.path.join(study_dir, f"detailed_report_{timestamp}.txt")
    with open(detailed_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("OPTUNA OPTIMIZATION - DETAILED REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"Total: {len(study.trials)} | ")
        f.write(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} | ")
        f.write(f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
        f.write(f"Pareto-optimal: {len(study.best_trials)}\n")
        f.write(f"Search space: {config.OPTUNA_SEARCH_SPACE}\n\n")

        for trial in study.trials:
            f.write("=" * 100 + "\n")
            f.write(f"TRIAL {trial.number} — {trial.state.name}\n")
            f.write("=" * 100 + "\n")

            if trial.datetime_start and trial.datetime_complete:
                duration = trial.datetime_complete - trial.datetime_start
                f.write(f"  Duration: {duration}\n")

            if trial.params:
                f.write("  Params: " + ", ".join(f"{k}={v}" for k, v in trial.params.items()) + "\n")

            if trial.values:
                f.write(f"  EdgeTotal: {trial.values[0]:.6f} | Gap: {trial.values[1]:.6f} | EdgePer: {trial.values[2]:.6f}\n")

            trial_name = f"optuna_{trial.number:03d}"
            f.write(f"  Files: {config.CHECKPOINT_DIR}/{trial_name}/ | {config.TENSORBOARD_DIR}/{trial_name}/\n\n")

    print(f"✓ Detailed report: {detailed_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main optimization loop with Optuna."""
    parser = argparse.ArgumentParser(
        description="Multi-Objective Hyperparameter Optimization with Optuna"
    )
    parser.add_argument('--trials', type=int, default=50, help='Number of trials (default: 50)')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--study-name', type=str, default='mamba_multiobjective', help='Study name')
    args = parser.parse_args()

    study_dir = os.path.join('optimize', args.study_name)
    os.makedirs(study_dir, exist_ok=True)
    storage = f'sqlite:///{study_dir}/study.db'

    if args.resume:
        print(f"\n{'=' * 80}")
        print(f"RESUMING STUDY: {args.study_name}")
        print(f"{'=' * 80}")
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print(f"Already completed: {len(study.trials)} trials")
    else:
        print(f"\n{'=' * 80}")
        print(f"CREATING NEW STUDY: {args.study_name}")
        print(f"{'=' * 80}")
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            directions=['maximize', 'minimize', 'maximize'],
            load_if_exists=False,
        )

    # Show search space
    search_space = config.OPTUNA_SEARCH_SPACE
    print(f"\nSearch space ({len(search_space)} params):")
    if search_space:
        for name, spec in search_space.items():
            current = getattr(config, name.upper(), '?')
            print(f"  {name}: {spec}  (config default: {current})")
    else:
        print("  (empty — all trials use config.py defaults)")
    print(f"\nTrials: {args.trials} | Epochs/trial: {config.EPOCHS}")
    print(f"Objectives: maximize edge_total_R, minimize gap_loss, maximize edge_per_trade")
    print(f"{'=' * 80}\n")

    study.optimize(objective, n_trials=args.trials)

    print(f"\n{'=' * 80}")
    print(f"OPTIMIZATION COMPLETE — {len(study.trials)} trials")
    print(f"{'=' * 80}")

    print_pareto_front(study)
    save_results(study, args.study_name)


if __name__ == "__main__":
    main()
