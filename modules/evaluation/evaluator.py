"""
Model Evaluator - Load checkpoint, run predictions, compute metrics
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.training.setup import resolve_num_workers

import config
from modules.data.loader import get_raw_data
from modules.data.transform import apply_transform_from_checkpoint
from modules.data.labeling import get_labels, get_labels_next_close, parse_close_horizon, remap_labels, TARGET_CLASS_NAMES, TRADE_DIRECTION
from modules.data.dataset import TimeSeriesWindowDataset, MultiTFWindowDataset
from modules.model.mamba import MambaPredictor, MultiTFMambaPredictor, ARCH_KEYS
from modules.training.trainer import get_device
from modules.training.metrics import compute_all_metrics, compute_confusion_matrix
from modules.data.multi_tf import compute_branch_vp, build_secondary_branches
from modules.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained model on a dataset.

    Responsibilities:
    - Load checkpoint and extract all information
    - Prepare data using checkpoint parameters
    - Run predictions
    - Compute metrics
    - Save results (CSV, JSON, TXT)
    """

    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = get_device(device)
        self.checkpoint = None
        self.model = None
        self.config_params = None
        self._long_pnl_R = None
        self._short_pnl_R = None
        self._dataset_long_pnl = None
        self._dataset_short_pnl = None

        logger.info(f"Initializing ModelEvaluator with checkpoint: {checkpoint_path}")
        logger.info(f"Device: {self.device}")

    def load_checkpoint(self):
        """
        Load checkpoint and extract all information.

        Returns:
            dict: Checkpoint contents
        """
        logger.info("=" * 80)
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        logger.info("=" * 80)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Extract config
        if 'config' not in self.checkpoint:
            raise ValueError("Checkpoint missing 'config' key")

        self.config_params = self.checkpoint['config']
        self.prediction_target = self.config_params['PREDICTION_TARGET']
        self.num_classes = 3 if self.prediction_target == 'triple' else 2

        # Verify required keys
        required_keys = ['normalization_params', 'WINDOW_SIZE']
        missing_keys = [key for key in required_keys if key not in self.config_params]
        if missing_keys:
            raise ValueError(f"Checkpoint config missing keys: {missing_keys}")

        # Create model from checkpoint architecture
        logger.info("Creating model from checkpoint architecture...")

        if 'model_state_dict' not in self.checkpoint:
            raise ValueError("Checkpoint missing 'model_state_dict' key")

        # Build arch_params from checkpoint
        arch_params = {k: self.config_params[k] for k in ARCH_KEYS}

        # Detect multi-timeframe mode from checkpoint config
        mtf_mode = self.config_params['MULTI_TF_MODE']
        state_dict = self.checkpoint['model_state_dict']

        if mtf_mode == "dual":
            # Dual model: detect input dims from encoder state dicts
            input_dim_primary = state_dict['encoder_primary.linear_proj.weight'].shape[1]
            input_dim_secondary = state_dict['encoder_secondary.linear_proj.weight'].shape[1]

            self.model = MultiTFMambaPredictor(
                input_dim_primary=input_dim_primary,
                input_dim_secondary=input_dim_secondary,
                d_model=self.config_params['D_MODEL'],
                d_state_primary=self.config_params['D_STATE'],
                d_state_secondary=self.config_params['MULTI_TF_D_STATE'],
                d_conv=self.config_params['D_CONV'],
                expand=self.config_params['EXPAND'],
                n_layers_primary=self.config_params['N_LAYERS'],
                n_layers_secondary=self.config_params['MULTI_TF_N_LAYERS'],
                arch_params=arch_params,
                num_classes=self.num_classes,
            )
            model_arch = f"MultiTF dual: primary({input_dim_primary}feat), secondary({input_dim_secondary}feat)"
            logger.info(f"  {model_arch}")
        else:
            # Standard or calibration model
            input_dim = state_dict['linear_proj.weight'].shape[1]

            model_arch = {
                'input_dim': input_dim,
                'd_model': self.config_params['D_MODEL'],
                'd_state': self.config_params['D_STATE'],
                'd_conv': self.config_params['D_CONV'],
                'expand': self.config_params['EXPAND'],
                'n_layers': self.config_params['N_LAYERS'],
                'arch_params': arch_params,
                'num_classes': self.num_classes,
            }

            # In calibration mode, use MTF-specific params
            if mtf_mode == "calibration":
                model_arch['d_state'] = self.config_params['MULTI_TF_D_STATE']
                model_arch['n_layers'] = self.config_params['MULTI_TF_N_LAYERS']

            self.model = MambaPredictor(**model_arch)

        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Mixed precision: controlled by config.EVAL_MIXED_PRECISION (independent from training)
        self.use_amp = (self.device.type == "cuda" and config.EVAL_MIXED_PRECISION)

        # Log info
        logger.info(f"✓ Checkpoint loaded successfully")
        logger.info(f"  Epoch: {self.checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Test Accuracy: {self.checkpoint.get('test_acc', 'N/A'):.4f}" if self.checkpoint.get('test_acc') else "  Test Accuracy: N/A")
        logger.info(f"  Model: {model_arch}")
        logger.info(f"  Window Size: {self.config_params['WINDOW_SIZE']}")

        return self.checkpoint

    def _compute_labels(self, df, window_size, inference_only):
        """Compute labels and P&L arrays from a DataFrame (raw or aggregated)."""
        if inference_only:
            self._long_pnl_R = None
            self._short_pnl_R = None
            return None

        close_horizon = parse_close_horizon(self.prediction_target)
        if close_horizon is not None:
            labels, _, _, long_pnl_R, short_pnl_R = get_labels_next_close(
                df, horizon=close_horizon,
                window_size=window_size, atr_period=self.config_params['ATR_PERIOD'])
        else:
            labels, _, _, long_pnl_R, short_pnl_R = get_labels(
                df,
                window_size=window_size,
                atr_multiplier_tp=self.config_params['ATR_MULTIPLIER_TP'],
                atr_multiplier_sl=self.config_params['ATR_MULTIPLIER_SL'],
                max_horizon=self.config_params['MAX_HORIZON'],
                atr_period=self.config_params['ATR_PERIOD']
            )
            labels, _ = remap_labels(labels, self.prediction_target)
        self._long_pnl_R = long_pnl_R
        self._short_pnl_R = short_pnl_R
        return labels

    @staticmethod
    def _extract_raw_hlc(df, window_size, n_data):
        """Extract raw H/L/C array for per-window zigzag computation."""
        return np.column_stack([
            df['High'].values[window_size:window_size + n_data],
            df['Low'].values[window_size:window_size + n_data],
            df['Close'].values[window_size:window_size + n_data]
        ]).astype(np.float64)

    def prepare_data(self, data_path: str, split: str = 'test', inference_only: bool = False):
        """
        Prepare data for evaluation.

        Handles standard, calibration, and dual multi-timeframe modes.

        Args:
            data_path: Path to CSV data file
            split: 'train', 'test', or 'all'
            inference_only: If True, skip label computation (no metrics)

        Returns:
            tuple: (dataset, labels, timestamps, split_info)
        """
        logger.info("=" * 80)
        logger.info(f"Preparing data from: {data_path}")
        logger.info(f"Split: {split}, Inference only: {inference_only}")
        logger.info("=" * 80)

        mtf_mode = self.config_params['MULTI_TF_MODE']

        # Load raw data (cached as attribute for reuse in evaluate/save)
        self._df_raw = get_raw_data(data_path)
        df_raw = self._df_raw
        logger.info(f"✓ Loaded {len(df_raw)} rows")

        window_size = self.config_params['WINDOW_SIZE']
        train_test_split = self.config_params['TRAIN_TEST_SPLIT']
        zigzag_length = self.config_params['ZIGZAG_LENGTH']

        # ====== CALIBRATION MODE ======
        if mtf_mode == "calibration":
            from modules.data.aggregation import aggregate_candles
            divisor = self.config_params['MULTI_TF_DIVISOR']
            align = self.config_params['MULTI_TF_ALIGN']

            df_agg, _, _ = aggregate_candles(df_raw, divisor, align)
            logger.info(f"✓ Aggregated: {len(df_raw)} → {len(df_agg)} candles")

            labels_sec = self._compute_labels(df_agg, window_size, inference_only)

            transformed_data = apply_transform_from_checkpoint(df_agg, self.config_params)
            # Strip time features (layout: [n_norm, 4 time, 2 regime])
            n_norm = transformed_data.shape[1] - 6  # 6 = 4 time + 2 regime
            data_sec = np.delete(transformed_data, np.s_[n_norm:n_norm+4], axis=1)

            n_data_sec = len(data_sec)
            raw_hlc_sec = self._extract_raw_hlc(df_agg, window_size, n_data_sec)

            if not inference_only:
                # Truncate data to labels length (same alignment as main.py)
                data_aligned = data_sec[:len(labels_sec)]
                labels_aligned = labels_sec[:len(data_aligned)]
                raw_hlc_sec = raw_hlc_sec[:len(data_aligned)]
            else:
                data_aligned = data_sec
                labels_aligned = None

            # VP for calibration mode (precomputed)
            vp_features_cal = compute_branch_vp(df_agg, self.config_params, window_size, len(data_aligned))

            return self._build_dataset_and_info(
                data_aligned, labels_aligned, df_agg, window_size, train_test_split,
                split, inference_only, n_norm + 1, raw_hlc=raw_hlc_sec,
                zigzag_length=zigzag_length, vp_features=vp_features_cal
            )

        # ====== DUAL MODE ======
        if mtf_mode == "dual":
            labels = self._compute_labels(df_raw, window_size, inference_only)

            # Primary transform
            transformed_pri = apply_transform_from_checkpoint(df_raw, self.config_params)

            # Align to labels length (same truncation as main.py)
            if not inference_only:
                data_pri = transformed_pri[:len(labels)]
                labels_aligned = labels[:len(data_pri)]
            else:
                data_pri = transformed_pri
                labels_aligned = np.zeros(len(data_pri), dtype=np.int64)

            ws = window_size
            n_data_pri = len(data_pri)
            n_valid = n_data_pri

            raw_hlc_pri = self._extract_raw_hlc(df_raw, window_size, n_data_pri)
            split_index = int(n_valid * train_test_split)
            n_norm_pri = data_pri.shape[1] - 6  # 6 = 4 time + 2 regime

            # Secondary branches (N-offset automatically handled)
            def sec_transform_fn(df_agg):
                return apply_transform_from_checkpoint(df_agg, self.config_params)

            sec = build_secondary_branches(df_raw, sec_transform_fn, self.config_params, n_data_pri)
            n_norm_sec = sec['n_norm_secondary']
            min_pri_start = sec['min_primary_start']

            logger.info(f"  [MTF dual] {self.config_params['MULTI_TF_ALIGN']} mode, min_pri_start={min_pri_start}")

            # Primary VP
            vp_features_pri = compute_branch_vp(df_raw, self.config_params, ws, len(data_pri))

            if split == 'train':
                indices = list(range(max(min_pri_start, 0), split_index - ws))
            elif split == 'test':
                indices = list(range(max(split_index - ws, min_pri_start), n_valid - ws))
            elif split == 'all':
                indices = list(range(min_pri_start, n_valid - ws))
            else:
                raise ValueError(f"Invalid split: {split}")

            dataset = MultiTFWindowDataset(
                data_primary=data_pri, data_secondary=sec['data_secondary_list'], labels=labels_aligned,
                indices_primary=indices, mapping_primary_to_secondary=sec['mapping_list'],
                window_size=ws, raw_hlc_primary=raw_hlc_pri, raw_hlc_secondary=sec['raw_hlc_secondary_list'],
                vol_secondary=sec['vol_secondary_list'],
                n_norm_features_primary=n_norm_pri + 1, n_norm_features_secondary=n_norm_sec + 1,
                partial_context=sec['partial_ctx_list'],
                zigzag_length=zigzag_length,
                vp_features_primary=vp_features_pri, vp_features_secondary=sec['vp_secondary_list'],
                vp_params=self.config_params,
                cross_tf_normalize=self.config_params['CROSS_TF_NORMALIZE'],
            )

            if not inference_only:
                label_indices = np.array(indices) + ws - 1
                dataset_labels = labels_aligned[label_indices]
                # Align P&L arrays for edge metrics
                if self._long_pnl_R is not None:
                    pnl_aligned_long = self._long_pnl_R[:len(data_pri)]
                    self._dataset_long_pnl = pnl_aligned_long[label_indices]
                    pnl_aligned_short = self._short_pnl_R[:len(data_pri)]
                    self._dataset_short_pnl = pnl_aligned_short[label_indices]
                else:
                    self._dataset_long_pnl = None
                    self._dataset_short_pnl = None
            else:
                dataset_labels = None
                self._dataset_long_pnl = None
                self._dataset_short_pnl = None

            raw_start_idx = ws + indices[0] + ws - 1
            timestamps = df_raw['Open time'].iloc[raw_start_idx:raw_start_idx + len(indices)].values

            split_info = {
                'split': split,
                'total_samples': len(dataset),
                'train_split_index': split_index,
                'window_size': ws,
                'nan_offset': 0,
                'raw_start_idx': raw_start_idx,
            }

            return dataset, dataset_labels, timestamps, split_info

        # ====== STANDARD MODE (off) ======
        labels = self._compute_labels(df_raw, window_size, inference_only)
        if labels is not None:
            logger.info(f"✓ Computed {len(labels)} labels (target={self.prediction_target})")
        else:
            logger.info("Skipping label computation (inference_only=True)")

        # Transform data using checkpoint parameters
        logger.info("Transforming data with checkpoint parameters...")
        transformed_data = apply_transform_from_checkpoint(
            df_raw,
            self.config_params
        )
        logger.info(f"✓ Transformed data: {transformed_data.shape}")

        n_data = len(transformed_data)
        raw_hlc = self._extract_raw_hlc(df_raw, window_size, n_data)

        # Truncate data to labels length (same alignment as main.py)
        if not inference_only:
            data_aligned = transformed_data[:len(labels)]
            labels_aligned = labels[:len(data_aligned)]
            raw_hlc = raw_hlc[:len(data_aligned)]
        else:
            data_aligned = transformed_data
            labels_aligned = None

        total_features = data_aligned.shape[1]
        n_norm_features = total_features - 6  # 6 = 4 time + 2 regime

        # VP features (precomputed)
        vp_features_std = compute_branch_vp(df_raw, self.config_params, window_size, len(data_aligned))

        return self._build_dataset_and_info(
            data_aligned, labels_aligned, df_raw, window_size, train_test_split,
            split, inference_only, n_norm_features + 1, raw_hlc=raw_hlc,
            zigzag_length=zigzag_length, vp_features=vp_features_std
        )

    def _build_dataset_and_info(self, data_aligned, labels_aligned, df_source,
                                window_size, train_test_split, split, inference_only,
                                n_norm_features, raw_hlc=None, zigzag_length=None,
                                vp_features=None):
        """Build standard TimeSeriesWindowDataset and split info."""
        n_valid = len(data_aligned)
        split_index = int(n_valid * train_test_split)

        if split == 'train':
            indices = list(range(split_index - window_size))
        elif split == 'test':
            indices = list(range(split_index - window_size, n_valid - window_size))
        elif split == 'all':
            indices = list(range(n_valid - window_size))
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'all'")

        logger.info(f"Split '{split}' indices: [{indices[0]}, ..., {indices[-1]}] (count: {len(indices)})")

        if not inference_only and labels_aligned is not None:
            dataset = TimeSeriesWindowDataset(
                data=data_aligned, labels=labels_aligned, indices=indices,
                window_size=window_size, raw_hlc=raw_hlc,
                n_norm_features=n_norm_features,
                zigzag_length=zigzag_length, vp_features=vp_features,
            )
        else:
            dummy_labels = np.zeros(len(data_aligned), dtype=np.int64)
            dataset = TimeSeriesWindowDataset(
                data=data_aligned, labels=dummy_labels, indices=indices,
                window_size=window_size, raw_hlc=raw_hlc,
                n_norm_features=n_norm_features,
                zigzag_length=zigzag_length, vp_features=vp_features,
            )

        logger.info(f"✓ Dataset created: {len(dataset)} samples")

        if not inference_only and labels_aligned is not None:
            label_indices = np.array(indices) + window_size - 1
            dataset_labels = labels_aligned[label_indices]
            logger.info(f"✓ Extracted {len(dataset_labels)} labels for dataset (end of window)")
            # Align P&L arrays for edge metrics
            if self._long_pnl_R is not None:
                pnl_aligned_long = self._long_pnl_R[:len(data_aligned)]
                self._dataset_long_pnl = pnl_aligned_long[label_indices]
                pnl_aligned_short = self._short_pnl_R[:len(data_aligned)]
                self._dataset_short_pnl = pnl_aligned_short[label_indices]
            else:
                self._dataset_long_pnl = None
                self._dataset_short_pnl = None
        else:
            dataset_labels = None
            self._dataset_long_pnl = None
            self._dataset_short_pnl = None

        raw_start_idx = window_size + indices[0] + window_size - 1
        timestamps = df_source['Open time'].iloc[raw_start_idx:raw_start_idx + len(indices)].values

        logger.info(f"✓ First predictable timestamp: {timestamps[0]} (end of first window)")

        split_info = {
            'split': split,
            'total_samples': len(dataset),
            'train_split_index': split_index,
            'window_size': window_size,
            'nan_offset': 0,
            'raw_start_idx': raw_start_idx
        }

        return dataset, dataset_labels, timestamps, split_info

    def predict(self, dataset):
        """
        Run predictions on dataset.

        Args:
            dataset: TimeSeriesWindowDataset

        Returns:
            tuple: (predictions, probabilities)
        """
        logger.info("=" * 80)
        logger.info("Running predictions...")
        logger.info("=" * 80)

        # Create dataloader
        batch_size = self.config_params['EVAL_BATCH_SIZE']
        num_workers = resolve_num_workers(self.config_params['NUM_WORKERS'])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=num_workers > 0
        )

        all_preds = []
        all_probs = []

        is_dual = hasattr(self.model, 'encoder_primary')

        with torch.no_grad():
            for batch_data in dataloader:
                # Detect mono (2 elements) vs dual (4+ elements: pri, sec, tf_map, label, [conf])
                if is_dual:
                    windows_pri = batch_data[0].to(self.device)
                    windows_sec = batch_data[1].to(self.device)
                    tf_mapping = batch_data[2].to(self.device)
                    with autocast(self.device.type, enabled=self.use_amp):
                        logits = self.model(windows_pri, windows_sec, tf_mapping)
                else:
                    windows = batch_data[0].to(self.device)
                    with autocast(self.device.type, enabled=self.use_amp):
                        logits = self.model(windows)

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)

        logger.info(f"✓ Predictions complete: {len(predictions)} samples")
        logger.info(f"  Prediction distribution: {np.bincount(predictions)}")

        return predictions, probabilities

    def compute_metrics(self, y_true, y_pred, y_probs):
        """
        Compute all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities

        Returns:
            dict: All metrics
        """
        logger.info("=" * 80)
        logger.info("Computing metrics...")
        logger.info("=" * 80)

        metrics = compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_probs=y_probs,
            model=self.model,
            num_classes=self.num_classes,
            prediction_target=self.prediction_target,
            long_pnl_R=self._dataset_long_pnl,
            short_pnl_R=self._dataset_short_pnl,
        )

        # Add confusion matrix
        metrics['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        logger.info("✓ Metrics computed")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        logger.info(f"  F1 Macro: {metrics.get('f1_macro', 0):.4f}")

        return metrics

    def save_results(self, output_dir: str, predictions, probabilities, timestamps,
                     labels=None, metrics=None, split_info=None, df_raw=None):
        """
        Save evaluation results to disk.

        Args:
            output_dir: Output directory path
            predictions: Predicted labels
            probabilities: Prediction probabilities
            timestamps: Timestamps for each prediction
            labels: True labels (None if inference_only)
            metrics: Metrics dict (None if inference_only)
            split_info: Split information dict
            df_raw: Raw dataframe (for OHLC in CSV)
        """
        logger.info("=" * 80)
        logger.info(f"Saving results to: {output_dir}")
        logger.info("=" * 80)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save predictions.csv
        logger.info("Saving predictions.csv...")

        class_names_csv = TARGET_CLASS_NAMES[self.prediction_target]
        predictions_data = {
            'timestamp': timestamps,
            'label_pred': predictions,
        }
        for i in range(self.num_classes):
            predictions_data[f'prob_{class_names_csv[i].lower()}'] = probabilities[:, i]

        # Add true labels if available
        if labels is not None:
            # Labels should already correspond to predictions (same length)
            # They were extracted in evaluate() from the dataset indices
            predictions_data['label_true'] = labels
            predictions_data['correct'] = (labels == predictions)

        # Add OHLC if df_raw provided
        if df_raw is not None and split_info is not None and 'raw_start_idx' in split_info:
            first_idx = split_info['raw_start_idx']
            ohlc_data = df_raw.iloc[first_idx:first_idx + len(predictions)]
            predictions_data['open'] = ohlc_data['Open'].values
            predictions_data['high'] = ohlc_data['High'].values
            predictions_data['low'] = ohlc_data['Low'].values
            predictions_data['close'] = ohlc_data['Close'].values

        df_predictions = pd.DataFrame(predictions_data)
        csv_path = output_path / 'predictions.csv'
        df_predictions.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved: {csv_path}")

        # Save metrics.json (if available)
        if metrics is not None:
            logger.info("Saving metrics.json...")
            metrics_path = output_path / 'metrics.json'

            # Convert numpy types to Python types for JSON serialization
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_serializable[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                else:
                    metrics_serializable[key] = value

            with open(metrics_path, 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            logger.info(f"✓ Saved: {metrics_path}")

        # Save summary.txt (if metrics available)
        if metrics is not None and labels is not None:
            logger.info("Saving summary.txt...")
            summary_path = output_path / 'summary.txt'

            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Checkpoint: {self.checkpoint_path}\n")
                f.write(f"Dataset: {split_info.get('data_path', 'N/A')}\n")
                f.write(f"Split: {split_info['split']}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("=" * 80 + "\n")
                f.write("DATASET INFO\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total samples: {split_info['total_samples']}\n")
                if labels is not None:
                    label_counts = Counter(labels)
                    f.write(f"Label distribution: {dict(sorted(label_counts.items()))}\n")
                f.write(f"Date range: {timestamps[0]} to {timestamps[-1]}\n\n")

                f.write("=" * 80 + "\n")
                f.write("OVERALL METRICS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Accuracy:           {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}\n")
                f.write(f"F1 Macro:           {metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"Final_Accuracy:     {metrics.get('final_accuracy', 0):.4f} ({int(metrics.get('final_accuracy_count', 0))} preds)\n")
                # Edge metrics: only meaningful when trade directions exist
                trade_dir = TRADE_DIRECTION.get(self.prediction_target, {})
                if trade_dir:
                    f.write(f"\nEdge (expected R per trade):\n")
                    f.write(f"  edge_per_trade:   {metrics.get('edge_per_trade', 0):.4f}\n")
                    f.write(f"  edge_total_R:     {metrics.get('edge_total_R', 0):.2f}\n")
                f.write(f"\nMargin Confidence (top-N% by margin = top1-top2 proba):\n")
                for p in ("05", "10", "25", "50"):
                    acc = metrics.get(f'confmargin_top{p}_acc', 0)
                    cnt = int(metrics.get(f'confmargin_top{p}_count', 0))
                    f.write(f"  top {int(p)}%: acc={acc:.4f} ({cnt} preds)\n")
                f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("PER-CLASS METRICS\n")
                f.write("=" * 80 + "\n")

                class_names_map = TARGET_CLASS_NAMES[self.prediction_target]
                class_names = [class_names_map[i] for i in range(self.num_classes)]
                for i, name in enumerate(class_names):
                    f.write(f"Class {i} ({name}):\n")
                    f.write(f"  Precision:  {metrics.get(f'precision_class_{i}', 0):.4f}\n")
                    f.write(f"  Recall:     {metrics.get(f'recall_class_{i}', 0):.4f}\n")
                    f.write(f"  F1-score:   {metrics.get(f'f1_class_{i}', 0):.4f}\n\n")

                f.write("=" * 80 + "\n")

            logger.info(f"✓ Saved: {summary_path}")

        logger.info("=" * 80)
        logger.info("Results saved successfully")
        logger.info("=" * 80)

    def evaluate(self, data_path: str, split: str = 'test', output_dir: str = None,
                 inference_only: bool = False):
        """
        Complete evaluation pipeline.

        Args:
            data_path: Path to CSV data file
            split: 'train', 'test', or 'all'
            output_dir: Output directory (auto-generated if None)
            inference_only: If True, skip metrics (no labels needed)

        Returns:
            dict: Evaluation results
        """
        # Load checkpoint
        self.load_checkpoint()

        # Prepare data
        dataset, labels, timestamps, split_info = self.prepare_data(
            data_path, split, inference_only
        )
        split_info['data_path'] = data_path

        # Run predictions
        predictions, probabilities = self.predict(dataset)

        # Compute metrics (if not inference_only)
        if not inference_only:
            metrics = self.compute_metrics(labels, predictions, probabilities)
        else:
            metrics = None
            logger.info("Skipping metrics computation (inference_only=True)")

        # Generate output directory name if not provided
        if output_dir is None:
            checkpoint_name = Path(self.checkpoint_path).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"{config.EVAL_OUTPUT_DIR}/{checkpoint_name}_eval_{timestamp}"

        # Reuse raw data loaded in prepare_data
        df_raw = self._df_raw

        # Save results
        self.save_results(
            output_dir=output_dir,
            predictions=predictions,
            probabilities=probabilities,
            timestamps=timestamps,
            labels=labels,
            metrics=metrics,
            split_info=split_info,
            df_raw=df_raw
        )

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'timestamps': timestamps,
            'labels': labels,
            'metrics': metrics,
            'split_info': split_info,
            'output_dir': output_dir
        }
