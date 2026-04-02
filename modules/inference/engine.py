"""
Prediction engine for live inference.

Uses Dataset.__getitem__() for feature assembly and normalization,
ensuring exact parity with training and evaluation pipelines.
Data preparation (transform, aggregate, VP, mapping) mirrors evaluator.py.
Supports all 3 modes: off (mono), calibration, dual.
"""

import hashlib
import os

import numpy as np
import torch
from torch.amp import autocast

import config
from modules.data.transform import apply_transform_from_checkpoint
from modules.data.labeling import TARGET_CLASS_NAMES
from modules.data.dataset import TimeSeriesWindowDataset, MultiTFWindowDataset
from modules.data.multi_tf import build_secondary_branches, compute_branch_vp
from modules.training.trainer import get_device
from modules.model.mamba import MambaPredictor, MultiTFMambaPredictor, ARCH_KEYS
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint using arch_params (no global config mutation).
    Supports all 3 modes: off, calibration, dual (mirrors evaluator.py load_checkpoint).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # weights_only=False required: checkpoint contains config dict, metrics, scalars (not just tensors)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint missing 'config' key")
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict' key")

    config_params = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    target = config_params.get('PREDICTION_TARGET', 'triple')
    num_classes = 3 if target == 'triple' else 2
    mtf_mode = config_params['MULTI_TF_MODE']

    # Build arch_params from checkpoint
    arch_params = {k: config_params[k] for k in ARCH_KEYS}

    if mtf_mode == "dual":
        # Dual model: detect input dims from encoder state dicts
        input_dim_pri = state_dict['encoder_primary.linear_proj.weight'].shape[1]
        input_dim_sec = state_dict['encoder_secondary.linear_proj.weight'].shape[1]

        model = MultiTFMambaPredictor(
            input_dim_primary=input_dim_pri,
            input_dim_secondary=input_dim_sec,
            d_model=config_params['D_MODEL'],
            d_state_primary=config_params['D_STATE'],
            d_state_secondary=config_params['MULTI_TF_D_STATE'],
            d_conv=config_params['D_CONV'],
            expand=config_params['EXPAND'],
            n_layers_primary=config_params['N_LAYERS'],
            n_layers_secondary=config_params['MULTI_TF_N_LAYERS'],
            arch_params=arch_params,
            num_classes=num_classes,
        )
        logger.info(f"  Mode: dual, primary({input_dim_pri}feat), secondary({input_dim_sec}feat)")
    else:
        # Standard (off) or calibration model
        input_dim = state_dict['linear_proj.weight'].shape[1]

        model_kwargs = {
            'input_dim': input_dim,
            'd_model': config_params['D_MODEL'],
            'd_state': config_params['D_STATE'],
            'd_conv': config_params['D_CONV'],
            'expand': config_params['EXPAND'],
            'n_layers': config_params['N_LAYERS'],
            'arch_params': arch_params,
            'num_classes': num_classes,
        }

        # In calibration mode, use MTF-specific params
        if mtf_mode == "calibration":
            model_kwargs['d_state'] = config_params['MULTI_TF_D_STATE']
            model_kwargs['n_layers'] = config_params['MULTI_TF_N_LAYERS']

        model = MambaPredictor(**model_kwargs)
        logger.info(f"  Mode: {mtf_mode}, input({input_dim}feat)")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', '?')}, D_MODEL={config_params.get('D_MODEL')}")
    logger.info(f"  Attention: {config_params.get('ATTENTION_POSITION')}, heads={config_params.get('ATTENTION_NUM_HEADS')}")

    return model, config_params


def _build_raw_hlc(df, ws, n_data):
    """Extract raw H/L/C array for per-window zigzag computation."""
    return np.column_stack([
        df['High'].values[ws:ws + n_data],
        df['Low'].values[ws:ws + n_data],
        df['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)


class PredictionEngine:
    """
    Wraps model loading and inference into a reusable engine.

    predict() builds a Dataset with a single index and calls
    dataset[0] to get the feature tensors — same code path as training
    and evaluation. No manual assembly or normalization.
    Supports all 3 modes: off (mono), calibration, dual.
    """

    def __init__(self, checkpoint_path, device='auto'):
        self.device = get_device(device)
        self.model, self.config_params = load_model(checkpoint_path, self.device)
        target = self.config_params.get('PREDICTION_TARGET', 'triple')
        self.num_classes = 3 if target == 'triple' else 2
        self.class_names = TARGET_CLASS_NAMES[target]
        self.prediction_target = target
        self.mtf_mode = self.config_params['MULTI_TF_MODE']
        self.is_dual = hasattr(self.model, 'encoder_primary')
        self.use_amp = (self.device.type == "cuda" and config.LIVE_MIXED_PRECISION)
        logger.info(f"PredictionEngine ready on {self.device} (AMP={'on' if self.use_amp else 'off'})")

    def predict(self, df_buffer):
        """
        Run a SINGLE isolated inference on the raw data buffer.

        Data preparation mirrors evaluator.py prepare_data().
        Feature assembly + normalization delegated to Dataset.__getitem__().

        Args:
            df_buffer: DataFrame with columns [Open time, Open, High, Low, Close, Volume]

        Returns:
            dict with keys: pred, confidence, probs, class_name
            or None if not enough data.
        """
        if self.is_dual:
            return self._predict_dual(df_buffer)
        elif self.mtf_mode == "calibration":
            return self._predict_calibration(df_buffer)
        else:
            return self._predict_standard(df_buffer)

    def _predict_dual(self, df_buffer):
        """Dual mode: primary + secondary branches via MultiTFWindowDataset."""
        ws = self.config_params['WINDOW_SIZE']
        zigzag_length = self.config_params['ZIGZAG_LENGTH']

        # === PRIMARY BRANCH ===
        transformed_pri = apply_transform_from_checkpoint(df_buffer, self.config_params)
        if len(transformed_pri) < ws:
            logger.warning(f"Not enough primary data: {len(transformed_pri)} < {ws}")
            return None

        n_data_pri = len(transformed_pri)

        # Raw H/L/C for primary per-window zigzag
        raw_hlc_pri = _build_raw_hlc(df_buffer, ws, n_data_pri)

        # === SECONDARY BRANCHES (N-offset automatically handled) ===
        def sec_transform_fn(df_agg):
            return apply_transform_from_checkpoint(df_agg, self.config_params)

        sec = build_secondary_branches(df_buffer, sec_transform_fn, self.config_params, n_data_pri)

        # Check secondary data sufficiency
        if len(sec['data_secondary_list'][0]) < ws:
            logger.warning(f"Not enough secondary data: {len(sec['data_secondary_list'][0])} < {ws}")
            return None

        # === VOLUME PROFILE ===
        vp_features_pri = compute_branch_vp(df_buffer, self.config_params, ws, n_data_pri)

        # === BUILD DATASET WITH SINGLE INDEX (last valid window) ===
        n_norm_pri = transformed_pri.shape[1] - 6  # 6 = 4 time + 2 regime
        last_index = n_data_pri - ws
        dummy_labels = np.zeros(n_data_pri, dtype=np.int64)

        dataset = MultiTFWindowDataset(
            data_primary=transformed_pri,
            data_secondary=sec['data_secondary_list'],
            labels=dummy_labels,
            indices_primary=[last_index],
            mapping_primary_to_secondary=sec['mapping_list'],
            window_size=ws,
            raw_hlc_primary=raw_hlc_pri,
            raw_hlc_secondary=sec['raw_hlc_secondary_list'],
            vol_secondary=sec['vol_secondary_list'],
            n_norm_features_primary=n_norm_pri + 1,
            n_norm_features_secondary=sec['n_norm_secondary'] + 1,
            partial_context=sec['partial_ctx_list'],
            zigzag_length=zigzag_length,
            vp_features_primary=vp_features_pri,
            vp_features_secondary=sec['vp_secondary_list'],
            vp_params=self.config_params,
            cross_tf_normalize=self.config_params['CROSS_TF_NORMALIZE'],
        )

        # === GET FEATURE TENSORS FROM DATASET (same path as training/eval) ===
        window_pri, window_sec, tf_map, _ = dataset[0]

        # === DIAGNOSTIC: log tensor fingerprint (DEBUG only) ===
        if logger.isEnabledFor(10):  # logging.DEBUG
            pri_bytes = window_pri.numpy().tobytes()
            sec_bytes = window_sec.numpy().tobytes()
            fingerprint = hashlib.md5(pri_bytes + sec_bytes).hexdigest()[:12]
            logger.debug(f"predict fingerprint: {fingerprint} "
                         f"pri[0:3]={window_pri[0,:3].tolist()} "
                         f"sec[0:3]={window_sec[0,:3].tolist()}")

        # === MODEL INFERENCE ===
        return self._run_inference_dual(window_pri, window_sec, tf_map)

    def _predict_calibration(self, df_buffer):
        """Calibration mode: aggregate → transform → TimeSeriesWindowDataset."""
        from modules.data.aggregation import aggregate_candles

        ws = self.config_params['WINDOW_SIZE']
        zigzag_length = self.config_params['ZIGZAG_LENGTH']
        divisor = self.config_params['MULTI_TF_DIVISOR']
        align = self.config_params['MULTI_TF_ALIGN']

        # Aggregate candles (same as evaluator calibration path)
        df_agg, _, _ = aggregate_candles(df_buffer, divisor, align)

        transformed_data = apply_transform_from_checkpoint(df_agg, self.config_params)
        if len(transformed_data) < ws:
            logger.warning(f"Not enough calibration data: {len(transformed_data)} < {ws}")
            return None

        # Strip time features (layout: [n_norm, 4 time, 2 regime])
        n_norm = transformed_data.shape[1] - 6
        data_cal = np.delete(transformed_data, np.s_[n_norm:n_norm+4], axis=1)
        n_data = len(data_cal)

        # Raw H/L/C for per-window zigzag
        raw_hlc = _build_raw_hlc(df_agg, ws, n_data)

        # VP features
        vp_features = compute_branch_vp(df_agg, self.config_params, ws, n_data)

        # Build dataset with single index (last valid window)
        last_index = n_data - ws
        dummy_labels = np.zeros(n_data, dtype=np.int64)

        dataset = TimeSeriesWindowDataset(
            data=data_cal, labels=dummy_labels, indices=[last_index],
            window_size=ws, raw_hlc=raw_hlc,
            n_norm_features=n_norm + 1,
            zigzag_length=zigzag_length, vp_features=vp_features,
        )

        # GET FEATURE TENSOR (mono: window, label)
        window = dataset[0][0]

        # === MODEL INFERENCE ===
        return self._run_inference(window)

    def _predict_standard(self, df_buffer):
        """Standard (off) mode: transform → TimeSeriesWindowDataset."""
        ws = self.config_params['WINDOW_SIZE']
        zigzag_length = self.config_params['ZIGZAG_LENGTH']

        transformed_data = apply_transform_from_checkpoint(df_buffer, self.config_params)
        if len(transformed_data) < ws:
            logger.warning(f"Not enough data: {len(transformed_data)} < {ws}")
            return None

        n_data = len(transformed_data)
        n_norm = transformed_data.shape[1] - 6  # 6 = 4 time + 2 regime

        # Raw H/L/C for per-window zigzag
        raw_hlc = _build_raw_hlc(df_buffer, ws, n_data)

        # VP features
        vp_features = compute_branch_vp(df_buffer, self.config_params, ws, n_data)

        # Build dataset with single index (last valid window)
        last_index = n_data - ws
        dummy_labels = np.zeros(n_data, dtype=np.int64)

        dataset = TimeSeriesWindowDataset(
            data=transformed_data, labels=dummy_labels, indices=[last_index],
            window_size=ws, raw_hlc=raw_hlc,
            n_norm_features=n_norm + 1,
            zigzag_length=zigzag_length, vp_features=vp_features,
        )

        # GET FEATURE TENSOR (mono: window, label)
        window = dataset[0][0]

        # === DIAGNOSTIC: log tensor fingerprint (DEBUG only) ===
        if logger.isEnabledFor(10):  # logging.DEBUG
            w_bytes = window.numpy().tobytes()
            fingerprint = hashlib.md5(w_bytes).hexdigest()[:12]
            logger.debug(f"predict fingerprint: {fingerprint} "
                         f"w[0:3]={window[0,:3].tolist()}")

        # === MODEL INFERENCE ===
        return self._run_inference(window)

    def _run_inference(self, window):
        """Run model inference on a single mono window tensor, return result dict."""
        with torch.no_grad():
            w = window.unsqueeze(0).to(self.device)
            with autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(w)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return self._build_result(probs)

    def _run_inference_dual(self, window_pri, window_sec, tf_map):
        """Run model inference on dual-branch tensors, return result dict."""
        with torch.no_grad():
            w_pri = window_pri.unsqueeze(0).to(self.device)
            w_sec = window_sec.unsqueeze(0).to(self.device)
            w_map = tf_map.unsqueeze(0).to(self.device)
            with autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(w_pri, w_sec, w_map)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return self._build_result(probs)

    def _build_result(self, probs):
        """Build prediction result dict from probability array."""
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        return {
            'pred': pred,
            'confidence': conf,
            'probs': probs,
            'class_name': self.class_names[pred],
        }
