"""
Data preprocessing pipeline for training and optimization.

Orchestrates the complete data preparation: load → label → transform → align → VP → split → datasets.
Used by main.py (training) and optimize.py (hyperparameter search).
"""

import numpy as np

import config
from modules.data.loader import get_raw_data
from modules.data.transform import transform_pipeline
from modules.data.labeling import get_labels, get_labels_next_close, parse_close_horizon, remap_labels, compute_atr
from modules.data.dataset import TimeSeriesWindowDataset, MultiTFWindowDataset
from modules.data.aggregation import aggregate_candles
from modules.data.multi_tf import compute_branch_vp, build_secondary_branches


def process_single_dataset(csv_path, logger):
    """
    Process a single CSV dataset through the complete preprocessing pipeline.

    Supports 3 multi-timeframe modes:
    - "off": standard primary pipeline (unchanged)
    - "calibration": secondary branch only (test secondary branch alone)
    - "dual": primary + secondary branches with aligned windows

    Args:
        csv_path: Path to CSV file
        logger: Logger instance

    Returns:
        dict with keys:
            - train_dataset, test_dataset
            - label_dist, label_pcts, n_valid, n_train, n_test
            - normalization_params, n_features, n_norm_features
            - multi_tf_mode: current mode
            - n_features_sec: (dual only) feature count for secondary branch
            - n_norm_features_sec: (dual only) norm feature count for secondary
    """
    mtf_mode = config.MULTI_TF_MODE

    # --- DATA LOADING ---
    df_raw = get_raw_data(csv_path)
    logger.debug(f"  Date range: {df_raw['Open time'].min()} to {df_raw['Open time'].max()}")

    # ===================================================================
    # MODE: CALIBRATION (secondary branch only)
    # ===================================================================
    if mtf_mode == "calibration":
        divisor = config.MULTI_TF_DIVISOR
        align = config.MULTI_TF_ALIGN

        logger.debug(f"  [MTF calibration] Aggregating {divisor}x candles (align={align})...")
        df_agg, _, _ = aggregate_candles(df_raw, divisor, align)

        # Labels on aggregated data
        close_horizon = parse_close_horizon(config.PREDICTION_TARGET)
        if close_horizon is not None:
            labels_sec, atr_sec, confidence_sec, long_pnl_sec, short_pnl_sec = get_labels_next_close(
                df_agg, horizon=close_horizon,
                window_size=config.WINDOW_SIZE, atr_period=config.ATR_PERIOD)
        else:
            labels_sec, atr_sec, confidence_sec, long_pnl_sec, short_pnl_sec = get_labels(
                df_agg,
                window_size=config.WINDOW_SIZE,
                atr_multiplier_tp=config.ATR_MULTIPLIER_TP, atr_multiplier_sl=config.ATR_MULTIPLIER_SL,
                max_horizon=config.MAX_HORIZON,
                atr_period=config.ATR_PERIOD
            )
            # Remap labels based on prediction target
            labels_sec, _ = remap_labels(labels_sec, config.PREDICTION_TARGET)

        # Transform aggregated data (produces 12 global features)
        result_sec = transform_pipeline(
            df_agg,
            atr_array=atr_sec,
            split_ratio=config.TRAIN_TEST_SPLIT,
            window_size=config.WINDOW_SIZE,
        )

        data_sec_full = result_sec['data']
        n_norm_features_sec = result_sec['n_norm_features']

        # Strip time features (4 columns after n_norm_features)
        # Layout: [0..n_norm-1] norm, [n_norm..n_norm+3] time, [n_norm+4..] regime
        n_norm = n_norm_features_sec
        data_sec = np.delete(data_sec_full, np.s_[n_norm:n_norm+4], axis=1)
        n_features_sec = data_sec.shape[1]  # 8 = 6 norm + 2 regime

        logger.debug(f"  ✓ Secondary features: {data_sec_full.shape[1]} → {n_features_sec} (stripped 4 time features)")

        # Extract raw H/L/C for per-window zigzag (aligned with data_sec)
        ws = config.WINDOW_SIZE
        n_data_sec = len(data_sec)
        raw_hlc_sec = np.column_stack([
            df_agg['High'].values[ws:ws + n_data_sec],
            df_agg['Low'].values[ws:ws + n_data_sec],
            df_agg['Close'].values[ws:ws + n_data_sec]
        ]).astype(np.float64)

        # Align
        min_len = min(len(data_sec), len(labels_sec))
        data_aligned = data_sec[:min_len]
        labels_aligned = labels_sec[:min_len]
        confidence_aligned = confidence_sec[:min_len]
        raw_hlc_sec = raw_hlc_sec[:min_len]
        long_pnl_aligned = long_pnl_sec[:min_len] if long_pnl_sec is not None else None
        short_pnl_aligned = short_pnl_sec[:min_len] if short_pnl_sec is not None else None

        # Split
        n_valid = len(data_aligned)
        split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

        if split_index < config.WINDOW_SIZE:
            raise ValueError(f"Not enough train samples in {csv_path}: {split_index} < {config.WINDOW_SIZE}")
        if (n_valid - split_index) < config.WINDOW_SIZE:
            raise ValueError(f"Not enough test samples in {csv_path}: {n_valid - split_index} < {config.WINDOW_SIZE}")

        train_indices = list(range(split_index - config.WINDOW_SIZE))
        test_indices = list(range(split_index - config.WINDOW_SIZE, n_valid - config.WINDOW_SIZE))

        # n_norm_features=7 after assembly (6 global OHLCV+ATR + 1 per-window log_zz)
        train_dataset = TimeSeriesWindowDataset(
            data=data_aligned, labels=labels_aligned, indices=train_indices,
            window_size=config.WINDOW_SIZE, raw_hlc=raw_hlc_sec,
            n_norm_features=n_norm + 1,
            confidence=confidence_aligned,
            long_pnl_R=long_pnl_aligned,
            short_pnl_R=short_pnl_aligned,
        )
        test_dataset = TimeSeriesWindowDataset(
            data=data_aligned, labels=labels_aligned, indices=test_indices,
            window_size=config.WINDOW_SIZE, raw_hlc=raw_hlc_sec,
            n_norm_features=n_norm + 1,
            confidence=confidence_aligned,
            long_pnl_R=long_pnl_aligned,
            short_pnl_R=short_pnl_aligned,
        )

        # Label distribution: all 3 classes
        unique, counts = np.unique(labels_aligned, return_counts=True)
        label_dist = {int(label): int(count) for label, count in zip(unique, counts)}
        label_pcts = {int(label): count / len(labels_aligned) * 100 for label, count in zip(unique, counts)}

        logger.info(f"  ✓ Dataset ready: {len(train_dataset)} train / {len(test_dataset)} test windows")

        # After per-window assembly: +6 features (1 log_zz + 5 struct)
        n_features_sec_assembled = n_features_sec + 6

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'label_dist': label_dist,
            'label_pcts': label_pcts,
            'n_valid': n_valid,
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            'normalization_params': result_sec['normalization_params'],
            'n_features': n_features_sec_assembled,
            'n_norm_features': n_norm + 1,  # +1 for per-window log_zz
            'multi_tf_mode': 'calibration',
        }

    # ===================================================================
    # SHARED: Load primary data + labels (used by "off" and "dual")
    # ===================================================================
    close_horizon = parse_close_horizon(config.PREDICTION_TARGET)
    if close_horizon is not None:
        labels, atr_array, confidence, long_pnl_R, short_pnl_R = get_labels_next_close(
            df_raw, horizon=close_horizon,
            window_size=config.WINDOW_SIZE, atr_period=config.ATR_PERIOD)
    else:
        labels, atr_array, confidence, long_pnl_R, short_pnl_R = get_labels(
            df_raw,
            window_size=config.WINDOW_SIZE,
            atr_multiplier_tp=config.ATR_MULTIPLIER_TP, atr_multiplier_sl=config.ATR_MULTIPLIER_SL,
            max_horizon=config.MAX_HORIZON,
            atr_period=config.ATR_PERIOD
        )
        # Remap labels based on prediction target
        labels, _ = remap_labels(labels, config.PREDICTION_TARGET)

    # --- DATA TRANSFORMATION (primary) ---

    transform_result = transform_pipeline(
        df_raw,
        atr_array=atr_array,
        split_ratio=config.TRAIN_TEST_SPLIT,
        window_size=config.WINDOW_SIZE,
    )

    data = transform_result['data']
    normalization_params = transform_result['normalization_params']
    n_features = transform_result['n_features']
    n_norm_features = transform_result['n_norm_features']

    logger.debug(f"  ✓ Transformed data shape: {data.shape}")
    logger.debug(f"  ✓ Norm features: {n_norm_features}")

    # Extract raw H/L/C for per-window zigzag (aligned with transformed data)
    ws = config.WINDOW_SIZE
    n_data = len(data)
    raw_hlc = np.column_stack([
        df_raw['High'].values[ws:ws + n_data],
        df_raw['Low'].values[ws:ws + n_data],
        df_raw['Close'].values[ws:ws + n_data]
    ]).astype(np.float64)

    # --- ALIGN DATA AND LABELS ---
    min_len = min(len(data), len(labels))
    data_aligned = data[:min_len]
    labels_aligned = labels[:min_len]
    confidence_aligned = confidence[:min_len]
    raw_hlc = raw_hlc[:min_len]
    long_pnl_aligned = long_pnl_R[:min_len] if long_pnl_R is not None else None
    short_pnl_aligned = short_pnl_R[:min_len] if short_pnl_R is not None else None

    if len(data_aligned) != len(labels_aligned):
        raise ValueError(f"Data/labels mismatch: {len(data_aligned)} vs {len(labels_aligned)}")

    # Label distribution (computed after alignment to reflect actual data used)
    unique, counts = np.unique(labels_aligned, return_counts=True)
    label_dist = {int(label): int(count) for label, count in zip(unique, counts)}
    label_pcts = {int(label): count / len(labels_aligned) * 100 for label, count in zip(unique, counts)}

    # --- VOLUME PROFILE (precomputed globally, no look-ahead) ---
    vp_config = {
        'VP_LOOKBACK': config.VP_LOOKBACK,
        'VP_BINS': config.VP_BINS, 'VP_VA_PCT': config.VP_VA_PCT,
    }
    vp_features = compute_branch_vp(df_raw, vp_config, ws, min_len)
    logger.debug(f"  ✓ VP features precomputed: {vp_features.shape}")

    # ===================================================================
    # MODE: DUAL (primary + secondary)
    # ===================================================================
    if mtf_mode == "dual":
        logger.debug(f"  [MTF dual] Aggregating {config.MULTI_TF_DIVISOR}x candles (align={config.MULTI_TF_ALIGN})...")

        # Secondary transform callable
        def sec_transform_fn(df_agg):
            atr_full = compute_atr(df_agg, period=config.ATR_PERIOD)
            # Slice at WINDOW_SIZE: transform_pipeline expects atr_array[0] = raw index ws
            atr = atr_full[config.WINDOW_SIZE:]
            result = transform_pipeline(df_agg, atr_array=atr,
                                        split_ratio=config.TRAIN_TEST_SPLIT,
                                        window_size=config.WINDOW_SIZE)
            return result['data']

        config_params_dict = {
            'MULTI_TF_DIVISOR': config.MULTI_TF_DIVISOR,
            'MULTI_TF_ALIGN': config.MULTI_TF_ALIGN,
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS,
            'VP_VA_PCT': config.VP_VA_PCT,
            'REGIME_LENGTH': config.REGIME_LENGTH,
            'ATR_PERIOD': config.ATR_PERIOD,
        }

        sec = build_secondary_branches(df_raw, sec_transform_fn, config_params_dict, len(data_aligned))
        n_norm_sec = sec['n_norm_secondary']
        n_features_sec = sec['data_secondary_list'][0].shape[1]
        min_pri_start = sec['min_primary_start']

        logger.debug(f"  [MTF dual] {config.MULTI_TF_ALIGN} mode, min_pri_start={min_pri_start}")
        logger.debug(f"  ✓ Secondary features: {n_features_sec} (stripped 4 time features)")

        # Split
        n_valid = len(data_aligned)
        split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

        train_start = max(min_pri_start, 0)
        train_indices = list(range(train_start, split_index - ws))
        test_indices = list(range(max(split_index - ws, min_pri_start), n_valid - ws))

        if len(train_indices) == 0:
            raise ValueError(f"No valid train indices in dual mode for {csv_path}")
        if len(test_indices) == 0:
            raise ValueError(f"No valid test indices in dual mode for {csv_path}")

        # Create MultiTFWindowDataset (n_norm_features +1 for per-window log_zz)
        train_dataset = MultiTFWindowDataset(
            data_primary=data_aligned, data_secondary=sec['data_secondary_list'], labels=labels_aligned,
            indices_primary=train_indices, mapping_primary_to_secondary=sec['mapping_list'],
            window_size=ws, raw_hlc_primary=raw_hlc, raw_hlc_secondary=sec['raw_hlc_secondary_list'],
            vol_secondary=sec['vol_secondary_list'],
            n_norm_features_primary=n_norm_features + 1, n_norm_features_secondary=n_norm_sec + 1,
            partial_context=sec['partial_ctx_list'],
            vp_features_primary=vp_features, vp_features_secondary=sec['vp_secondary_list'],
            vp_params=config_params_dict,
            cross_tf_normalize=config.CROSS_TF_NORMALIZE,
            confidence=confidence_aligned,
            long_pnl_R=long_pnl_aligned,
            short_pnl_R=short_pnl_aligned,
        )

        test_dataset = MultiTFWindowDataset(
            data_primary=data_aligned, data_secondary=sec['data_secondary_list'], labels=labels_aligned,
            indices_primary=test_indices, mapping_primary_to_secondary=sec['mapping_list'],
            window_size=ws, raw_hlc_primary=raw_hlc, raw_hlc_secondary=sec['raw_hlc_secondary_list'],
            vol_secondary=sec['vol_secondary_list'],
            n_norm_features_primary=n_norm_features + 1, n_norm_features_secondary=n_norm_sec + 1,
            partial_context=sec['partial_ctx_list'],
            vp_features_primary=vp_features, vp_features_secondary=sec['vp_secondary_list'],
            vp_params=config_params_dict,
            cross_tf_normalize=config.CROSS_TF_NORMALIZE,
            confidence=confidence_aligned,
            long_pnl_R=long_pnl_aligned,
            short_pnl_R=short_pnl_aligned,
        )

        logger.info(f"  ✓ Dual dataset ready: {len(train_dataset)} train / {len(test_dataset)} test windows")

        # After per-window assembly: primary gets +6 features (1 log_zz + 5 struct)
        # secondary gets +6 features too. VP adds +5.
        n_features_assembled = n_features + 6 + 5    # 12 + 6 + 5 = 23
        n_features_sec_assembled = n_features_sec + 6 + 5  # 8 + 6 + 5 = 19

        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'label_dist': label_dist,
            'label_pcts': label_pcts,
            'n_valid': n_valid,
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            'normalization_params': normalization_params,
            'n_features': n_features_assembled,
            'n_norm_features': n_norm_features + 1,  # +1 for per-window log_zz
            'multi_tf_mode': 'dual',
            'n_features_sec': n_features_sec_assembled,
            'n_norm_features_sec': n_norm_sec + 1,
        }

    # ===================================================================
    # MODE: OFF (standard pipeline, unchanged)
    # ===================================================================
    n_valid = len(data_aligned)
    split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

    if split_index < config.WINDOW_SIZE:
        raise ValueError(f"Not enough train samples in {csv_path}: {split_index} < {config.WINDOW_SIZE}")
    if (n_valid - split_index) < config.WINDOW_SIZE:
        raise ValueError(f"Not enough test samples in {csv_path}: {n_valid - split_index} < {config.WINDOW_SIZE}")

    # --- CREATE INDEX LISTS ---
    train_indices = list(range(split_index - config.WINDOW_SIZE))
    test_indices = list(range(split_index - config.WINDOW_SIZE, n_valid - config.WINDOW_SIZE))

    # --- CREATE DATASETS (n_norm_features +1 for per-window log_zz) ---
    train_dataset = TimeSeriesWindowDataset(
        data=data_aligned,
        labels=labels_aligned,
        indices=train_indices,
        window_size=config.WINDOW_SIZE,
        raw_hlc=raw_hlc,
        n_norm_features=n_norm_features + 1,
        vp_features=vp_features,
        confidence=confidence_aligned,
        long_pnl_R=long_pnl_aligned,
        short_pnl_R=short_pnl_aligned,
    )

    test_dataset = TimeSeriesWindowDataset(
        data=data_aligned,
        labels=labels_aligned,
        indices=test_indices,
        window_size=config.WINDOW_SIZE,
        raw_hlc=raw_hlc,
        n_norm_features=n_norm_features + 1,
        vp_features=vp_features,
        confidence=confidence_aligned,
        long_pnl_R=long_pnl_aligned,
        short_pnl_R=short_pnl_aligned,
    )

    # After per-window assembly: +6 features (1 log_zz + 5 struct) + 5 VP
    n_features_assembled = n_features + 6 + 5  # 12 + 6 + 5 = 23

    logger.info(f"  ✓ Dataset ready: {len(train_dataset)} train / {len(test_dataset)} test windows")

    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'label_dist': label_dist,
        'label_pcts': label_pcts,
        'n_valid': n_valid,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'normalization_params': normalization_params,
        'n_features': n_features_assembled,
        'n_norm_features': n_norm_features + 1,  # +1 for per-window log_zz
        'multi_tf_mode': 'off',
    }
