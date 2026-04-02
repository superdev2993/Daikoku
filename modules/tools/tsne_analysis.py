"""
t-SNE Analysis Tool — Visualize data structure and model embeddings.

Two analysis levels:
- Raw features: flattened windows → does the directional signal exist in the data?
- Model embeddings: pre-classifier representation → has the model learned to separate classes?

Usage:
    python -m modules.tools.tsne_analysis --split test
    python -m modules.tools.tsne_analysis --checkpoint models/latest.pt --split all
    python -m modules.tools.tsne_analysis --checkpoint models/latest.pt --perplexity 50 --workers 6
"""

import sys
import os

# CRITICAL: Limit BLAS threads BEFORE importing numpy/sklearn
# Without this, each ProcessPoolExecutor worker tries to use all CPU cores
# via OpenBLAS/MKL, causing thread contention and near-zero throughput.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import config
from modules.data.loader import get_raw_data
from modules.data.labeling import get_labels, remap_labels
from modules.data.transform import transform_pipeline, apply_transform_from_checkpoint
from modules.data.dataset import TimeSeriesWindowDataset, MultiTFWindowDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Class colors consistent with visualizer.py
CLASS_NAMES = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}
CLASS_COLORS = {0: '#EF553B', 1: '#FFA726', 2: '#00CC96'}
DATASET_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
]


# ============================================================================
# 1. Dataset window extraction (raw features)
# ============================================================================

def extract_dataset_windows(dataset, batch_size=512):
    """
    Extract all windows from a dataset as flattened numpy arrays.

    Detects mono (2-tuple) vs dual (3-tuple) automatically.

    Returns:
        dict with keys:
            'raw_primary': np.array (N, D_pri) — flattened primary windows
            'raw_concat': np.array (N, D_pri+D_sec) or None — flattened primary+secondary
            'labels': np.array (N,)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_primary = []
    all_concat = []
    all_labels = []
    is_dual = None

    for batch in loader:
        if is_dual is None:
            # First batch: detect format
            # Mono: (window, label) or (window, label, conf)
            # Dual: (pri, sec, label) or (pri, sec, label, conf)
            if len(batch) >= 3 and batch[0].dim() == 3 and batch[1].dim() == 3:
                is_dual = True
            else:
                is_dual = False

        if is_dual:
            pri, sec, labels = batch[0], batch[1], batch[2]
            B = pri.shape[0]
            pri_flat = pri.reshape(B, -1).numpy()
            sec_flat = sec.reshape(B, -1).numpy()
            all_primary.append(pri_flat)
            all_concat.append(np.concatenate([pri_flat, sec_flat], axis=1))
            all_labels.append(labels.numpy())
        else:
            windows, labels = batch[0], batch[1]
            B = windows.shape[0]
            all_primary.append(windows.reshape(B, -1).numpy())
            all_labels.append(labels.numpy())

    result = {
        'raw_primary': np.concatenate(all_primary, axis=0),
        'raw_concat': np.concatenate(all_concat, axis=0) if is_dual else None,
        'labels': np.concatenate(all_labels, axis=0),
    }
    return result


# ============================================================================
# 2. Model embedding extraction (forward hook)
# ============================================================================

def get_hook_layer(model):
    """
    Detect the correct layer to hook for embedding extraction.

    Returns the first Linear layer of the classifier head (its input = embedding).
    """
    from modules.model.mamba import MultiTFMambaPredictor, MambaPredictor, UnifiedHead

    if isinstance(model, MultiTFMambaPredictor):
        # MultiTF: always UnifiedHead → classifier[0]
        return model.head.classifier[0]
    elif isinstance(model, MambaPredictor):
        if isinstance(model.head, UnifiedHead):
            # Mono + attention: UnifiedHead → classifier[0]
            return model.head.classifier[0]
        elif isinstance(model.head, nn.Sequential):
            # Mono no attention: Sequential → head[0] (Linear)
            return model.head[0]
    raise ValueError(f"Cannot find hook layer for model type {type(model)}")


def extract_embeddings(model, dataset, device, batch_size=256):
    """
    Extract pre-classifier embeddings via forward hook.

    Args:
        model: Trained model (eval mode)
        dataset: Dataset to extract from
        device: torch device
        batch_size: Batch size for inference

    Returns:
        tuple: (embeddings: np.array(N, D_emb), confidences: np.array(N, 3))
    """
    hook_layer = get_hook_layer(model)

    embeddings_list = []
    confidences_list = []

    def hook_fn(module, input, output):
        # input[0] is the embedding fed to this Linear layer
        embeddings_list.append(input[0].detach().cpu().numpy())

    handle = hook_layer.register_forward_hook(hook_fn)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                # Detect mono vs dual
                if len(batch) >= 3 and batch[0].dim() == 3 and batch[1].dim() == 3:
                    pri = batch[0].to(device)
                    sec = batch[1].to(device)
                    logits = model(pri, sec)
                else:
                    windows = batch[0].to(device)
                    logits = model(windows)

                probs = torch.softmax(logits, dim=-1)
                confidences_list.append(probs.cpu().numpy())
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"CUDA OOM with batch_size={batch_size}, retrying with {batch_size // 2}")
            handle.remove()
            torch.cuda.empty_cache()
            embeddings_list.clear()
            confidences_list.clear()
            return extract_embeddings(model, dataset, device, batch_size=batch_size // 2)
        raise
    finally:
        handle.remove()

    embeddings = np.concatenate(embeddings_list, axis=0)
    confidences = np.concatenate(confidences_list, axis=0)
    return embeddings, confidences


# ============================================================================
# 3. t-SNE computation (parallelized)
# ============================================================================

PCA_TARGET_DIM = 50  # Standard pre-t-SNE reduction (recommended by sklearn/van der Maaten)


def _pca_reduce(data, target_dim=PCA_TARGET_DIM):
    """
    Reduce dimensionality with PCA before t-SNE.

    Applied BEFORE pickling to workers — reduces serialization from ~160MB to ~1.6MB per job.
    Standard practice: t-SNE does not benefit from dimensions beyond ~50.

    Args:
        data: np.array (N, D) with D >> target_dim
        target_dim: number of PCA components

    Returns:
        np.array (N, min(target_dim, D))
    """
    from sklearn.decomposition import PCA

    if data.shape[1] <= target_dim:
        return data

    pca = PCA(n_components=target_dim, random_state=42)
    return pca.fit_transform(data)


def run_single_tsne(data, perplexity, seed=42):
    """
    Run a single t-SNE computation. Top-level function for pickling.

    Args:
        data: np.array (N, D) — already PCA-reduced
        perplexity: float
        seed: int

    Returns:
        np.array (N, 2)
    """
    from sklearn.manifold import TSNE

    n_samples = data.shape[0]

    # Adjust perplexity if too high
    max_perp = n_samples / 3.0
    if perplexity >= max_perp:
        adjusted = max(5.0, max_perp - 1)
        perplexity = adjusted

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        method='barnes_hut',
        random_state=seed,
        n_jobs=1,
        init='pca',
        learning_rate='auto',
    )
    return tsne.fit_transform(data)


def run_all_tsne(jobs, perplexity, max_workers):
    """
    Run all t-SNE jobs in parallel with progress tracking.

    Args:
        jobs: list of (name, data_array) tuples
        perplexity: float
        max_workers: int

    Returns:
        dict: {name: tsne_2d_array} for successful jobs, {name: None} for failures
    """
    results = {}
    total = len(jobs)

    if total == 0:
        return results

    # PCA reduction in main process BEFORE dispatching to workers
    # Reduces pickle overhead from ~160MB to ~1.6MB per job
    print(f"\n{'='*60}", flush=True)
    print(f"  PCA reduction ({PCA_TARGET_DIM}D) + t-SNE: {total} jobs, {max_workers} workers", flush=True)
    print(f"{'='*60}", flush=True)

    pca_start = time.time()
    jobs_reduced = []
    for name, data in jobs:
        orig_dim = data.shape[1]
        reduced = _pca_reduce(data)
        jobs_reduced.append((name, reduced, data.shape))
        print(f"  PCA: {name} ({data.shape[0]:,}x{orig_dim}) -> ({reduced.shape[0]:,}x{reduced.shape[1]})", flush=True)

    pca_elapsed = time.time() - pca_start
    print(f"  PCA done in {pca_elapsed:.1f}s", flush=True)
    print(f"  {'─'*56}", flush=True)

    start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {}
        for name, data_reduced, orig_shape in jobs_reduced:
            future = executor.submit(run_single_tsne, data_reduced, perplexity)
            future_to_info[future] = (name, orig_shape)

        for i, future in enumerate(as_completed(future_to_info), 1):
            name, orig_shape = future_to_info[future]
            pct = i / total * 100
            elapsed = time.time() - start
            try:
                results[name] = future.result()
                print(f"  [{i:2d}/{total}] {pct:5.1f}% | {name} ({orig_shape[0]:,}x{orig_shape[1]}) | {elapsed:.0f}s elapsed", flush=True)
            except Exception as e:
                print(f"  [{i:2d}/{total}] {pct:5.1f}% | {name} FAILED: {e} | {elapsed:.0f}s elapsed", flush=True)
                results[name] = None

    elapsed = time.time() - start
    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"{'='*60}", flush=True)
    print(f"  t-SNE complete: {n_ok}/{total} succeeded in {elapsed:.1f}s", flush=True)
    print(f"{'='*60}\n", flush=True)
    return results


# ============================================================================
# 4. Data pipeline (with or without checkpoint)
# ============================================================================

def load_dataset_with_checkpoint(evaluator, path, split):
    """
    Load a dataset using ModelEvaluator (checkpoint available).

    Returns:
        dict with keys: raw_primary, raw_concat, labels, timestamps, embeddings, confidences, name
        or None if loading fails
    """
    name = Path(path).stem
    logger.info(f"Loading {name} with checkpoint (split={split})...")

    try:
        dataset, labels, timestamps, split_info = evaluator.prepare_data(path, split)
    except Exception as e:
        logger.warning(f"  Skipping {name}: {e}")
        return None

    if len(dataset) == 0:
        logger.warning(f"  Skipping {name}: empty dataset after split")
        return None

    # Extract raw windows
    raw_data = extract_dataset_windows(dataset)

    # Extract embeddings
    emb, conf = extract_embeddings(evaluator.model, dataset, evaluator.device)

    return {
        'name': name,
        'raw_primary': raw_data['raw_primary'],
        'raw_concat': raw_data['raw_concat'],
        'labels': raw_data['labels'],
        'timestamps': timestamps,
        'embeddings': emb,
        'confidences': conf,
        'n_samples': len(dataset),
    }


def load_dataset_without_checkpoint(path, split, max_horizon=None):
    """
    Load a dataset without checkpoint (raw features only).

    Args:
        max_horizon: Override MAX_HORIZON for labeling (None = use config)

    Returns:
        dict with keys: raw_primary, raw_concat, labels, timestamps, name
        or None if loading fails
    """
    name = Path(path).stem
    logger.info(f"Loading {name} without checkpoint (split={split})...")

    try:
        df_raw = get_raw_data(path)
    except Exception as e:
        logger.warning(f"  Skipping {name}: {e}")
        return None

    ws = config.WINDOW_SIZE

    # Labels on raw data
    labels, atr, _, _, _ = get_labels(df_raw, max_horizon=max_horizon)
    labels, _ = remap_labels(labels, config.PREDICTION_TARGET)

    mtf_mode = config.MULTI_TF_MODE

    if mtf_mode == "dual":
        from modules.data.multi_tf import build_secondary_branches, compute_branch_vp

        # Primary transform
        result = transform_pipeline(df_raw, atr_array=atr)
        data_pri = result['data'][:len(labels)]
        labels_aligned = labels[:len(data_pri)]

        n_data = len(data_pri)
        raw_hlc_pri = np.column_stack([
            df_raw['High'].values[ws:ws + n_data],
            df_raw['Low'].values[ws:ws + n_data],
            df_raw['Close'].values[ws:ws + n_data]
        ]).astype(np.float64)

        n_norm_pri = data_pri.shape[1] - 6
        split_index = int(n_data * config.TRAIN_TEST_SPLIT)

        # Secondary branches
        from modules.data.labeling import compute_atr

        def sec_transform_fn(df_agg):
            atr_full = compute_atr(df_agg, period=config.ATR_PERIOD)
            atr_sec = atr_full[config.WINDOW_SIZE:]
            sec_result = transform_pipeline(df_agg, atr_array=atr_sec)
            return sec_result['data']

        sec = build_secondary_branches(df_raw, sec_transform_fn, {
            'MULTI_TF_DIVISOR': config.MULTI_TF_DIVISOR,
            'MULTI_TF_ALIGN': config.MULTI_TF_ALIGN,
            'REGIME_LENGTH': config.REGIME_LENGTH,
            'ATR_PERIOD': config.ATR_PERIOD,
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS,
            'VP_VA_PCT': config.VP_VA_PCT,
        }, n_data)

        n_norm_sec = sec['n_norm_secondary']
        min_pri_start = sec['min_primary_start']

        # VP features
        vp_features_pri = compute_branch_vp(df_raw, {
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS,
            'VP_VA_PCT': config.VP_VA_PCT,
            'WINDOW_SIZE': config.WINDOW_SIZE,
        }, ws, len(data_pri))

        # Indices
        if split == 'train':
            indices = list(range(max(min_pri_start, 0), split_index - ws))
        elif split == 'test':
            indices = list(range(max(split_index - ws, min_pri_start), n_data - ws))
        elif split == 'all':
            indices = list(range(min_pri_start, n_data - ws))
        else:
            raise ValueError(f"Invalid split: {split}")

        dataset = MultiTFWindowDataset(
            data_primary=data_pri, data_secondary=sec['data_secondary_list'],
            labels=labels_aligned,
            indices_primary=indices, mapping_primary_to_secondary=sec['mapping_list'],
            window_size=ws, raw_hlc_primary=raw_hlc_pri,
            raw_hlc_secondary=sec['raw_hlc_secondary_list'],
            vol_secondary=sec['vol_secondary_list'],
            n_norm_features_primary=n_norm_pri + 1, n_norm_features_secondary=n_norm_sec + 1,
            partial_context=sec['partial_ctx_list'],
            zigzag_length=config.ZIGZAG_LENGTH,
            vp_features_primary=vp_features_pri, vp_features_secondary=sec['vp_secondary_list'],
            vp_params={
                'VP_LOOKBACK': config.VP_LOOKBACK,
                'VP_BINS': config.VP_BINS,
                'VP_VA_PCT': config.VP_VA_PCT,
            },
            cross_tf_normalize=config.CROSS_TF_NORMALIZE,
        )

        # Timestamps
        raw_start_idx = ws + indices[0] + ws - 1
        timestamps = df_raw['Open time'].iloc[raw_start_idx:raw_start_idx + len(indices)].values

    elif mtf_mode == "calibration":
        from modules.data.aggregation import aggregate_candles
        from modules.data.multi_tf import compute_branch_vp

        divisor = config.MULTI_TF_DIVISOR
        align = config.MULTI_TF_ALIGN
        df_agg, _, _ = aggregate_candles(df_raw, divisor, align)

        labels_sec, atr_sec, _, _, _ = get_labels(df_agg, max_horizon=max_horizon)
        labels_sec, _ = remap_labels(labels_sec, config.PREDICTION_TARGET)

        result = transform_pipeline(df_agg, atr_array=atr_sec)
        data_sec = result['data']
        # Strip time features
        n_norm = data_sec.shape[1] - 6
        data_sec = np.delete(data_sec, np.s_[n_norm:n_norm+4], axis=1)

        n_data = len(data_sec)
        raw_hlc = np.column_stack([
            df_agg['High'].values[ws:ws + n_data],
            df_agg['Low'].values[ws:ws + n_data],
            df_agg['Close'].values[ws:ws + n_data]
        ]).astype(np.float64)

        data_aligned = data_sec[:len(labels_sec)]
        labels_aligned = labels_sec[:len(data_aligned)]
        raw_hlc = raw_hlc[:len(data_aligned)]

        n_valid = len(data_aligned)
        split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

        if split == 'train':
            indices = list(range(split_index - ws))
        elif split == 'test':
            indices = list(range(split_index - ws, n_valid - ws))
        elif split == 'all':
            indices = list(range(n_valid - ws))
        else:
            raise ValueError(f"Invalid split: {split}")

        vp_features = compute_branch_vp(df_agg, {
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS,
            'VP_VA_PCT': config.VP_VA_PCT,
            'WINDOW_SIZE': config.WINDOW_SIZE,
        }, ws, len(data_aligned))

        dataset = TimeSeriesWindowDataset(
            data=data_aligned, labels=labels_aligned, indices=indices,
            window_size=ws, raw_hlc=raw_hlc,
            n_norm_features=n_norm + 1,
            zigzag_length=config.ZIGZAG_LENGTH,
            vp_features=vp_features,
        )

        raw_start_idx = ws + indices[0] + ws - 1
        timestamps = df_agg['Open time'].iloc[raw_start_idx:raw_start_idx + len(indices)].values

    else:
        # Standard mode (off)
        result = transform_pipeline(df_raw, atr_array=atr)
        data = result['data'][:len(labels)]
        labels_aligned = labels[:len(data)]

        n_data = len(data)
        raw_hlc = np.column_stack([
            df_raw['High'].values[ws:ws + n_data],
            df_raw['Low'].values[ws:ws + n_data],
            df_raw['Close'].values[ws:ws + n_data]
        ]).astype(np.float64)
        raw_hlc = raw_hlc[:len(data)]

        n_valid = len(data)
        split_index = int(n_valid * config.TRAIN_TEST_SPLIT)
        total_features = data.shape[1]
        n_norm_features = total_features - 6

        if split == 'train':
            indices = list(range(split_index - ws))
        elif split == 'test':
            indices = list(range(split_index - ws, n_valid - ws))
        elif split == 'all':
            indices = list(range(n_valid - ws))
        else:
            raise ValueError(f"Invalid split: {split}")

        from modules.data.multi_tf import compute_branch_vp
        vp_features = compute_branch_vp(df_raw, {
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS,
            'VP_VA_PCT': config.VP_VA_PCT,
            'WINDOW_SIZE': config.WINDOW_SIZE,
        }, ws, len(data))

        dataset = TimeSeriesWindowDataset(
            data=data, labels=labels_aligned, indices=indices,
            window_size=ws, raw_hlc=raw_hlc,
            n_norm_features=n_norm_features + 1,
            zigzag_length=config.ZIGZAG_LENGTH,
            vp_features=vp_features,
        )

        raw_start_idx = ws + indices[0] + ws - 1
        timestamps = df_raw['Open time'].iloc[raw_start_idx:raw_start_idx + len(indices)].values

    if len(dataset) == 0:
        logger.warning(f"  Skipping {name}: empty dataset after split")
        return None

    raw_data = extract_dataset_windows(dataset)

    return {
        'name': name,
        'raw_primary': raw_data['raw_primary'],
        'raw_concat': raw_data['raw_concat'],
        'labels': raw_data['labels'],
        'timestamps': timestamps,
        'embeddings': None,
        'confidences': None,
        'n_samples': len(dataset),
    }


# ============================================================================
# 5. Silhouette score computation
# ============================================================================

def compute_silhouette(tsne_2d, labels):
    """
    Compute silhouette score. Returns float or None if computation fails.
    """
    from sklearn.metrics import silhouette_score

    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    try:
        # Subsample if too large (>20k) for speed
        n = len(labels)
        if n > 20000:
            idx = np.random.RandomState(42).choice(n, 20000, replace=False)
            return float(silhouette_score(tsne_2d[idx], labels[idx]))
        return float(silhouette_score(tsne_2d, labels))
    except Exception:
        return None


# ============================================================================
# 6. HTML report generation (Plotly)
# ============================================================================

def build_tsne_scatter(tsne_2d, values, color_mode, title, class_names=None, dataset_names=None):
    """
    Build a Plotly Scattergl figure for t-SNE results.

    Args:
        tsne_2d: np.array (N, 2)
        values: np.array — labels, timestamps, dataset_ids, or confidences
        color_mode: 'label', 'time', 'dataset', 'confidence'
        title: str
        class_names: dict {0: 'Bear', ...} for label mode
        dataset_names: list of str for dataset mode

    Returns:
        go.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if color_mode == 'label':
        if class_names is None:
            class_names = CLASS_NAMES
        for cls_id in sorted(class_names.keys()):
            mask = values == cls_id
            if not np.any(mask):
                continue
            fig.add_trace(go.Scatter(
                x=tsne_2d[mask, 0], y=tsne_2d[mask, 1],
                mode='markers',
                marker=dict(size=3, opacity=0.6, color=CLASS_COLORS.get(cls_id, '#888')),
                name=f"{class_names[cls_id]} ({int(np.sum(mask))})",
                hovertemplate=f"{class_names[cls_id]}<br>x:%{{x:.1f}}<br>y:%{{y:.1f}}<extra></extra>",
            ))

    elif color_mode == 'time':
        # Convert to numeric for colorscale
        import pandas as pd
        ts = pd.to_datetime(values)
        numeric = (ts - ts.min()).total_seconds().values
        fig.add_trace(go.Scatter(
            x=tsne_2d[:, 0], y=tsne_2d[:, 1],
            mode='markers',
            marker=dict(
                size=3, opacity=0.6,
                color=numeric,
                colorscale='Viridis',
                colorbar=dict(
                    title='Time',
                    tickvals=[numeric.min(), numeric.max()],
                    ticktext=[str(ts.min().date()), str(ts.max().date())],
                ),
            ),
            hovertemplate="x:%{x:.1f}<br>y:%{y:.1f}<extra></extra>",
        ))

    elif color_mode == 'dataset':
        unique_ids = sorted(np.unique(values))
        for i, ds_id in enumerate(unique_ids):
            mask = values == ds_id
            ds_name = dataset_names[ds_id] if dataset_names and ds_id < len(dataset_names) else f"Dataset {ds_id}"
            color = DATASET_COLORS[i % len(DATASET_COLORS)]
            fig.add_trace(go.Scatter(
                x=tsne_2d[mask, 0], y=tsne_2d[mask, 1],
                mode='markers',
                marker=dict(size=3, opacity=0.6, color=color),
                name=f"{ds_name} ({int(np.sum(mask))})",
                hovertemplate=f"{ds_name}<br>x:%{{x:.1f}}<br>y:%{{y:.1f}}<extra></extra>",
            ))

    elif color_mode == 'confidence':
        # Max probability as confidence
        max_conf = np.max(values, axis=1) if values.ndim > 1 else values
        fig.add_trace(go.Scatter(
            x=tsne_2d[:, 0], y=tsne_2d[:, 1],
            mode='markers',
            marker=dict(
                size=3, opacity=0.6,
                color=max_conf,
                colorscale='RdYlGn',
                cmin=0.33, cmax=1.0,
                colorbar=dict(title='Confidence'),
            ),
            hovertemplate="conf:%{marker.color:.3f}<br>x:%{x:.1f}<br>y:%{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template='plotly_dark',
        width=650, height=550,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font=dict(size=10)),
    )
    return fig


def build_stats_table(all_datasets, tsne_results, has_checkpoint):
    """
    Build an HTML table with silhouette scores per dataset and mode.
    """
    rows = []

    for ds in all_datasets:
        name = ds['name']
        n = ds['n_samples']

        for mode in ['raw_primary', 'raw_concat', 'embeddings']:
            key = f"{name}_{mode}"
            tsne_2d = tsne_results.get(key)
            if tsne_2d is None:
                continue

            sil = compute_silhouette(tsne_2d, ds['labels'])
            dim_key = mode
            if dim_key == 'raw_primary':
                dim = ds['raw_primary'].shape[1]
            elif dim_key == 'raw_concat' and ds['raw_concat'] is not None:
                dim = ds['raw_concat'].shape[1]
            elif dim_key == 'embeddings' and ds.get('embeddings') is not None:
                dim = ds['embeddings'].shape[1]
            else:
                dim = '?'

            rows.append({
                'dataset': name, 'n_samples': n, 'mode': mode,
                'dimensions': dim,
                'silhouette': f"{sil:.4f}" if sil is not None else 'N/A',
            })

    # Global
    for mode in ['raw_primary', 'raw_concat', 'embeddings']:
        key = f"global_{mode}"
        tsne_2d = tsne_results.get(key)
        if tsne_2d is None:
            continue

        global_labels = np.concatenate([ds['labels'] for ds in all_datasets])
        sil = compute_silhouette(tsne_2d, global_labels)
        rows.append({
            'dataset': 'GLOBAL', 'n_samples': len(global_labels), 'mode': mode,
            'dimensions': '-',
            'silhouette': f"{sil:.4f}" if sil is not None else 'N/A',
        })

    if not rows:
        return "<p>No silhouette scores computed.</p>"

    html = '<table style="width:100%;border-collapse:collapse;margin:20px 0;">'
    html += '<tr style="border-bottom:2px solid #555;">'
    for h in ['Dataset', 'N', 'Mode', 'Dim', 'Silhouette']:
        html += f'<th style="padding:8px;text-align:left;">{h}</th>'
    html += '</tr>'

    for row in rows:
        html += '<tr style="border-bottom:1px solid #333;">'
        html += f'<td style="padding:6px;">{row["dataset"]}</td>'
        html += f'<td style="padding:6px;">{row["n_samples"]:,}</td>'
        html += f'<td style="padding:6px;">{row["mode"]}</td>'
        html += f'<td style="padding:6px;">{row["dimensions"]}</td>'
        html += f'<td style="padding:6px;">{row["silhouette"]}</td>'
        html += '</tr>'

    html += '</table>'
    return html


def assemble_html(sections, config_info, stats_html, output_path):
    """
    Assemble final HTML report from all sections.

    Args:
        sections: list of (section_title, list_of_plotly_figures)
        config_info: dict of configuration parameters
        stats_html: HTML string for stats section
        output_path: output file path
    """
    import plotly.io as pio

    # Navigation links
    nav_ids = ['config'] + [s[0].lower().replace(' ', '_').replace('—', '').strip('_') for s in sections] + ['stats']
    nav_labels = ['Config'] + [s[0].split('—')[0].strip() for s in sections] + ['Stats']

    nav_html = '<div id="nav" style="position:sticky;top:0;z-index:1000;background:#1a1a2e;padding:10px 20px;border-bottom:1px solid #444;display:flex;gap:12px;flex-wrap:wrap;">'
    for nid, nlabel in zip(nav_ids, nav_labels):
        nav_html += f'<a href="#{nid}" style="color:#7eb8da;text-decoration:none;padding:4px 10px;border-radius:4px;background:#252540;font-size:13px;">{nlabel}</a>'
    nav_html += '</div>'

    # Config section
    config_html = '<div id="config" style="padding:20px;"><h2 style="color:#ddd;">Configuration</h2>'
    config_html += '<table style="border-collapse:collapse;margin:10px 0;">'
    for k, v in config_info.items():
        config_html += f'<tr><td style="padding:4px 12px;color:#999;">{k}</td><td style="padding:4px 12px;color:#ddd;">{v}</td></tr>'
    config_html += '</table></div><hr style="border-color:#333;">'

    # Build figure HTML
    figures_html = ''
    for section_title, figures in sections:
        section_id = section_title.lower().replace(' ', '_').replace('—', '').strip('_')
        figures_html += f'<div id="{section_id}" style="padding:20px;">'
        figures_html += f'<h2 style="color:#ddd;">{section_title}</h2>'
        figures_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(600px, 1fr));gap:10px;">'

        for fig in figures:
            if fig is None:
                figures_html += '<div style="padding:40px;text-align:center;color:#888;background:#1e1e2e;border-radius:8px;">t-SNE computation failed</div>'
                continue
            fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
            figures_html += f'<div>{fig_html}</div>'

        figures_html += '</div></div><hr style="border-color:#333;">'

    # Stats section
    stats_section = f'<div id="stats" style="padding:20px;"><h2 style="color:#ddd;">Silhouette Scores</h2>{stats_html}</div>'

    # Full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>t-SNE Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ background: #0f0f23; color: #ccc; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; }}
        h2 {{ margin-top: 0; }}
        a {{ color: #7eb8da; }}
        hr {{ margin: 0; }}
    </style>
</head>
<body>
{nav_html}
{config_html}
{figures_html}
{stats_section}
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    logger.info(f"Report saved to {output_path}")


# ============================================================================
# 7. Main pipeline
# ============================================================================

def main():
    global CLASS_NAMES, CLASS_COLORS

    parser = argparse.ArgumentParser(description="t-SNE analysis of raw features and model embeddings")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for embedding extraction (optional)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'all'],
                        help='Data split to analyze (default: test)')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--output', type=str, default='evaluation/tsne_analysis.html',
                        help='Output HTML path (default: evaluation/tsne_analysis.html)')
    parser.add_argument('--workers', type=int, default=min(cpu_count() // 2, 8),
                        help=f'Number of parallel workers (default: {min(cpu_count() // 2, 8)})')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Override MAX_HORIZON for labeling (default: use config.py)')
    parser.add_argument('--binary', type=int, default=None, choices=[0, 1, 2],
                        help='Binary mode: isolate one class vs rest (0=Bear, 1=Uncertain, 2=Bull)')
    args = parser.parse_args()

    has_checkpoint = args.checkpoint is not None

    # Validate checkpoint if provided
    if has_checkpoint and not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("t-SNE ANALYSIS")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Perplexity: {args.perplexity}")
    logger.info(f"  Checkpoint: {args.checkpoint or 'None (raw only)'}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Datasets: {len(config.INPUT_FILES)}")
    if args.horizon is not None:
        logger.info(f"  Horizon override: {args.horizon} (config: {config.MAX_HORIZON})")
    if args.binary is not None:
        binary_name = CLASS_NAMES[args.binary]
        logger.info(f"  Binary mode: {binary_name} vs rest")
    logger.info("=" * 70)

    # Warn if checkpoint + horizon (embeddings won't reflect new labels)
    if has_checkpoint and args.horizon is not None:
        logger.warning("--horizon with --checkpoint: labels are recomputed but embeddings come from the original model")

    # Load evaluator once if checkpoint provided
    evaluator = None
    if has_checkpoint:
        from modules.evaluation.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(args.checkpoint)
        evaluator.load_checkpoint()

    # Load all datasets
    all_datasets = []
    for path in config.INPUT_FILES:
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue

        if has_checkpoint:
            ds_data = load_dataset_with_checkpoint(evaluator, path, args.split)
        else:
            ds_data = load_dataset_without_checkpoint(path, args.split, max_horizon=args.horizon)

        if ds_data is not None:
            all_datasets.append(ds_data)

    if not all_datasets:
        logger.error("No datasets loaded. Exiting.")
        sys.exit(1)

    logger.info(f"\nLoaded {len(all_datasets)} datasets, total {sum(d['n_samples'] for d in all_datasets):,} samples")

    # Binary remapping: isolate one class vs rest
    if args.binary is not None:
        target_class = args.binary
        target_name = {0: 'Bear', 1: 'Uncertain', 2: 'Bull'}[target_class]
        original_color = {0: '#EF553B', 1: '#FFA726', 2: '#00CC96'}[target_class]
        CLASS_NAMES = {0: f'Not-{target_name}', 1: target_name}
        CLASS_COLORS = {0: '#888888', 1: original_color}
        for ds in all_datasets:
            ds['labels'] = (ds['labels'] == target_class).astype(int)
        logger.info(f"  Binary remapping: {target_name}=1 vs rest=0")

    # Prepare t-SNE jobs
    tsne_jobs = []

    # Per-dataset jobs
    for ds in all_datasets:
        name = ds['name']
        tsne_jobs.append((f"{name}_raw_primary", ds['raw_primary']))
        if ds['raw_concat'] is not None:
            tsne_jobs.append((f"{name}_raw_concat", ds['raw_concat']))
        if ds.get('embeddings') is not None:
            tsne_jobs.append((f"{name}_embeddings", ds['embeddings']))

    # Global (concatenated) jobs
    global_raw_primary = np.concatenate([d['raw_primary'] for d in all_datasets])
    global_labels = np.concatenate([d['labels'] for d in all_datasets])
    global_dataset_ids = np.concatenate([np.full(len(d['labels']), i) for i, d in enumerate(all_datasets)])
    tsne_jobs.append(("global_raw_primary", global_raw_primary))

    has_dual = any(d['raw_concat'] is not None for d in all_datasets)
    if has_dual:
        global_raw_concat = np.concatenate([d['raw_concat'] for d in all_datasets if d['raw_concat'] is not None])
        tsne_jobs.append(("global_raw_concat", global_raw_concat))

    has_emb = any(d.get('embeddings') is not None for d in all_datasets)
    if has_emb:
        global_embeddings = np.concatenate([d['embeddings'] for d in all_datasets if d.get('embeddings') is not None])
        tsne_jobs.append(("global_embeddings", global_embeddings))

    # Global timestamps and confidences
    global_timestamps = np.concatenate([d['timestamps'] for d in all_datasets])
    if has_emb:
        global_confidences = np.concatenate([d['confidences'] for d in all_datasets if d.get('confidences') is not None])

    # Run all t-SNE in parallel
    logger.info(f"\n{len(tsne_jobs)} t-SNE jobs to run:")
    for name, data in tsne_jobs:
        logger.info(f"  {name}: {data.shape}")

    tsne_results = run_all_tsne(tsne_jobs, args.perplexity, args.workers)

    # Build HTML report
    import plotly.graph_objects as go

    sections = []
    dataset_names = [d['name'] for d in all_datasets]

    # Per-dataset sections
    for ds in all_datasets:
        name = ds['name']
        section_title = f"{name} — {ds['n_samples']:,} samples"
        figures = []

        # Raw primary
        key = f"{name}_raw_primary"
        tsne_2d = tsne_results.get(key)
        dim_pri = ds['raw_primary'].shape[1]
        if tsne_2d is not None:
            figures.append(build_tsne_scatter(tsne_2d, ds['labels'], 'label',
                                              f"Raw Primary ({dim_pri}D) — by Label"))
            figures.append(build_tsne_scatter(tsne_2d, ds['timestamps'], 'time',
                                              f"Raw Primary ({dim_pri}D) — by Time"))
        else:
            figures.extend([None, None])

        # Raw concat (dual)
        if ds['raw_concat'] is not None:
            key = f"{name}_raw_concat"
            tsne_2d = tsne_results.get(key)
            dim_concat = ds['raw_concat'].shape[1]
            if tsne_2d is not None:
                figures.append(build_tsne_scatter(tsne_2d, ds['labels'], 'label',
                                                  f"Raw Pri+Sec ({dim_concat}D) — by Label"))
                figures.append(build_tsne_scatter(tsne_2d, ds['timestamps'], 'time',
                                                  f"Raw Pri+Sec ({dim_concat}D) — by Time"))
            else:
                figures.extend([None, None])

        # Embeddings
        if ds.get('embeddings') is not None:
            key = f"{name}_embeddings"
            tsne_2d = tsne_results.get(key)
            dim_emb = ds['embeddings'].shape[1]
            if tsne_2d is not None:
                figures.append(build_tsne_scatter(tsne_2d, ds['labels'], 'label',
                                                  f"Embeddings ({dim_emb}D) — by Label"))
                figures.append(build_tsne_scatter(tsne_2d, ds['timestamps'], 'time',
                                                  f"Embeddings ({dim_emb}D) — by Time"))
                figures.append(build_tsne_scatter(tsne_2d, ds['confidences'], 'confidence',
                                                  f"Embeddings ({dim_emb}D) — by Confidence"))
            else:
                figures.extend([None, None, None])

        sections.append((section_title, figures))

    # Global section
    global_figures = []

    # Global raw primary
    tsne_2d = tsne_results.get("global_raw_primary")
    if tsne_2d is not None:
        global_figures.append(build_tsne_scatter(tsne_2d, global_labels, 'label',
                                                  f"Global Raw Primary — by Label"))
        global_figures.append(build_tsne_scatter(tsne_2d, global_timestamps, 'time',
                                                  f"Global Raw Primary — by Time"))
        global_figures.append(build_tsne_scatter(tsne_2d, global_dataset_ids, 'dataset',
                                                  f"Global Raw Primary — by Dataset",
                                                  dataset_names=dataset_names))
    else:
        global_figures.extend([None, None, None])

    # Global raw concat
    if has_dual:
        tsne_2d = tsne_results.get("global_raw_concat")
        if tsne_2d is not None:
            # Dataset IDs for concat (only datasets that have concat)
            concat_ds_ids = np.concatenate([
                np.full(len(d['labels']), i)
                for i, d in enumerate(all_datasets) if d['raw_concat'] is not None
            ])
            concat_labels = np.concatenate([
                d['labels'] for d in all_datasets if d['raw_concat'] is not None
            ])
            concat_timestamps = np.concatenate([
                d['timestamps'] for d in all_datasets if d['raw_concat'] is not None
            ])
            global_figures.append(build_tsne_scatter(tsne_2d, concat_labels, 'label',
                                                      f"Global Raw Pri+Sec — by Label"))
            global_figures.append(build_tsne_scatter(tsne_2d, concat_timestamps, 'time',
                                                      f"Global Raw Pri+Sec — by Time"))
            global_figures.append(build_tsne_scatter(tsne_2d, concat_ds_ids, 'dataset',
                                                      f"Global Raw Pri+Sec — by Dataset",
                                                      dataset_names=dataset_names))
        else:
            global_figures.extend([None, None, None])

    # Global embeddings
    if has_emb:
        tsne_2d = tsne_results.get("global_embeddings")
        if tsne_2d is not None:
            emb_ds_ids = np.concatenate([
                np.full(len(d['labels']), i)
                for i, d in enumerate(all_datasets) if d.get('embeddings') is not None
            ])
            emb_labels = np.concatenate([
                d['labels'] for d in all_datasets if d.get('embeddings') is not None
            ])
            emb_timestamps = np.concatenate([
                d['timestamps'] for d in all_datasets if d.get('embeddings') is not None
            ])
            global_figures.append(build_tsne_scatter(tsne_2d, emb_labels, 'label',
                                                      f"Global Embeddings — by Label"))
            global_figures.append(build_tsne_scatter(tsne_2d, emb_timestamps, 'time',
                                                      f"Global Embeddings — by Time"))
            global_figures.append(build_tsne_scatter(tsne_2d, global_confidences, 'confidence',
                                                      f"Global Embeddings — by Confidence"))
            global_figures.append(build_tsne_scatter(tsne_2d, emb_ds_ids, 'dataset',
                                                      f"Global Embeddings — by Dataset",
                                                      dataset_names=dataset_names))
        else:
            global_figures.extend([None, None, None, None])

    n_total = sum(d['n_samples'] for d in all_datasets)
    sections.append((f"Global — {n_total:,} samples", global_figures))

    # Stats
    stats_html = build_stats_table(all_datasets, tsne_results, has_checkpoint)

    # Config info
    config_info = {
        'Perplexity': args.perplexity,
        'Split': args.split,
        'Window Size': config.WINDOW_SIZE,
        'Multi-TF Mode': config.MULTI_TF_MODE,
        'VP': f"lookback={config.VP_LOOKBACK}, bins={config.VP_BINS}",
        'Checkpoint': args.checkpoint or 'None',
        'Horizon': args.horizon if args.horizon is not None else f"{config.MAX_HORIZON} (config)",
        'Mode': f"Binary ({target_name} vs rest)" if args.binary is not None else "3-class",
        'Datasets': len(all_datasets),
        'Total Samples': f"{n_total:,}",
    }

    # Assemble and write HTML
    assemble_html(sections, config_info, stats_html, args.output)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
