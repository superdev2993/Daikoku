"""
MLP Cascade-Correlation Baseline — Architecture-agnostic test.

Tests whether a growing MLP (Cascade-Correlation with frozen blocks)
rivals Mamba on the same data. Answers the question: is the predictive
signal fundamentally sequential or does a non-recurrent model suffice?

Supports two training modes:
  - Standard (default): trains new block + output jointly with CrossEntropy
  - Residual correlation (--residual-loss): Fahlman 1989 original algorithm.
    New block is trained to maximize |correlation(output, residual error)|.
    Each neuron explicitly targets what the current network gets wrong.

Block size modes:
  - Fixed (default): all blocks have --block-size neurons
  - Growing (--growing): sizes double each step (2→4→8→16→32→64...)

Usage:
    python -m modules.tools.mlp_cascade_baseline
    python -m modules.tools.mlp_cascade_baseline --residual-loss --growing --modes last,stats
    python -m modules.tools.mlp_cascade_baseline --residual-loss --block-size 1 --max-steps 30
    python -m modules.tools.mlp_cascade_baseline --target bull --residual-loss
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path

import config
from modules.data.loader import get_raw_data
from modules.data.labeling import get_labels, remap_labels, TARGET_CLASS_NAMES
from modules.data.transform import transform_pipeline
from modules.data.dataset import TimeSeriesWindowDataset


# ============================================================================
# Data Pipeline
# ============================================================================

def build_pipeline(target_override=None):
    """
    Reuse the exact same pipeline as main.py to build the dataset.

    Returns dataset + metadata WITHOUT materializing all windows in RAM.
    """
    prediction_target = target_override or config.PREDICTION_TARGET

    all_data = []
    all_labels = []
    all_raw_hlc = []
    all_vp = []

    ws = config.WINDOW_SIZE
    num_classes = None

    for csv_path in config.INPUT_FILES:
        df_raw = get_raw_data(csv_path)
        labels, atr_array, _confidence, _, _ = get_labels(
            df_raw,
            window_size=config.WINDOW_SIZE,
            atr_multiplier_tp=config.ATR_MULTIPLIER_TP,
            atr_multiplier_sl=config.ATR_MULTIPLIER_SL,
            max_horizon=config.MAX_HORIZON,
            atr_period=config.ATR_PERIOD,
        )

        labels, num_classes = remap_labels(labels, prediction_target)

        transform_result = transform_pipeline(
            df_raw,
            atr_array=atr_array,
            split_ratio=config.TRAIN_TEST_SPLIT,
            window_size=config.WINDOW_SIZE,
        )
        data = transform_result['data']

        n_data = len(data)
        raw_hlc = np.column_stack([
            df_raw['High'].values[ws:ws + n_data],
            df_raw['Low'].values[ws:ws + n_data],
            df_raw['Close'].values[ws:ws + n_data],
        ]).astype(np.float64)

        # Align
        min_len = min(len(data), len(labels))
        data = data[:min_len]
        labels = labels[:min_len]
        raw_hlc = raw_hlc[:min_len]

        # Volume Profile
        from modules.data.multi_tf import compute_branch_vp
        vp_config = {
            'VP_LOOKBACK': config.VP_LOOKBACK,
            'VP_BINS': config.VP_BINS, 'VP_VA_PCT': config.VP_VA_PCT,
        }
        vp_features = compute_branch_vp(df_raw, vp_config, ws, min_len)

        all_data.append(data)
        all_labels.append(labels)
        all_raw_hlc.append(raw_hlc)
        all_vp.append(vp_features)

    # Concatenate
    data_cat = np.concatenate(all_data, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    raw_hlc_cat = np.concatenate(all_raw_hlc, axis=0)
    if all_vp[0] is not None:
        vp_cat = np.concatenate(all_vp, axis=0)
    else:
        vp_cat = None

    # Build dataset for per-window features
    n_valid = len(data_cat)
    n_norm = transform_result.get('n_norm_features', 0)
    split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

    n_train = split_index - ws
    n_total = n_valid - ws

    dataset = TimeSeriesWindowDataset(
        data=data_cat,
        labels=labels_cat,
        indices=list(range(n_total)),
        window_size=ws,
        raw_hlc=raw_hlc_cat,
        n_norm_features=n_norm + 1,
        vp_features=vp_cat,
    )

    class_names = TARGET_CLASS_NAMES[prediction_target]

    return {
        'dataset': dataset,
        'n_train': n_train,
        'n_total': n_total,
        'class_names': class_names,
        'num_classes': num_classes,
        'target': prediction_target,
    }


def extract_tabular_for_mode(dataset, n_train, mode):
    """
    Stream through dataset, compute tabular features per window on-the-fly.
    Never stores full 3D windows in RAM — only the compact 2D result.

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
    """
    n_total = len(dataset)

    # Probe first window to get feature dimensions
    w0 = dataset[0][0].numpy()  # (ws, F)
    ws, F = w0.shape

    if mode == "last":
        feat_dim = F
    elif mode == "stats":
        feat_dim = F * 5
    elif mode == "flatten":
        feat_dim = ws * F
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ram_mb = n_total * feat_dim * 4 / (1024 ** 2)
    print(f"  Streaming {n_total} windows → {mode} ({feat_dim} features, ~{ram_mb:.0f} MB)...")
    t0 = time.time()

    # Pre-allocate output arrays (no intermediate 3D storage)
    X = np.empty((n_total, feat_dim), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int64)

    for i in range(n_total):
        result = dataset[i]
        w = result[0].numpy()  # (ws, F)
        y[i] = result[1].item()

        if mode == "last":
            X[i] = w[-1, :]
        elif mode == "stats":
            X[i] = np.concatenate([
                w[-1, :],        # last
                w.mean(axis=0),  # mean
                w.std(axis=0),   # std
                w.min(axis=0),   # min
                w.max(axis=0),   # max
            ])
        elif mode == "flatten":
            X[i] = w.reshape(-1)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


# ============================================================================
# Cascade-Correlation Network
# ============================================================================

class CascadeBlock(nn.Module):
    """A hidden block that will be frozen after training."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class CascadeCorrelationNet(nn.Module):
    """
    Cascade-Correlation network: grows by adding frozen hidden blocks.

    Each new block sees original features + all previous blocks' outputs.
    Only the newest block or the output layer is trained at any time.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # Output layer: initially maps raw features to classes
        self.output = nn.Linear(input_dim, num_classes)

    def _current_dim(self):
        """Total feature dimension = original + all block outputs."""
        return self.input_dim + sum(b.output_dim for b in self.blocks)

    def add_block(self, hidden_dim=16):
        """Add a new trainable block. Input = features + all previous block outputs.
        Does NOT touch the output layer — caller must rebuild it."""
        current_dim = self._current_dim()
        new_block = CascadeBlock(current_dim, hidden_dim)
        self.blocks.append(new_block)

    def rebuild_output(self, device=None):
        """Recreate output layer to match current total dimension."""
        new_dim = self._current_dim()
        self.output = nn.Linear(new_dim, self.num_classes)
        if device is not None:
            self.output = self.output.to(device)

    def freeze_last_block(self):
        """Freeze the last added block (requires_grad=False)."""
        if len(self.blocks) > 0:
            for p in self.blocks[-1].parameters():
                p.requires_grad = False

    def forward(self, x):
        """x = original features. Pass through all blocks, concat, classify."""
        features = [x]
        for block in self.blocks:
            block_input = torch.cat(features, dim=1)
            block_out = block(block_input)
            features.append(block_out)
        combined = torch.cat(features, dim=1)
        return self.output(combined)


# ============================================================================
# Residual Correlation Loss (Fahlman 1989)
# ============================================================================

def compute_residual_correlation_loss(block_output, residual_error):
    """
    Fahlman's Cascade-Correlation objective: maximize |correlation|
    between new block's output and residual error.

    S = sum_outputs sum_samples |( V_p - V_mean ) * ( E_p,o - E_mean_o )|

    We minimize -S to maximize correlation.

    Args:
        block_output: (batch, hidden_dim) — new block's activations
        residual_error: (batch, num_classes) — target - current_prediction

    Returns:
        scalar loss (negative correlation, to minimize)
    """
    # Center both
    V = block_output - block_output.mean(dim=0, keepdim=True)   # (batch, H)
    E = residual_error - residual_error.mean(dim=0, keepdim=True)  # (batch, C)

    # Correlation: for each hidden unit h and each output o,
    # corr(h,o) = sum_samples V[:,h] * E[:,o]
    # We want to maximize the sum of |corr| across all (h,o) pairs
    corr_matrix = V.T @ E  # (H, C)
    S = corr_matrix.abs().sum()

    # Normalize by batch size for stability
    S = S / block_output.shape[0]

    # Minimize negative correlation
    return -S


# ============================================================================
# Cascade Training Loop
# ============================================================================

def _get_block_size_for_step(step, block_size, growing):
    """Compute block size for a given step."""
    if growing:
        # Growing: 2, 4, 8, 16, 32, 64, 128...
        return 2 ** (step)  # step 1→2, step 2→4, step 3→8, ...
    return block_size


def _compute_frozen_features(model, x):
    """Forward through all FROZEN blocks only, return concatenated features.
    Excludes the last block (which is the one being trained)."""
    features = [x]
    for block in model.blocks[:-1]:
        block_input = torch.cat(features, dim=1)
        block_out = block(block_input)
        features.append(block_out)
    return torch.cat(features, dim=1)


def run_cascade(X_train, X_test, y_train, y_test, num_classes,
                block_size=16, max_steps=10, epochs_per_block=50,
                epochs_output=30, patience=2, lr=0.001,
                residual_loss=False, growing=False):
    """
    Full Cascade-Correlation training procedure.

    Args:
        residual_loss: if True, use Fahlman's correlation with residual error
                       to train new blocks (instead of joint CrossEntropy).
        growing: if True, block sizes double each step (2→4→8→16→32...).

    Returns:
        preds, probs, history (list of dicts per step)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    y_te = torch.tensor(y_test, dtype=torch.long, device=device)

    # One-hot targets for residual computation
    y_tr_onehot = torch.zeros(len(y_train), num_classes, device=device)
    y_tr_onehot.scatter_(1, y_tr.unsqueeze(1), 1.0)

    input_dim = X_train.shape[1]
    model = CascadeCorrelationNet(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    mode_str = "residual-corr" if residual_loss else "joint-CE"
    size_str = "growing" if growing else f"fixed-{block_size}"
    print(f"    Mode: {mode_str}, blocks: {size_str}")

    history = []

    # ── Step 0: train output layer only (linear baseline) ──────────────
    optimizer = torch.optim.AdamW(model.output.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(epochs_output):
        model.train()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate step 0
    model.eval()
    with torch.no_grad():
        logits_test = model(X_te)
        preds_t = logits_test.argmax(dim=1)
        acc = (preds_t == y_te).float().mean().item()
        test_loss = criterion(logits_test, y_te).item()

    history.append({
        'step': 0, 'n_blocks': 0, 'block_size': 0,
        'total_params': sum(p.numel() for p in model.parameters()),
        'test_loss': test_loss, 'test_acc': acc,
    })
    print(f"    Step 0 (linear): test_acc={acc:.4f}  test_loss={test_loss:.4f}")

    best_acc = acc
    no_improve = 0

    # ── Steps 1..max_steps: add blocks ─────────────────────────────────
    for step in range(1, max_steps + 1):
        step_block_size = _get_block_size_for_step(step, block_size, growing)

        # 1. Add new block + rebuild output
        model.add_block(hidden_dim=step_block_size)
        model.blocks[-1].to(device)
        model.rebuild_output(device=device)

        if residual_loss:
            # ── Fahlman: train block to maximize correlation with residual ──

            # Compute residual BEFORE adding the block's influence.
            # Use frozen features (all previous blocks) + a temp output to get
            # the current network's best prediction error.
            with torch.no_grad():
                frozen_feats = _compute_frozen_features(model, X_tr)

            # Train a temporary output on frozen features to get current best prediction
            temp_output = nn.Linear(frozen_feats.shape[1], num_classes).to(device)
            temp_optim = torch.optim.AdamW(temp_output.parameters(), lr=lr * 5)
            for _e in range(20):
                logits_tmp = temp_output(frozen_feats.detach())
                l = criterion(logits_tmp, y_tr)
                temp_optim.zero_grad()
                l.backward()
                temp_optim.step()

            with torch.no_grad():
                current_probs = torch.softmax(temp_output(frozen_feats), dim=1)
                residual = y_tr_onehot - current_probs  # (batch, C) — fixed target
            del temp_output, temp_optim  # free memory

            # Freeze everything except the new block
            for block in model.blocks[:-1]:
                for p in block.parameters():
                    p.requires_grad = False
            for p in model.output.parameters():
                p.requires_grad = False
            for p in model.blocks[-1].parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.blocks[-1].parameters(), lr=lr, weight_decay=0.01,
            )

            for epoch in range(epochs_per_block):
                model.train()

                # Forward through frozen blocks only (no grad), then new block (with grad)
                with torch.no_grad():
                    block_input = _compute_frozen_features(model, X_tr)
                new_block_out = model.blocks[-1](block_input)

                # Maximize |correlation(block_output, residual)|
                loss = compute_residual_correlation_loss(new_block_out, residual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        else:
            # ── Standard: train new block + output jointly with CrossEntropy ──
            for block in model.blocks[:-1]:
                for p in block.parameters():
                    p.requires_grad = False
            for p in model.blocks[-1].parameters():
                p.requires_grad = True
            for p in model.output.parameters():
                p.requires_grad = True

            trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

            for epoch in range(epochs_per_block):
                model.train()
                logits = model(X_tr)
                loss = criterion(logits, y_tr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 3. Freeze the new block
        model.freeze_last_block()

        # 4. Retrain output layer (all blocks frozen, CrossEntropy)
        for block in model.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in model.output.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(model.output.parameters(), lr=lr, weight_decay=0.01)

        for epoch in range(epochs_output):
            model.train()
            logits = model(X_tr)
            loss = criterion(logits, y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 5. Evaluate
        model.eval()
        with torch.no_grad():
            logits_test = model(X_te)
            preds_t = logits_test.argmax(dim=1)
            acc = (preds_t == y_te).float().mean().item()
            test_loss = criterion(logits_test, y_te).item()

        total_params = sum(p.numel() for p in model.parameters())
        history.append({
            'step': step, 'n_blocks': step, 'block_size': step_block_size,
            'total_params': total_params,
            'test_loss': test_loss, 'test_acc': acc,
        })
        print(f"    Step {step} ({step_block_size}n, {total_params} params): "
              f"test_acc={acc:.4f}  test_loss={test_loss:.4f}")

        # 6. Early stopping on accuracy
        if acc > best_acc + 1e-4:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    Early stop: no improvement for {patience} steps")
            break

    # Final predictions
    model.eval()
    with torch.no_grad():
        logits_final = model(X_te)
        probs_final = torch.softmax(logits_final, dim=1).cpu().numpy()
        preds_final = logits_final.argmax(dim=1).cpu().numpy()

    return preds_final, probs_final, history


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true, preds, probs, num_classes, class_names, prediction_target="triple"):
    """Compute classification metrics with directional accuracy."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    metrics = {}
    metrics['accuracy'] = float((preds == y_true).mean())
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, preds))

    # Directional accuracy: among predictions that are actionable, how many correct?
    if prediction_target == "triple":
        # Actionable = Bear(0) or Bull(2), exclude Uncertain(1) predictions
        dir_mask = (preds == 0) | (preds == 2)
        dir_count = int(dir_mask.sum())
        if dir_count > 0:
            dir_acc = float((preds[dir_mask] == y_true[dir_mask]).mean())
        else:
            dir_acc = 0.0
        # Per-direction breakdown
        bear_pred_mask = preds == 0
        bull_pred_mask = preds == 2
        bear_pred_count = int(bear_pred_mask.sum())
        bull_pred_count = int(bull_pred_mask.sum())
        bear_pred_acc = float((y_true[bear_pred_mask] == 0).mean()) if bear_pred_count > 0 else 0.0
        bull_pred_acc = float((y_true[bull_pred_mask] == 2).mean()) if bull_pred_count > 0 else 0.0
        metrics['dir_bear_acc'] = bear_pred_acc
        metrics['dir_bear_count'] = bear_pred_count
        metrics['dir_bull_acc'] = bull_pred_acc
        metrics['dir_bull_count'] = bull_pred_count
    else:
        # Binary: precision of class 1 (positive = what we detect)
        pos_mask = preds == 1
        dir_count = int(pos_mask.sum())
        dir_acc = float((y_true[pos_mask] == 1).mean()) if dir_count > 0 else 0.0
    metrics['final_accuracy'] = dir_acc
    metrics['final_count'] = dir_count

    # ROC-AUC
    if num_classes == 2:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, probs, multi_class='ovr'))
        except ValueError:
            metrics['roc_auc'] = 0.0

    # Per-class precision / recall / F1
    for c in range(num_classes):
        name = class_names[c].lower().replace("-", "")
        mask_pred = preds == c
        mask_true = y_true == c

        tp = int(((preds == c) & (y_true == c)).sum())
        precision = tp / mask_pred.sum() if mask_pred.sum() > 0 else 0.0
        recall = tp / mask_true.sum() if mask_true.sum() > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision_{name}'] = float(precision)
        metrics[f'recall_{name}'] = float(recall)
        metrics[f'f1_{name}'] = float(f1)
        metrics[f'support_{name}'] = int(mask_true.sum())

    # High confidence metrics
    for thresh in [0.5, 0.6, 0.7]:
        max_prob = probs.max(axis=1)
        hc_mask = max_prob >= thresh
        hc_count = int(hc_mask.sum())

        if hc_count > 0:
            hc_preds = preds[hc_mask]
            hc_true = y_true[hc_mask]

            metrics[f'hc{int(thresh*10)}_count'] = hc_count
            metrics[f'hc{int(thresh*10)}_accuracy'] = float((hc_preds == hc_true).mean())

            # HC directional accuracy (triple mode)
            if prediction_target == "triple":
                hc_dir_mask = (hc_preds == 0) | (hc_preds == 2)
                hc_dir_count = int(hc_dir_mask.sum())
                if hc_dir_count > 0:
                    hc_dir_acc = float((hc_preds[hc_dir_mask] == hc_true[hc_dir_mask]).mean())
                else:
                    hc_dir_acc = 0.0
                metrics[f'hc{int(thresh*10)}_dir_acc'] = hc_dir_acc
                metrics[f'hc{int(thresh*10)}_dir_count'] = hc_dir_count

            # Per-class HC
            for c in range(num_classes):
                name = class_names[c].lower().replace("-", "")
                c_mask = hc_preds == c
                c_count = int(c_mask.sum())
                if c_count > 0:
                    c_acc = float((hc_true[c_mask] == c).mean())
                    metrics[f'hc{int(thresh*10)}_{name}_acc'] = c_acc
                    metrics[f'hc{int(thresh*10)}_{name}_n'] = c_count
                else:
                    metrics[f'hc{int(thresh*10)}_{name}_acc'] = 0.0
                    metrics[f'hc{int(thresh*10)}_{name}_n'] = 0
        else:
            metrics[f'hc{int(thresh*10)}_count'] = 0
            metrics[f'hc{int(thresh*10)}_accuracy'] = 0.0
            if prediction_target == "triple":
                metrics[f'hc{int(thresh*10)}_dir_acc'] = 0.0
                metrics[f'hc{int(thresh*10)}_dir_count'] = 0

    return metrics


# ============================================================================
# Report
# ============================================================================

def print_report(pipeline_data, results, residual_loss=False, growing=False):
    """Print the full report with cascade progression."""
    target = pipeline_data['target']
    class_names = pipeline_data['class_names']
    num_classes = pipeline_data['num_classes']

    # Header
    mode_label = "Residual-Correlation" if residual_loss else "Joint-CrossEntropy"
    size_label = "Growing (2→4→8→...)" if growing else "Fixed"
    print("\n" + "=" * 72)
    print(f"  MLP Cascade-Correlation Baseline — {mode_label}, {size_label}")
    print("=" * 72)
    print(f"  Dataset     : {len(config.INPUT_FILES)} files, WINDOW_SIZE={config.WINDOW_SIZE}")
    print(f"  Target      : {target} ({num_classes} classes)")
    print(f"  Classes     : {class_names}")
    print(f"  Split       : {config.TRAIN_TEST_SPLIT:.0%} train / {1 - config.TRAIN_TEST_SPLIT:.0%} test (chronological)")
    print(f"  Train       : {pipeline_data['n_train']} samples")
    print(f"  Test        : {pipeline_data['n_test']} samples")

    # Label distribution
    y_test = pipeline_data['y_test']
    print(f"\n  Label distribution (test):")
    for c in range(num_classes):
        n = (y_test == c).sum()
        pct = n / len(y_test) * 100
        print(f"    {class_names[c]:>12s} : {n:>5d} ({pct:.1f}%)")

    random_acc = 1.0 / num_classes

    # Results per mode
    for mode_name, res in results.items():
        m = res['metrics']
        n_feat = res['n_features']
        history = res['history']

        print(f"\n{'─' * 72}")
        print(f"  Feature Mode: {mode_name} ({n_feat} features)")
        print(f"{'─' * 72}")

        # Cascade progression
        print(f"\n  Cascade Progression:")
        print(f"  {'Step':>5s}  {'BlkSz':>6s}  {'Params':>8s}  {'Test Acc':>9s}  {'Test Loss':>10s}")
        for h in history:
            bsz = h.get('block_size', 0)
            bsz_str = f"{bsz}" if bsz > 0 else "—"
            print(f"  {h['step']:>5d}  {bsz_str:>6s}  {h['total_params']:>8d}  "
                  f"{h['test_acc']:>9.4f}  {h['test_loss']:>10.4f}")

        # Classification report
        print(f"\n  {'':>12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
        for c in range(num_classes):
            name = class_names[c].lower().replace("-", "")
            print(f"  {class_names[c]:>12s}  {m[f'precision_{name}']:>10.4f}  "
                  f"{m[f'recall_{name}']:>10.4f}  {m[f'f1_{name}']:>10.4f}  "
                  f"{m[f'support_{name}']:>10d}")

        roc = m.get('roc_auc', 0)
        print(f"\n  Accuracy: {m['accuracy']:.4f}    Balanced Acc: {m['balanced_accuracy']:.4f}    ROC-AUC: {roc:.4f}")

        # Directional accuracy (the key metric)
        fa = m.get('final_accuracy', 0)
        fc = m.get('final_count', 0)
        print(f"  >>> Directional Accuracy: {fa:.4f}  ({fc} directional predictions)")

        if target == "triple":
            db_acc = m.get('dir_bear_acc', 0)
            db_n = m.get('dir_bear_count', 0)
            du_acc = m.get('dir_bull_acc', 0)
            du_n = m.get('dir_bull_count', 0)
            print(f"      Bear precision: {db_acc:.4f} ({db_n} preds)    "
                  f"Bull precision: {du_acc:.4f} ({du_n} preds)")

        # High confidence
        print(f"\n  High Confidence:")
        for thresh in [5, 6, 7]:
            key_count = f'hc{thresh}_count'
            key_acc = f'hc{thresh}_accuracy'
            if m.get(key_count, 0) > 0:
                line = f"    prob >= 0.{thresh}: {m[key_count]:>5d} preds, acc={m[key_acc]:.4f}"
                # Directional HC
                if target == "triple":
                    hc_dir_acc = m.get(f'hc{thresh}_dir_acc', 0)
                    hc_dir_count = m.get(f'hc{thresh}_dir_count', 0)
                    line += f"  dir_acc={hc_dir_acc:.4f}({hc_dir_count})"
                # Per-class detail
                parts = []
                for c in range(num_classes):
                    name = class_names[c].lower().replace("-", "")
                    k_acc = f'hc{thresh}_{name}_acc'
                    k_n = f'hc{thresh}_{name}_n'
                    if m.get(k_n, 0) > 0:
                        parts.append(f"{class_names[c]}={m[k_acc]:.1%}({m[k_n]})")
                line += "  [" + ", ".join(parts) + "]"
                print(line)
            else:
                print(f"    prob >= 0.{thresh}: 0 preds")

    # Comparison table
    print(f"\n{'=' * 72}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 72}")

    key_metrics = [('final_accuracy', 'DIR ACC'), ('final_count', 'DIR COUNT')]

    if target == "triple":
        key_metrics += [
            ('dir_bear_acc', 'Bear prec'),
            ('dir_bull_acc', 'Bull prec'),
            ('roc_auc', 'ROC-AUC'),
            ('balanced_accuracy', 'bal_acc'),
            ('accuracy', 'accuracy'),
        ]
    elif num_classes == 2:
        pos_name = class_names[1].lower().replace("-", "")
        key_metrics += [
            ('roc_auc', 'ROC-AUC'),
            (f'precision_{pos_name}', f'prec_{class_names[1]}'),
            (f'recall_{pos_name}', f'recall_{class_names[1]}'),
            ('balanced_accuracy', 'bal_acc'),
        ]

    modes = list(results.keys())
    header = f"  {'Metric':>20s}"
    for mode in modes:
        header += f"  {mode:>12s}"
    print(header)

    for metric_key, label in key_metrics:
        line = f"  {label:>20s}"
        for mode in modes:
            v = results[mode]['metrics'].get(metric_key, 0)
            if metric_key == 'final_count' or metric_key.endswith('_count'):
                line += f"  {int(v):>12d}"
            else:
                line += f"  {v:>12.4f}"
        print(line)

    # Verdict
    print(f"\n{'=' * 72}")
    print("  VERDICT")
    print(f"{'=' * 72}")

    best_dir = max(results[m]['metrics'].get('final_accuracy', 0) for m in results)
    best_mode = max(results, key=lambda m: results[m]['metrics'].get('final_accuracy', 0))
    best_bal = max(results[m]['metrics']['balanced_accuracy'] for m in results)

    print(f"  Best directional accuracy: {best_dir:.4f} ({best_mode})")
    print(f"  Best balanced accuracy:    {best_bal:.4f} (random={random_acc:.4f})")

    if best_bal < random_acc + 0.02:
        print(f"\n  >> MLP FINDS NO EDGE: features insufficient for {num_classes}-class.")
    elif best_dir < 0.40:
        print(f"\n  >> WEAK DIRECTIONAL SIGNAL: MLP directional acc < 40%.")
        print(f"     Signal may require sequence modeling (Mamba advantage).")
    else:
        print(f"\n  >> MLP FINDS DIRECTIONAL EDGE: dir_acc={best_dir:.4f}")
        print(f"     If Mamba doesn't beat this, the signal is NOT sequential.")

    # Best cascade step analysis
    for mode_name, res in results.items():
        h = res['history']
        accs = [s['test_acc'] for s in h]
        best_step = h[np.argmax(accs)]['step']
        if best_step == 0:
            print(f"  [{mode_name}] Best at step 0 (linear) — hidden layers don't help.")
        elif best_step <= 2:
            print(f"  [{mode_name}] Best at step {best_step} — shallow nonlinearity sufficient.")
        else:
            print(f"  [{mode_name}] Best at step {best_step} — model benefits from depth.")

    print(f"{'=' * 72}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLP Cascade-Correlation baseline")
    parser.add_argument("--target", type=str, default=None,
                        help="Override PREDICTION_TARGET (triple/bull/bear/uncertain)")
    parser.add_argument("--modes", type=str, default="last,stats,flatten",
                        help="Comma-separated feature modes (last,stats,flatten)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Hidden neurons per cascade block (default: 16)")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum cascade blocks to add (default: 10)")
    parser.add_argument("--epochs-block", type=int, default=50,
                        help="Epochs to train each new block (default: 50)")
    parser.add_argument("--epochs-output", type=int, default=30,
                        help="Epochs to retrain output layer (default: 30)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Steps without improvement before stopping (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--residual-loss", action="store_true",
                        help="Use Fahlman residual correlation loss (default: joint CrossEntropy)")
    parser.add_argument("--growing", action="store_true",
                        help="Growing block sizes: 2→4→8→16→32... (default: fixed)")
    args = parser.parse_args()

    target = args.target
    modes = [m.strip() for m in args.modes.split(",")]

    print(f"\n  Loading data pipeline...")
    pipeline_data = build_pipeline(target_override=target)
    dataset = pipeline_data['dataset']
    n_train = pipeline_data['n_train']

    # y_test needed for report — extract from first mode run
    y_test_ref = None

    results = {}
    for mode in modes:
        print(f"\n  Extracting features: {mode}...")
        X_train, X_test, y_train, y_test = extract_tabular_for_mode(
            dataset, n_train, mode,
        )

        if y_test_ref is None:
            y_test_ref = y_test

        # Replace NaN/Inf
        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  Training Cascade MLP ({mode}: {X_train.shape[1]} features)...")
        t0 = time.time()
        preds, probs, history = run_cascade(
            X_train, X_test,
            y_train, y_test,
            pipeline_data['num_classes'],
            block_size=args.block_size,
            max_steps=args.max_steps,
            epochs_per_block=args.epochs_block,
            epochs_output=args.epochs_output,
            patience=args.patience,
            lr=args.lr,
            residual_loss=args.residual_loss,
            growing=args.growing,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({len(history)} steps)")

        metrics = compute_metrics(
            y_test, preds, probs,
            pipeline_data['num_classes'], pipeline_data['class_names'],
            prediction_target=pipeline_data['target'],
        )

        results[mode] = {
            'metrics': metrics,
            'n_features': X_train.shape[1],
            'history': history,
        }

        # Free memory between modes
        del X_train, X_test

    # Build report-compatible dict
    pipeline_data['y_test'] = y_test_ref
    pipeline_data['n_test'] = len(y_test_ref)

    print_report(pipeline_data, results,
                 residual_loss=args.residual_loss, growing=args.growing)


if __name__ == "__main__":
    main()
