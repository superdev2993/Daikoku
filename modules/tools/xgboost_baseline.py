"""
XGBoost Baseline — Reality Check for feature informativeness.

Tests whether the current features contain predictive information
using gradient boosting (architecture-agnostic). If XGBoost finds no edge,
the problem is the features, not the model.

Usage:
    python -m modules.tools.xgboost_baseline
    python -m modules.tools.xgboost_baseline --target bull
    python -m modules.tools.xgboost_baseline --target triple --modes last,stats
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path

import config
from modules.data.loader import get_raw_data
from modules.data.labeling import get_labels, remap_labels, TARGET_CLASS_NAMES
from modules.data.transform import transform_pipeline
from modules.data.dataset import TimeSeriesWindowDataset


EXTRA_FEATURE_NAMES = []  # Populated by compute_extra_features


def _rsi(close, period):
    """Wilder RSI, returns array same length as close, scaled to [0, 1]."""
    n = len(close)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    if len(gain) >= period:
        avg_gain[period] = gain[:period].mean()
        avg_loss[period] = loss[:period].mean()

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = np.nan
    return rsi / 100.0


def _ema(close, period):
    """EMA with warmup NaN."""
    n = len(close)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    ema[period - 1] = close[:period].mean()
    k = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = close[i] * k + ema[i - 1] * (1 - k)
    return ema


def _rolling_hh(high, period):
    """Rolling highest high."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = high[i - period + 1:i + 1].max()
    return out


def _rolling_ll(low, period):
    """Rolling lowest low."""
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = low[i - period + 1:i + 1].min()
    return out


def _safe_log_ratio(a, b):
    """log(a/b) with NaN propagation."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((a > 0) & (b > 0), np.log(a / b), np.nan)


INDICATOR_GROUPS = {
    "rsi": "RSI (7, 14, 28)",
    "ema": "EMA 9/18 (dist + cross)",
    "bb": "Bollinger Bands (20, 2σ)",
    "ichi": "Ichimoku (tenkan, kijun, cloud) — no Chikou (look-ahead)",
}


def compute_extra_features(df_raw, groups=None):
    """
    Compute extra technical indicators on raw OHLCV data.

    Args:
        df_raw: DataFrame with OHLCV columns.
        groups: set of group names to include. None = all.
                Valid: "rsi", "ema", "bb", "ichi"

    Returns:
        (features, names): numpy array (n_rows, n_extra) and list of names.
    """
    if groups is None:
        groups = set(INDICATOR_GROUPS.keys())

    close = df_raw['Close'].values.astype(np.float64)
    high = df_raw['High'].values.astype(np.float64)
    low = df_raw['Low'].values.astype(np.float64)
    n = len(close)

    features = []
    names = []

    # ── RSI (7, 14, 28) ─────────────────────────────────────────────
    if "rsi" in groups:
        for p in [7, 14, 28]:
            features.append(_rsi(close, p))
            names.append(f"rsi_{p}")

    # ── EMA 9, 18 ───────────────────────────────────────────────────
    if "ema" in groups:
        ema9 = _ema(close, 9)
        ema18 = _ema(close, 18)
        features.append(_safe_log_ratio(close, ema9))       # ema_dist_9
        names.append("ema_dist_9")
        features.append(_safe_log_ratio(close, ema18))      # ema_dist_18
        names.append("ema_dist_18")
        features.append(_safe_log_ratio(ema9, ema18))       # ema_cross
        names.append("ema_cross")

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────
    if "bb" in groups:
        bb_period = 20
        bb_std_mult = 2.0
        sma20 = np.full(n, np.nan)
        std20 = np.full(n, np.nan)
        for i in range(bb_period - 1, n):
            window = close[i - bb_period + 1:i + 1]
            sma20[i] = window.mean()
            std20[i] = window.std()

        bb_upper = sma20 + bb_std_mult * std20
        bb_lower = sma20 - bb_std_mult * std20

        features.append(_safe_log_ratio(close, bb_upper))   # bb_dist_upper
        names.append("bb_dist_upper")
        features.append(_safe_log_ratio(close, bb_lower))   # bb_dist_lower
        names.append("bb_dist_lower")
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_width = np.where(sma20 > 0, (bb_upper - bb_lower) / sma20, np.nan)
        features.append(bb_width)                            # bb_width
        names.append("bb_width")

    # ── Ichimoku (sans Chikou — look-ahead bias) ─────────────────────
    if "ichi" in groups:
        # Tenkan-sen (9)
        hh9 = _rolling_hh(high, 9)
        ll9 = _rolling_ll(low, 9)
        tenkan = (hh9 + ll9) / 2.0

        # Kijun-sen (26)
        hh26 = _rolling_hh(high, 26)
        ll26 = _rolling_ll(low, 26)
        kijun = (hh26 + ll26) / 2.0

        # Senkou Span A: (tenkan + kijun) / 2, shifted +26
        senkou_a_raw = (tenkan + kijun) / 2.0
        senkou_a = np.full(n, np.nan)
        senkou_a[26:] = senkou_a_raw[:-26]

        # Senkou Span B: (HH52 + LL52) / 2, shifted +26
        hh52 = _rolling_hh(high, 52)
        ll52 = _rolling_ll(low, 52)
        senkou_b_raw = (hh52 + ll52) / 2.0
        senkou_b = np.full(n, np.nan)
        senkou_b[26:] = senkou_b_raw[:-26]

        features.append(_safe_log_ratio(close, tenkan))      # ichi_dist_tenkan
        names.append("ichi_dist_tenkan")
        features.append(_safe_log_ratio(close, kijun))       # ichi_dist_kijun
        names.append("ichi_dist_kijun")
        features.append(_safe_log_ratio(close, senkou_a))    # ichi_dist_cloud
        names.append("ichi_dist_cloud")
        features.append(_safe_log_ratio(senkou_a, senkou_b)) # ichi_cloud_width
        names.append("ichi_cloud_width")

    if not features:
        # No indicators selected — return empty
        return np.zeros((n, 0), dtype=np.float32), []

    result = np.column_stack(features).astype(np.float32)  # (n, n_extra)
    return result, names


def extract_windows_and_labels(target_override=None, indicator_groups=None):
    """
    Reuse the exact same pipeline as main.py to build windows and labels.

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, class_names, num_classes, target
        Each X is a dict of {mode_name: numpy 2D array}
    """
    prediction_target = target_override or config.PREDICTION_TARGET

    global EXTRA_FEATURE_NAMES

    all_data = []
    all_labels = []
    all_raw_hlc = []
    all_vp = []
    all_extra = []

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

        # Extra technical indicators (computed globally on raw data, aligned)
        extra_full, extra_names = compute_extra_features(df_raw, groups=indicator_groups)
        EXTRA_FEATURE_NAMES = extra_names
        extra_aligned = extra_full[ws:ws + n_data][:min_len]  # (min_len, 14)

        all_data.append(data)
        all_labels.append(labels)
        all_raw_hlc.append(raw_hlc)
        all_vp.append(vp_features)
        all_extra.append(extra_aligned)

    # Concatenate all datasets
    data_cat = np.concatenate(all_data, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    raw_hlc_cat = np.concatenate(all_raw_hlc, axis=0)
    extra_cat = np.concatenate(all_extra, axis=0)  # (total, 14)
    if all_vp[0] is not None:
        vp_cat = np.concatenate(all_vp, axis=0)
    else:
        vp_cat = None

    # Build dataset to reuse exact same windowing + per-window features
    n_valid = len(data_cat)
    n_norm = transform_result.get('n_norm_features', 0)
    split_index = int(n_valid * config.TRAIN_TEST_SPLIT)

    train_indices = list(range(split_index - ws))
    test_indices = list(range(split_index - ws, n_valid - ws))

    dataset = TimeSeriesWindowDataset(
        data=data_cat,
        labels=labels_cat,
        indices=list(range(n_valid - ws)),
        window_size=ws,
        raw_hlc=raw_hlc_cat,
        n_norm_features=n_norm + 1,
        vp_features=vp_cat,
    )

    # Extract all windows as numpy
    print(f"  Extracting {len(dataset)} windows...")
    t0 = time.time()

    windows = []
    window_labels = []
    for i in range(len(dataset)):
        result = dataset[i]
        w, l = result[0], result[1]  # handles (w, l) and (w, l, conf)
        windows.append(w.numpy())
        window_labels.append(l.item())

    windows = np.array(windows)     # (N, ws, n_features)
    window_labels = np.array(window_labels)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {windows.shape}")

    # Inject extra indicators into windows (if any)
    n_extra = extra_cat.shape[1] if extra_cat.shape[1] > 0 else 0
    if n_extra > 0:
        n_windows = len(windows)
        extra_windows = np.zeros((n_windows, ws, n_extra), dtype=np.float32)
        for i in range(n_windows):
            extra_windows[i] = extra_cat[i:i + ws]
        windows = np.concatenate([windows, extra_windows], axis=2)
        print(f"  + Extra indicators injected: {windows.shape} (added {n_extra}: {', '.join(EXTRA_FEATURE_NAMES)})")
    else:
        print(f"  No extra indicators (baseline features only: {windows.shape})")

    # Split
    n_train = len(train_indices)
    X_train_raw = windows[:n_train]
    X_test_raw = windows[n_train:]
    y_train = window_labels[:n_train]
    y_test = window_labels[n_train:]

    class_names = TARGET_CLASS_NAMES[prediction_target]

    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'y_train': y_train,
        'y_test': y_test,
        'class_names': class_names,
        'num_classes': num_classes,
        'target': prediction_target,
        'n_train': n_train,
        'n_test': len(X_test_raw),
    }


def get_feature_names(n_features):
    """Return human-readable names for assembled features."""
    # Assembly order: [5 OHLCV | log_zz | log_atr | 4 time | 5 struct | (5 VP) | 2 regime | extra]
    base = ["log_price", "log_open", "log_wick_hi", "log_wick_lo", "log_volume",
            "log_zz", "log_atr",
            "sin_hour", "cos_hour", "sin_weekday", "cos_weekday",
            "struct_rank", "dist_res1", "dist_res2", "dist_sup1", "dist_sup2"]
    if n_features >= 23 + len(EXTRA_FEATURE_NAMES):
        # VP enabled (23 base features)
        base += ["vp_dist_poc", "vp_dist_vah", "vp_dist_val", "vp_width_va", "vp_skew"]
    base += ["regime_trend", "regime_voltrend"]
    # Extra technical indicators (appended by XGBoost baseline)
    base += EXTRA_FEATURE_NAMES
    return base[:n_features]


def build_tabular_features(X_raw, mode):
    """
    Convert 3D windows (N, ws, F) into 2D tabular features (N, D).

    Modes:
        last: just the last candle (N, F)
        stats: mean/std/min/max/last per feature (N, F*5)
        flatten: full window flattened (N, ws*F)

    Returns:
        tuple: (X_2d, feature_names)
    """
    N, ws, F = X_raw.shape
    base_names = get_feature_names(F)

    if mode == "last":
        names = [f"{n}" for n in base_names]
        return X_raw[:, -1, :], names

    elif mode == "stats":
        last = X_raw[:, -1, :]
        mean = X_raw.mean(axis=1)
        std = X_raw.std(axis=1)
        vmin = X_raw.min(axis=1)
        vmax = X_raw.max(axis=1)
        names = []
        for prefix in ["last", "mean", "std", "min", "max"]:
            names += [f"{prefix}_{n}" for n in base_names]
        return np.concatenate([last, mean, std, vmin, vmax], axis=1), names

    elif mode == "flatten":
        names = [f"t{t}_{n}" for t in range(ws) for n in base_names]
        return X_raw.reshape(N, -1), names

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_xgboost(X_train, X_test, y_train, y_test, num_classes):
    """
    Train XGBoost with early stopping, return predictions and probabilities.
    Uses scale_pos_weight for binary to handle class imbalance.
    """
    import xgboost as xgb

    # Class balance for binary
    spw = None
    if num_classes == 2:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
        'num_class': num_classes if num_classes > 2 else None,
        'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
        'scale_pos_weight': spw if num_classes == 2 else None,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'random_state': config.SEED,
        'verbosity': 0,
    }
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    n_estimators = params.pop('n_estimators')

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dtest, 'test')],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # Predictions
    probs_raw = model.predict(dtest)
    if num_classes == 2:
        # binary: xgb returns P(class=1)
        probs = np.column_stack([1 - probs_raw, probs_raw])
        preds = (probs_raw >= 0.5).astype(int)
    else:
        probs = probs_raw
        preds = probs.argmax(axis=1)

    best_round = model.best_iteration
    return preds, probs, best_round, model


def compute_metrics(y_true, preds, probs, num_classes, class_names, prediction_target="triple"):
    """Compute and return metrics dict."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    metrics = {}
    metrics['accuracy'] = (preds == y_true).mean()
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, preds)

    # Final Accuracy: precision of the actionable class(es)
    if prediction_target == "triple":
        # Directional: among preds Bear(0) or Bull(2), how many correct?
        dir_mask = (preds == 0) | (preds == 2)
        final_count = int(dir_mask.sum())
        final_acc = float((preds[dir_mask] == y_true[dir_mask]).mean()) if final_count > 0 else 0.0
    else:
        # Binary: precision of class 1 (positive = what we detect)
        pos_mask = preds == 1
        final_count = int(pos_mask.sum())
        final_acc = float((y_true[pos_mask] == 1).mean()) if final_count > 0 else 0.0
    metrics['final_accuracy'] = final_acc
    metrics['final_count'] = final_count

    # ROC-AUC (threshold-free metric)
    if num_classes == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, probs[:, 1])
    else:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, probs, multi_class='ovr')
        except ValueError:
            metrics['roc_auc'] = 0.0

    for c in range(num_classes):
        name = class_names[c].lower().replace("-", "")
        mask_pred = preds == c
        mask_true = y_true == c

        tp = ((preds == c) & (y_true == c)).sum()
        precision = tp / mask_pred.sum() if mask_pred.sum() > 0 else 0.0
        recall = tp / mask_true.sum() if mask_true.sum() > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision_{name}'] = precision
        metrics[f'recall_{name}'] = recall
        metrics[f'f1_{name}'] = f1
        metrics[f'support_{name}'] = int(mask_true.sum())

    # High confidence metrics
    for thresh in [0.5, 0.6, 0.7]:
        max_prob = probs.max(axis=1)
        hc_mask = max_prob >= thresh
        if hc_mask.sum() > 0:
            hc_preds = preds[hc_mask]
            hc_true = y_true[hc_mask]
            hc_probs = probs[hc_mask]

            metrics[f'hc{int(thresh*10)}_count'] = int(hc_mask.sum())
            metrics[f'hc{int(thresh*10)}_accuracy'] = (hc_preds == hc_true).mean()

            # Per-class HC
            for c in range(num_classes):
                name = class_names[c].lower().replace("-", "")
                c_mask = hc_preds == c
                if c_mask.sum() > 0:
                    c_acc = (hc_true[c_mask] == c).mean()
                    metrics[f'hc{int(thresh*10)}_{name}_acc'] = c_acc
                    metrics[f'hc{int(thresh*10)}_{name}_n'] = int(c_mask.sum())
                else:
                    metrics[f'hc{int(thresh*10)}_{name}_acc'] = 0.0
                    metrics[f'hc{int(thresh*10)}_{name}_n'] = 0
        else:
            metrics[f'hc{int(thresh*10)}_count'] = 0
            metrics[f'hc{int(thresh*10)}_accuracy'] = 0.0

    return metrics


def print_report(pipeline_data, results):
    """Print the full report."""
    target = pipeline_data['target']
    class_names = pipeline_data['class_names']
    num_classes = pipeline_data['num_classes']

    # Header
    print("\n" + "=" * 72)
    print("  XGBoost Baseline — Reality Check")
    print("=" * 72)
    print(f"  Dataset     : {len(config.INPUT_FILES)} files, WINDOW_SIZE={config.WINDOW_SIZE}")
    print(f"  Target      : {target} ({num_classes} classes)")
    print(f"  Classes     : {class_names}")
    print(f"  Split       : {config.TRAIN_TEST_SPLIT:.0%} train / {1 - config.TRAIN_TEST_SPLIT:.0%} test (chronological)")
    print(f"  Train       : {pipeline_data['n_train']} samples")
    print(f"  Test        : {pipeline_data['n_test']} samples")

    # Label distribution
    y_train = pipeline_data['y_train']
    y_test = pipeline_data['y_test']
    print(f"\n  Label distribution (test):")
    for c in range(num_classes):
        n = (y_test == c).sum()
        pct = n / len(y_test) * 100
        print(f"    {class_names[c]:>12s} : {n:>5d} ({pct:.1f}%)")

    # Identify positive class
    if target == "triple":
        positive_label = "All classes"
    else:
        positive_label = class_names[1]

    base_rate = None
    if num_classes == 2:
        base_rate = (y_test == 1).mean()
        print(f"\n  Base rate ({class_names[1]}): {base_rate:.1%}")
        if target in ("bull", "bear"):
            breakeven = 1 / 3  # R:R = 2:1
            print(f"  Breakeven (R:R 2:1)      : {breakeven:.1%}")

    # Results per mode
    for mode_name, res in results.items():
        m = res['metrics']
        n_feat = res['n_features']
        best_round = res['best_round']

        print(f"\n{'─' * 72}")
        print(f"  Feature Mode: {mode_name} ({n_feat} features, best_round={best_round})")
        print(f"{'─' * 72}")

        # Classification report
        print(f"\n  {'':>12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>10s}")
        for c in range(num_classes):
            name = class_names[c].lower().replace("-", "")
            print(f"  {class_names[c]:>12s}  {m[f'precision_{name}']:>10.4f}  {m[f'recall_{name}']:>10.4f}  {m[f'f1_{name}']:>10.4f}  {m[f'support_{name}']:>10d}")

        roc = m.get('roc_auc', 0)
        print(f"\n  Accuracy: {m['accuracy']:.4f}    Balanced Acc: {m['balanced_accuracy']:.4f}    ROC-AUC: {roc:.4f}")
        fa = m.get('final_accuracy', 0)
        fc = m.get('final_count', 0)
        print(f"  >>> Final_Accuracy: {fa:.4f}  ({fc} predictions)")

        # High confidence
        print(f"\n  High Confidence:")
        for thresh in [5, 6, 7]:
            key_count = f'hc{thresh}_count'
            key_acc = f'hc{thresh}_accuracy'
            if m.get(key_count, 0) > 0:
                line = f"    prob >= 0.{thresh}: {m[key_count]:>5d} preds, acc={m[key_acc]:.4f}"
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

        # Feature importance
        importance = res.get('importance', {})
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_n = min(20, len(sorted_imp))
            total_gain = sum(v for _, v in sorted_imp)
            print(f"\n  Top {top_n} Features (by gain, total={total_gain:.0f}):")
            cumul = 0
            for i, (fname, gain) in enumerate(sorted_imp[:top_n]):
                pct = gain / total_gain * 100
                cumul += pct
                bar = "█" * int(pct / 2)
                print(f"    {i+1:>2d}. {fname:<30s} {gain:>10.1f} ({pct:>5.1f}%) {bar}")
            print(f"        {'Top-' + str(top_n) + ' cumulative':.<30s} {cumul:>17.1f}%")

            # Aggregate by base feature name (for flatten mode)
            if mode_name == "flatten":
                agg = {}
                for fname, gain in sorted_imp:
                    # t42_log_price -> log_price
                    parts = fname.split("_", 1)
                    if parts[0].startswith("t") and parts[0][1:].isdigit():
                        base = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        base = fname
                    agg[base] = agg.get(base, 0) + gain

                sorted_agg = sorted(agg.items(), key=lambda x: x[1], reverse=True)
                print(f"\n  Aggregated by feature type (flatten):")
                for fname, gain in sorted_agg:
                    pct = gain / total_gain * 100
                    bar = "█" * int(pct / 2)
                    print(f"    {fname:<30s} {gain:>10.1f} ({pct:>5.1f}%) {bar}")

                # Aggregate by timestep bucket (for flatten)
                ws = len(set(int(f.split("_")[0][1:]) for f in importance if f.startswith("t") and f.split("_")[0][1:].isdigit()))
                if ws > 0:
                    buckets = {}
                    for fname, gain in sorted_imp:
                        parts = fname.split("_", 1)
                        if parts[0].startswith("t") and parts[0][1:].isdigit():
                            t = int(parts[0][1:])
                            bucket = f"t{t//10*10:02d}-t{t//10*10+9:02d}"
                            buckets[bucket] = buckets.get(bucket, 0) + gain

                    sorted_buckets = sorted(buckets.items())
                    print(f"\n  Importance by timestep bucket (flatten):")
                    for bname, gain in sorted_buckets:
                        pct = gain / total_gain * 100
                        bar = "█" * int(pct)
                        print(f"    {bname:<12s} {pct:>5.1f}% {bar}")

    # Comparison table
    print(f"\n{'=' * 72}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 72}")

    # Key metric based on target
    # Final Accuracy always first in comparison
    key_metrics = [('final_accuracy', 'FINAL ACC')]

    if num_classes == 2:
        pos_name = class_names[1].lower().replace("-", "")
        key_metrics += [
            ('roc_auc', 'ROC-AUC'),
            (f'precision_{pos_name}', f'prec_{class_names[1]}'),
            (f'recall_{pos_name}', f'recall_{class_names[1]}'),
            (f'f1_{pos_name}', f'f1_{class_names[1]}'),
            ('balanced_accuracy', 'bal_acc'),
        ]
    else:
        key_metrics += [
            ('roc_auc', 'ROC-AUC'),
            ('balanced_accuracy', 'bal_acc'),
            ('accuracy', 'accuracy'),
        ]
        for c in range(num_classes):
            name = class_names[c].lower().replace("-", "")
            key_metrics.append((f'precision_{name}', f'prec_{class_names[c]}'))

    modes = list(results.keys())
    header = f"  {'Metric':>20s}"
    for mode in modes:
        header += f"  {mode:>12s}"
    print(header)

    for metric_key, label in key_metrics:
        line = f"  {label:>20s}"
        for mode in modes:
            v = results[mode]['metrics'].get(metric_key, 0)
            line += f"  {v:>12.4f}"
        print(line)

    # Verdict
    print(f"\n{'=' * 72}")
    print("  VERDICT")
    print(f"{'=' * 72}")

    if num_classes == 2:
        pos_name = class_names[1].lower().replace("-", "")
        best_prec = max(results[m]['metrics'][f'precision_{pos_name}'] for m in results)
        best_mode = max(results, key=lambda m: results[m]['metrics'][f'precision_{pos_name}'])

        print(f"  Best {class_names[1]} precision: {best_prec:.4f} ({best_mode})")
        print(f"  Base rate:                {base_rate:.4f}")
        edge = best_prec - base_rate
        print(f"  Edge over base rate:      {edge:+.4f}")

        if best_prec < base_rate + 0.02:
            print(f"\n  >> FEATURES INSUFFICIENT: XGBoost finds no edge.")
            print(f"     The predictive information is not in the current features.")
            print(f"     Changing the neural network architecture will NOT help.")
        elif best_prec < base_rate + 0.05:
            print(f"\n  >> WEAK SIGNAL: XGBoost finds marginal edge ({edge:+.1%}).")
            print(f"     Features contain some information, but very limited.")
        else:
            print(f"\n  >> SIGNAL FOUND: XGBoost finds edge ({edge:+.1%}).")
            print(f"     Features contain predictive information.")
            print(f"     Mamba may be overfitting the temporal sequence.")
    else:
        best_bal = max(results[m]['metrics']['balanced_accuracy'] for m in results)
        random_bal = 1.0 / num_classes
        print(f"  Best balanced accuracy: {best_bal:.4f} (random={random_bal:.4f})")
        if best_bal < random_bal + 0.02:
            print(f"\n  >> FEATURES INSUFFICIENT for {num_classes}-class separation.")
        else:
            print(f"\n  >> SIGNAL FOUND: edge={best_bal - random_bal:+.4f} over random.")

    print(f"{'=' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="XGBoost baseline reality check")
    parser.add_argument("--target", type=str, default=None,
                        help="Override PREDICTION_TARGET (triple/bull/bear/uncertain)")
    parser.add_argument("--modes", type=str, default="last,stats,flatten",
                        help="Comma-separated feature modes (last,stats,flatten)")
    parser.add_argument("--indicators", type=str, default="all",
                        help="Indicator groups: all, none, or comma-separated (rsi,ema,bb,ichi)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Override WINDOW_SIZE (default: from config.py)")
    args = parser.parse_args()

    target = args.target
    modes = [m.strip() for m in args.modes.split(",")]

    # Override window size if requested
    if args.window_size is not None:
        config.WINDOW_SIZE = args.window_size
        print(f"  WINDOW_SIZE overridden to {config.WINDOW_SIZE}")

    # Parse indicator groups
    if args.indicators == "all":
        indicator_groups = None  # all
    elif args.indicators == "none":
        indicator_groups = set()
    else:
        indicator_groups = set(g.strip() for g in args.indicators.split(","))

    print(f"\n  Loading data pipeline...")
    pipeline_data = extract_windows_and_labels(target_override=target, indicator_groups=indicator_groups)

    results = {}
    for mode in modes:
        print(f"\n  Building features: {mode}...")
        X_train, feat_names = build_tabular_features(pipeline_data['X_train_raw'], mode)
        X_test, _ = build_tabular_features(pipeline_data['X_test_raw'], mode)

        # Replace NaN/Inf (XGBoost handles them but cleaner this way)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  Training XGBoost ({mode}: {X_train.shape[1]} features)...")
        t0 = time.time()
        preds, probs, best_round, model = run_xgboost(
            X_train, X_test,
            pipeline_data['y_train'], pipeline_data['y_test'],
            pipeline_data['num_classes'],
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s (best_round={best_round})")

        metrics = compute_metrics(
            pipeline_data['y_test'], preds, probs,
            pipeline_data['num_classes'], pipeline_data['class_names'],
            prediction_target=pipeline_data['target'],
        )

        # Feature importance
        importance_raw = model.get_score(importance_type='gain')
        # Map fX index to feature name
        importance = {}
        for fkey, gain in importance_raw.items():
            fidx = int(fkey.replace('f', ''))
            fname = feat_names[fidx] if fidx < len(feat_names) else fkey
            importance[fname] = gain

        results[mode] = {
            'metrics': metrics,
            'n_features': X_train.shape[1],
            'best_round': best_round,
            'importance': importance,
        }

    print_report(pipeline_data, results)


if __name__ == "__main__":
    main()
