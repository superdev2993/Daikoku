# TensorBoard Metrics Reference

*[Version francaise](Metrics_TB_FR.md)*

All metrics logged to TensorBoard during training. Classes: Bear=0, Uncertain=1, Bull=2.

---

## Loss & Accuracy (base)

- `Loss/train` — `mean(CrossEntropyLoss)` on training batches. Measures model's ability to fit training data.
- `Loss/test` — `mean(CrossEntropyLoss)` on test set. Measures generalization; divergence from train = overfitting.
- `Accuracy/train` — `correct / total` on train. Overall training accuracy.
- `Accuracy/test` — `correct / total` on test. Overall test accuracy.

## Training per-class

- `Train/Balanced_Accuracy` — `mean(recall per class)` (sklearn) on train. Class-imbalance-corrected accuracy.
- `Train/F1_Class_{0,1,2}` — `2 * P * R / (P + R)` per class, train. Per-class F1 to detect prediction bias.
- `Train/Precision_Class_{0,1,2}` — `TP / (TP + FP)` per class, train. Correct prediction ratio per class.
- `Train/Recall_Class_{0,1,2}` — `TP / (TP + FN)` per class, train. Detection coverage per class.
- `Training/learning_rate` — `optimizer.param_groups[0]['lr']`. Current LR tracking (scheduler, warmup).

## Cascade Training

- `Cascade/trainable_params` — `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Trainable parameter count (increases progressively in cascade mode).
- `Cascade/active_depth` — `min(epoch + 1, n_layers_primary)`. Number of unfrozen Mamba blocks at current epoch.

## Overall Performance (test)

Prefix: `Metrics/`

- `accuracy` — `correct / total` (sklearn accuracy_score). Overall test set accuracy.
- `balanced_accuracy` — `mean(recall per class)` (sklearn balanced_accuracy_score). Class-imbalance-corrected accuracy.
- `f1_macro` — `mean(F1 per class)`. Mean F1 across all classes, main generalization metric.
- `precision_macro` — `mean(Precision per class)`. Average precision across 3 classes.
- `recall_macro` — `mean(Recall per class)`. Average recall across 3 classes.

## Per-class (test)

Prefix: `Metrics/`

- `precision_class_{0,1,2}` — `TP / (TP + FP)` for class i. Prediction reliability per class on test.
- `recall_class_{0,1,2}` — `TP / (TP + FN)` for class i. Detection coverage per class on test.
- `f1_class_{0,1,2}` — `2 * P * R / (P + R)` for class i. Precision/recall balance per class on test.

## Confidence

Prefix: `Metrics/`

- `entropy_mean` — `mean(scipy.entropy(probs.T))`. Average prediction uncertainty; low = decisive model.
- `confidence_correct_mean` — `mean(max(probs))` on correct predictions. Average confidence when the model is right.
- `confidence_incorrect_mean` — `mean(max(probs))` on misclassified predictions. Average confidence when wrong; gap with correct = calibration quality.

## Overfitting Gaps

Prefix: `Metrics/`

- `gap_loss` — `test_loss - train_loss`. Loss gap; positive = overfitting, increasing = model memorizing.
- `gap_accuracy` — `train_acc - test_acc`. Accuracy gap; positive = overfitting.

## Final Accuracy (actionable predictions)

Prefix: `Metrics/`

- `final_accuracy` — `(Bear correct + Bull correct) / (total Bear + Bull preds)`. Directional prediction accuracy (excludes Uncertain).
- `final_accuracy_count` — Number of Bear + Bull predictions. Actionable prediction volume; too low = overly conservative model.

## High Confidence Directional

Prefix: `Metrics/highconf_`

Thresholds: `{05, 06, 07}` = confidence `max(probs) >= {0.5, 0.6, 0.7}`

- `{T}_bear_acc` — `(true == 0).mean()` among Bear predictions with confidence >= T. Bear accuracy filtered by confidence threshold.
- `{T}_bull_acc` — `(true == 2).mean()` among Bull predictions with confidence >= T. Bull accuracy filtered by confidence threshold.
- `{T}_count` — `n_bear + n_bull` above threshold T. Volume of directional predictions above threshold.
- `{T}_dir_acc` — `(bear_correct + bull_correct) / (n_bear + n_bull)` at threshold T. Combined directional accuracy at this confidence threshold.

> Higher threshold = fewer trades but expected better accuracy. Sample < 50 = not significant.

## Margin Confidence

Prefix: `Metrics/confmargin_`

Percentiles: `{05, 10, 25, 50}` = top N% of actionable predictions (Bear+Bull) sorted by `margin = proba_top1 - proba_top2` descending.

- `top{P}_acc` — `accuracy(y_pred == y_true)` on top P% by margin. Accuracy of the most decisive predictions (large gap between top1/top2).
- `top{P}_count` — `max(1, n_actionable * P / 100)`. Number of samples in this slice.

> Margin filters by gap between the two highest probabilities. More effective than highconf for identifying the most reliable predictions.

## Edge (exact P&L per trade)

Prefix: `Metrics/`

Exact expected gain computed from pre-calculated P&L in labelling (long_pnl_R / short_pnl_R).

- `edge_per_trade` — `sum(pnl_R[trades]) / n_trades`. Expected gain per trade in R units. Positive = profitable edge.
- `edge_total_R` — `sum(pnl_R[trades])`. Total cumulated R on all test set trades. Allows comparing absolute potential between configs.

Exact P&L per trade (triple/bull/bear mode):
- Bull prediction → `long_pnl_R[i]`: TP hit = +RR, SL hit = -1R, timeout = (close_horizon - entry) / (SL_mult x ATR)
- Bear prediction → `short_pnl_R[i]`: TP hit = +RR, SL hit = -1R, timeout = (entry - close_horizon) / (SL_mult x ATR)
- P&L arrays are pre-computed in `get_labels()` (same loop, same window, zero look-ahead bias)

Uncertain/closeN mode (no TP/SL barriers):
- Convention RR=1:1 → `edge_per_trade = 2 x final_accuracy - 1`
- `edge_total_R = edge_per_trade x count`

## Diagnostics (gradients & weights)

Prefix: `Metrics/`

- `grad_norm_mean` — `mean(L2_norm(param.grad))` all params. Gradient health; increasing = overfitting, >1.0 = monitor.
- `grad_norm_max` — `max(L2_norm(param.grad))`. Max gradient; >10 = potential instability.
- `weight_norm_mean` — `mean(L2_norm(param.data))`. Average weight norm; should stay stable if weight decay works.
- `weight_norm_max` — `max(L2_norm(param.data))`. Max weight norm.

## Activation Health

- `Health/dead_neurons_total_pct` — `100 * (neurons with std < 1e-6) / total`. Overall dead neuron percentage in the model.
- `Activations/{layer}/mean` — `mean(activations)` of the layer. Average activation value; drift = instability.
- `Activations/{layer}/std` — `std(activations)` of the layer. Activation spread; ~0 = dead layer.
- `Activations/{layer}/dead_pct` — `100 * (dims with std < 1e-6) / total_dims`. Dead neuron percentage in this layer.

> Layers: `linear_proj`, `mamba_block_0..N` (mono) or `primary_*`, `secondary_*` (multi-TF).

## Input Gate (CNN gated residual)

- `InputGate/{branch}/mean` — `mean(sigmoid(gate_raw))`. Average gate; ~0.5 = balanced CNN/skip, ~1 = bypass CNN.
- `InputGate/{branch}/std` — `std(sigmoid(gate_raw))`. Gate variability; high = temporal selectivity.
- `InputGate/{branch}/early_mean` — `mean(gate)` first half of sequence. Gate on window start (distant context).
- `InputGate/{branch}/late_mean` — `mean(gate)` second half of sequence. Gate on window end (recent context); early/late gap = temporal bias.

> `{branch}`: `main` (mono), `primary` or `secondary` (multi-TF).

## SSM Delta (Mamba selectivity)

- `Delta/{branch}_block_{i}/mean` — `mean(softplus(dt_proj(x) + dt_bias))`. Average SSM discretization step; reflects block reactivity.
- `Delta/{branch}_block_{i}/std` — `std(softplus(...))`. Delta variability; high = selective block, ~0 = static block.
- `Delta/{branch}_block_{i}/max` — `max(softplus(...))`. Max delta; very high values = state reset.

> Compare std across blocks: increasing with depth = progressive specialization (healthy). Frozen block (delta~0 from E0 to Efinal) = static feature extraction.

## Branch Diagnostics (multi-TF only)

### Representation norms

- `BranchDiagnostics/norm_primary_last` — `mean(L2_norm(h_pri[:, -1, :]))`. Primary last token norm; reflects representation richness.
- `BranchDiagnostics/norm_primary_max` — `mean(L2_norm(max_pool(h_pri)))`. Primary max pool norm.
- `BranchDiagnostics/norm_secondary_last` — `mean(L2_norm(h_sec[:, -1, :]))`. Secondary last token norm.
- `BranchDiagnostics/norm_secondary_max` — `mean(L2_norm(max_pool(h_sec)))`. Secondary max pool norm.
- `BranchDiagnostics/norm_attn_primary` — Attention pooled primary norm. Active if `ATTENTION_POSITION='pre_gate'`.
- `BranchDiagnostics/norm_attn_secondary` — Attention pooled secondary norm. Active if `ATTENTION_POSITION='pre_gate'`.
- `BranchDiagnostics/norm_cross_tf_attn` — Cross-TF attention norm. Active if `ATTENTION_POSITION='post_gate'`.
- `BranchDiagnostics/norm_enriched_last` — Enriched primary norm. Active if `ATTENTION_POSITION='aligned'`.

### Gate V2 (scalar softmax)

- `BranchDiagnostics/gate_w_{name}_mean` — `mean(softmax(gate_logits)[:, i])`. Average path weight in fusion; shows which branch dominates.
- `BranchDiagnostics/gate_w_primary_mean` — Sum of primary path weights. Total primary branch share in decision.
- `BranchDiagnostics/gate_w_secondary_mean` — Sum of secondary path weights. Total secondary branch share.
- `BranchDiagnostics/gate_w_primary_std` — Std of primary weights. Gate variability; high = adaptive per-sample decision.

> `{name}`: `std_primary`, `attn_primary`, `std_secondary`, `attn_secondary`, `cross_tf_attn` depending on config.

### Gate V3 (MLP per-feature sigmoid)

- `GateV3/gate_mean_{name}` — `mean(gate[:, i, :])`. Average gate per path; ~0.5 = shared, ~1 = dominates.
- `GateV3/gate_std_{name}` — `std(gate[:, i, :])`. Gate variability; high = fine per-feature selection.
- `GateV3/contribution_norm_{name}` — `mean(L2_norm(proj * gate))`. Effective path contribution after gating.

### Concat fusion

- `BranchDiagnostics/fusion_weight_primary_pct` — `norm(W_primary) / (norm(W_pri) + norm(W_sec)) * 100`. Learned static weight for primary in fusion layer.
- `BranchDiagnostics/fusion_weight_secondary_pct` — `norm(W_secondary) / total * 100`. Learned static weight for secondary.

### AlignedFusion gate

- `AlignedFusion/gate_mean` — `mean(sigmoid(gate_mlp(cat[h_pri, h_sec_aligned])))`. Average aligned fusion gate; ~1 = favors primary, ~0 = favors secondary.
- `AlignedFusion/gate_std` — `std(gate)`. Variability; high = adaptive fusion per position.
- `AlignedFusion/gate_early` — `mean(gate[:, :T/2, :])`. Gate first half of sequence.
- `AlignedFusion/gate_late` — `mean(gate[:, T/2:, :])`. Gate second half; early/late gap = context-dependent fusion.

## Feature Importance (figure)

- `FeatureImportance/bar_chart` — Figure (add_figure) generated at last epoch. Gradient-based feature importance bar chart. Non-scalar (TensorBoard image).

---

*Estimated total: ~100-150 metrics depending on configuration (mono/multi-TF, attention position, gate version, number of Mamba blocks).*
