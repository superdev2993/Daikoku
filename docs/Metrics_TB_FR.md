# Reference des Metriques TensorBoard

*[English version](Metrics_TB.md)*

Toutes les metriques loguees dans TensorBoard pendant l'entrainement. Classes : Bear=0, Uncertain=1, Bull=2.

---

## Loss & Accuracy (base)

- `Loss/train` — `mean(CrossEntropyLoss)` sur batches train. Mesure la capacite du modele a fitter les donnees d'entrainement.
- `Loss/test` — `mean(CrossEntropyLoss)` sur test set. Mesure la capacite de generalisation ; divergence vs train = overfit.
- `Accuracy/train` — `correct / total` sur train. Accuracy globale sur train.
- `Accuracy/test` — `correct / total` sur test. Accuracy globale sur test.

## Training per-class

- `Train/Balanced_Accuracy` — `mean(recall par classe)` (sklearn) sur train. Accuracy corrigee du desequilibre de classes.
- `Train/F1_Class_{0,1,2}` — `2 * P * R / (P + R)` par classe, train. F1 par classe pour detecter un biais de prediction.
- `Train/Precision_Class_{0,1,2}` — `TP / (TP + FP)` par classe, train. Proportion de predictions correctes par classe.
- `Train/Recall_Class_{0,1,2}` — `TP / (TP + FN)` par classe, train. Proportion de vrais detectes par classe.
- `Training/learning_rate` — `optimizer.param_groups[0]['lr']`. Suivi du LR courant (scheduler, warmup).

## Cascade Training

- `Cascade/trainable_params` — `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Nombre de parametres entrainables (augmente progressivement en cascade).
- `Cascade/active_depth` — `min(epoch + 1, n_layers_primary)`. Nombre de blocs Mamba degeles a l'epoch courante.

## Performance globale (test)

Prefix : `Metrics/`

- `accuracy` — `correct / total` (sklearn accuracy_score). Accuracy globale sur test set.
- `balanced_accuracy` — `mean(recall par classe)` (sklearn balanced_accuracy_score). Accuracy corrigee pour desequilibre de classes.
- `f1_macro` — `mean(F1 par classe)`. Score F1 moyen toutes classes, metrique principale de generalisation.
- `precision_macro` — `mean(Precision par classe)`. Precision moyenne sur les 3 classes.
- `recall_macro` — `mean(Recall par classe)`. Recall moyen sur les 3 classes.

## Per-class (test)

Prefix : `Metrics/`

- `precision_class_{0,1,2}` — `TP / (TP + FP)` pour classe i. Fiabilite des predictions par classe sur test.
- `recall_class_{0,1,2}` — `TP / (TP + FN)` pour classe i. Couverture de detection par classe sur test.
- `f1_class_{0,1,2}` — `2 * P * R / (P + R)` pour classe i. Equilibre precision/recall par classe sur test.

## Confidence

Prefix : `Metrics/`

- `entropy_mean` — `mean(scipy.entropy(probs.T))`. Incertitude moyenne des predictions ; bas = modele decisif.
- `confidence_correct_mean` — `mean(max(probs))` sur predictions correctes. Confiance moyenne quand le modele a raison.
- `confidence_incorrect_mean` — `mean(max(probs))` sur predictions incorrectes. Confiance moyenne quand le modele a tort ; ecart avec correct = calibration.

## Overfitting Gaps

Prefix : `Metrics/`

- `gap_loss` — `test_loss - train_loss`. Ecart de loss ; positif = overfitting, croissant = modele memorise.
- `gap_accuracy` — `train_acc - test_acc`. Ecart d'accuracy ; positif = overfit.

## Final Accuracy (predictions actionnables)

Prefix : `Metrics/`

- `final_accuracy` — `(Bear correct + Bull correct) / (total Bear + Bull preds)`. Precision des predictions directionnelles (exclut Uncertain).
- `final_accuracy_count` — Nombre de predictions Bear + Bull. Volume de predictions actionables ; trop bas = modele trop conservateur.

## High Confidence Directional

Prefix : `Metrics/highconf_`

Seuils : `{05, 06, 07}` = confidence `max(probs) >= {0.5, 0.6, 0.7}`

- `{T}_bear_acc` — `(true == 0).mean()` parmi preds Bear avec confidence >= T. Accuracy Bear filtre par seuil de confiance.
- `{T}_bull_acc` — `(true == 2).mean()` parmi preds Bull avec confidence >= T. Accuracy Bull filtre par seuil de confiance.
- `{T}_count` — `n_bear + n_bull` au-dessus du seuil T. Volume de predictions directionnelles au-dessus du seuil.
- `{T}_dir_acc` — `(bear_correct + bull_correct) / (n_bear + n_bull)` au seuil T. Accuracy directionnelle combinee a ce seuil de confiance.

> Seuil plus haut = moins de trades mais accuracy attendue meilleure. Echantillon < 50 = non significatif.

## Margin Confidence

Prefix : `Metrics/confmargin_`

Percentiles : `{05, 10, 25, 50}` = top N% des predictions actionables (Bear+Bull) triees par `margin = proba_top1 - proba_top2` decroissant.

- `top{P}_acc` — `accuracy(y_pred == y_true)` sur top P% par margin. Accuracy des predictions les plus decisives (forte marge entre top1/top2).
- `top{P}_count` — `max(1, n_actionable * P / 100)`. Nombre de samples dans cette tranche.

> Le margin filtre par ecart entre les 2 probas les plus hautes. Plus efficace que le highconf pour identifier les predictions les plus fiables.

## Edge (P&L exact par trade)

Prefix : `Metrics/`

Esperance de gain exacte calculee a partir du P&L pre-calcule dans le labelling (long_pnl_R / short_pnl_R).

- `edge_per_trade` — `sum(pnl_R[trades]) / n_trades`. Esperance de gain par trade en unites de R. Positif = edge profitable.
- `edge_total_R` — `sum(pnl_R[trades])`. R total cumule sur tous les trades du test set. Permet de comparer le potentiel absolu entre configs.

P&L exact par trade (mode triple/bull/bear) :
- Bull prediction → `long_pnl_R[i]` : TP hit = +RR, SL hit = -1R, timeout = (close_horizon - entry) / (SL_mult x ATR)
- Bear prediction → `short_pnl_R[i]` : TP hit = +RR, SL hit = -1R, timeout = (entry - close_horizon) / (SL_mult x ATR)
- Les P&L arrays sont pre-calcules dans `get_labels()` (meme boucle, meme fenetre, zero look-ahead bias)

Mode uncertain/closeN (pas de barrieres TP/SL) :
- Convention RR=1:1 → `edge_per_trade = 2 x final_accuracy - 1`
- `edge_total_R = edge_per_trade x count`

## Diagnostics (gradients & poids)

Prefix : `Metrics/`

- `grad_norm_mean` — `mean(L2_norm(param.grad))` tous params. Sante des gradients ; croissant = overfit, >1.0 = surveiller.
- `grad_norm_max` — `max(L2_norm(param.grad))`. Gradient max ; >10 = instabilite potentielle.
- `weight_norm_mean` — `mean(L2_norm(param.data))`. Norme moyenne des poids ; doit rester stable si weight decay fonctionne.
- `weight_norm_max` — `max(L2_norm(param.data))`. Norme max des poids.

## Activation Health

- `Health/dead_neurons_total_pct` — `100 * (neurons avec std < 1e-6) / total`. Pourcentage global de neurones morts dans le modele.
- `Activations/{layer}/mean` — `mean(activations)` de la couche. Valeur moyenne des activations ; drift = instabilite.
- `Activations/{layer}/std` — `std(activations)` de la couche. Dispersion des activations ; ~0 = couche morte.
- `Activations/{layer}/dead_pct` — `100 * (dims avec std < 1e-6) / total_dims`. Pourcentage de neurones morts dans cette couche.

> Layers : `linear_proj`, `mamba_block_0..N` (mono) ou `primary_*`, `secondary_*` (multi-TF).

## Input Gate (CNN gated residual)

- `InputGate/{branch}/mean` — `mean(sigmoid(gate_raw))`. Gate moyen ; ~0.5 = partage equilibre CNN/skip, ~1 = bypass CNN.
- `InputGate/{branch}/std` — `std(sigmoid(gate_raw))`. Variabilite du gate ; haut = selectivite temporelle.
- `InputGate/{branch}/early_mean` — `mean(gate)` premiere moitie de sequence. Gate sur le debut de fenetre (contexte lointain).
- `InputGate/{branch}/late_mean` — `mean(gate)` seconde moitie de sequence. Gate sur la fin de fenetre (contexte recent) ; ecart early/late = biais temporel.

> `{branch}` : `main` (mono), `primary` ou `secondary` (multi-TF).

## SSM Delta (selectivite Mamba)

- `Delta/{branch}_block_{i}/mean` — `mean(softplus(dt_proj(x) + dt_bias))`. Pas de discretisation moyen du SSM ; reflete la reactivite du bloc.
- `Delta/{branch}_block_{i}/std` — `std(softplus(...))`. Variabilite du delta ; haut = bloc selectif, ~0 = bloc statique.
- `Delta/{branch}_block_{i}/max` — `max(softplus(...))`. Delta max ; valeurs tres hautes = reset du state.

> Comparer std entre blocs : croissant avec la profondeur = specialisation progressive (sain). Bloc gele (delta~0 entre E0 et Efinal) = feature extraction statique.

## Branch Diagnostics (multi-TF uniquement)

### Representation norms

- `BranchDiagnostics/norm_primary_last` — `mean(L2_norm(h_pri[:, -1, :]))`. Norme du last token primary ; reflete la richesse de la representation.
- `BranchDiagnostics/norm_primary_max` — `mean(L2_norm(max_pool(h_pri)))`. Norme du max pool primary.
- `BranchDiagnostics/norm_secondary_last` — `mean(L2_norm(h_sec[:, -1, :]))`. Norme du last token secondary.
- `BranchDiagnostics/norm_secondary_max` — `mean(L2_norm(max_pool(h_sec)))`. Norme du max pool secondary.
- `BranchDiagnostics/norm_attn_primary` — Norme attention pooled primary. Actif si `ATTENTION_POSITION='pre_gate'`.
- `BranchDiagnostics/norm_attn_secondary` — Norme attention pooled secondary. Actif si `ATTENTION_POSITION='pre_gate'`.
- `BranchDiagnostics/norm_cross_tf_attn` — Norme cross-TF attention. Actif si `ATTENTION_POSITION='post_gate'`.
- `BranchDiagnostics/norm_enriched_last` — Norme enriched primary. Actif si `ATTENTION_POSITION='aligned'`.

### Gate V2 (scalar softmax)

- `BranchDiagnostics/gate_w_{name}_mean` — `mean(softmax(gate_logits)[:, i])`. Poids moyen du path i dans la fusion ; montre quelle branche domine.
- `BranchDiagnostics/gate_w_primary_mean` — Somme poids paths primary. Part totale de la branche primary dans la decision.
- `BranchDiagnostics/gate_w_secondary_mean` — Somme poids paths secondary. Part totale de la branche secondary.
- `BranchDiagnostics/gate_w_primary_std` — Std poids primary. Variabilite du gate ; haut = decision adaptative par sample.

> `{name}` : `std_primary`, `attn_primary`, `std_secondary`, `attn_secondary`, `cross_tf_attn` selon config.

### Gate V3 (MLP per-feature sigmoid)

- `GateV3/gate_mean_{name}` — `mean(gate[:, i, :])`. Gate moyen par path ; ~0.5 = partage, ~1 = domine.
- `GateV3/gate_std_{name}` — `std(gate[:, i, :])`. Variabilite du gate ; haut = selection fine par feature.
- `GateV3/contribution_norm_{name}` — `mean(L2_norm(proj * gate))`. Contribution effective du path apres gating.

### Concat fusion

- `BranchDiagnostics/fusion_weight_primary_pct` — `norm(W_primary) / (norm(W_pri) + norm(W_sec)) * 100`. Poids statique appris pour primary dans la couche de fusion.
- `BranchDiagnostics/fusion_weight_secondary_pct` — `norm(W_secondary) / total * 100`. Poids statique appris pour secondary.

### AlignedFusion gate

- `AlignedFusion/gate_mean` — `mean(sigmoid(gate_mlp(cat[h_pri, h_sec_aligned])))`. Gate moyen de fusion aligned ; ~1 = favorise primary, ~0 = favorise secondary.
- `AlignedFusion/gate_std` — `std(gate)`. Variabilite ; haut = fusion adaptative par position.
- `AlignedFusion/gate_early` — `mean(gate[:, :T/2, :])`. Gate premiere moitie de sequence.
- `AlignedFusion/gate_late` — `mean(gate[:, T/2:, :])`. Gate seconde moitie ; ecart early/late = fusion dependante du contexte temporel.

## Feature Importance (figure)

- `FeatureImportance/bar_chart` — Figure (add_figure) generee a la derniere epoch. Bar chart des importances de features basees sur les gradients. Non scalar (image TensorBoard).

---

*Total estime : ~100-150 metriques selon configuration (mono/multi-TF, attention position, gate version, nombre de blocs Mamba).*
