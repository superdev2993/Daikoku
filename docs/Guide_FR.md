# Daikoku -- Guide

*[English version](Guide.md)*

Daikoku est un systeme de prediction de trading de cryptomonnaies base sur l'architecture **Mamba** (Selective State Space Model). Il predit la direction du marche -- Haussier (Bull), Baissier (Bear) ou Incertain (Uncertain) -- a partir de donnees de chandeliers OHLCV en utilisant une methode de labellisation Triple Barrier avec des niveaux de take-profit et stop-loss bases sur l'ATR.

---

## Table des matieres

1. [Prerequis](#prerequis)
2. [Installation](#installation)
3. [Preparation des donnees](#preparation-des-donnees)
4. [Configuration](#configuration)
5. [Entrainement](#entrainement)
6. [Optimisation des hyperparametres](#optimisation-des-hyperparametres)
7. [Evaluation](#evaluation)
8. [Inference en direct](#inference-en-direct)
9. [Outils](#outils)
10. [Structure du projet](#structure-du-projet)
11. [Tests](#tests)
12. [Depannage](#depannage)

---

## Prerequis

- **Python** 3.11
- **RAM** : 16 Go minimum
- **GPU** (fortement recommande) : carte graphique NVIDIA compatible CUDA avec au moins 4 Go de VRAM
- **PyTorch** 2.4.1 avec CUDA 12.4 (`torch 2.4.1+cu124`)
- **CUDA toolkit** 12.4+ (recommandé, pour l'acceleration GPU)

Le systeme peut fonctionner entierement sur CPU (toutes les fonctionnalites y compris l'entrainement et les tests), mais l'entrainement et les tests seront extremement plus lents. Une carte graphique CUDA est fortement recommandee pour toute charge de travail serieuse.

### Environnement teste

Les versions suivantes sont testees et fonctionnent ensemble :

| Paquet            | Version     | Utilite                              |
|-------------------|-------------|--------------------------------------|
| `torch`           | 2.4.1+cu124 | Framework d'apprentissage profond    |
| `numpy`           | 2.4.1       | Calcul numerique                     |
| `pandas`          | 3.0.0       | Manipulation de donnees              |
| `scikit-learn`    | 1.8.0       | Metriques et evaluation              |
| `scipy`           | 1.17.0      | Operations statistiques              |
| `optuna`          | 4.7.0       | Optimisation d'hyperparametres       |
| `tensorboard`     | 2.20.0      | Visualisation de l'entrainement      |
| `matplotlib`      | 3.10.8      | Graphiques                           |
| `plotly`          | 6.5.2       | Visualisations interactives          |
| `numba`           | 0.64.0      | Routines numeriques compilees en JIT |
| `tqdm`            | 4.67.1      | Barres de progression                |
| `ccxt`            | 4.5.36      | API d'echanges de cryptomonnaies     |
| `python-dateutil` | 2.9.0       | Utilitaires de parsing de dates      |

### Paquets optionnels (acceleration GPU)

| Paquet          | Version | Utilite                                           |
|-----------------|---------|---------------------------------------------------|
| `mamba-ssm`     | 2.3.0   | Noyaux CUDA optimises pour Mamba (GPU uniquement) |
| `causal-conv1d` | 1.5.2   | Convolution causale rapide (GPU uniquement)       |

Si `mamba-ssm` n'est pas installe (par exemple dans les environnements CPU uniquement), Daikoku bascule automatiquement vers une implementation PyTorch pure (`mamba_cpu.py`).

---

## Installation

### 1. Cloner le depot

```bash
git clone https://github.com/yannpointud/Daikoku.git
cd Daikoku
```

### 2. Installer PyTorch

Installez PyTorch separement avec la version CUDA appropriee pour votre systeme. Consultez [pytorch.org](https://pytorch.org/get-started/locally/) pour la commande correcte.

Version testee (CUDA 12.4) :
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

Pour CPU uniquement :
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

Si vous etes en CPU uniquement et que l'installation de `mamba-ssm` echoue, vous pouvez l'ignorer -- le systeme utilisera automatiquement le mode de repli CPU.

Pour installer les noyaux Mamba acceleres GPU (optionnel) :
```bash
pip install mamba-ssm==2.3.0 causal-conv1d==1.5.2
```

### 4. Creer les repertoires de donnees

```bash
mkdir -p data models
```

---

## Preparation des donnees

### Format CSV

Daikoku attend des fichiers CSV OHLCV avec les colonnes suivantes :

```
Open time,Open,High,Low,Close,Volume
```

Exigences :
- **Open time** : Chaine de date/heure interpretable (par exemple `2023-01-01 00:00:00`)
- **Ordre chronologique** : Les lignes doivent etre triees par date, sans horodatages dupliques
- **Coherence OHLC** : `High >= Low`, et `Open`/`Close` doivent etre dans l'intervalle `[Low, High]`
- **Volume** : Doit etre >= 0

Exemple de donnees :
```csv
Open time,Open,High,Low,Close,Volume
2023-01-01 00:00:00,16541.77,16553.12,16530.01,16547.23,1234.56
2023-01-01 01:00:00,16547.23,16560.00,16540.00,16555.89,987.65
2023-01-01 02:00:00,16555.89,16558.45,16545.30,16550.12,1456.78
```

### Telechargement des donnees

Utilisez l'outil de telechargement integre pour recuperer les donnees OHLCV historiques depuis les echanges de cryptomonnaies.

**Echanges supportes** : binance, bybit, bitget, kucoin

**Intervalles de temps supportes** : 5m, 15m, 30m, 1h, 2h, 4h, 1d

#### Mode CLI

```bash
# Echange unique
python modules/tools/download_candles.py \
    --exchange binance \
    --pair BTC/USDT \
    --timeframe 1h \
    --start 2020-01-01 \
    --output data/BTC_1h_bin.csv

# Plusieurs echanges (agregation ponderee par le volume)
python modules/tools/download_candles.py \
    --exchanges binance,bybit \
    --pair ETH/USDT \
    --timeframe 1h \
    --start 2021-01-01 \
    --end 2024-12-31 \
    --output data/ETH_1h_agg.csv
```

#### Mode interactif

```bash
python modules/tools/download_candles.py
```

Cela vous demandera le(s) echange(s), la paire, l'intervalle de temps, la plage de dates et le fichier de sortie.

#### Arguments CLI

| Argument      | Description                                         | Defaut                           |
|---------------|-----------------------------------------------------|----------------------------------|
| `--exchange`  | Nom d'un seul echange                               | --                               |
| `--exchanges` | Liste d'echanges separee par des virgules (agreges) | --                               |
| `--pair`      | Paire de trading                                    | `BTC/USDT`                       |
| `--timeframe` | Intervalle de temps des chandeliers                 | --                               |
| `--start`     | Date de debut (format ISO)                          | `2019-08-01`                     |
| `--end`       | Date de fin (format ISO)                            | maintenant                       |
| `--output`    | Chemin du fichier CSV de sortie                     | `data/download/btc_usdt_agg.csv` |
| `--limit`     | Nombre maximum de chandeliers par appel API         | `1000`                           |
| `--sleep`     | Pause entre les appels API (secondes)               | `1.0`                            |

Lorsque plusieurs echanges sont selectionnes, les donnees sont agregees en utilisant une moyenne ponderee par le volume pour les prix OHLC et un volume somme.

---

## Configuration

Tous les parametres sont centralises dans un seul fichier : **`config.py`**. Editez ce fichier avant l'entrainement.

### Parametres de donnees

```python
INPUT_FILES = [
    "data/BTC_1h_agg.csv",
    # "data/ETH_1h_bin.csv",    # Decommentez pour ajouter d'autres jeux de donnees
    # "data/SOL_1h_bin.csv",
]

PREDICTION_TARGET = "triple"    # "triple" (3 classes : bull/bear/uncertain)
                                # "bull", "bear", "uncertain" (binaire)
                                # "close1", "close2", "close3" (binaire prochain close)

WINDOW_SIZE = 120               # Nombre de chandeliers par fenetre d'entree
TRAIN_TEST_SPLIT = 0.9          # 90% entrainement, 10% test (chronologique)
NORMALIZE = True                # Normalisation mediane/IQR par fenetre
```

### Labellisation Triple Barrier

Controle la maniere dont les chandeliers sont labellises pour l'entrainement. Utilise l'ATR (Average True Range) pour definir des niveaux dynamiques de TP/SL.

```python
ATR_PERIOD = 48           # Periode de lissage ATR (Wilder)
ATR_MULTIPLIER_TP = 2     # Take profit = 2 x ATR
ATR_MULTIPLIER_SL = 1     # Stop loss = 1 x ATR  (donne un ratio R:R de 2:1)
MAX_HORIZON = 12          # Nombre max de chandeliers avant de labelliser "uncertain"
```

### Architecture du modele

```python
D_MODEL = 48              # Dimension interne du modele
N_LAYERS = 5              # Nombre de blocs Mamba
D_STATE = 32              # Dimension de l'etat SSM
D_CONV = 4                # Taille du noyau de convolution
EXPAND = 2                # Facteur d'expansion pour la dimension interne
```

### Multi-Timeframe (MTF)

Daikoku supporte une architecture a double branche qui traite deux intervalles de temps simultanement.

```python
MULTI_TF_MODE = "dual"          # "off"         = intervalle de temps unique
                                # "calibration" = TF secondaire uniquement (pour les tests)
                                # "dual"        = les deux intervalles de temps

MULTI_TF_DIVISOR = 4            # Facteur d'agregation (ex : 1h primaire -> 4h secondaire)
MULTI_TF_N_LAYERS = 4           # Couches Mamba pour la branche secondaire
MULTI_TF_D_STATE = 24           # Dimension de l'etat SSM pour la branche secondaire
MULTI_TF_ALIGN = "unstructured" # "structured" (limites horaires) / "unstructured"
CROSS_TF_NORMALIZE = True       # Normaliser le secondaire avec les statistiques du primaire
```

### Attention et Gate

```python
ATTENTION_POSITION = "aligned"  # "off"       = pas d'attention
                                # "pre_gate"  = auto-attention intra-TF
                                # "post_gate" = attention croisee inter-TF
                                # "aligned"   = alignement temporel inter-TF (recommande)

ATTENTION_NUM_HEADS = 4         # Doit diviser D_MODEL
GATE_VERSION = "v3"             # "v2" (scalaire) / "v3" (MLP par feature)
GATE_BOTTLENECK = 64            # Dimension du goulot d'etranglement pour le MLP v3
```

### Entrainement

```python
EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.03
DROPOUT = 0.1
DROPOUT_LAST_N_LAYERS = 8      # Appliquer le dropout uniquement aux N dernieres couches
DROPOUT_HEAD = 0.1              # Dropout dans la tete de classification
DEVICE = "auto"                 # "auto" (GPU si disponible), "cuda", "cpu"
SEED = 42
SHUFFLE_TRAIN = True            # Melanger les fenetres d'entrainement
SHUFFLE_CHUNKS = 8              # 1 = melange complet, N>1 = melange par blocs
```

### Planification du taux d'apprentissage

```python
LR_SCHEDULER = "none"          # "none" / "cosine" / "wsd" (warmup-stable-decay)
LR_MIN = 0.000005              # Taux d'apprentissage minimum cible
LR_WARMUP_PCT = 0.4            # WSD uniquement : proportion de warmup
LR_STABLE_PCT = 0.2            # WSD uniquement : proportion stable
```

### Fonction de perte

```python
FOCAL_GAMMA = 0                # 0 = entropie croisee standard, >0 = focal loss
CLASS_WEIGHT_ALPHA = 0.5       # 0 = pas de ponderation de classe, 0.5 = racine de la frequence inverse, 1.0 = complete
```

### Performance (GPU)

```python
MIXED_PRECISION = True         # AMP float16 sur la passe avant
CUDNN_BENCHMARK = True         # Auto-reglage des algorithmes de convolution
NUM_WORKERS = "auto"           # "auto" (adapte aux coeurs CPU et au device) ou int
PERSISTENT_WORKERS = True      # Garder les workers actifs entre les epoques
NON_BLOCKING_TRANSFER = True   # Chevauchement des transferts CPU/GPU
```

### Ingenierie des features

Daikoku calcule automatiquement 23 features d'entree :

| Categorie                      | Features                                                     |
|--------------------------------|--------------------------------------------------------------|
| **OHLCV** (5)                  | log_price, log_open, log_wick_high, log_wick_low, log_volume |
| **Structure** (1)              | log_zz (zigzag)                                              |
| **Volatilite** (1)             | log_atr                                                      |
| **Temps** (4)                  | sin_hour, cos_hour, sin_weekday, cos_weekday                 |
| **Structurel** (5)             | struct_rank, dist_res_1/2, dist_sup_1/2                      |
| **Profil de volume** (5)       | Features VP (POC, VAH, VAL, width, skew)                     |
| **Regime** (2)                 | regime_trend, regime_voltrend                                |

---

## Entrainement

### Entrainement de base

```bash
python main.py
```

Cela execute le pipeline complet :
1. Charger et valider les donnees CSV
2. Calculer les labels Triple Barrier
3. Transformer les features (echelle logarithmique, normalisation)
4. Creer les ensembles train/test (chronologique)
5. Initialiser le modele Mamba
6. Entrainer avec sauvegarde de points de controle
7. Sauvegarder le modele dans `models/latest.pt`

### Reprendre l'entrainement

Reprendre a partir d'un point de controle precedent :

```bash
python main.py --resume models/latest.pt
```

### Entrainement multi-jeux de donnees

Ajoutez plusieurs fichiers CSV a `INPUT_FILES` dans `config.py` :

```python
INPUT_FILES = [
    "data/BTC_1h_agg.csv",
    "data/ETH_1h_bin.csv",
    "data/SOL_1h_bin.csv",
]
```

Les jeux de donnees sont traites independamment, puis fusionnes via `ConcatDataset` pour l'entrainement.

### Surveiller l'entrainement

Les metriques d'entrainement sont enregistrees dans TensorBoard :

```bash
tensorboard --logdir runs/
```

Ouvrez `http://localhost:6006` dans votre navigateur pour visualiser :
- Courbes de perte (train/test)
- Metriques de precision
- Planification du taux d'apprentissage
- Diagnostics de surveillance des activations
- Equilibre des branches (pour le mode dual MTF)

![TensorBoard](images/tensorboard.png)

Pour la liste complete de toutes les metriques TensorBoard (100+), voir **[Metrics_TB_FR.md](Metrics_TB_FR.md)**.

Pour surveiller l'utilisation GPU en temps reel, nous recommandons [nvitop](https://github.com/XuehaiPan/nvitop) :

```bash
pip install nvitop
nvitop
```

### Points de controle (Checkpoints)

Les points de controle sont sauvegardes dans `models/` et contiennent :
- `model_state_dict` : Poids du modele
- `optimizer_state` : Etat de l'optimiseur pour la reprise
- `config` : Tous les parametres de configuration au moment de l'entrainement
- `epoch` : Derniere epoque completee
- `metrics` : Metriques d'entrainement

Une copie est egalement placee dans le repertoire de la session TensorBoard (par exemple `runs/run_001/latest.pt`).

### Gestion des interruptions

L'entrainement peut etre interrompu en toute securite avec `Ctrl+C`. Le gestionnaire de signal sauvegarde un point de controle avant de quitter, de sorte qu'aucune progression n'est perdue.

---

## Optimisation des hyperparametres

Daikoku utilise **Optuna** pour l'optimisation multi-objectifs des hyperparametres.

### Objectifs

1. **Maximiser** `edge_total_R` -- Rentabilite totale en multiples de R (qualite x volume)
2. **Minimiser** `gap_loss` -- Difference entre la perte d'entrainement et la perte de test (prevention du surapprentissage)
3. **Maximiser** `edge_per_trade` -- Qualite par trade en multiples de R

### Definir l'espace de recherche

Editez `OPTUNA_SEARCH_SPACE` dans `config.py` :

```python
OPTUNA_SEARCH_SPACE = {
    'd_model': [32, 48, 64],              # Categoriel : choix dans une liste
    'n_layers': (3, 10),                   # Intervalle entier
    'learning_rate': (0.00001, 0.001),     # Intervalle flottant (echelle log)
    'weight_decay': [0.001, 0.01, 0.03],  # Categoriel
    'dropout': (0.05, 0.3),                # Intervalle flottant (echelle log)
    'window_size': (60, 200),              # Intervalle entier
}
```

Regles de format :
- **Liste** `[v1, v2, ...]` -- choix categoriel
- **Tuple d'entiers** `(min, max)` -- intervalle entier
- **Tuple de flottants** `(min, max)` -- intervalle flottant avec echelle logarithmique

Tout parametre non liste utilise sa valeur par defaut de `config.py`.

### Parametres explorables

Tout parametre UPPERCASE de `config.py` peut etre explore — il suffit d'ajouter son nom en minuscules comme cle dans `OPTUNA_SEARCH_SPACE`. Choix courants :

| Categorie    | Parametres                                                                 |
|--------------|----------------------------------------------------------------------------|
| Modele       | d_model, n_layers, d_state, d_conv, expand                                 |
| Entrainement | learning_rate, batch_size, weight_decay, dropout, dropout_last_n_layers    |
| Donnees      | window_size, atr_period, atr_multiplier_tp, atr_multiplier_sl, max_horizon |
| Perte        | focal_gamma, class_weight_alpha                                            |
| CNN          | cnn_kernel_size                                                            |
| Attention    | attention_position, gate_version, gate_bottleneck                          |
| Multi-TF     | multi_tf_divisor, multi_tf_n_layers, multi_tf_d_state                      |
| Label        | label_confidence_enabled, label_confidence_curve                           |

### Lancer l'optimisation

```bash
# Nouvelle etude avec 50 essais
python optimize.py --trials 50

# Reprendre une etude existante
python optimize.py --trials 50 --resume

# Nom d'etude personnalise
python optimize.py --trials 100 --study-name my_experiment
```

### Arguments CLI

| Argument       | Description                                  | Defaut                 |
|----------------|----------------------------------------------|------------------------|
| `--trials`     | Nombre d'essais d'optimisation               | `50`                   |
| `--resume`     | Reprendre une etude existante                | `False`                |
| `--study-name` | Nom de l'etude (utilise pour la base SQLite) | `mamba_multiobjective` |

### Resultats

- **Base de donnees SQLite** : `optimize/<study_name>/study.db` (persistante, permet la reprise)
- **Fichier de resultats** : `optimize/<study_name>/results_<timestamp>.txt`
- **Rapport detaille** : `optimize/<study_name>/detailed_report_<timestamp>.txt`
- **TensorBoard** : Chaque essai enregistre sous `runs/optuna_<NNN>/`
- **Points de controle** : Chaque essai sauvegarde sous `models/optuna_<NNN>/`

Apres l'optimisation, le front de Pareto est affiche, montrant les meilleurs compromis entre la precision directionnelle et la robustesse.

---

## Evaluation

### Evaluation de base

```bash
python evaluate.py
```

Sans arguments, tous les defauts proviennent de `config.py` (chemin du checkpoint, fichier de donnees, split, device). Cela evalue le point de controle par defaut sur l'ensemble de test du jeu de donnees configure.

### Evaluation personnalisee

```bash
# Point de controle et fichier de donnees specifiques
python evaluate.py --checkpoint models/best.pt --data data/test.csv

# Evaluer sur toutes les donnees (train + test)
python evaluate.py --split all

# Evaluer sur l'ensemble d'entrainement uniquement
python evaluate.py --split train

# Forcer le CPU
python evaluate.py --device cpu
```

### Arguments CLI

| Argument           | Description                                    | Defaut                                 |
|--------------------|------------------------------------------------|----------------------------------------|
| `--checkpoint`     | Chemin du point de controle du modele          | `models/latest.pt`                     |
| `--data`           | Chemin du fichier CSV de donnees               | Valeur de `EVAL_DATA_FILE` dans config |
| `--split`          | Ensemble de donnees : `train`, `test`, `all`   | `test`                                 |
| `--output`         | Repertoire de sortie                           | Genere automatiquement                 |
| `--device`         | Peripherique : `auto`, `cuda`, `cpu`           | `auto`                                 |

### Sorties

Les resultats sont sauvegardes dans `evaluation/YYYYMMDD_HHMMSS/` :

| Fichier                   | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| `predictions.csv`         | Predictions par echantillon avec probabilites                       |
| `metrics.json`            | Toutes les metriques calculees                                      |
| `interactive_report.html` | Visualisation interactive des predictions sur le graphique des prix |
| `input_features.html`     | Visualisation des features d'entree transformees                    |
| `latest.pt`               | Copie du point de controle evalue                                   |

![Rapport d'evaluation](images/evaluate1.png)

### Metriques principales

| Metrique              | Description                                                        |
|-----------------------|--------------------------------------------------------------------|
| `accuracy`            | Precision globale de classification                                |
| `balanced_accuracy`   | Precision equilibree entre les classes                             |
| `f1_macro`            | Score F1 macro-moyenne                                             |
| `final_accuracy`      | Precision directionnelle (taux de predictions correctes bull/bear) |
| `edge_per_trade`      | Rendement attendu en multiples de R par trade                      |
| `gap_loss`            | Ecart de perte train-test (indicateur de surapprentissage)         |
| `highconf_XX_dir_acc` | Precision directionnelle au seuil de confiance XX                  |

---

## Inference en direct

L'inference en direct telecharge des chandeliers en temps reel depuis un echange, execute des predictions sur chaque nouveau chandelier cloture, suit les trades virtuels et sert un tableau de bord web.

### Utilisation de base

```bash
python inference.py
```

Cela utilise les valeurs par defaut de `config.py` (BTC/USDT, 15m, binance, port 7777).

### Utilisation personnalisee

```bash
# Paire et intervalle de temps differents
python inference.py --pair ETH/USDT --timeframe 1h --port 8888

# Point de controle specifique
python inference.py --checkpoint models/best.pt

# Forcer le GPU
python inference.py --device cuda

# Demarrage frais (effacer la session precedente)
python inference.py --fresh

# Echange different
python inference.py --exchange bybit
```

### Arguments CLI

| Argument        | Description                                          | Defaut                     |
|-----------------|------------------------------------------------------|----------------------------|
| `--pair`        | Paire de trading                                     | `BTC/USDT` (depuis config) |
| `--timeframe`   | Intervalle de temps des chandeliers                  | `15m` (depuis config)      |
| `--checkpoint`  | Chemin du point de controle du modele                | `models/latest.pt`         |
| `--port`        | Port HTTP du tableau de bord                         | `7777`                     |
| `--buffer-size` | Taille du tampon de chandeliers bruts                | `2000`                     |
| `--device`      | Peripherique : `auto`, `cuda`, `cpu`                 | `auto`                     |
| `--exchange`    | Nom de l'echange                                     | `binance`                  |
| `--fresh`       | Forcer un demarrage frais, effacer l'etat sauvegarde | `False`                    |

### Pipeline

1. **Charger le modele** depuis le point de controle
2. **Initialiser le tracker** avec les parametres TP/SL bases sur l'ATR
3. **Telecharger les chandeliers** pour remplir le tampon (ou reprendre depuis le tampon sauvegarde)
4. **Demarrer le tableau de bord** sur le port HTTP specifie
5. **Boucle principale** : Attendre la cloture du chandelier -> executer la prediction -> suivre le trade -> mettre a jour le tableau de bord

### Tableau de bord

Ouvrez `http://localhost:7777` (ou votre port configure) pour visualiser le tableau de bord en direct affichant :
- Predictions actuelles et confiance
- Trades virtuels ouverts et clotures
- Statistiques de gains/pertes
- Graphique des prix avec signaux

![Dashboard Live](images/live1.png)

### Reprise de session

Lorsque vous arretez l'inference avec `Ctrl+C`, le systeme sauvegarde :
- `live/predictions_<PAIR>_<TF>.csv` -- toutes les predictions
- `live/trades_<PAIR>_<TF>.csv` -- trades clotures
- `live/buffer_<PAIR>_<TF>.csv` -- tampon de chandeliers
- `live/state_<PAIR>_<TF>.json` -- trades ouverts et etat en attente

Au redemarrage (sans `--fresh`), la session est automatiquement reprise :
- Les predictions et trades precedents sont recharges
- Les chandeliers manquants depuis l'arret sont telecharges et rejoues
- Les trades ouverts sont mis a jour avec les eventuels TP/SL atteints pendant la periode hors ligne

### Mode comparaison

Tous les `LIVE_COMPARE_EVERY` chandeliers (par defaut : 1), le systeme re-execute l'inference sur toutes les predictions precedentes en utilisant le tampon sauvegarde. Cela verifie la reproductibilite deterministe -- toute divergence entre les predictions en direct et la re-inference est signalee comme une erreur sur le tableau de bord.

---

## Outils

### Augmentation de donnees

Generer des donnees d'entrainement synthetiques en utilisant un processus d'Ornstein-Uhlenbeck :

```bash
python -m modules.tools.augment_data

# Parametres personnalises
python -m modules.tools.augment_data --file data/BTC_1h_agg.csv --variants 4
python -m modules.tools.augment_data --file data/ETH_1h_bin.csv --pct 5 --variants 6
```

Configuration dans `config.py` :

```python
AUGMENT_VARIANTS = 4     # Nombre de copies augmentees par fichier original
AUGMENT_PCT = 50          # Pourcentage de perturbation
```

L'augmentation modifie :
- **Open/Close** : Processus OU avec pas = max(PCT% du corps, 0.1% du prix), avec continuite assuree
- **High/Low** : Perturbation aleatoire de la taille des meches dans la limite de PCT%
- **Volume** : Perturbation aleatoire uniforme dans la limite de PCT%

### Telechargement de donnees

Voir [Preparation des donnees](#preparation-des-donnees) pour la documentation complete du telechargement.

---

## Structure du projet

```
Daikoku/
|-- main.py                  # Point d'entree pour l'entrainement
|-- evaluate.py              # Point d'entree pour l'evaluation
|-- inference.py             # Point d'entree pour l'inference en direct
|-- optimize.py              # Point d'entree pour l'optimisation Optuna
|-- config.py                # Tous les parametres de configuration
|-- requirements.txt         # Dependances Python
|
|-- data/                    # Jeux de donnees d'entrainement/evaluation (CSV)
|
|-- models/                  # Points de controle sauvegardes (fichiers .pt)
|-- runs/                    # Repertoires de logs TensorBoard
|-- evaluation/              # Rapports de sortie d'evaluation
|-- live/                    # Fichiers de session d'inference en direct
|-- logs/                    # Logs applicatifs
|
|-- modules/
    |-- model/
    |   |-- mamba.py         # Architecture du modele Mamba (GPU)
    |   |-- mamba_cpu.py     # Mode de repli CPU en PyTorch pur
    |
    |-- data/
    |   |-- loader.py        # Chargement et validation des donnees
    |   |-- labeling.py      # Calcul des labels Triple Barrier
    |   |-- pipeline.py      # Pipeline de traitement des donnees
    |
    |-- training/
    |   |-- trainer.py       # Boucle d'entrainement et evaluation
    |   |-- loss.py          # Fonctions de perte (Focal, CE ponderee par classe)
    |   |-- metrics.py       # Calcul des metriques
    |   |-- setup.py         # Creation du modele/optimiseur/dataloader
    |
    |-- inference/
    |   |-- engine.py        # Moteur de prediction
    |   |-- feed.py          # Flux de chandeliers (connexion a l'echange)
    |   |-- tracker.py       # Tracker de trades (TP/SL/timeout)
    |   |-- dashboard.py     # Serveur du tableau de bord web
    |
    |-- evaluation/
    |   |-- evaluator.py     # Evaluateur de modele
    |   |-- visualizer.py    # Visualisations des predictions
    |
    |-- tools/
    |   |-- download_candles.py  # Telechargement de donnees OHLCV
    |   |-- augment_data.py      # Augmentation de donnees
    |
    |-- utils/
        |-- logger.py        # Configuration du logging
        |-- seed.py          # Gestion de la graine aleatoire
```

---

## Tests

### Lancer les tests

```bash
python -m pytest tests/ -v
```

La plupart des tests necessitent le jeu de donnees de test `tests/data/test_Dataset.csv`, qui est inclus dans le depot. Si le fichier est absent, ces tests seront automatiquement ignores (skipped).

### Jeu de donnees de test

Le fichier `tests/data/test_Dataset.csv` (~6400 lignes de donnees BTC 1h OHLCV) est utilise par la suite de tests pour :
- Entrainer un mini-modele (2 epoques) partage entre les tests via la fixture `trained_checkpoint`
- Valider le pipeline complet (chargement, labellisation, transformation, fenetrage, entrainement, evaluation)
- Verifier l'absence de biais look-ahead, l'assemblage des features et la normalisation

Tous les fichiers de test importent le chemin du jeu de donnees depuis `conftest.py` via la constante `TEST_CSV`.

---

## Depannage

### L'installation de mamba-ssm echoue

C'est attendu sur les systemes CPU uniquement. Le paquet `mamba-ssm` necessite CUDA. Daikoku utilisera automatiquement le mode de repli en PyTorch pur (`mamba_cpu.py`). Vous pouvez ignorer cette erreur en toute securite et ne pas installer `mamba-ssm` ni `causal-conv1d`.

### Memoire CUDA insuffisante

- Reduire `BATCH_SIZE` dans `config.py` (essayez 64 ou 32)
- Reduire `WINDOW_SIZE` (essayez 60 ou 80)
- Reduire la taille du modele : diminuer `D_MODEL`, `N_LAYERS` ou `D_STATE`
- Definir `MIXED_PRECISION = True` (par defaut) pour utiliser float16
- Reduire `NUM_WORKERS` pour liberer de la memoire systeme

### L'entrainement est lent sur CPU

- Definir `NUM_WORKERS = 0` (le surcout du multi-worker est pire sur CPU)
- Definir `MIXED_PRECISION = False` (AMP n'apporte aucun benefice sur CPU)
- Reduire `WINDOW_SIZE` et `BATCH_SIZE`
- Utiliser des dimensions de modele plus petites

### Forcer le mode CPU

```python
# Dans config.py
DEVICE = "cpu"
```

Ou via CLI :
```bash
python evaluate.py --device cpu
python inference.py --device cpu
```

### Fichier de donnees introuvable

Assurez-vous que vos fichiers CSV sont au bon chemin relatif a la racine de Daikoku. Les chemins dans `INPUT_FILES` sont relatifs au repertoire de travail. Executez les commandes depuis le repertoire racine de Daikoku.

### Limites de debit de l'API de l'echange

Si le telechargement de donnees est ralenti, augmentez le parametre `--sleep` :
```bash
python modules/tools/download_candles.py --exchange binance --timeframe 1h --start 2020-01-01 --sleep 2.0
```

### Bitget necessite des cles API

Bitget ne renvoie qu'environ 100 chandeliers recents sans authentification API. Pour les donnees historiques completes, vous devez configurer les cles API dans le dictionnaire `API_KEYS` en haut de `modules/tools/download_candles.py`.

### TensorBoard n'affiche pas de donnees

Assurez-vous de pointer vers le bon repertoire de logs :
```bash
tensorboard --logdir runs/
```

Les sessions sont numerotees sequentiellement (`run_001`, `run_002`, etc.). Vous pouvez visualiser une session specifique :
```bash
tensorboard --logdir runs/run_001
```

### Compatibilite des points de controle

Les points de controle stockent la configuration complete utilisee lors de l'entrainement. Lors du chargement d'un point de controle pour l'evaluation ou l'inference, l'architecture du modele est reconstruite a partir de la configuration sauvegardee, et non a partir du `config.py` actuel. Cela signifie que vous pouvez modifier `config.py` en toute securite entre les sessions d'entrainement sans casser les points de controle existants.

---

## Flux de travail typique

```
1. Telecharger les donnees
   python modules/tools/download_candles.py --exchange binance --pair BTC/USDT --timeframe 1h --start 2020-01-01 --output data/BTC_1h_bin.csv

2. Editer config.py
   Definir INPUT_FILES, PREDICTION_TARGET, parametres du modele, parametres d'entrainement

3. Entrainer
   python main.py

4. Surveiller
   tensorboard --logdir runs/

5. Evaluer
   python evaluate.py

6. (Optionnel) Optimiser les hyperparametres
   python optimize.py --trials 50

7. Passer en production
   python inference.py --pair BTC/USDT --timeframe 1h
```
