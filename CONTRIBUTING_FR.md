# Contribuer à Daikoku

*[English version](CONTRIBUTING.md)*

Merci de votre intérêt pour contribuer à Daikoku ! Ce document explique comment participer.

---

## Signaler un bug

Ouvrez une [Issue GitHub](../../issues) avec :

- **Version Python**, **version PyTorch**, **modèle GPU** (ou CPU uniquement)
- **Étapes pour reproduire** le problème
- **Traceback complet de l'erreur**
- **Modifications de config** (le cas échéant) par rapport aux défauts de `config.py`

## Proposer une fonctionnalité

Ouvrez une [Issue GitHub](../../issues) avec le label "feature request" **avant d'écrire du code**. Décrivez :

- Quel problème cela résout-il ?
- Comment cela s'intègre-t-il dans l'architecture existante ?

Cela évite un travail inutile sur des fonctionnalités qui ne correspondent pas à la direction du projet.

## Soumettre du code

### Workflow

1. **Forkez** le dépôt
2. **Créez une branche feature** depuis `master` : `git checkout -b feature/ma-feature`
3. **Faites vos modifications** — gardez des commits ciblés et descriptifs
4. **Lancez les tests** : `python -m pytest tests/ -v`
5. **Ouvrez une Pull Request** vers `master` avec une description claire

### Conventions de code

- **Commentaires en anglais**, toujours
- **Configuration** : tous les paramètres sont dans `config.py` — ne jamais utiliser de fallbacks `getattr(config, 'X', default)`
- **Pas de compatibilité ascendante** : les changements cassants sur les checkpoints ou la config sont acceptables
- **Parité** : `main.py` et `optimize.py` doivent produire des résultats identiques pour la même config — toute modification de l'un doit être reflétée dans l'autre
- **Labels sur données brutes** : l'étiquetage Triple Barrier est toujours calculé avant toute transformation des données (pas de fuite d'information)
- **Les tests doivent passer** : ne jamais modifier un test juste pour qu'il passe — corriger le code source à la place

### Ce que nous acceptons

- Corrections de bugs avec des étapes de reproduction claires
- Améliorations de performance appuyées par des benchmarks
- Nouvelles fonctionnalités dans le périmètre de la prédiction crypto basée sur Mamba
- Améliorations de la documentation

### Ce que nous n'acceptons pas

- Fonctionnalités hors du périmètre du projet (ex. actifs non-crypto, architectures non-Mamba)
- Modifications qui cassent les tests existants sans justification

## Environnement de développement

Suivez les instructions complètes d'installation dans le [Guide d'utilisation](docs/Guide_FR.md#installation). En résumé :

1. **CUDA 12.4+** — requis pour l'entraînement GPU. Installer depuis [NVIDIA](https://developer.nvidia.com/cuda-toolkit)
2. **Python 3.11**
3. **PyTorch 2.4.1** avec support CUDA :
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
5. **Noyaux CUDA Mamba** (fortement recommandé) :
   ```bash
   pip install mamba-ssm==2.3.0 causal-conv1d==1.5.2
   ```
6. **Lancer les tests** :
   ```bash
   python -m pytest tests/ -v
   ```

La suite de tests nécessite `tests/data/test_Dataset.csv` (inclus dans le dépôt). Voir le [Guide d'utilisation](docs/Guide_FR.md#tests) pour plus de détails.

## Structure du projet

- `config.py` — source unique de vérité pour tous les paramètres
- `main.py` — pipeline d'entraînement
- `optimize.py` — optimisation des hyperparamètres (doit rester synchronisé avec `main.py`)
- `evaluate.py` — évaluation hors ligne
- `inference.py` — inférence en direct avec connexion à un échange
- `modules/` — tout le code d'implémentation
- `tests/` — suite de tests (pytest)

Pour les détails techniques complets, voir [Architecture_FR.md](docs/Architecture_FR.md).
