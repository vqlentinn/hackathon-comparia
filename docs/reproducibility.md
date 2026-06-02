# Reproductibilité

## Environnement

- **Python 3.12** (le 3.14 par défaut de la machine est incompatible avec la
  stack scientifique : pas de wheels pour spaCy/torch/numpyro/hdbscan/prophet).
- venv isolé : `.venv/` à la racine.
- Versions épinglées : `requirements.txt` (contraintes) + `requirements.lock`
  (gel exact via `pip freeze`, à régénérer après chaque install).

```bash
.venv/bin/pip freeze > requirements.lock
```

## Graine aléatoire

`random_state = 42` partout (sklearn, numpy, xgboost, échantillonnages,
NumPyro `rng_key`). Constante exposée : `compariawatch.RANDOM_STATE`.

## Données

Datasets gated HF non versionnés (volume). Pour reproduire :
1. obtenir l'accès aux datasets `ministere-culture/comparia-*` ;
2. `HF_TOKEN` dans `.env` ;
3. lancer `notebooks/01_ingestion.ipynb` → régénère `data/interim/df_paired.parquet`.

## Ordre d'exécution des notebooks

`01_ingestion` → `02_features_stylo` → `03_R1_convergence` →
`04_R3_counterfactual` → `05_R2_gap` → `06_R4_forecast` →
`07_figures_paper` → `08_robustness`.
