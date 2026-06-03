# R1-bis — Résultats du run

> Strictement factuel, sans interprétation. L'interprétation Goodhart est à
> faire avec le binôme à partir de ces chiffres + de la figure
> `paper/figures/R1bis_cohorte_stable.png`.

## Run

- **Date** : 2026-06-03
- **Script** : `scripts/r1bis_cohorte_stable.py`
- **Seed** : 42 (OLS déterministe ; `np.random.seed(42)` posée par sûreté)
- **Sorties** :
  - `data/processed/dispersion_cohorte.csv` (12 lignes × 4 colonnes)
  - `paper/figures/R1bis_cohorte_stable.png`

## Périmètre — entonnoir de filtrage

| Étape | Lignes |
|---|---|
| `data/interim/battles_with_dates.parquet` (brut) | 142 243 |
| Filtre `source == "vote"` | 115 680 (81,3 %) |
| Exclusion `winner == "tie"` | 79 864 |
| `dropna(timestamp)` (NaT = 0,99 %) | 79 075 |
| `densify_features` (merge `votes_length` = 100 %) | **79 075** |

## Cohorte stable

Critère : **≥ 100 battles totales sur ≥ 12 mois distincts** (présence côté A ou B).
Fallback à 10 mois prévu si < 6 modèles → **non déclenché** (cohorte = 9).

| Modèle | Battles | Mois actifs |
|---|---:|---:|
| llama-3.1-405b | 5 760 | 13 |
| llama-3.1-8b | 5 201 | 15 |
| ministral-8b-instruct-2410 | 4 790 | 12 |
| phi-4 | 4 524 | 12 |
| llama-3.3-70b | 4 468 | 13 |
| gemma-3-4b | 4 287 | 12 |
| gemma-3-27b | 3 807 | 12 |
| gemma-3-12b | 3 690 | 12 |
| command-a | 3 428 | 12 |

## Fenêtre temporelle retenue

- **12 mois** : 2025-02 → 2026-01.
- **Mois écartés** (cohorte active < 5 ce mois-là, règle de skip) :
  2024-10, 2024-11, 2024-12, 2025-01, 2026-02.

## Métrique

Distance euclidienne moyenne entre centroïdes de modèles, **identique pour
global et cohorte** :

1. Pour chaque réponse, features = `[headers, lists, bold, code_blocks, emoji]`
   **densifiées** (`feature / assistant_chars` du même côté).
2. Z-score global sur toutes les réponses (fit unique de `StandardScaler`).
3. Centroïde par `(model, month)` sur les modèles avec ≥ 100 réponses ce
   mois-là.
4. `dispersion_globale` = `mean(pdist(centroïdes des modèles actifs))`.
5. `dispersion_cohorte` = `mean(pdist(centroïdes des modèles ∈ cohorte ∧ actifs))`.

## OLS — `dispersion ~ month_idx`

| Série | Pente | IC 95 % | p | R² |
|---|---:|---|---:|---:|
| dispersion_globale | +0,03839 | [+0,01474, +0,06204] | 0,0047 | 0,567 |
| dispersion_cohorte | −0,00330 | [−0,02228, +0,01569] | 0,71 | 0,015 |

## Série mensuelle

| Mois | dispersion_globale | dispersion_cohorte | n_models_global | n_models_cohorte |
|---|---:|---:|---:|---:|
| 2025-02 | 0,8909 | 0,5807 | 24 | 5 |
| 2025-03 | 0,9904 | 0,7352 | 32 | 9 |
| 2025-04 | 1,0059 | 0,7421 | 38 | 9 |
| 2025-05 | 0,9307 | 0,8013 | 27 | 9 |
| 2025-06 | 0,9677 | 0,8321 | 27 | 9 |
| 2025-07 | 0,9808 | 0,8207 | 24 | 8 |
| 2025-08 | 0,8868 | 0,6254 | 21 | 9 |
| 2025-09 | 0,9281 | 0,6854 | 29 | 9 |
| 2025-10 | 1,3608 | 0,7039 | 37 | 7 |
| 2025-11 | 1,3392 | 0,8718 | 28 | 7 |
| 2025-12 | 1,2694 | 0,6350 | 24 | 7 |
| 2026-01 | 1,2725 | 0,5964 | 23 | 5 |

## Note de comparabilité

Les features sont **densifiées** (`feature / assistant_chars`) pour neutraliser
l'effet longueur. L'échelle absolue diffère donc de l'analyse R1 historique
(`r1_robustness.py`, comptes bruts). Choix méthodologique assumé : ce qui
importe pour le test R1-bis est la comparabilité **interne au run** entre
`dispersion_globale` et `dispersion_cohorte` (mêmes features, même z-score,
même métrique, seul le sous-ensemble de modèles change), pas la réplication
des valeurs numériques de la courbe historique 0,20 → 0,74.
