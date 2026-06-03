# Handoff R1-bis — cohorte stable

> Pour le binôme. Données dans `comparia-R1bis-donnees.tar.gz` (Drive).

## Données (extraire à la racine du repo)

| Fichier | Rôle |
|---------|------|
| `data/interim/battles_with_dates.parquet` | Battles + features style + dates |
| `data/raw/votes_length.parquet` | Caractères réponse A/B → **densité** style |
| `data/raw/votes_timestamps.parquet` | Cache timestamps (repro jointure) |
| `style-control-analysis/battles_bt_styled.parquet` | Source Zilinskas |
| `data/processed/diversity_temporal.parquet` | R1 global (référence) |
| `data/processed/diversity_temporal_robustness.parquet` | R1b existant (≠ R1-bis) |

## Code à réutiliser

- `src/compariawatch/diversity.py` — `compute_monthly_diversity`, `battles_to_long`
- `src/compariawatch/data.py` — jointures
- `src/compariawatch/style_quality.py` — `load_votes_length`
- **Ne pas** utiliser `features.py` (non implémenté)

## Pièges

1. **Périmètre** : `source == "vote"` uniquement (~1 % timestamps manquants). Sans filtre → 19 % (réactions).
2. **Timestamps** : jointure via `comparia-votes`, pas conversations (OK si battles votées).
3. **Densité** : `feature / assistant_chars` après merge `votes_length`.
4. **Métrique R1 actuelle** : `method="centroid"` dans `r1_robustness.py` — même chose pour global et cohorte.
5. **R1b existant** (`r1_robustness.py`) ≠ spec R1-bis (6 mois récurrents, filtre A∩B).

## Livrables

- `paper/figures/R1bis_cohorte_stable.png`
- `data/processed/dispersion_cohorte.csv`
- Module dans `src/compariawatch/`
- Pas de push / pas d'interprétation Goodhart sans accord Valentin.

## Vérif rapide

```bash
python -c "
import pandas as pd
b = pd.read_parquet('data/interim/battles_with_dates.parquet')
v = b[b.source == 'vote']
print('votes', len(v), 'missing ts', v.timestamp.isna().mean())
assert (v.timestamp.isna().mean() < 0.15)
print('OK')
"
```

Spec complète : message Slack / brief R1-bis du binôme.

**Résultats du run** : voir [`R1bis_resultats.md`](R1bis_resultats.md).
