# Commit Log

Fichier à alimenter après chaque phase. Copier-coller la commande dans le terminal.

## Phase 0-1 — Setup + validation accès

```bash
git add .gitignore .env.example README.md pyproject.toml requirements.txt docs/ src/ scripts/check_access.py tests/ data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep paper/figures/.gitkeep paper/tables/.gitkeep slides/.gitkeep notebooks/01_validation_zilinskas.ipynb
git commit -m "$(cat <<'EOF'
chore: initialise le repo ComparIA hackathon

- crée la structure projet, le venv Python 3.12 et les fichiers de config
- ajoute le squelette compariawatch avec docstrings et types
- ajoute le script check_access pour valider HF, Mistral et Groq
EOF
)"
```

## Phase 1b — Jointure timestamps HF

```bash
git add src/compariawatch/data.py scripts/join_timestamps.py notebooks/02_timestamp_join.ipynb docs/data_dictionary.md data/raw/.gitkeep data/interim/.gitkeep
git commit -m "$(cat <<'EOF'
feat(data): joint les timestamps HF aux battles Zilinskas

- stream comparia-votes et met en cache les timestamps par conversation_pair_id
- produit data/interim/battles_with_dates.parquet pour les analyses temporelles
- documente le schéma réel des datasets ComparIA
EOF
)"
```

## Phase R1 — Convergence stylistique brute

```bash
git add src/compariawatch/diversity.py notebooks/03_R1_convergence.ipynb
git add -f data/processed/diversity_temporal.parquet paper/figures/R1_convergence.png
git commit -m "$(cat <<'EOF'
feat(R1): mesure la diversité stylistique mensuelle

- calcule la diversité inter-modèles par mois sur les features de style Zilinskas
- estime une tendance OLS sur 17 cohortes mensuelles
- génère la figure R1 et le parquet diversity_temporal
EOF
)"
```

## Phase R1b — Robustesse à faire

```bash
git add scripts/r1_robustness.py docs/project_status.md docs/commit_log.md
git add -f data/processed/diversity_temporal_robustness.parquet paper/figures/R1_convergence_robustness.png
git commit -m "$(cat <<'EOF'
test(R1): ajoute les contrôles de robustesse de convergence

- contrôle la tendance brute par le nombre de modèles et le volume mensuel
- compare fenêtres mensuelles, trimestrielles et cohortes à modèles constants
- clarifie si l'augmentation observée reflète l'arrivée de nouveaux modèles
EOF
)"
```

## Documentation — Suivi projet

```bash
git add docs/project_status.md docs/commit_log.md
git commit -m "$(cat <<'EOF'
docs: ajoute le suivi projet et les hypothèses révisées

- documente l'état du pipeline et les artefacts produits
- consigne le résultat R1 brut et son interprétation
- ajoute les hypothèses révisées et les prochaines analyses de robustesse
EOF
)"
```

## Phase R2bis — Style Premium longitudinal

```bash
git add src/compariawatch/style_premium.py scripts/r2bis_style_premium.py notebooks/04_R2bis_style_premium.ipynb docs/project_status.md docs/commit_log.md
git add -f data/processed/style_premium_temporal.parquet data/processed/style_premium_trends.parquet paper/figures/R2bis_style_premium_longitudinal.png
git commit -m "$(cat <<'EOF'
feat(R2bis): mesure le Style Premium longitudinal

- recalcule les coefficients Bradley-Terry style-controlled par mois
- teste la tendance temporelle de bold, lists et headers avec bootstrap
- produit les parquets R2bis et la figure longitudinale
EOF
)"
```

## Phase R3 — Smoke test contrefactuels

```bash
git add src/compariawatch/counterfactual.py scripts/r3_smoke_counterfactual.py notebooks/05_R3_counterfactual_smoke.ipynb docs/project_status.md docs/commit_log.md
git add -f data/interim/rewrites_smoke.parquet
git commit -m "$(cat <<'EOF'
feat(R3): valide le smoke test de contrefactuels

- extrait 10 réponses gagnantes depuis comparia-votes
- génère trois styles via Mistral avec retry et sauvegarde incrémentale
- produit rewrites_smoke.parquet pour la vérification sémantique suivante
EOF
)"
```

## Phase R3 — Scoring smoke

```bash
git add src/compariawatch/judge.py scripts/r3_score_smoke.py docs/project_status.md docs/commit_log.md
git add -f data/interim/rewrites_smoke_scored.parquet data/processed/causal_style_votes_smoke.parquet paper/figures/R3_style_premium_smoke.png
git commit -m "$(cat <<'EOF'
feat(R3): score les contrefactuels smoke

- calcule la similarité cosine original/réécriture sur les 30 contrefactuels
- juge concise et verbose contre neutre avec ordre A/B randomisé
- produit les votes smoke et la figure R3 provisoire
EOF
)"
```

## Phase R3 — Batch N=100

À lancer après fin du batch + scoring :

```bash
git add scripts/r3_smoke_counterfactual.py scripts/r3_score_smoke.py docs/project_status.md docs/commit_log.md
git add -f data/interim/rewrites_n100.parquet data/interim/rewrites_n100_scored.parquet data/processed/causal_style_votes_n100.parquet paper/figures/R3_style_premium_n100.png
git commit -m "$(cat <<'EOF'
feat(R3): étend les contrefactuels causaux à 100 réponses

- génère 300 réécritures Mistral sur 100 réponses source
- filtre les paires jugées avec similarité cosine suffisante
- produit les votes causaux N100 et la figure R3 principale
EOF
)"
```

## Phase R5 — Style vs qualité

```bash
git add src/compariawatch/style_quality.py scripts/r5_style_vs_quality.py docs/project_status.md docs/commit_log.md
git add -f data/raw/votes_quality.parquet data/processed/style_quality_dataset.parquet data/processed/style_quality_model_summary.parquet data/processed/style_quality_coefficients.parquet paper/figures/R5_style_vs_quality.png
git commit -m "$(cat <<'EOF'
feat(R5): contrôle le style par les labels qualité ComparIA

- joint les features de style aux labels qualité de comparia-votes
- compare des modèles logit qualité seule, style seul et qualité plus style
- montre que le style améliore la prédiction du vote après contrôle qualité
EOF
)"
```
