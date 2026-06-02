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
