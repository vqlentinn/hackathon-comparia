
# 🦎 L'Arène se mord la queue

> Audit de **Compar:IA** à l'épreuve de la loi de Goodhart : pas de convergence
> sur cohorte fixe (R1-bis), vulnérabilité au style (R3/R5/R6).

Hackathon Compar:IA — ACSS Institute, Université Paris-Dauphine PSL — juin 2026.
Équipe : Valentin Proux & coéquipier.

---

## Thèse

> Compar:IA ne s'effondre pas et ne converge pas stylistiquement sur une cohorte
> fixe, mais reste **vulnérable au formatage** (style, longueur, structure).

| Module | Question | Méthode |
|---|---|---|
| **R1 / R1-bis** | Convergence ou artefact de composition ? | diversité mensuelle + cohorte stable densifiée |
| **R2bis** | Le style premium augmente-t-il ? | BT mensuel style-controlled |
| **R3** | Effet causal du format ? | contrefactuels + juges LLM |
| **R5 / R5bis** | Style vs qualité + longueur ? | logit + AUC |
| **R6** | Gaming des modèles faibles ? | tiers + interactions BT |

Docs : [`docs/pitch_equipe_reference.md`](docs/pitch_equipe_reference.md) · [`docs/R1bis_resultats.md`](docs/R1bis_resultats.md)

---

## Installation

```bash
# Python 3.12 requis (3.14 incompatible avec la stack scientifique)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_lg
```

Puis copier le template d'environnement et y mettre vos clés :

```bash
cp .env.example .env
# éditer .env : HF_TOKEN, MISTRAL_API_KEY, GROQ_API_KEY
```

## Documentation pitch & présentation

| Fichier | Usage |
|---|---|
| [`docs/pitch_dust_brief.md`](docs/pitch_dust_brief.md) | Brief pour Dust — génération des slides |
| [`docs/pitch_equipe_reference.md`](docs/pitch_equipe_reference.md) | Référence équipe — définitions, chiffres, FAQ jury |
| [`docs/project_status.md`](docs/project_status.md) | État technique complet du projet |

---

```bash
.venv/bin/python scripts/check_access.py
```

Teste les 3 clés API et affiche le schéma réel des datasets HF gated
(`comparia-conversations`, `comparia-votes`).

---

## Structure

```
arene-mord-queue/
├── data/{raw,interim,processed}/   # gitignored (datasets ~9 Go)
├── notebooks/                      # 01_ingestion … 08_robustness
├── src/compariawatch/             # librairie (data, features, diversity,
│                                   #   counterfactual, judge, forecast)
├── scripts/                        # check_access, download_data, …
├── tests/
├── paper/                          # note LaTeX 5 pages + figures
└── slides/
```

## Pipeline (sorties clés)

1. `data/interim/df_paired.parquet` — paires votées FR (Phase 1)
2. `data/interim/df_with_stylo.parquet` — features stylo (Phase 2)
3. `paper/figures/R1_convergence.png` — convergence (Phase 3)
4. `paper/figures/R3_style_premium.png` — Style Premium causal (Phase 4)
5. `paper/figures/R2_gap.png`, `R4_forecast.png` (Phase 5)

## Reproductibilité

`random_state = 42` partout. Versions exactes dans `requirements.lock`
(`pip freeze` après install). Voir `docs/reproducibility.md`.


