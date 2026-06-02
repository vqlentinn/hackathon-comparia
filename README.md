
# 🦎 L'Arène se mord la queue

> Audit causal de **Compar:IA** à l'épreuve de la loi de Goodhart : convergence
> stylistique, dérive arène-réalité, et projection d'effondrement du benchmark.

Hackathon Compar:IA — ACSS Institute, Université Paris-Dauphine PSL — juin 2026.
Équipe : Valentin Proux & coéquipier.

---

## Thèse

Compar:IA, devenu cible d'entraînement implicite des LLM (RLHF/DPO), perd
progressivement son pouvoir de mesurer la qualité. On le démontre via 4 modules :

| Module | Question | Méthode |
|---|---|---|
| **R1** | La diversité stylistique inter-modèles décroît-elle ? | features stylo + Wasserstein (POT) + OLS |
| **R2** | L'écart Elo Compar:IA vs benchmarks externes croît-il ? | rank gap + OLS sur l'âge des modèles |
| **R3** | Quelle part causale du style dans la préférence ? | contrefactuels LLM, NLI+cosine, judge, ATE |
| **R4** | Quand le benchmark cessera-t-il de discriminer ? | state-space bayésien NumPyro (fallback Prophet) |

Spec complète : [`projet_arene_se_mord_la_queue.md`](projet_arene_se_mord_la_queue.md).

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

## Validation de l'accès aux données

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
=======
# hackathon-comparia

