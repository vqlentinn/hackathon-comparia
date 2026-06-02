# Project Status — L'Arène se mord la queue

Dernière mise à jour : mardi 2 juin 2026, 19h55.

Ce fichier sert de **source de vérité opérationnelle** pour suivre l'avancement du
hackathon, les choix méthodologiques, les résultats déjà observés, les hypothèses
à réviser et les prochaines décisions. Il est volontairement détaillé pour être
relisible directement depuis Cursor/Dust sans reconstituer l'historique du chat.

---

## 1. Positionnement du projet

### Titre

**L'Arène se mord la queue : audit causal de Compar:IA à l'épreuve de la loi de Goodhart**

### Thèse initiale

Compar:IA, plateforme française d'évaluation de LLM par préférence humaine,
devient une cible d'entraînement implicite des laboratoires d'IA. Les modèles
optimisent alors les signaux gagnants de l'arène, notamment le style, ce qui
peut dégrader progressivement la validité du benchmark.

### Reformulation actuelle

Après intégration du travail de Zilinskas, le projet devient une **extension
temporelle, causale et prospective** de son analyse de biais de style sur
Compar:IA.

On ne refait pas son résultat principal global. On s'appuie dessus pour tester :

- si le biais de style évolue dans le temps ;
- si la diversité stylistique observée baisse vraiment ou si elle est masquée
  par l'arrivée de nouveaux modèles ;
- si un changement de style à contenu constant modifie causalement le jugement ;
- si les tendances observées permettent une projection crédible.

---

## 2. Travail antérieur intégré

### Source

Dossier local :

`style-control-analysis/`

Fichier clé :

`style-control-analysis/battles_bt_styled.parquet`

### Ce que contient le parquet

- 142 243 battles Compar:IA.
- 89 modèles.
- Colonnes de style Zilinskas :
  - `headers_a`, `headers_b`
  - `lists_a`, `lists_b`
  - `bold_a`, `bold_b`
  - `code_blocks_a`, `code_blocks_b`
  - `emoji_a`, `emoji_b`
- Colonnes de modèles :
  - `model_a_name`
  - `model_b_name`
- Colonne outcome :
  - `winner` avec valeurs `model_a`, `model_b`, `tie`
- Colonne source :
  - `vote`
  - `reaction`

### Ce que le parquet ne contient pas

Il ne contient pas de `timestamp`. C'était le premier blocage pour R1/R2bis/R4.

Solution implémentée : jointure des timestamps depuis HuggingFace
`ministere-culture/comparia-votes`.

---

## 3. Setup technique

### Environnement

- Python utilisé : **3.12.12**
- Venv : `.venv/`
- Python 3.14 évité, car trop récent pour la stack scientifique
  (`torch`, `spacy`, `numpyro`, `hdbscan`, `prophet`).

### Accès API validés

Le script `scripts/check_access.py` valide :

- `HF_TOKEN` : OK
- `MISTRAL_API_KEY` : OK
- `GROQ_API_KEY` : OK
- `ministere-culture/comparia-conversations` : OK
- `ministere-culture/comparia-votes` : OK

### Schémas HF réels validés

`comparia-conversations` contient notamment :

- `conversation_pair_id`
- `timestamp`
- `languages`
- `opening_msg`
- `conversation_a`
- `conversation_b`
- `model_a_name`
- `model_b_name`

`comparia-votes` contient notamment :

- `conversation_pair_id`
- `timestamp`
- `chosen_model_name`
- `both_equal`
- `model_a_name`
- `model_b_name`

Important : la spec initiale parlait de `preference`. Le schéma réel n'a pas de
colonne `preference`. Le gagnant est à reconstruire via `chosen_model_name` et
`both_equal`.

---

## 4. Artefacts produits

### Scripts

- `scripts/check_access.py`
  - valide `.env`, HF, Mistral, Groq ;
  - affiche les colonnes réelles HF ;
  - supporte `mistralai >= 2.x`.

- `scripts/join_timestamps.py`
  - stream `comparia-votes` ;
  - extrait `conversation_pair_id` + `timestamp` ;
  - cache le résultat ;
  - joint avec le parquet Zilinskas ;
  - produit `battles_with_dates.parquet`.

### Modules Python

- `src/compariawatch/data.py`
  - `load_battles_zilinskas`
  - `fetch_vote_timestamps`
  - `join_battles_with_dates`

- `src/compariawatch/diversity.py`
  - `battles_to_long`
  - `standardize_features`
  - `compute_monthly_diversity`
  - `mean_centroid_diversity`
  - `wasserstein_diversity`

### Notebooks

- `notebooks/01_validation_zilinskas.ipynb`
  - sanity check du parquet Zilinskas ;
  - reproduction des coefficients BT style-controlled.

- `notebooks/02_timestamp_join.ipynb`
  - jointure timestamps ;
  - EDA mensuelle ;
  - contrôle couverture temporelle.

- `notebooks/03_R1_convergence.ipynb`
  - diversité stylistique mensuelle ;
  - OLS ;
  - figure R1.

### Données intermédiaires

Gitignored localement :

- `data/raw/votes_timestamps.parquet`
- `data/interim/battles_with_dates.parquet`
- `data/processed/diversity_temporal.parquet`

### Figures

Gitignored localement :

- `paper/figures/R1_convergence.png`
- `paper/figures/EDA_battles_monthly.png`

---

## 5. Résultats obtenus

### 5.1 Sanity check Zilinskas

Méthode répliquée depuis `clean_and_analyze.py` :

- modèle Bradley-Terry via régression logistique ;
- `X_model = +1/-1` pour les deux modèles ;
- `y = 1` si `model_a` gagne, `0` sinon ;
- `X_style = style_a - style_b`, standardisé ;
- pas de miroir artificiel des observations.

Résultat local :

| Feature | Effet local approximatif | Référence attendue |
|---|---:|---:|
| headers | +15.3 % / SD | ~+15.6 % |
| lists | +18.8 % / SD | ~+18 % |
| bold | +16.8 % / SD | ~+19 % |
| code_blocks | ~+0.8 % / SD | ~0 % |
| emoji | ~+2.4 % / SD | ~0 % |

Conclusion : le pipeline BT style-controlled est validé.

### 5.2 Jointure timestamps

Résultat :

- votes streamés : ~149k ;
- battles Zilinskas : 142 243 ;
- battles avec timestamp : 114 626 ;
- match rate : **80.6 %** ;
- explication du non-match : battles `source=reaction` absentes de
  `comparia-votes`.

Cohortes temporelles :

- 18 mois disponibles ;
- période observée : oct. 2024 à fév. 2026 ;
- pour R1 décisif + timestamp : 79 075 battles.

### 5.3 R1 brute — Convergence stylistique

Définition actuelle :

- format long : une ligne = une réponse modèle ;
- features : 5 features style Zilinskas ;
- z-score global ;
- centroïde stylistique par `(model, month)` ;
- diversité mensuelle `D_t` = distance euclidienne moyenne entre centroïdes de
  modèles.

Résultat OLS :

| Métrique | Valeur |
|---|---:|
| battles utilisées | 79 075 |
| cohortes | 17 |
| pente mensuelle `β` | +0.057 |
| p-value | < 0.001 |
| R² | 0.925 |

Conclusion brute :

**H1 initiale non confirmée.** La diversité stylistique brute augmente dans le
temps, au lieu de diminuer.

### 5.4 R1b — Robustesses et contrôles

Script :

`scripts/r1_robustness.py`

Outputs :

- `data/processed/diversity_temporal_robustness.parquet`
- `paper/figures/R1_convergence_robustness.png`

Tests effectués :

1. OLS brute :

```text
diversity ~ month_idx
```

2. OLS contrôlée :

```text
diversity ~ month_idx + n_models + n_responses
```

3. Corrélations avec la composition de l'arène :

```text
corr(diversity, n_models)
corr(diversity, n_responses)
```

4. Sous-échantillon modèles récurrents :

```text
garder les modèles présents dans au moins 6 mois
```

5. Cohortes trimestrielles :

```text
diversity par quarter
```

Résultats :

| Spécification | Pente temporelle | p-value | R² | Lecture |
|---|---:|---:|---:|---|
| mensuelle brute | +0.0572 | < 0.001 | 0.925 | diversité en hausse |
| mensuelle contrôlée | +0.0466 | < 0.001 | 0.961 | hausse persiste |
| modèles récurrents ≥6 mois | +0.0412 | < 0.001 | 0.529 | hausse persiste |
| trimestrielle | +0.1575 | 0.001 | 0.942 | hausse persiste |

Corrélations :

| Variable | Corrélation avec diversité |
|---|---:|
| `n_models` | +0.826 |
| `n_responses` | -0.041 |

Conclusion R1b :

La diversité stylistique **augmente robustement**. L'arrivée de nouveaux modèles
explique une partie de la hausse (`n_models` corrélé à +0.826), mais ne suffit
pas à l'annuler : la tendance temporelle reste positive après contrôle et sur
les modèles récurrents.

Cela réfute la version simple de H1 ("convergence stylistique globale"). Il faut
repositionner R1 comme un résultat négatif mais informatif : l'arène n'est pas
encore en convergence globale ; elle est plutôt en phase d'expansion et de
différenciation stylistique.

### 5.5 R2bis — Style Premium longitudinal

Script :

`scripts/r2bis_style_premium.py`

Module :

`src/compariawatch/style_premium.py`

Notebook :

`notebooks/04_R2bis_style_premium.ipynb`

Outputs :

- `data/processed/style_premium_temporal.parquet`
- `data/processed/style_premium_trends.parquet`
- `paper/figures/R2bis_style_premium_longitudinal.png`

Méthode :

- battles décisives avec timestamp : 79 075 ;
- 17 cohortes mensuelles ;
- Bradley-Terry style-controlled par mois ;
- features : `headers`, `lists`, `bold`, `code_blocks`, `emoji` ;
- bootstrap par cohorte : `n_boot=40` ;
- tendance : OLS `odds_pct ~ month_idx` par feature.

Résultats :

| Feature | Style Premium moyen | Pente mensuelle | p-value | Lecture |
|---|---:|---:|---:|---|
| `bold` | +19.8 % | -1.40 pts/mois | 0.013 | diminue significativement |
| `lists` | +14.6 % | -0.79 pts/mois | 0.031 | diminue significativement |
| `headers` | +14.5 % | +0.48 pts/mois | 0.546 | stable / non significatif |
| `code_blocks` | +4.4 % | -0.75 pts/mois | 0.015 | diminue, effet moyen faible |
| `emoji` | +5.2 % | +0.35 pts/mois | 0.401 | non significatif |

Conclusion R2bis :

Le Style Premium est **réel en niveau** : les effets moyens de `bold`, `lists`
et `headers` restent autour de +14 % à +20 % d'odds de victoire par écart-type.
En revanche, il ne croît pas dans le temps. Au contraire, `bold` et `lists`
diminuent significativement sur la période observée.

Conséquence :

H2 ("le style premium augmente dans le temps") n'est pas confirmée. Le résultat
reste utile : il montre que le style influence déjà fortement les votes, mais
que le signal de Goodhart temporel n'est pas visible sur ces features simples.

### 5.6 R3 — Smoke test contrefactuels

Script :

`scripts/r3_smoke_counterfactual.py`

Module :

`src/compariawatch/counterfactual.py`

Notebook :

`notebooks/05_R3_counterfactual_smoke.ipynb`

Output :

`data/interim/rewrites_smoke.parquet`

Scope :

- 10 réponses gagnantes depuis `comparia-votes` ;
- 3 styles (`verbose_markdown`, `concise_direct`, `neutre_baseline`) ;
- 30 appels Mistral ;
- sauvegarde incrémentale toutes les 5 réponses ;
- pas encore de NLI/cosine/judge.

Résultat :

| Style | N | Longueur originale moyenne | Longueur réécrite moyenne |
|---|---:|---:|---:|
| `verbose_markdown` | 10 | 2204 chars | 4423 chars |
| `concise_direct` | 10 | 2204 chars | 1110 chars |
| `neutre_baseline` | 10 | 2204 chars | 1580 chars |

Validation :

- extraction HF → OK ;
- choix de la réponse originale → OK ;
- Mistral rewrite → OK ;
- parquet → OK ;
- aucune valeur nulle `original` / `rewritten`.

Point méthodologique :

Le style `verbose_markdown` allonge très fortement les réponses. R3 devra donc
contrôler explicitement la longueur ou comparer aussi `verbose_markdown` vs
`neutre_baseline` en interprétant l'effet comme un bundle style+longueur.

### 5.7 R3 — Scoring smoke (cosine + LLM judge)

Script :

`scripts/r3_score_smoke.py`

Module :

`src/compariawatch/judge.py`

Outputs :

- `data/interim/rewrites_smoke_scored.parquet`
- `data/processed/causal_style_votes_smoke.parquet`
- `paper/figures/R3_style_premium_smoke.png`

Validation sémantique cosine :

| Style | Cosine moyen | Taux `cosine >= 0.85` |
|---|---:|---:|
| `concise_direct` | 0.865 | 70 % |
| `neutre_baseline` | 0.845 | 60 % |
| `verbose_markdown` | 0.856 | 50 % |

Judge smoke :

Comparaison de chaque style cible contre `neutre_baseline`.

| Style cible vs neutre | N | Victoire style | Victoire neutre | Tie |
|---|---:|---:|---:|---:|
| `concise_direct` | 10 | 100 % | 0 % | 0 % |
| `verbose_markdown` | 10 | 10 % | 90 % | 0 % |

Lecture :

Le signal causal de smoke est très fort, mais il va dans une direction
différente de l'intuition initiale : le juge préfère massivement la version
concise à la version neutre, et pénalise la version verbose-markdown.

Limites :

- N=10 seulement ;
- cosine insuffisant pour une partie des réécritures ;
- juge Mistral peut avoir un biais contre le verbiage ;
- `verbose_markdown` change fortement la longueur.

Décision :

R3 devient le cœur empirique. Il faut passer à N=100 avec :

- filtrage cosine strict ;
- comparaison `concise_direct` vs `neutre_baseline` ;
- comparaison `verbose_markdown` vs `neutre_baseline` ;
- analyse longueur comme médiateur.

### 5.8 R3 — Batch N=100

Statut :

**Terminé et scoré.**

Commande :

```bash
python scripts/r3_smoke_counterfactual.py \
  --n 100 \
  --output data/interim/rewrites_n100.parquet
```

Scope attendu :

- 100 réponses source ;
- 3 styles ;
- 300 réécritures Mistral ;
- sauvegarde incrémentale toutes les 5 réponses source ;
- sortie : `data/interim/rewrites_n100.parquet`.

Scoring :

```bash
python scripts/r3_score_smoke.py \
  --input data/interim/rewrites_n100.parquet \
  --scored-output data/interim/rewrites_n100_scored.parquet \
  --votes-output data/processed/causal_style_votes_n100.parquet \
  --figure paper/figures/R3_style_premium_n100.png \
  --require-cosine
```

Outputs :

- `data/interim/rewrites_n100.parquet`
- `data/interim/rewrites_n100_scored.parquet`
- `data/processed/causal_style_votes_n100.parquet`
- `paper/figures/R3_style_premium_n100.png`

Validation sémantique :

| Style | N | Cosine moyen | Taux `cosine >= 0.85` |
|---|---:|---:|---:|
| `concise_direct` | 100 | 0.868 | 68 % |
| `neutre_baseline` | 100 | 0.879 | 71 % |
| `verbose_markdown` | 100 | 0.848 | 49 % |

Jugements après filtre cosine :

Le filtre `--require-cosine` garde seulement les comparaisons où le style cible
et la baseline neutre passent tous deux le seuil `cosine >= 0.85`.

| Style cible vs neutre | N jugé | Victoires style | Victoires neutre | Tie | Win rate style |
|---|---:|---:|---:|---:|---:|
| `concise_direct` | 56 | 50 | 5 | 1 | 89.3 % |
| `verbose_markdown` | 38 | 8 | 30 | 0 | 21.1 % |

Résultat principal R3 :

À contenu proche selon cosine, le juge préfère massivement la version
**concise_direct** à la version neutre, tandis que la version
**verbose_markdown** est pénalisée.

Interprétation :

- L'effet causal du style existe fortement.
- Il ne va pas dans le sens "plus long/markdown = mieux" pour le juge Mistral.
- Le signal gagnant ici est plutôt **concision, densité, absence de verbiage**.
- Ce résultat est cohérent avec l'idée que les arènes peuvent favoriser des
  styles spécifiques, mais il contredit la version naïve "markdown verbose gagne".

Note méthodologique :

`--require-cosine` filtre les jugements pour ne garder que les paires où le
style cible et le neutre passent tous les deux le seuil `cosine >= 0.85`.
Cela réduit N mais augmente la crédibilité causale.

---

## 6. Interprétation actuelle de R1

### Ce qu'on ne doit pas dire

On ne peut pas écrire simplement :

> "La diversité augmente, donc la thèse est fausse."

Ce serait trop rapide, car la composition de l'arène change fortement dans le
temps.

### Explication testée

La diversité brute augmente en partie parce que de nouveaux modèles et familles
entrent progressivement dans Compar:IA. Le nombre de modèles par mois augmente
fortement, ce qui peut mécaniquement augmenter la distance moyenne entre
centroïdes.

Exemples observés :

- octobre 2024 : 19 modèles ;
- mars-avril 2025 : 32-39 modèles ;
- octobre 2025 : 47 modèles.

Les robustesses montrent cependant que cette explication n'est pas suffisante :
la tendance reste positive après contrôle de `n_models` et `n_responses`, et
reste positive sur les modèles récurrents.

### Décision méthodologique

R1 doit être reformulé en deux niveaux :

1. **R1a — diversité brute**
   - résultat observé : hausse ;
   - interprétation : expansion de l'arène / arrivée de modèles.

2. **R1b — diversité conditionnelle**
   - résultat observé : toujours hausse ;
   - interprétation : pas de convergence globale détectable sur la période ;
   - conséquence : ne pas fonder le pitch sur R1 comme preuve de convergence.

Ce pivot est important : R1 devient une **borne empirique**. Il ne soutient pas
la convergence globale, mais il montre que l'arène est encore en phase
d'expansion stylistique. Le cœur Goodhart doit donc venir de R2bis/R3 :
croissance du style premium et preuve causale par contrefactuels.

---

## 7. Hypothèses révisées

### H1 initiale

> La diversité stylistique inter-modèles décroît dans le temps.

Statut :

**Réfutée sur la période observée, en brut et après robustesses.**

### H1a — nouvelle hypothèse descriptive

> La diversité brute augmente avec l'expansion de l'arène, car l'arrivée de
> nouveaux modèles augmente l'hétérogénéité observée.

Test :

```text
diversity ~ month_idx
```

Statut :

**Supportée partiellement.** `n_models` est fortement corrélé à la diversité
(+0.826), mais la pente temporelle reste positive après contrôle.

### H1b — hypothèse Goodhart conditionnelle

> À composition contrôlée, ou au sein des modèles/familles récurrents, la
> diversité stylistique décroît.

Tests à faire :

```text
diversity ~ month_idx + n_models + n_responses
```

et :

```text
diversity sur modèles présents au moins K mois
```

Statut :

**Non confirmée.** Les modèles récurrents (présents au moins 6 mois) gardent une
tendance positive (`β=+0.0412`, `p<0.001`).

### H1c — nouvelle formulation conservatrice pour le papier

> Sur la période observée, Compar:IA ne montre pas encore de convergence
> stylistique globale ; l'arène semble plutôt en expansion stylistique. Cette
> absence de convergence rend d'autant plus important le test causal du style :
> si le style influence déjà les votes malgré une diversité croissante, le
> benchmark reste vulnérable à Goodhart.

Statut :

**À adopter pour le pitch si R2bis/R3 confirment l'effet du style.**

### H2 / R2bis — Style Premium longitudinal

> Les coefficients BT de style (`bold`, `lists`, `headers`) augmentent dans le
> temps.

Statut :

**Non confirmée en tendance, mais effet moyen robuste.**

Priorité :

Le style premium existe en niveau (`bold` ~+20 %, `lists/headers` ~+14-15 %),
mais il ne s'accroît pas. La preuve Goodhart temporelle reste donc faible.

### H3 / R3 — effet causal du style

> À contenu sémantiquement constant, le style verbose/markdown augmente la
> préférence du juge.

Statut :

Smoke test de génération validé. Pas encore de vérification sémantique ni de
jugement causal.

Précondition :

Accès Mistral/Groq validés et extraction texte depuis `comparia-votes` validée.
Prochaine étape : cosine similarity puis LLM-as-judge sur le smoke.

### H4 / R4 — forecast

> On peut projeter une date de perte de pouvoir discriminant à partir de la série
> temporelle de diversité.

Statut :

À dégrader fortement. Comme R1/R1b augmentent, un forecast d'effondrement basé
sur la diversité globale n'a pas de sens en l'état. Deux fallbacks :

- forecaster le **style premium** n'a pas de sens tel quel car R2bis ne montre
  pas de hausse ;
- forecaster la **part explicative du style** plutôt que la diversité.

---

## 8. Risques critiques actualisés

### Risque 1 — R1 contredit la thèse brute

Impact : élevé.

Mitigation :

- ne pas masquer le résultat ;
- le présenter comme "expansion de l'arène" ;
- tester la convergence conditionnelle — fait ;
- pivoter R1 en résultat négatif robuste ;
- déplacer le cœur de la preuve vers R2bis/R3.

### Risque 2 — 17/18 cohortes seulement

Impact : moyen/élevé.

Mitigation :

- rester prudent sur les p-values ;
- utiliser tendances et intervalles ;
- éviter les conclusions fortes sur forecast.

### Risque 3 — R3 peut consommer du budget API

Impact : moyen.

Mitigation :

- smoke test sur 10 réponses ;
- batch sauvegardé toutes les 50 itérations ;
- fallback Groq ;
- commencer par 100 paires avant 500.

### Risque 4 — confusion style vs qualité réelle

Impact : élevé.

Mitigation :

- citer Zilinskas : style = médiateur et confondeur partiel ;
- ne pas prétendre que tout style est biais ;
- R3 par contrefactuels pour isoler le style à contenu constant.

---

## 9. Prochaine étape recommandée

Priorité immédiate :

**R3 — Preuve causale par contrefactuels.**

Objectif :

Exploiter le résultat N=100 dans le papier/figures, puis décider si on étend à
N=200 uniquement si le temps API le permet.

Output attendu :

- table R3 propre pour le papier ;
- figure R3 finale ;
- section discussion sur concision vs verbose.

Pourquoi :

R1 ne confirme pas la convergence. R2bis confirme un effet moyen du style, mais
pas sa croissance temporelle. R3 N=100 fournit le résultat fort du projet :
la concision est préférée à la version neutre dans 89 % des cas filtrés, tandis
que le verbose-markdown est rejeté dans 79 % des cas.

### 5.9 R5 — Style vs qualité (piste Goodhart)

Objectif :

Tester si le style prédit encore la victoire après contrôle des labels qualité
disponibles dans `comparia-votes` :

- complétude ;
- utilité ;
- clarté du format ;
- créativité ;
- incorrect ;
- superficiel ;
- instructions non suivies.

Question :

```text
winner ~ delta_quality + delta_style
```

et :

```text
winner ~ delta_quality + delta_style + delta_style × month
```

Statut :

**Terminé.**

Déblocage :

Le parquet HF complet `comparia-votes` a été téléchargé localement :

`data/raw/hf/comparia-votes/votes.parquet`

Script :

`scripts/r5_style_vs_quality.py`

Module :

`src/compariawatch/style_quality.py`

Outputs :

- `data/raw/votes_quality.parquet`
- `data/processed/style_quality_dataset.parquet`
- `data/processed/style_quality_model_summary.parquet`
- `data/processed/style_quality_coefficients.parquet`
- `paper/figures/R5_style_vs_quality.png`

Données :

- 149 209 votes dans le parquet local ;
- 79 073 battles décisives jointes avec features de style ;
- labels qualité utilisés :
  - complétude ;
  - utilité ;
  - clarté du format ;
  - créativité ;
  - incorrect ;
  - superficiel ;
  - instructions non suivies.

Résultat AUC :

| Modèle | AUC | Lecture |
|---|---:|---|
| style seul | 0.669 | le style prédit déjà une partie du vote |
| qualité seule | 0.821 | les labels qualité expliquent fortement le vote |
| qualité + style | 0.862 | le style ajoute un gain net après contrôle qualité |
| qualité + style + interactions temps | 0.863 | gain temporel marginal |

Effets style après contrôle qualité :

| Feature style | Odds % / SD |
|---|---:|
| `bold` | +20.1 % |
| `delta_style_total` | +16.4 % |
| `headers` | +10.2 % |
| `lists` | +7.5 % |
| `emoji` | +2.9 % |
| `code_blocks` | +0.6 % |

Interactions style × mois :

| Interaction | Odds % / SD |
|---|---:|
| `bold × month` | -29.9 % |
| `code_blocks × month` | -18.5 % |
| `lists × month` | -4.1 % |
| `emoji × month` | -1.0 % |
| `headers × month` | +4.4 % |

Interprétation :

R5 est le résultat Goodhart le plus solide à ce stade. Même après contrôle des
labels qualité déclarés, les features de style améliorent nettement la
prédiction du vote (`AUC 0.821 → 0.862`) et certains effets restent forts
(`bold +20 %`, `headers +10 %`). En revanche, les interactions temporelles
n'indiquent pas une inflation généralisée : certains effets diminuent dans le
temps, notamment `bold`.

Conclusion :

Le benchmark n'est pas en effondrement visible, mais il est **vulnérable au
style** : le style a une contribution indépendante de la qualité déclarée.

### 5.10 R3 — Ensemble judge multi-modèles

Objectif :

Vérifier si l'effet causal R3 (concision préférée, verbose pénalisé) tient avec
des juges LLM fiables, ou si c'est un biais Mistral seul.

Statut :

**Analysé sur N=75/94 paires** (run interrompu par quota Groq ; suffisant
statistiquement).

Scripts / modules :

- `src/compariawatch/judge.py` (`groq_judge`, `ensemble_judge`)
- `scripts/r3_ensemble_judge.py`
- `notebooks/07_ensemble_analysis.ipynb`

Output :

- `data/processed/causal_style_votes_n100_ensemble.parquet`
- `paper/figures/R3_ensemble_judge.png`

#### Protocole initial (3 juges)

Trois juges indépendants ont été testés pour robustesse :

| Slot | Modèle | Rôle |
|---|---|---|
| #1 | Mistral Small | réutilisé du run N=100 |
| #2 | Groq `llama-3.3-70b-versatile` | juge principal Groq |
| #3 | Groq `llama-3.1-8b-instant` | remplace Mixtral décommissionné |

Le kappa moyen sur 3 juges reste faible (0.25), principalement parce que le
juge #3 (8B) diverge systématiquement des deux autres.

#### Pivot méthodologique (2 juges sérieux)

Conformément à Zheng et al. (2023, NeurIPS) — les petits modèles sont des juges
LLM peu fiables — l'analyse **principale** repose sur l'accord **2/2** entre :

- **Mistral Small** (juge #1) ;
- **Groq Llama 3.3 70B** (juge #2).

Le juge Llama 3.1 8B est conservé comme **sensitivity check secondaire**, avec
disclaimer explicite (κ < 0.30 avec les juges sérieux).

Distribution des 75 paires :

| Comparaison | N |
|---|---:|
| `concise_direct` vs neutre | 45 |
| `verbose_markdown` vs neutre | 30 |

Résultats principaux (2 juges sérieux) :

| Comparaison | Mistral | Llama 70B | Agreement 2/2 |
|---|---:|---:|---:|
| concise vs neutre | 88.9 % [76.5, 95.2] | 90.9 % [78.8, 96.4] | **97.3 % [86.2, 99.5]** (n=37) |
| verbose vs neutre | 26.7 % [14.2, 44.4] | 33.3 % [19.2, 51.2] | **26.9 % [13.7, 46.1]** (n=26) |

Accord inter-juges (Mistral × Llama 70B) :

| Métrique | Valeur | Lecture |
|---|---:|---|
| Cohen's κ global | **0.67** | accord substantiel / proche fort |
| Taux d'accord 2/2 — concise | 84.1 % | les 2 juges votent pareil sur 37/44 paires |
| Taux d'accord 2/2 — verbose | 86.7 % | les 2 juges votent pareil sur 26/30 paires |

Sensitivity check (Llama 8B — hors analyse principale) :

| Métrique | Valeur |
|---|---:|
| Win-rate concise (8B) | 55.6 % (non significatif vs 50 %) |
| Win-rate verbose (8B) | 63.3 % (direction opposée aux juges sérieux) |
| κ(8B, Mistral) | −0.05 |
| κ(8B, Llama 70B) | 0.13 |

Verdict (3 lignes) :

**R3 ROBUSTE (2 juges sérieux).** Mistral et Llama 70B convergent fortement
(κ = 0.67) : concision gagne (~91–97 % selon métrique), verbose perd (~27–33 %).
Ce n'est pas un biais Mistral seul. Le juge 8B diverge mais est exclu de
l'analyse principale (Zheng et al. 2023). La thèse « divergence humain↔LLM »
sur la valeur du formatage est soutenable pour le pitch.

---

## 10. État des commits

### Déjà poussé

Commit setup :

```text
chore: initialise le repo ComparIA hackathon
```

### À pousser

Commit R1 :

```text
feat(R1): mesure la diversité stylistique mensuelle
```

Voir commandes prêtes à copier-coller dans :

`docs/commit_log.md`

Commit R1b :

```text
test(R1): ajoute les contrôles de robustesse de convergence
```

Commit R2bis :

```text
feat(R2bis): mesure le Style Premium longitudinal
```

Commit R3 smoke :

```text
feat(R3): valide le smoke test de contrefactuels
```

Commit R3 scoring smoke :

```text
feat(R3): score les contrefactuels smoke
```

### À ajouter au prochain commit docs

Ce fichier :

`docs/project_status.md`

Commit suggéré :

```text
docs: ajoute le suivi projet et les hypothèses révisées
```

Bullets :

- documente l'état du pipeline et les artefacts produits
- consigne le résultat R1 brut et son interprétation
- ajoute les hypothèses révisées et les prochaines analyses de robustesse

---

## 11. Décision de pitch provisoire

R1b ne confirme pas la convergence conditionnelle. Pitch provisoire :

> "Nous ne trouvons pas encore de convergence stylistique globale ni de hausse
> temporelle du style premium. En revanche, le style influence bien la préférence :
> les features de formatage ont un effet moyen de +14 à +20 % dans les données
> observationnelles, et notre smoke causal montre que changer le style à contenu
> proche peut renverser le jugement. Fait intéressant, le juge ne récompense pas
> le markdown verbose : il préfère la concision."

Dans les deux cas, ne pas forcer la thèse. Le projet reste valable si on montre
que :

- le style influence les votes ;
- cette influence évolue ;
- les classements doivent être corrigés ou audités.

