# Project Status — L'Arène se mord la queue

Dernière mise à jour : mardi 2 juin 2026, 18h10.

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

Tester la préservation sémantique puis le jugement LLM sur les contrefactuels
déjà générés.

Output attendu :

- `data/interim/rewrites_smoke_scored.parquet`
- `data/processed/causal_style_votes.parquet`
- `paper/figures/R3_style_premium.png`

Pourquoi :

R1 ne confirme pas la convergence. R2bis confirme un effet moyen du style, mais
pas sa croissance temporelle. Le meilleur résultat à aller chercher maintenant
est donc causal : à contenu constant, le style change-t-il la préférence ?

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
> temporelle du style premium. En revanche, le niveau du biais de style est déjà
> élevé : bold, listes et headers donnent encore environ +14 à +20 % d'odds de
> victoire par écart-type. La prochaine étape est donc causale : vérifier, par
> contrefactuels à contenu constant, si ce premium persiste quand on neutralise
> le contenu."

Dans les deux cas, ne pas forcer la thèse. Le projet reste valable si on montre
que :

- le style influence les votes ;
- cette influence évolue ;
- les classements doivent être corrigés ou audités.

