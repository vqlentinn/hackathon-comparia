# Référence équipe — Tout savoir pour présenter et répondre aux questions

> **Objectif** : document de travail pour Valentin & coéquipier. Contient définitions,
> chiffres expliqués, méthodes, limites, FAQ jury, et lien avec Zilinskas.
> Pour les slides, Dust utilise [`pitch_dust_brief.md`](pitch_dust_brief.md).

**Dernière mise à jour** : juin 2026 — recheck chiffres validé sur parquets locaux.

---

## Table des matières

1. [Glossaire — notions à maîtriser](#1-glossaire--notions-à-maîtriser)
2. [Thèse du projet — version finale](#2-thèse-du-projet--version-finale)
3. [Données — ce qu'on utilise et pourquoi](#3-données--ce-quon-utilise-et-pourquoi)
4. [R1 — Diversité stylistique temporelle](#4-r1--diversité-stylistique-temporelle)
5. [R2bis — Style Premium longitudinal](#5-r2bis--style-premium-longitudinal)
6. [R3 — Contrefactuels causaux](#6-r3--contrefactuels-causaux)
7. [R5 — Style vs qualité déclarée](#7-r5--style-vs-qualité-déclarée)
8. [R5bis — Structure vs longueur](#8-r5bis--structure-vs-longueur)
9. [R6 — Endogénéité par tier](#9-r6--endogénéité-par-tier)
10. [Synthèse narrative — comment tout recolle](#10-synthèse-narrative--comment-tout-recolle)
11. [Rapport avec Zilinskas](#11-rapport-avec-zilinskas)
12. [Limites méthodologiques — à connaître par cœur](#12-limites-méthodologiques--à-connaître-par-cœur)
13. [FAQ jury — questions & réponses](#13-faq-jury--questions--réponses)
14. [Scripts de reproduction](#14-scripts-de-reproduction)
15. [Annexe — tous les chiffres clés](#15-annexe--tous-les-chiffres-clés)

---

## 1. Glossaire — notions à maîtriser

### Loi de Goodhart

**Définition** : « Quand une mesure devient une cible, elle cesse d'être une bonne
mesure. » (Charles Goodhart, 1975)

**Dans notre projet** : Compar:IA mesure la qualité des LLM via votes humains. Si
les labos optimisent pour gagner sur Compar:IA, ils peuvent optimiser des **proxies**
(style, longueur, format) plutôt que la vraie qualité.

**Ce qu'on teste** : est-ce que ce mécanisme se voit déjà (convergence, dérive
temporelle, vulnérabilité au style) ?

**Ce qu'on NE prouve PAS** : qu'un effondrement est imminent ou que les labos
optimisent déjà activement Compar:IA.

---

### Compar:IA

Plateforme d'évaluation de LLM du ministère de la Culture / DINUM. Deux modèles
anonymes côte à côte, l'utilisateur vote. ~89 modèles, ~145k battles nettoyées
(Zilinskas).

---

### Battle (combat / duel)

Une comparaison entre `model_a` et `model_b` sur un même prompt. Outcome :
`winner` = `model_a`, `model_b`, ou `tie`.

---

### Feature stylistique (style feature)

Mesure de **formatage markdown** extraite par regex sur le texte de réponse :

| Feature | Exemple | Pattern |
|---|---|---|
| `headers` | `# Titre` | lignes commençant par `#` |
| `lists` | `- item` ou `1. item` | puces / numérotations |
| `bold` | `**texte**` | gras markdown |
| `code_blocks` | ` ```python ` | blocs de code |
| `emoji` | 😀 | caractères emoji Unicode |

**Important** : ce ne sont PAS des mesures sémantiques ou de qualité du contenu.
Ce sont des comptages de structure de surface.

---

### Delta style (Δstyle)

Pour une battle : `feature_a - feature_b`. Positif = le modèle A a plus de cette
feature que B.

---

### Bradley-Terry (BT)

Modèle classique de ranking pairwise. Probabilité que A bat B :

```
P(A gagne) = σ(β_A - β_B + γ·Δstyle)
```

- `β_i` = force intrinsèque du modèle i
- `γ·Δstyle` = effet du différentiel de style

**Style-controlled BT** (Zilinskas, LMSYS) : on estime γ en même temps que les β,
puis on lit les β « net de style ».

---

### Style Premium

Coefficient γ d'une feature style dans le BT style-controlled. Interprété en
**% d'augmentation des odds de victoire par écart-type** de Δstyle :

```
odds_pct = (exp(γ) - 1) × 100
```

Exemple : bold +20 % → si A a 1 SD de bold de plus que B, les odds de victoire
de A augmentent d'environ 20 %.

---

### Odds / Odds ratio

- **Odds** = P(victoire) / P(défaite)
- **Odds ratio** = odds_A / odds_B
- **% odds / SD** = notre métrique standardisée (features z-scorées avant régression)

---

### AUC (Area Under Curve)

Métrique de performance d'un classifieur (ici régression logistique). 0.5 = hasard,
1.0 = parfait. On compare des **modèles nested** : si ajouter des variables style
augmente l'AUC au-delà des variables qualité, le style apporte un signal
additionnel.

**Limite** : nos AUC sont **in-sample** (pas de train/test split).

---

### Diversité stylistique (R1)

Distance moyenne entre centroïdes de features style par (modèle, mois). Mesure
si les modèles **se ressemblent** stylistiquement ou non. Hausse = plus de
différenciation ; baisse = convergence.

Métrique utilisée : distance euclidienne entre centroïdes (Wasserstein aussi testé).

---

### Contrefactuel (R3)

Réponse réelle réécrite par un LLM (Mistral) dans un style cible, en préservant
le contenu. Permet de manipuler **style** tout en gardant le **sémantique** proche.

---

### Cosine similarity (R3)

Similarité entre embeddings sentence-transformers de l'original et de la réécriture.
Seuil 0.85 = filtre de préservation sémantique avant jugement.

---

### LLM-as-judge (R3)

Un LLM (Mistral, Groq Llama 70B) choisit entre deux réponses sans voir le modèle
source. Ce n'est **pas** un vote humain Compar:IA.

---

### Cohen's κ (kappa)

Mesure d'accord entre juges au-delà du hasard. Échelle de Landis & Koch :
- κ < 0.20 : faible
- 0.21–0.40 : fair
- 0.41–0.60 : modéré
- 0.61–0.80 : substantiel
- > 0.80 : quasi parfait

Notre κ Mistral × Llama 70B = **0.67** → accord substantiel.

---

### Labels qualité (R5)

Annotations utilisateur dans `comparia-votes` après le vote :

| Label | Sens |
|---|---|
| `conv_complete` | réponse complète |
| `conv_useful` | utile |
| `conv_clear_formatting` | format clair |
| `conv_creative` | créative |
| `conv_incorrect` | incorrecte (négatif) |
| `conv_superficial` | superficielle (négatif) |
| `conv_instructions_not_followed` | consignes non suivies (négatif) |

Les labels négatifs sont **inversés** dans les deltas pour que positif = mieux pour A.

---

### Endogénéité (R6)

Problème : le style peut être corrélé à la qualité réelle du modèle (les bons
modèles formatent peut-être mieux **parce qu'ils sont bons**, pas pour tricher).

**Deux histoires causales** :
1. **Confondant** : modèles faibles sur-compensent par le format
2. **Médiateur** : le format est une composante légitime de la qualité perçue

R6 teste quelle histoire tient via tiers de modèles et corrélation rating↔format.

---

### Tier (R6)

Découpage des 89 modèles en thirds par rating BT standard Zilinskas :
- **top** : 29 modèles les mieux classés
- **middle** : 29 suivants
- **bottom** : 31 restants

**Paire bottom-top** = battle entre un modèle faible et un modèle fort.

---

## 2. Thèse du projet — version finale

### Thèse initiale (mail à Simon)

Compar:IA subit un effet Goodhart : convergence stylistique, style > contenu,
projection d'effondrement.

### Thèse retenue (après données)

> Compar:IA **ne montre pas encore** de convergence globale ni d'amplification
> temporelle du style premium, **mais** reste **vulnérable au style** : le format
> prédit les votes indépendamment de la qualité déclarée et de la longueur, et
> des contrefactuels montrent que changer le style renverse le jugement.

### Pitch en une phrase

« Pas encore effondré, mais hackable par la forme. »

---

## 3. Données — ce qu'on utilise et pourquoi

| Source | N | Usage | Chemin |
|---|---:|---|---|
| Zilinskas battles | 142 243 | style features, winners | `style-control-analysis/battles_bt_styled.parquet` + join timestamps |
| Battles datés | 114 626 | R1, R2bis, R6 (votes + timestamp) | `data/interim/battles_with_dates.parquet` |
| comparia-votes local | 149 209 | qualité, longueur, R5/R5bis | `data/raw/hf/comparia-votes/votes.parquet` |
| Battles R5 jointes | 79 073 | régressions qualité+style | `data/processed/style_quality_dataset.parquet` |

**Non utilisé** : `comparia-reactions` (nice-to-have, triangulation future).

**Pourquoi 79k vs 114k ?** R5 joint battles Zilinskas + votes HF sur
`conversation_pair_id` ; seules les battles avec vote explicite et labels qualité
matchent.

---

## 4. R1 — Diversité stylistique temporelle

### Question

Les modèles récents se ressemblent-ils de plus en plus stylistiquement ?

### Méthode

1. Long format : une ligne par (modèle, mois) avec centroïde des 5 features style
2. Diversité mensuelle = distance moyenne entre centroïdes de modèles actifs ce mois
3. OLS : `diversity ~ month_idx` (+ contrôles)

### Chiffres

| Spec | β (pente/mois) | p-value | R² |
|---|---:|---:|---:|
| Brute | +0.0572 | <0.001 | 0.925 |
| + n_models + n_responses | +0.0466 | <0.001 | 0.961 |
| Modèles récurrents ≥6 mois | +0.0412 | <0.001 | 0.529 |
| Trimestrielle | +0.1575 | 0.001 | 0.942 |

**Corrélations** :
- `n_models` ↔ diversité : **+0.826** (plus de modèles → plus de diversité apparente)
- `n_responses` ↔ diversité : −0.041 (négligeable)

**Composition** : oct 2024 = 19 modèles → oct 2025 = 47 modèles.

### Interprétation

**H1 refutée** : pas de convergence, **expansion** stylistique.

### Comment répondre

> « On a testé la convergence — elle n'est pas là. L'arène accueille plus de
> modèles différents, donc la diversité mesurée augmente. Même en contrôlant ça,
> la pente reste positive. »

### Figures / scripts

- `paper/figures/R1_convergence_robustness.png`
- `scripts/r1_robustness.py`

---

## 5. R2bis — Style Premium longitudinal

### Question

Le style premium (γ du BT) **augmente-t-il** dans le temps ?

### Méthode

Par mois (17 cohortes, min 500 battles) :
1. BT style-controlled → coefficients style
2. Bootstrap n=40 → IC 95 %
3. OLS tendance : `odds_pct ~ month_idx`

### Chiffres niveau (Style Premium moyen)

| Feature | Premium moyen | Pente/mois | p-value |
|---|---:|---:|---:|
| bold | +19.8 % | **−1.40** | 0.013 |
| lists | +14.6 % | **−0.79** | 0.031 |
| headers | +14.5 % | +0.48 | 0.546 |
| code_blocks | +4.4 % | −0.75 | 0.015 |
| emoji | +5.2 % | +0.35 | 0.401 |

### Interprétation

- **En niveau** : le style compte (+14 à +20 %), cohérent avec Zilinskas (+16–19 %)
- **Dans le temps** : pas d'amplification ; bold et lists **diminuent**

**H2 refutée** : pas de Goodhart temporel visible sur ces features.

### Comment répondre

> « Le biais existe depuis le début et reste fort, mais il ne s'aggrave pas
> mesurablement sur 17 mois. »

### Figures / scripts

- `paper/figures/R2bis_style_premium_longitudinal.png`
- `scripts/r2bis_style_premium.py`

---

## 6. R3 — Contrefactuels causaux

### Question

À contenu sémantiquement proche, changer le style change-t-il le jugement ?

### Protocole

1. **Source** : 100 réponses gagnantes (`comparia-votes`)
2. **Réécriture** Mistral en 3 styles :
   - `concise_direct` — court, direct
   - `verbose_markdown` — long, titres, listes
   - `neutre_baseline` — reformulation neutre
3. **Filtre sémantique** : cosine ≥ 0.85 (sentence-transformers)
4. **Jugement** : paire (style vs neutre), le juge ne voit pas le modèle

### Validation sémantique (N=100 générés)

| Style | Cosine moyen | Taux ≥0.85 |
|---|---:|---:|
| concise | 0.868 | 68 % |
| neutre | 0.879 | 71 % |
| verbose | 0.848 | 49 % |

Verbose plus dur à préserver → moins de paires passent le filtre.

### Résultats Mistral seul (N=100, filtre cosine)

| Style vs neutre | N jugé | Win rate style |
|---|---:|---:|
| concise | 56 | **89.3 %** |
| verbose | 38 | **21.1 %** |

94 paires générées au total ; 56+38=94 jugées après filtre.

### Résultats ensemble (analyse principale)

**Juges** : Mistral Small + Groq Llama 3.3 70B (κ=0.67)  
**N=75** paires avec les deux juges Groq (run partiel, suffisant)

| Style vs neutre | Mistral | Llama 70B | Accord 2/2 |
|---|---:|---:|---:|
| concise (n=45) | 88.9 % | 90.9 % | **97.3 %** (n=37) |
| verbose (n=30) | 26.7 % | 33.3 % | **26.9 %** (n=26) |

IC Wilson 95 % accord concise : [86.2, 99.5].

**Juge 8B exclu** : κ<0.30 avec les sérieux, direction opposée sur verbose.

### Interprétation

- Effet causal **fort** et **robuste** (2 juges)
- Surprise : **concision** gagne, **verbose** perd — pas « markdown = mieux »
- Le contrefactuel verbose manipule longueur **et** structure (bundle)

### Limites R3

- Juges LLM, pas humains Compar:IA
- N modeste (75–94)
- verbose_markdown = style + longueur confondus

### Comment répondre

> « On ne prétend pas reproduire le vote humain. On montre qu'à contenu proche,
> le style seul suffit à renverser un juge — et deux LLM indépendants convergent. »

### Figures / scripts

- `paper/figures/R3_ensemble_judge.png`
- `paper/tables/table_r3_counterfactual.md`
- `scripts/r3_smoke_counterfactual.py`, `scripts/r3_score_smoke.py`, `scripts/r3_ensemble_judge.py`

---

## 7. R5 — Style vs qualité déclarée

### Question

Le style prédit-il le vote **après contrôle** des labels qualité utilisateur ?

### Méthode

Régression logistique : `P(A gagne) ~ Δqualité + Δstyle`  
79 073 battles, features standardisées, 7 labels qualité + 5 style.

**Specs anti-colinéarité** : composantes séparées (pas totaux + composantes).

### AUC par spécification

| Modèle | AUC | n_features |
|---|---:|---:|
| Style seul | 0.669 | 5 |
| Qualité seule | 0.821 | 7 |
| Qualité + style | **0.862** | 12 |
| + interactions temps | 0.863 | 17 |

**Gain style** : 0.821 → 0.862 = **+0.041 AUC**

### Coefficients style (modèle qualité + style)

| Feature | % odds / SD |
|---|---:|
| bold | **+30.6** |
| lists | +15.4 |
| headers | +12.8 |
| emoji | +4.2 |
| code_blocks | +1.5 |

### Interprétation

Même quand l'utilisateur a tagué complétude, utilité, etc., le **format markdown**
ajoute du pouvoir prédictif. C'est le mécanisme Goodhart direct : **proxy (style)
au-delà de la variable déclarée (qualité)**.

### Limites R5

- Labels qualité corrélés au vote (même utilisateur)
- AUC in-sample
- Pas de causalité (observationnel)

### Figures / scripts

- `paper/figures/R5_style_vs_quality.png`
- `paper/tables/table_r5_auc.md`, `table_r5_style_coefficients.md`
- `scripts/r5_style_vs_quality.py`

---

## 8. R5bis — Structure vs longueur

### Question

Bold/headers aident-ils encore après contrôle **qualité + longueur** ?

Réconcilie R3 (verbose perd) et R5 (structure gagne).

### Méthode

Ajout de `delta_log_assistant_chars` = log1p(chars_A) − log1p(chars_B)  
Longueur = somme caractères messages `assistant` dans `conversation_a/b`.

### AUC

| Modèle | AUC |
|---|---:|
| Longueur seule | 0.661 |
| Qualité + longueur | 0.860 |
| Qualité + style (R5) | 0.862 |
| **Qualité + longueur + style** | **0.865** |

### Coefficients modèle complet

| Variable | % odds / SD | vs R5 sans longueur |
|---|---:|---|
| Longueur log | **+40.5** | (nouveau) |
| bold | **+14.5** | était +30.6 |
| headers | **+13.8** | était +12.8 |
| lists | +5.3 | était +15.4 |
| emoji | +6.2 | était +4.2 |

### Interprétation

1. **Longueur** : associée positivement au vote même après qualité (+40 %)
2. **Bold/headers** : **survivent** → effet structurel, pas juste verbosité
3. **Lists** : surtout corrélé à la longueur (15 % → 5 %)

**Deux dimensions du style** :
- Verbosité → aide observationnellement, pénalisée causalement en contrefactuel verbose
- Structure (bold, headers) → levier indépendant

Simon **exclut** volontairement la longueur de son style control ; nous montrons
ce qui reste quand on la contrôle.

### Figures / scripts

- `paper/figures/R5bis_structure_vs_length.png`
- `paper/tables/table_r5bis_structure_vs_length.md`
- `scripts/r5bis_structure_vs_length.py`

---

## 9. R6 — Endogénéité par tier

### Question

Est-ce que seuls les modèles **faibles** « gamment » le format ?

Extension de `style-control-analysis/endogeneity_analysis.py`.

### Méthode

1. Tiers par rating BT standard (JSON Zilinskas)
2. BT style-controlled **par type de paire** (bottom-bottom, top-top, bottom-top…)
3. Corrélation rating ↔ intensité moyenne formatage par modèle

### Corrélation rating ↔ formatage (87 modèles)

| Feature | Pearson r | Spearman ρ |
|---|---:|---:|
| bold | **+0.569** | +0.646 |
| lists | +0.519 | +0.577 |
| headers | +0.532 | +0.541 |
| composite | +0.592 | +0.687 |

**Les modèles mieux classés formatent PLUS**, pas moins.

### Intensité bold moyenne par tier

| Tier | Bold moyen |
|---|---:|
| bottom | 7.33 |
| middle | 15.07 |
| top | **27.81** |

### Style premium bold par type de paire (% odds/SD)

| Paire | bold | n_battles |
|---|---:|---:|
| bottom-bottom | +24.2 | 17 293 |
| middle-middle | +8.3 | 10 837 |
| top-top | +18.1 | 6 229 |
| **bottom-top** | **+41.6** | 9 449 |

### Interprétation

1. **Pas un gaming des faibles** — les tops formatent davantage (r≈+0.6)
2. **Le biais persiste** partout (bottom-bottom +24 %)
3. **Maximum en cross-tier** bottom-top (+42 %) : quand un faible affronte un fort,
   le delta de format devient un levier énorme

Double mécanisme Goodhart :
- Format corrélé à la qualité réelle (confondant)
- Format exploitable indépendamment (R3/R5)

### Figures / scripts

- `paper/figures/R6_endogeneity_tiers.png`
- `paper/tables/table_r6_endogeneity_tiers.md`
- `scripts/r6_endogeneity_tiers.py`

---

## 10. Synthèse narrative — comment tout recolle

```
Mail initial          →  Données           →  Conclusion
─────────────────────────────────────────────────────────
Convergence style     →  R1/R1b            →  NON (hausse)
Style premium ↑       →  R2bis             →  NON (stable/↓)
Effondrement R4       →  abandon           →  pas de tendance
Part causale style    →  R3                →  OUI (concision)
Style vs qualité      →  R5                →  OUI (+0.041 AUC)
Structure vs longueur →  R5bis             →  OUI (décomposition)
Qui triche ?          →  R6                →  NON (tops formatent +)
```

**Histoire finale** : Compar:IA est sain en surface (pas de collapse), mais
l'arène reste optimisable par la forme — surtout structure markdown et matchups
déséquilibrés.

---

## 11. Rapport avec Zilinskas

### Ce qu'il a fait

- Style-controlled BT sur 145k battles français
- bold/lists/headers : +16–19 % odds/SD
- Reshuffling rankings (r=0.976)
- Papier : formatting bias, pas Goodhart temporel ni causal

### Ce qu'on ajoute

| Extension | Module |
|---|---|
| Dynamique temporelle | R1, R2bis |
| Preuve causale (contrefactuels) | R3 |
| Style vs labels qualité | R5 |
| Contrôle longueur (qu'il exclut) | R5bis |
| Endogénéité tiers | R6 |

### Mail à Simon — comment le cadrer

> « On a testé votre crainte Goodhart temporelle : pas de convergence ni
> d'amplification du style premium. En revanche, on complète votre analyse
> observationnelle par une preuve causale (contrefactuels), un contrôle qualité
> + longueur, et une stratification par tiers. Surprise : concision gagne
> causalement, pas le verbose-markdown. »

---

## 12. Limites méthodologiques — à connaître par cœur

| Limite | Modules | Comment l'assumer |
|---|---|---|
| AUC in-sample | R5, R5bis | « ordre de grandeur, pas prédiction out-of-sample » |
| Labels qualité post-vote | R5 | « contrôle partiel, pas ground truth » |
| LLM judge ≠ humain | R3 | « robustesse multi-juges, pas external validity directe » |
| N modeste contrefactuels | R3 | « effet large, intervalles larges mais cohérents » |
| verbose = style + longueur | R3 | « R5bis décompose ; verbose bundle les deux » |
| Tiers from same data family | R6 | « extension directe Zilinskas, pas indépendant » |
| Longueur = chars pas tokens | R5bis | « proxy cohérent ; tokens HF non jointes ici » |
| Pas R4 forecast | — | « pas de tendance à projeter honnêtement » |
| Réactions HF non exploitées | — | « votes suffisent pour la thèse retenue » |

---

## 13. FAQ jury — questions & réponses

### « Compar:IA s'effondre ? »

**Non.** R1 montre une diversité croissante, R2bis pas d'amplification du style
premium. On parle de **vulnérabilité**, pas d'effondrement.

### « C'est pas juste Mistral qui juge ? »

**Non pour l'analyse principale.** Ensemble Mistral + Llama 70B, κ=0.67. Concise
97.3 % en accord 2/2.

### « Vous avez juste refait Zilinskas ? »

**Non.** Il fait du BT observationnel statique. Nous ajoutons : temporel (R1/R2bis),
causal (R3), qualité+longueur (R5/R5bis), tiers (R6).

### « Pourquoi pas comparia-reactions ? »

Signal différent, 46 % de ties, overlap quasi nul avec votes. Nice-to-have, pas
nécessaire pour la thèse. `comparia-votes` suffit (qualité + longueur + votes).

### « +30 % bold c'est énorme, c'est crédible ? »

C'est par **écart-type** de Δstyle sur 79k battles. Zilinskas trouve +19 % en BT
joint — même ordre de grandeur. R5bis ramène bold à +14.5 % après longueur.

### « Pourquoi verbose perd si lists/bold aident ? »

**Deux dimensions.** R5bis : lists corrèle avec longueur ; verbose contrefactuel
allonge beaucoup (2204→4423 chars). Le juge pénalise le bundle verbosité, pas le
gras isolé.

### « Les faibles trichent avec le format ? »

**Non.** R6 : r=+0.6 rating↔format. Les tops formatent 4× plus. Le biais existe
partout ; il est **maximal** quand faible vs fort s'affrontent (+42 % bold).

### « Goodhart sans collapse, c'est pas faible ? »

C'est **plus honnête et plus défendable**. Goodhart ne dit pas « ça s'effondre
demain » — il dit « la mesure devient optimisable ». On le montre causalement
et observationnellement.

### « AUC 0.86 c'est bien ? »

Oui pour des features simples sur vote binaire bruité. Le **delta** +0.041 après
qualité est l'argument clé.

### « Pourquoi pas contrôler la longueur dans R3 ? »

Le contrefactuel verbose **est** une manipulation de longueur — c'est le test.
R5bis contrôle longueur observationnellement ; les deux se complètent.

### « Quelle recommandation concrète pour Compar:IA ? »

1. Style-control BT systématique (comme LMSYS)
2. Contrôler longueur séparément
3. Auditer paires cross-tier (bottom-top)
4. Ne pas optimiser pour verbose-markdown

---

## 14. Scripts de reproduction

```bash
cd comparia-hackathon && source .venv/bin/activate

# R1 robustesse
python scripts/r1_robustness.py

# R2bis
python scripts/r2bis_style_premium.py

# R5
python scripts/r5_style_vs_quality.py

# R5bis (extrait longueur la 1ère fois, ~15s)
python scripts/r5bis_structure_vs_length.py

# R6
python scripts/r6_endogeneity_tiers.py

# Tables finales
python scripts/make_final_tables.py
```

---

## 15. Annexe — tous les chiffres clés

### Données

| Metric | Valeur |
|---|---:|
| Battles Zilinskas total | 142 243 |
| Battles avec timestamp | 114 626 |
| Battles votes datés (R6) | 114 626 |
| Battles R5 jointes | 79 073 |
| Cohortes mensuelles R1/R2bis | 17 |
| Modèles (tiers R6) | 87–89 |

### R1

| Metric | Valeur |
|---|---:|
| β diversité brute | +0.0572/mois |
| β contrôlée | +0.0466/mois |
| corr(diversity, n_models) | +0.826 |

### R2bis

| Feature | Premium | Pente/mois |
|---|---:|---:|
| bold | +19.8 % | −1.40 |
| lists | +14.6 % | −0.79 |
| headers | +14.5 % | +0.48 |

### R3 ensemble

| Style | Win rate accord 2/2 | n |
|---|---:|---:|
| concise | 97.3 % | 37 |
| verbose | 26.9 % | 26 |
| κ Mistral×Llama70B | 0.67 | — |

### R5

| Metric | Valeur |
|---|---:|
| AUC qualité | 0.821 |
| AUC qualité+style | 0.862 |
| bold après qualité | +30.6 % |

### R5bis

| Metric | Valeur |
|---|---:|
| AUC complet | 0.865 |
| longueur log | +40.5 % |
| bold après qualité+longueur | +14.5 % |

### R6

| Metric | Valeur |
|---|---:|
| r rating↔bold | +0.569 |
| bold bottom-top | +41.6 % |
| bold moyen top tier | 27.81 |

---

*Fin du document. Pour l'état technique live du repo, voir [`project_status.md`](project_status.md).*
