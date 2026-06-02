# 🦎 L'Arène se mord la queue

> **Audit causal de Compar:IA à l'épreuve de la loi de Goodhart : convergence stylistique, dérive arène-réalité, et projection d'effondrement du benchmark**

---

## 📑 Table des matières

1. [Contexte général](#1-contexte-général)
2. [Le problème](#2-le-problème)
3. [Notre thèse](#3-notre-thèse)
4. [Cadre théorique](#4-cadre-théorique)
5. [Questions de recherche](#5-questions-de-recherche)
6. [Hypothèses préenregistrées](#6-hypothèses-préenregistrées)
7. [Méthodologie détaillée](#7-méthodologie-détaillée)
8. [Stack technique complète](#8-stack-technique-complète)
9. [Données utilisées](#9-données-utilisées)
10. [Architecture du pipeline](#10-architecture-du-pipeline)
11. [Modélisation statistique](#11-modélisation-statistique)
12. [Validation et robustesse](#12-validation-et-robustesse)
13. [Livrables](#13-livrables)
14. [Hardware et budget](#14-hardware-et-budget)
15. [Plan d'exécution jour par jour](#15-plan-dexécution-jour-par-jour)
16. [Répartition à 2](#16-répartition-à-2)
17. [Risques et mitigations](#17-risques-et-mitigations)
18. [Structure du repo GitHub](#18-structure-du-repo-github)
19. [Template du papier](#19-template-du-papier)
20. [Pitch et slides](#20-pitch-et-slides)
21. [Questions à poser à l'équipe Compar:IA](#21-questions-à-poser-à-léquipe-compariac)
22. [Références bibliographiques](#22-références-bibliographiques)
23. [Checklist pré-démarrage](#23-checklist-pré-démarrage)

---

## 1. Contexte général

### 1.1 Le hackathon

- **Organisateur** : Christophe Benavent, DRM - ACSS Institute, Université Dauphine PSL
- **Sujet** : *Les styles de l'IA* — exploration des biais (extrinsèques et intrinsèques) des LLM via le dataset Compar:IA
- **Date** : semaine du 1er juin 2026
- **Format** : équipes de 2 personnes, 4 jours de travail, pitch 10 min en fin de semaine
- **Rendu** : note d'étude 4-5 pages + pitch + livrables annexes

### 1.2 Compar:IA, qu'est-ce que c'est ?

**Compar:IA** est une plateforme française développée par le Ministère de la Culture (en partenariat avec HuggingFace, sous égide de Termignon et al., 2026) qui permet aux utilisateurs de comparer en aveugle les réponses de 110+ LLM à un même prompt et de voter pour la meilleure. C'est l'équivalent français de **LMArena/Chatbot Arena**.

Les votes accumulés (~100k+) servent à construire un **classement Elo** des LLM en français, classement de plus en plus utilisé comme **référence pour la politique publique française** en matière d'IA (DINUM, programme Albert, achats publics).

### 1.3 L'équipe et le projet

- **Équipe** : 2 personnes
- **Membres** :
  - Valentin Proux (data eng + modélisation)
  - Coéquipier (NLP + viz + rédaction)
- **Durée effective** : Lun → Ven (4-5 jours pleins)

---

## 2. Le problème

### 2.1 Le problème scientifique

Compar:IA collecte des **préférences humaines** sur des paires de réponses. Cette préférence est traitée comme un **proxy direct de la qualité du LLM**. Or :

1. **Les utilisateurs ne lisent pas toujours en détail** — décision souvent prise à l'œil (heuristique de surface)
2. **Le style (longueur, markdown, politesse) influence massivement le vote** — biais documenté dans LMArena (Chiang et al., 2024)
3. **Les votes sont sujets à des biais de position, de familiarité, de topic**
4. **Les LLM sont désormais entraînés POUR gagner sur les arènes** (via DPO, RLHF, distillation des préférences) → **boucle de rétroaction**

### 2.2 Le problème politique

Le Ministère de la Culture, la DINUM, et la commande publique française vont s'appuyer sur Compar:IA pour des **décisions d'achat de plusieurs millions d'euros** (programme Albert, modèles souverains). Si le classement est biaisé, **on choisit les mauvais modèles**.

### 2.3 Le problème théorique

C'est un cas d'école de **Loi de Goodhart** : une métrique (Elo Compar:IA) devient une cible d'optimisation pour les développeurs de LLM, ce qui **dégrade sa validité** comme mesure.

---

## 3. Notre thèse

> **Compar:IA, devenu cible d'entraînement implicite des LLM, perd progressivement son pouvoir de mesurer la qualité. Nous mesurons empiriquement cette dégradation, démontrons causalement le rôle du style (vs contenu) dans la préférence humaine, et projetons la date à laquelle le benchmark cessera d'être discriminant.**

### Reformulation en 3 affirmations testables

1. **Affirmation descriptive** : la diversité stylistique inter-modèles décroît dans le temps
2. **Affirmation causale** : une part substantielle de la préférence humaine est attribuable au style indépendamment du contenu
3. **Affirmation prospective** : si les tendances se poursuivent, Compar:IA atteindra son point d'effondrement discriminant en [DATE PROJETÉE]

---

## 4. Cadre théorique

### 4.1 Loi de Goodhart (Goodhart, 1975)

> *"Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."*

Variante de Manheim & Garrabrant (2018) appliquée à l'IA, qui distingue 4 types :
- **Regressional Goodhart** : optimiser un proxy bruité dégrade la vraie variable
- **Extremal Goodhart** : aux extrêmes, le proxy diverge de la cible
- **Causal Goodhart** : l'intervention casse la corrélation observée
- **Adversarial Goodhart** : agents stratégiques exploitent le proxy

**Notre cas combine surtout regressional + adversarial Goodhart.**

### 4.2 Sociologie de la performativité des classements (Espeland & Sauder, 2007)

> *"Engines of Anxiety: Academic Rankings, Reputation, and Accountability"*

Les classements ne mesurent pas seulement, **ils transforment l'objet mesuré**. Appliqué aux LLM : les labos modifient leurs modèles pour gagner sur l'arène, ce qui change la nature de ce que l'arène mesure.

### 4.3 Inférence causale en NLP (Feder et al., TACL 2022)

> *"Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond"*

Méthodologie de génération de **contrefactuels textuels** par LLM, vérifiés par NLI, pour mesurer des effets causaux dans des textes observationnels.

### 4.4 Évaluation des LLM (Chiang et al., 2024 ; Boubdir et al., 2023)

> *"Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference"*

Documente les biais de l'évaluation par préférence : position, longueur, formatage. Justifie l'approche Bradley-Terry étendu.

---

## 5. Questions de recherche

### Q1 — Convergence stylistique
> *La diversité stylistique entre LLM diminue-t-elle dans le temps ?*

### Q2 — Causalité du style
> *Quelle part de la préférence humaine sur Compar:IA est causalement attribuable au style (indépendamment du contenu) ?*

### Q3 — Dérive arène-réalité
> *L'écart entre l'Elo Compar:IA et les benchmarks externes objectifs (MT-Bench-FR, FrenchBench) s'accroît-il avec la date de release des modèles ?*

### Q4 — Projection
> *À quelle date Compar:IA perdra-t-il son pouvoir discriminant si les tendances observées se poursuivent ?*

---

## 6. Hypothèses préenregistrées

### H1 — Convergence
$$ \text{Cov}\Big(\text{date\_release}_m, D_t\Big) < 0 $$
où $$D_t$$ est la distance stylistique moyenne inter-modèles dans la cohorte $$t$$.

### H2 — Effet causal du style
$$ \mathbb{E}\big[P(\text{vote}|r_{\text{stylé}}) - P(\text{vote}|r_{\text{neutre}})\big] > 0 $$
à contenu sémantiquement vérifié constant.

### H3 — Arena-Reality Gap croissant
$$ \frac{\partial}{\partial t}|E_m - U_m| > 0 $$
où $$E_m$$ = Elo Compar:IA et $$U_m$$ = score benchmark externe normalisé.

### H4 — Projection
Modèle d'état bayésien sur $$D_t$$ permettant d'estimer $$P(D_t < \tau_{\text{crit}})$$ pour $$t$$ futur.

---

## 7. Méthodologie détaillée

### 7.1 Vue d'ensemble

Le projet se décompose en **5 modules indépendants** qui produisent chacun un résultat publiable :

1. **M1 — Espace stylistique** : extraction de 25 features stylo + embeddings dense
2. **M2 — Convergence temporelle (R1)** : mesure de la diversité stylistique par cohorte de release
3. **M3 — Inférence causale (R3)** : génération + jugement de contrefactuels stylistiques
4. **M4 — Arena-Reality Gap (R2)** : corrélation Elo Compar:IA × benchmarks externes
5. **M5 — Forecast (R4)** : modèle bayésien de projection temporelle

### 7.2 Module 1 — Espace stylistique

#### 7.2.1 Features handcrafted (NLP symbolique)

| Dimension | Feature | Méthode |
|---|---|---|
| **Longueur** | n_tokens, n_sents, avg_sent_len | spaCy |
| **Complexité syntaxique** | max_dep_depth, % subordonnées | spaCy parser |
| **Formalité** | % subjonctifs, ratio mots longs | spaCy morph + textstat |
| **Densité lexicale** | TTR (Type-Token Ratio) | calcul direct |
| **Registre** | tu_count, vous_count, ratio_tu/vous | regex |
| **Ponctuation** | ratio !, ?, …, emojis | regex |
| **Markdown** | bullets, headers, bold, code blocks | regex |
| **POS ratios** | %ADJ, %ADV, %VERB, %NOUN, %PRON | spaCy POS |
| **Hedging** | "peut-être", "il semble", "généralement"… | lexique regex |
| **Flatterie** | "excellente question", "bravo"… | lexique regex |
| **Anglicismes** | rate / 100 tokens | lexique curatée |
| **Lisibilité** | Flesch FR | textstat |
| **Sentiment** | valence VADER-fr / LIWC-fr | feel-it |

Total : **~25 dimensions interprétables**.

#### 7.2.2 Embeddings dense (NLP transformer)

- **Modèle** : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768d)
- **Usage** : représentation sémantique des prompts (pour topic modeling) et représentation "globale" des réponses (pour mesures de similarité dense)

#### 7.2.3 Standardisation

Z-score sur l'ensemble du corpus, par dimension :

$$ z_d(x) = \frac{x_d - \mu_d}{\sigma_d} $$

avec $$\mu_d, \sigma_d$$ calculés sur **prompts ∪ réponses_A ∪ réponses_B**.

### 7.3 Module 2 — Convergence temporelle (R1)

#### 7.3.1 Définition de la diversité inter-modèles

Pour chaque cohorte temporelle $$t$$ (mois ou trimestre) :

$$ D_t = \mathbb{E}_{m, m' \in \mathcal{C}_t, m \neq m'} \Big[ W_2\big(\mathcal{S}_m, \mathcal{S}_{m'}\big) \Big] $$

où $$\mathcal{S}_m$$ est la distribution empirique des features stylo standardisées des réponses du modèle $$m$$, et $$W_2$$ est la **distance de Wasserstein-2** (transport optimal).

#### 7.3.2 Mesures alternatives (robustness)

- **Variance pooled** : variance intra-cohort des centroïdes stylistiques
- **Coefficient de Gini stylistique** : inégalité de la distribution des styles
- **Entropie de Shannon** sur clusters HDBSCAN des features stylo

#### 7.3.3 Test statistique

- **Régression** : $$D_t = \alpha + \beta \cdot t + \epsilon_t$$
- **Test** : $$H_0 : \beta = 0$$ vs $$H_1 : \beta < 0$$
- **Robustness** : régression locale (LOESS) pour visualiser la forme

### 7.4 Module 3 — Inférence causale par contrefactuels (R3)

#### 7.4.1 Pipeline

```
1. Sample 1000 paires de Compar:IA
2. Pour chaque réponse r : générer r' = rewrite(r, style_target)
   où style_target ∈ {verbose-markdown, concise-direct,
                       flatteur-poli, neutre-baseline}
3. Vérifier préservation sémantique :
   - NLI bidirectionnel : r |= r' ET r' |= r (score > 0.7)
   - Cosine sentence-embed > 0.85
   - Filtrer les paires qui échouent
4. Re-judge :
   - LLM-as-judge sur (r vs r') pour tous les couples
   - Validation humaine sur 200 paires (auto-fait, prend ~1h)
5. Estimer l'effet causal :
   - ATE = E[vote(r') - vote(r)] par style_target
   - Par modèle d'origine, par topic
```

#### 7.4.2 Garde-fous méthodologiques

- **Préservation vérifiée** par 2 méthodes indépendantes (NLI + cosine)
- **Échantillon balanced** par topic et modèle d'origine
- **Validation humaine** sur sous-échantillon (accord κ > 0.6 attendu)
- **Test placebo** : reformuler r en "même style" et vérifier que vote ≈ 50/50

#### 7.4.3 Métriques d'output

| Métrique | Définition |
|---|---|
| **Style ATE** | Effet moyen du changement de style sur la préférence |
| **Style Premium par modèle** | $$E_m^{\text{brut}} - E_m^{\text{neutre}}$$ |
| **Style sensitivity par topic** | Effet du style selon le sujet |

### 7.5 Module 4 — Arena-Reality Gap (R2)

#### 7.5.1 Données externes

- **MT-Bench-FR** : scores sur tâches diverses
- **FrenchBench** : benchmark spécifique FR
- **IFEval-FR** : instruction following

#### 7.5.2 Calcul du gap

Pour chaque modèle $$m$$ présent dans Compar:IA ET un benchmark externe $$B$$ :

$$ G_m^B = \text{rank}_{\text{Compar:IA}}(m) - \text{rank}_B(m) $$

(en utilisant des ranks normalisés sur [0, 1])

#### 7.5.3 Modélisation temporelle

$$ |G_m^B| = \alpha + \beta \cdot \text{age}_m + \epsilon_m $$

où $$\text{age}_m$$ = date de release. Test $$H_0: \beta = 0$$.

### 7.6 Module 5 — Forecast d'effondrement (R4)

#### 7.6.1 Modèle d'état bayésien

$$ D_t = \mu_t + \epsilon_t, \quad \mu_t = \mu_{t-1} + \nu_t $$

avec $$\epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$ et $$\nu_t \sim \mathcal{N}(\lambda, \sigma_\nu^2)$$.

Priors :
- $$\lambda \sim \mathcal{N}(0, 0.1)$$ (tendance)
- $$\sigma_\epsilon, \sigma_\nu \sim \text{HalfNormal}(0.5)$$

Inférence : NumPyro NUTS, 4 chaînes × 2000 samples.

#### 7.6.2 Seuil critique

Définition : $$\tau_{\text{crit}}$$ tel que la distance inter-modèles soit inférieure à la **distance intra-modèle** (variance d'un même modèle sur des prompts différents).

#### 7.6.3 Output prédictif

- $$P(D_t < \tau_{\text{crit}} | t)$$ pour $$t \in [2026, 2028]$$
- **Date espérée d'effondrement** : $$t^* = \arg\min_t P(D_t < \tau_{\text{crit}}) > 0.5$$
- Intervalle de crédibilité à 95%

---

## 8. Stack technique complète

### 8.1 Vue d'ensemble par famille

| Famille | Méthodes principales | Pourquoi |
|---|---|---|
| **NLP symbolique** | spaCy, regex, textstat | Features interprétables, rapide, pas de GPU |
| **NLP transformer (zero-shot)** | sentence-transformers, BERTopic, CamemBERT-NLI | Sémantique, topic, vérification |
| **LLM-as-instrument** | API Mistral/Groq | Génération de contrefactuels, jugement |
| **Inférence causale** | dowhy, econml, ATE | Estimation rigoureuse d'effets |
| **Statistiques avancées** | Wasserstein (POT), bootstrap, mixed-effects | Mesure de convergence |
| **Bayésien** | NumPyro, PyMC | Forecast probabiliste |
| **ML supervisé secondaire** | XGBoost + SHAP | Validation non-linéaire |
| **Visualisation** | matplotlib, seaborn, plotly, altair | Figures du papier |
| **Démo** | Streamlit | Dashboard interactif |

### 8.2 Ce qu'on n'utilise PAS (et pourquoi)

| Non utilisé | Raison |
|---|---|
| Fine-tuning de transformers | Inutile et coûteux, zero-shot suffit |
| RNN/LSTM | Obsolète, transformers font mieux |
| Réseaux de neurones from scratch | Pas d'apport vs features handcrafted + logit (interprétabilité critique) |
| Generative AI pour features | Hallucinations, on préfère le déterministe |
| Causal Discovery (DoWhy structural) | Trop ambitieux pour 4j |

### 8.3 Liste exhaustive des dépendances Python

```
# Data
pandas==2.x
polars==1.x
numpy==1.26.x
scipy==1.11.x

# HuggingFace
datasets==2.x
huggingface-hub==0.20.x

# NLP
spacy==3.7.x
textstat==0.7.x
sentence-transformers==2.7.x
transformers==4.40.x
torch==2.2.x

# Topic modeling
bertopic==0.16.x
umap-learn==0.5.x
hdbscan==0.8.x

# Stats / Causal
statsmodels==0.14.x
bambi==0.13.x
scikit-learn==1.4.x
dowhy==0.11.x
econml==0.15.x
pot==0.9.x  # Optimal Transport

# Bayésien
pymc==5.x  # OR
numpyro==0.13.x

# Forecasting
prophet==1.1.x  # OR statsmodels.tsa

# ML supervisé
xgboost==2.0.x
shap==0.45.x

# API LLM
mistralai==1.x
openai==1.x
groq==0.5.x

# Visualisation
matplotlib==3.8.x
seaborn==0.13.x
plotly==5.x
altair==5.x

# Démo
streamlit==1.32.x

# Dev
jupyter
ipykernel
swifter
tqdm
```

### 8.4 Modèles pré-entraînés téléchargés

| Modèle | Taille | Usage |
|---|---|---|
| `fr_core_news_lg` (spaCy) | ~500 MB | Features stylo FR |
| `paraphrase-multilingual-mpnet-base-v2` | ~1 GB | Embeddings sémantiques |
| `morit/french-nli` ou XNLI | ~500 MB | Vérification contrefactuels |
| `cmarkea/distilcamembert-base-nli` (alt.) | ~250 MB | NLI léger |

**Total disque pour modèles** : ~2.5 GB

---

## 9. Données utilisées

### 9.1 Datasets principaux

| Dataset | Source | Statut | Volume | Usage |
|---|---|---|---|---|
| `comparia-conversations` | HuggingFace (gated) | Acceptation requise | 8.74 GB | Conversations, réponses, prompts |
| `comparia-votes` | HuggingFace (gated) | Acceptation requise | ~50 MB | Préférences finales (`y`) |
| `comparia-reactions` | HuggingFace (gated) | Acceptation requise (optionnel) | ~100 MB | Réactions par message (V2) |
| `models.json` | GitHub betagouv/ComparIA | Public | ~200 KB | Métadonnées des 110+ modèles |
| `MT-Bench-FR` | HuggingFace | Public | Petit | Benchmark externe pour R2 |
| `FrenchBench` | HuggingFace | Public | Petit | Benchmark externe pour R2 |
| `IFEval-FR` | HuggingFace (si dispo) | Public | Petit | Benchmark externe pour R2 |

### 9.2 Schéma de `comparia-conversations`

Colonnes clés exploitées :
- `conversation_pair_id` : clé de jointure avec votes
- `conversation_a` / `conversation_b` : liste {role, content}
- `model_a_name` / `model_b_name`
- `system_prompt_a` / `system_prompt_b`
- `total_conv_a_output_tokens` / `_b_`
- `total_conv_a_kwh` / `_b_`
- `model_a_total_params` / `model_a_active_params` / idem b
- `opening_msg`
- `short_summary`, `keywords`, `categories`
- `languages`
- `visitor_id`
- `timestamp` (date du vote)

### 9.3 Schéma de `comparia-votes`

Colonnes clés :
- `conversation_pair_id` (jointure)
- `preference` : `{model_a, model_b, tie, both_bad}`
- `visitor_id`
- `timestamp`

### 9.4 Pipeline d'ingestion

```python
from datasets import load_dataset

conv  = load_dataset("ministere-culture/comparia-conversations", split="train")
votes = load_dataset("ministere-culture/comparia-votes",         split="train")

# Filtre FR
conv = conv.filter(lambda x: "fr" in (x["languages"] or []))

# Jointure
df = conv.to_pandas().merge(
    votes.to_pandas(),
    on="conversation_pair_id",
    how="inner"
)

# Préférence binaire
df = df[df["preference"].isin(["model_a", "model_b"])]
df["y"] = (df["preference"] == "model_a").astype(int)
```

### 9.5 Volumes attendus après nettoyage

- Total paires : ~100k
- Paires FR : ~70-80k (estimation)
- Paires avec préférence claire (hors tie/both_bad) : ~50-60k
- Modèles avec ≥50 votes : ~80-100 modèles
- Cohortes temporelles avec ≥5 modèles : ~12-18 (mois)

---

## 10. Architecture du pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                      DONNÉES BRUTES                           │
│  comparia-conversations ⨝ comparia-votes ⨝ models.json       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   NETTOYAGE + JOINTURE        │
            │   • filtre FR                 │
            │   • dedup                     │
            │   • binary preference         │
            └──────────────┬────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  M1 NLP      │  │  M1 sentence │  │  Topic       │
│  symbolique  │  │  embeddings  │  │  modeling    │
│  (25 feats)  │  │  (768d)      │  │  (BERTopic)  │
└───────┬──────┘  └──────┬───────┘  └──────┬───────┘
        │                │                  │
        └────────────────┼──────────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  TABLE FEATURES UNIFIÉE   │
            │  (1 ligne = 1 paire)      │
            └──────────────┬───────────┘
                           │
   ┌───────────────────────┼───────────────────────────┐
   │                       │                           │
   ▼                       ▼                           ▼
┌──────────┐         ┌──────────┐              ┌──────────┐
│ M2 Conv. │         │ M3 Causal│              │ M4 Arena │
│ temp.    │         │ rewrite  │              │ Reality  │
│ (R1)     │         │ + judge  │              │ Gap (R2) │
│          │         │ (R3)     │              │          │
│ POT      │         │ Mistral  │              │ corr.    │
│ Wasserst.│         │ API+NLI  │              │ MT-Bench │
└─────┬────┘         └────┬─────┘              └─────┬────┘
      │                   │                          │
      └───────────────────┼──────────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │   M5 FORECAST (R4)    │
              │   NumPyro state-space │
              │   → date d'effond.    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      RENDUS           │
              │  • Papier 5 pages     │
              │  • Dataset HF         │
              │  • compariawatch lib  │
              │  • Dashboard          │
              │  • Slides pitch       │
              └──────────────────────┘
```

---

## 11. Modélisation statistique

### 11.1 Modèle 1 — Régression de convergence (R1)

```python
import statsmodels.api as sm

# D_t = distance Wasserstein moyenne inter-modèles dans la cohorte t
X = sm.add_constant(cohorts_df[["month_idx"]])
y = cohorts_df["wasserstein_diversity"]

model_r1 = sm.OLS(y, X).fit()
print(model_r1.summary())
# H1 acceptée si: model_r1.params["month_idx"] < 0 et p-value < 0.05
```

### 11.2 Modèle 2 — Effet causal du style (R3)

```python
# Après génération de contrefactuels
# d_ij = 1 si paire i, condition j (verbose vs neutre)
# y_ij = vote synthétique LLM-as-judge

import bambi as bmb

model_r3 = bmb.Model(
    "y ~ style_target + (1|original_model) + (1|topic)",
    data=counterfactual_df,
    family="bernoulli"
)
trace = model_r3.fit(draws=2000, chains=4)
# ATE = posterior mean de "style_target"
```

### 11.3 Modèle 3 — Arena-Reality Gap (R2)

```python
import statsmodels.api as sm

X = sm.add_constant(models_df[["age_months"]])
y = models_df["abs_rank_gap"]

model_r2 = sm.OLS(y, X).fit()
# H3 acceptée si: model_r2.params["age_months"] > 0 et p-value < 0.05
```

### 11.4 Modèle 4 — Forecast bayésien (R4)

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def forecast_model(T, D_obs=None):
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(0.5))
    sigma_nu  = numpyro.sample("sigma_nu",  dist.HalfNormal(0.5))
    lam       = numpyro.sample("lambda",    dist.Normal(0, 0.1))

    mu_0 = numpyro.sample("mu_0", dist.Normal(0, 1))
    mu = [mu_0]
    for t in range(1, T):
        mu_t = numpyro.sample(f"mu_{t}", dist.Normal(mu[-1] + lam, sigma_nu))
        mu.append(mu_t)

    with numpyro.plate("obs", T):
        numpyro.sample("D", dist.Normal(jnp.stack(mu), sigma_eps), obs=D_obs)

nuts = NUTS(forecast_model)
mcmc = MCMC(nuts, num_samples=2000, num_chains=4)
mcmc.run(rng_key, T=24, D_obs=observed_D)
```

### 11.5 Modèle 5 — Validation par XGBoost + SHAP

```python
import xgboost as xgb
import shap

# Prédire le vote depuis les features
X = paired_features
y = df["y"]

xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=6).fit(X, y)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)  # ← figure du papier
```

---

## 12. Validation et robustesse

### 12.1 Robustness checks pour R1 (convergence)

1. **Cohortes alternatives** : refaire avec trimestres / semestres / fenêtres glissantes
2. **Métriques alternatives** : Wasserstein vs Variance pooled vs entropie clusters
3. **Sous-échantillon** : retirer les top-3 modèles (effets de queue)
4. **Familles séparées** : OpenAI / Mistral / Anthropic / Open-source séparément

### 12.2 Robustness checks pour R3 (causal)

1. **Préservation vérifiée** par double critère (NLI + cosine)
2. **Validation humaine** sur 200 paires (accord κ Cohen)
3. **Test placebo** : reformuler en style identique → effet attendu ≈ 0
4. **Inversion** : générer aussi `r ← rewrite(r', neutre)` pour tester symétrie

### 12.3 Robustness checks pour R2 (gap)

1. **Plusieurs benchmarks externes** (au moins 2-3, prendre le pooled)
2. **Stratification par taille de modèle** (params)
3. **Stratification par licence** (open vs fermé)

### 12.4 Robustness checks pour R4 (forecast)

1. **Hold-out temporel** : retirer les 3 derniers mois, vérifier que le modèle les retrouve
2. **Sensibilité aux priors** : refaire avec priors plus larges / plus serrés
3. **Modèles alternatifs** : Prophet, exponential smoothing, ARIMA, GP

### 12.5 Test d'identifiabilité général

**Simulation Monte Carlo** : générer un dataset synthétique où on connaît la vérité (vraie diversité décroît à un taux $$\lambda^*$$ connu) et vérifier que notre pipeline retrouve $$\hat{\lambda} \approx \lambda^*$$.

---

## 13. Livrables

### 13.1 Livrable principal — Note d'étude (5 pages PDF)

Structure :
1. **Introduction** (0.5 p) — problème politique + cadre Goodhart
2. **Données & méthode** (1 p) — pipeline, métriques
3. **Résultats** (2 p) — R1, R2, R3, R4 avec figures
4. **Discussion** (1 p) — implications, limites
5. **Conclusion & recommandations** (0.5 p) — 3 propositions de réforme

Format : LaTeX, double colonne, références bibliographiques.

### 13.2 Livrable dataset — Triplets contrefactuels (HF)

- 1000-2000 triplets `(prompt, response_original, response_restyled, llm_judge_vote)`
- Publié sur HuggingFace sous `[username]/comparia-counterfactuals-fr`
- Licence : CC-BY-SA 4.0
- README détaillé en anglais et français

### 13.3 Livrable code — Librairie `compariawatch`

```
compariawatch/
├── pyproject.toml
├── README.md
├── LICENSE (MIT)
├── compariawatch/
│   ├── __init__.py
│   ├── features.py        # extraction stylo
│   ├── diversity.py       # Wasserstein, Gini, entropie
│   ├── counterfactual.py  # pipeline rewrite + judge
│   ├── forecast.py        # state-space model
│   └── dashboard.py       # Streamlit app
├── tests/
└── examples/
    └── compariac_audit.ipynb
```

Pip-installable, documentée, testée.

### 13.4 Livrable visuel — Dashboard Streamlit

3 onglets :
1. **Diversity Tracker** : courbe $$D_t$$ + projection
2. **Style Premium** : tableau modèles + Style ATE
3. **Recommender** : prompt en input → modèle conseillé selon profil utilisateur

### 13.5 Livrable pitch — Slides

10-12 slides, 10 min de présentation :
1. Titre + équipe
2. Hook : "L'arène va mourir"
3. Le problème (Goodhart)
4. Les données
5. Méthode en 3 étapes
6. Résultat R1 (courbe convergence)
7. Résultat R2 (Arena-Reality Gap)
8. Résultat R3 (preuve causale par contrefactuel)
9. Résultat R4 (date d'effondrement projetée)
10. Recommandations politiques
11. Limites
12. Q&A / demo dashboard

---

## 14. Hardware et budget

### 14.1 Hardware

| Composant | Requis | Alternative |
|---|---|---|
| **CPU** | Laptop 8+ cœurs | Colab gratuit |
| **GPU** | Optionnel (utile pour embeddings) | Colab T4 gratuit, ou MPS sur Mac Apple Silicon |
| **RAM** | 16 GB minimum | Colab 12 GB |
| **Disque** | 15 GB libre | - |
| **Connexion** | Stable (pour download datasets) | - |

### 14.2 Budget API

| Service | Volume | Coût |
|---|---|---|
| Mistral API (réécritures) | ~1000 calls × 800 tokens out | ~5 € |
| Mistral Small (judge) | ~4000 calls × 500 tokens | ~0.4 € |
| Mistral Small (registre classif) | ~2000 calls × 100 tokens | ~0.04 € |
| HuggingFace | Modèles open-source | 0 € |
| Groq (alternative gratuite) | Llama 3.3 70B free tier | 0 € |
| **TOTAL** | | **< 10 €** |

### 14.3 Stockage

- Datasets : 9 GB
- Modèles pré-entraînés : 2.5 GB
- Features extraites (parquet) : 2-3 GB
- **TOTAL** : ~15 GB

### 14.4 Software gratuit

- Python 3.11+ : gratuit
- VS Code / Cursor : gratuit
- GitHub : gratuit
- HuggingFace Hub : gratuit
- LaTeX (Overleaf) : gratuit (free tier suffit)
- Streamlit Community Cloud : gratuit (déploiement)

---

## 15. Plan d'exécution jour par jour

### 🟢 Lundi — Setup & Foundations

**Matin (9h-12h)**
- ✅ Setup env Python + deps
- ✅ Demande d'accès gated HF + acceptation
- ✅ Création clés API (Mistral, Groq)
- ✅ Création repo GitHub + Notion partagé
- ✅ Download datasets en background (lance et oublie)

**Après-midi (14h-18h)**
- ✅ Ingestion + jointure conv ⨝ votes
- ✅ EDA volumes (combien de paires FR, par modèle, par mois ?)
- ✅ Récup `models.json` GitHub + parse métadonnées
- ✅ Extraction prompt / response_a / response_b en colonnes propres

**Soir (lance avant de partir)**
- 🔥 Lancer extraction features stylo sur 100% du corpus (4-8h)

### 🟢 Mardi — Features & Topics

**Matin (9h-12h)**
- ✅ Récupération des features stylo (lancées la veille)
- ✅ Sentence-transformers embeddings (1-3h en background)
- ✅ Standardisation z-score, sauvegarde parquet

**Après-midi (14h-18h)**
- ✅ Topic modeling BERTopic sur opening_msg
- ✅ Classification de registre (LLM-as-judge sur 2k prompts via Mistral)
- ✅ Premier draft de figures R1 (courbe convergence stylo)

**Soir**
- 🔥 Lancer génération de contrefactuels (Mistral API, 1000 réécritures, ~3h)

### 🟡 Mercredi — Modélisation & Causal

**Matin (9h-12h)**
- ✅ Récupération contrefactuels (lancés la veille)
- ✅ NLI + cosine vérification de préservation
- ✅ LLM-as-judge sur paires contrefactuelles
- ✅ Calcul ATE par condition

**Après-midi (14h-18h)**
- ✅ Régression R1 (convergence) finalisée
- ✅ Calcul gap Compar:IA × benchmarks externes (R2)
- ✅ Modèle bambi pour R3
- ✅ Validation par XGBoost + SHAP

**Soir**
- 🔥 Lancer forecast bayésien (NumPyro, 30 min)

### 🟡 Jeudi — Forecast & Viz & Lib

**Matin (9h-12h)**
- ✅ Forecast R4 — résultats et figures
- ✅ Finalisation de toutes les figures du papier
- ✅ Tests des robustness checks

**Après-midi (14h-18h)**
- ✅ Packaging `compariawatch` (structure, tests, README)
- ✅ Première version du dashboard Streamlit
- ✅ Publication du dataset HF (triplets contrefactuels)

**Soir**
- ✅ Début rédaction de la note (intro + méthode)

### 🟡 Vendredi — Rédaction & Pitch

**Matin (9h-12h)**
- ✅ Rédaction complète de la note (5 pages)
- ✅ Relecture croisée

**Après-midi (14h-16h)**
- ✅ Slides pitch
- ✅ Répétition du pitch (2-3 fois)
- ✅ Finition dashboard pour démo

**16h ou 17h — PITCH** 🎤

---

## 16. Répartition à 2

### 16.1 Profil A — "Data Eng + Modélisation" (Valentin)

**Responsabilités**
- Ingestion + jointure
- Pipeline features stylo
- Modèles statistiques (R1, R2, R3, R4)
- Forecast bayésien
- Code repo + librairie

**Outils**
- pandas, polars, numpy
- statsmodels, bambi, numpyro
- xgboost, shap
- POT (optimal transport)

### 16.2 Profil B — "NLP + Causal + Rédaction" (Coéquipier)

**Responsabilités**
- Embeddings + topic modeling
- Pipeline contrefactuels (Mistral API + NLI)
- LLM-as-judge
- Figures + viz
- Dashboard Streamlit
- Rédaction du papier

**Outils**
- sentence-transformers, BERTopic
- Mistral API, transformers (NLI)
- matplotlib, seaborn, plotly
- Streamlit
- LaTeX / Overleaf

### 16.3 Matrice de responsabilité (RACI)

| Tâche | A (Valentin) | B (Coéquipier) |
|---|---|---|
| Ingestion data | **R** | C |
| Features stylo | **R** | C |
| Embeddings sémantiques | A | **R** |
| Topic modeling | C | **R** |
| Génération contrefactuels | C | **R** |
| Vérification NLI | I | **R** |
| Modèle R1 | **R** | I |
| Modèle R2 | **R** | I |
| Modèle R3 (bambi) | **R** | C |
| Modèle R4 (forecast) | **R** | I |
| Viz figures | C | **R** |
| Dashboard | C | **R** |
| Librairie code | **R** | C |
| Dataset HF publié | C | **R** |
| Rédaction papier | C | **R** |
| Slides pitch | A | A |
| Pitch (présentation) | **R** | **R** |

R = Responsable, A = Approbateur, C = Consulté, I = Informé

---

## 17. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **Accès gated HF refusé / lent** | Moyenne | Critique | Lundi matin first thing, mail au mainteneur en backup |
| **MCMC bayésien trop lent** | Moyenne | Moyen | Variational inference (SVI) en fallback |
| **API Mistral rate limit** | Élevée | Moyen | Réduire échantillon, étaler dans le temps, async + retry |
| **Pas de "ground truth" pour R3** | Élevée | Moyen | Validation humaine sur 200 paires (1h de travail) |
| **Identifiabilité de θ_Q vs θ_S** | Moyenne | Élevé | Test sur données synthétiques d'abord |
| **Pas assez de modèles par cohorte temporelle** | Moyenne | Moyen | Élargir les fenêtres, semestres au lieu de mois |
| **Bug d'extraction texte des conversations** | Faible | Élevé | Tests unitaires sur 10 exemples avant de lancer sur 100k |
| **Coéquipier malade / absent** | Faible | Critique | Documenter au max, README clair, env reproductible |
| **Goulot sur features stylo (lent)** | Moyenne | Moyen | swifter, multiprocessing, échantillon 50k si vraiment trop |
| **Données externes (MT-Bench-FR) inaccessibles** | Faible | Moyen | Plan B : utiliser LMArena anglais comme proxy |
| **Hallucinations dans contrefactuels** | Moyenne | Moyen | NLI + cosine + filter strict |
| **Effets stylo confondus avec longueur** | Élevée | Élevé | Contrôle systématique pour longueur en covariable |

---

## 18. Structure du repo GitHub

```
arene-mord-queue/
├── README.md                        # vue d'ensemble + comment exécuter
├── LICENSE                          # MIT
├── pyproject.toml                   # dépendances + métadonnées
├── .gitignore
├── .env.example                     # template clés API
├── data/                            # gitignored (lourd)
│   ├── raw/                         # datasets HF téléchargés
│   ├── interim/                     # features extraites
│   └── processed/                   # données finales
├── notebooks/
│   ├── 01_ingestion_eda.ipynb
│   ├── 02_features_stylo.ipynb
│   ├── 03_embeddings_topics.ipynb
│   ├── 04_R1_convergence.ipynb
│   ├── 05_R2_arena_reality_gap.ipynb
│   ├── 06_R3_counterfactual.ipynb
│   ├── 07_R4_forecast.ipynb
│   ├── 08_robustness_checks.ipynb
│   └── 09_figures_paper.ipynb
├── src/
│   └── compariawatch/
│       ├── __init__.py
│       ├── data.py                  # ingestion + jointure
│       ├── features.py              # stylo
│       ├── diversity.py             # Wasserstein
│       ├── counterfactual.py        # rewrite + NLI
│       ├── judge.py                 # LLM-as-judge
│       ├── forecast.py              # state-space
│       └── dashboard.py             # Streamlit
├── tests/
│   ├── test_features.py
│   ├── test_diversity.py
│   └── test_judge.py
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── tables/
├── slides/
│   └── pitch.pdf
├── scripts/
│   ├── setup_env.sh
│   ├── download_data.py
│   ├── run_all.sh
│   └── publish_dataset.py
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    └── reproducibility.md
```

---

## 19. Template du papier

```latex
\documentclass[10pt, twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage{amsmath, amssymb, graphicx, booktabs, natbib}
\usepackage{geometry}
\geometry{a4paper, margin=2cm}

\title{L'Arène se mord la queue : audit causal de Compar:IA à
       l'épreuve de la loi de Goodhart}
\author{Valentin Proux \and [Coéquipier] \\
        Hackathon Compar:IA -- Université Paris-Dauphine PSL}
\date{Juin 2026}

\begin{document}
\maketitle

\begin{abstract}
Compar:IA, plateforme française d'évaluation des grands modèles
de langue (LLM) par préférence humaine, s'impose comme benchmark
de référence pour la commande publique en France. Nous montrons
empiriquement qu'elle est en train de subir un effet Goodhart :
les LLM convergent stylistiquement (...), une part causale
de Y\% de la préférence est attribuable au style indépendamment
du contenu (...), et nous projetons par modélisation bayésienne
que le benchmark perdra son pouvoir discriminant courant [DATE].
Nous fournissons un dataset de contrefactuels (...), une librairie
open-source de diagnostic (...) et trois recommandations de réforme.
\end{abstract}

\section{Introduction}
\section{Données et méthode}
\section{Résultats}
\subsection{R1 : convergence stylistique temporelle}
\subsection{R2 : creusement de l'Arena-Reality Gap}
\subsection{R3 : preuve causale du Style Premium}
\subsection{R4 : projection de l'effondrement}
\section{Discussion}
\section{Recommandations}
\section{Limites et travaux futurs}

\bibliographystyle{plain}
\bibliography{references}
\end{document}
```

---

## 20. Pitch et slides

### Script (10 min)

**[0:00-0:45] Hook**
> *"Bonjour. Imaginez : dans 18 mois, Compar:IA cesse de fonctionner. Plus aucun classement ne discrimine quoi que ce soit. C'est ce que nous prédisons. Voilà comment, et voilà ce qu'il faut faire pour l'éviter."*

**[0:45-2:00] Contexte & problème**
- Compar:IA = benchmark public français pour LLM
- Devenu cible d'entraînement des labos
- Risque : effet Goodhart

**[2:00-3:30] Cadre théorique + données**
- Goodhart's law (1975), Espeland (2007)
- Dataset Compar:IA + benchmarks externes
- 70k paires FR analysées

**[3:30-7:00] Résultats** (le cœur)
- R1 : courbe de convergence → -X% par an
- R2 : Arena-Reality Gap croissant
- R3 : preuve causale par contrefactuels → +Y% de préférence due au pur style
- R4 : projection → effondrement courant [DATE]

**[7:00-8:30] Recommandations & démo**
- 3 réformes possibles
- Démo dashboard Streamlit
- Dataset + lib publiés

**[8:30-10:00] Limites, conclusion, Q&A**
- Honnêteté sur les hypothèses
- Posture : alerte, pas prophétie
- Appel à l'action

---

## 21. Questions à poser à l'équipe Compar:IA

À poser dès que possible (déblocage du projet) :

1. **Sampling** : comment les paires de modèles sont-elles échantillonnées ?
2. **Visitor ID** : unique et persistant ?
3. **Temps de vote** : log du temps entre apparition et clic ?
4. **System prompts** : identiques A/B sur une même paire ?
5. **kWh** : mesure ou estimation ?
6. **Biais de position** : randomisé côté UI ?
7. **Power users** : part des votes des 1% les plus actifs ?
8. **Comparaison LMArena** : a-t-elle été faite par eux ?

---

## 22. Références bibliographiques

```bibtex
@article{goodhart1975problems,
  title={Problems of Monetary Management: The U.K. Experience},
  author={Goodhart, Charles A. E.},
  journal={Papers in Monetary Economics},
  year={1975}
}

@book{espeland2016engines,
  title={Engines of Anxiety: Academic Rankings, Reputation, and Accountability},
  author={Espeland, Wendy Nelson and Sauder, Michael},
  publisher={Russell Sage Foundation},
  year={2016}
}

@inproceedings{chiang2024chatbot,
  title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
  author={Chiang, Wei-Lin and others},
  booktitle={ICML},
  year={2024}
}

@article{feder2022causal,
  title={Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond},
  author={Feder, Amir and others},
  journal={TACL},
  year={2022}
}

@misc{manheim2018categorizing,
  title={Categorizing Variants of Goodhart's Law},
  author={Manheim, David and Garrabrant, Scott},
  year={2018},
  eprint={1803.04585},
  archivePrefix={arXiv}
}

@article{giles1973accent,
  title={Accent mobility: A model and some data},
  author={Giles, Howard},
  journal={Anthropological Linguistics},
  year={1973}
}

@inproceedings{boubdir2023elo,
  title={Elo Uncovered: Robustness and Best Practices in Language Model Evaluation},
  author={Boubdir, Meriem and others},
  booktitle={GEM Workshop},
  year={2023}
}

@misc{termignon2026comparia,
  title={Compar:IA: A French Platform for Comparative LLM Evaluation},
  author={Termignon and others},
  year={2026},
  note={Ministère de la Culture}
}
```

---

## 23. Checklist pré-démarrage

### Avant lundi 9h00

- [ ] Compte HuggingFace créé + token généré
- [ ] Demande d'accès aux 3 datasets gated soumise
- [ ] Compte Mistral créé + 10€ de crédit + clé API
- [ ] Compte Groq créé (gratuit) + clé API
- [ ] Repo GitHub privé créé + invitation coéquipier
- [ ] Notion ou Google Doc partagé créé
- [ ] Slack/Discord/WhatsApp groupe à 2 créé
- [ ] Overleaf ou installation LaTeX prête
- [ ] Python 3.11+ installé localement
- [ ] VS Code / Cursor avec extensions Python prêt
- [ ] 15+ GB d'espace disque libre
- [ ] Connexion internet stable

### Lundi 9h-9h30

- [ ] Vérification que tous les datasets sont accessibles (sinon : escalade)
- [ ] `pip install -r requirements.txt` réussi
- [ ] `python -m spacy download fr_core_news_lg` réussi
- [ ] Test des 2 clés API (1 appel chacune)
- [ ] Kickoff meeting de 30 min pour aligner sur le plan

### Lundi 12h00

- [ ] Datasets téléchargés (vérifier `~/.cache/huggingface/`)
- [ ] Premier notebook `01_ingestion_eda.ipynb` produit la jointure réussie
- [ ] Volumes attendus confirmés (~70k paires FR ?)

---

## 🎯 Résumé exécutif final

**Projet** : *L'Arène se mord la queue*
**Question** : Compar:IA devient-il sa propre cible et perd-il son pouvoir discriminant ?
**Méthode** : NLP (features + embeddings) + inférence causale par contrefactuels LLM-générés + modélisation bayésienne d'état + forecast
**Livrables** : 1 papier 5p + 1 dataset HF + 1 librairie pip-installable + 1 dashboard + 3 recommandations politiques
**Stack** : Python, spaCy, sentence-transformers, Mistral API, NumPyro, Streamlit
**Hardware** : laptop standard + Colab gratuit en backup
**Budget** : < 10 €
**Durée** : 4-5 jours, 2 personnes
**Niveau technique** : recherche (NeurIPS workshop, FAccT)
**Risque global** : modéré, plusieurs fallbacks identifiés

**Verdict de faisabilité** : 🟢 **GO**.

---

*Document généré le 1er juin 2026, version 1.0*
