# Brief Dust — Génération des slides Compar:IA

> **Objectif de ce fichier** : document unique pour que Dust (ou un autre agent) lise
> le repo et produise un deck de présentation hackathon (~10–12 slides, 5–7 min).
> Tout le détail méthodologique, définitions et Q&A sont dans
> [`pitch_equipe_reference.md`](pitch_equipe_reference.md).

---

## 1. Consignes de production

### Format attendu

- **Langue** : français
- **Durée cible** : 5 à 7 minutes (~1 slide/minute)
- **Public** : jury hackathon Compar:IA / ACSS, niveau technique modéré à élevé
- **Ton** : scientifique, honnête, pas alarmiste — on ne prétend pas que Compar:IA
  « s'effondre », on montre une **vulnérabilité Goodhart** mesurée
- **Style visuel** : sobre, figures du repo en priorité, peu de texte par slide

### Message central (à répéter)

> **Compar:IA passe le stress-test Goodhart ? Pas encore effondré, mais vulnérable au style.**

### Ce qu'il NE FAUT PAS dire

- « Compar:IA s'effondre » / « le benchmark est mort »
- « Les modèles faibles trichent avec le format » (R6 montre l'inverse : les tops
  formatent plus)
- « Plus de markdown = toujours mieux » (R3 montre que verbose perd)
- « On a tout découvert » (extension du travail de Simon Zilinskas)

### Ce qu'il FAUT dire

- Pas de convergence stylistique globale (R1) ni d'amplification temporelle du style
  premium (R2bis)
- Mais le style influence les votes : observationnellement (R5), causalement (R3),
  indépendamment de la qualité déclarée (R5) et de la longueur (R5bis)
- Deux dimensions du style : **structure markdown** vs **verbosité/longueur**
- Extension naturelle du papier Zilinskas : temporel + causal + longueur + tiers

---

## 2. Structure slides recommandée

### Slide 1 — Titre

**Titre** : Compar:IA passe le stress-test Goodhart ?  
**Sous-titre** : Pas encore effondré, mais vulnérable au style  
**Équipe** : Valentin Proux & coéquipier — Dauphine PSL — juin 2026

---

### Slide 2 — Problème & question

**Loi de Goodhart** : quand une mesure devient un objectif, elle cesse d'être une
bonne mesure.

**Question du projet** : les LLM optimisés pour gagner sur Compar:IA convergent-ils
stylistiquement et dégradent-ils le signal de qualité ?

**Plan en 4 blocs** :

1. Dynamique (R1, R2bis) — y a-t-il convergence / dérive temporelle ?
2. Causal (R3) — changer le style change-t-il le jugement ?
3. Mécanisme (R5, R5bis) — style vs qualité déclarée vs longueur
4. Hétérogénéité (R6) — qui formate, où le biais est-il maximal ?

---

### Slide 3 — Données & méthode (1 slide synthèse)

**Sources** :

- Parquet Zilinskas (`battles_bt_styled.parquet`) : 142k battles, 5 features style
- `comparia-votes` (HF) : timestamps + labels qualité — 79k battles jointes (R5)
- Contrefactuels Mistral + juges Mistral + Groq Llama 70B (R3)

**Features style** : headers, lists, bold, code_blocks, emoji (regex markdown)

**Figure** : aucune — schéma pipeline optionnel

---

### Slide 4 — R1/R2bis : pas d'effondrement visible

**Message** : l'arène **s'étend**, ne **converge** pas.

| Résultat | Chiffre clé |
|---|---|
| Diversité stylistique mensuelle | pente **+0.057/mois** (p<0.001) — **hausse** |
| Après contrôle n_models | pente **+0.047** — hausse persiste |
| Style premium bold (niveau moyen) | **+19.8 %** odds/SD |
| Tendance temporelle bold | **−1.4 pts/mois** (diminue, pas augmente) |

**Figure** : `paper/figures/R1_convergence_robustness.png`  
**Figure optionnelle** : `paper/figures/R2bis_style_premium_longitudinal.png`

**Takeaway** : pas de signal Goodhart « collapse imminent » sur la dynamique.

---

### Slide 5 — R3 : preuve causale (cœur empirique #1)

**Méthode en 1 ligne** : réécrire des réponses gagnantes dans 3 styles (concis,
verbose-markdown, neutre) → filtrer cosine ≥ 0.85 → re-juger paires.

**Résultats ensemble 2 juges sérieux** (Mistral + Llama 70B, κ=0.67) :

| Comparaison | Win rate style | IC 95 % (accord 2/2) |
|---|---:|---|
| concise vs neutre | **97.3 %** | [86.2, 99.5] |
| verbose vs neutre | **26.9 %** | [13.7, 46.1] |

**Figure** : `paper/figures/R3_ensemble_judge.png`

**Takeaway** : à contenu proche, **concision gagne**, verbose **perd** — ce n'est
pas « plus de markdown = mieux ».

---

### Slide 6 — R5 : style indépendant de la qualité déclarée

**Question** : le style prédit-il encore le vote après contrôle de 7 labels qualité
utilisateur (complétude, utilité, etc.) ?

| Modèle | AUC |
|---|---:|
| Qualité seule | 0.821 |
| Qualité + style | **0.862** |
| Gain style | **+0.041** |

**Effets style après contrôle qualité** :

| Feature | % odds / SD |
|---|---:|
| bold | +30.6 |
| lists | +15.4 |
| headers | +12.8 |

**Figure** : `paper/figures/R5_style_vs_quality.png`  
**Table** : `paper/tables/table_r5_style_coefficients.md`

**Takeaway** : le format est un **proxy** partiel — il prédit au-delà de ce que
l'utilisateur déclare sur la qualité.

---

### Slide 7 — R5bis : structure ≠ longueur (slide pivot ⭐)

**Question** : bold/headers aident-ils encore après contrôle **qualité + longueur** ?

| Modèle | AUC |
|---|---:|
| Qualité + style (R5) | 0.862 |
| Qualité + longueur + style (R5bis) | **0.865** |

**Après double contrôle** :

| Variable | % odds / SD |
|---|---:|
| Longueur (log) | +40.5 |
| bold | **+14.5** (survit) |
| headers | **+13.8** (survit) |
| lists | +5.3 (atténué) |

**Figure** : `paper/figures/R5bis_structure_vs_length.png`  
**Table** : `paper/tables/table_r5bis_structure_vs_length.md`

**Takeaway** : réconcilie R3 et R5 — **verbosité** et **structure markdown** sont
deux leviers distincts.

---

### Slide 8 — R6 : endogénéité & tiers de modèles

**Question** : est-ce que seuls les modèles faibles « gamment » le format ?

| Fait | Chiffre |
|---|---|
| Corrélation rating ↔ formatage (bold) | **r = +0.57** |
| Bold moyen bottom → top tier | 7.3 → **27.8** |
| Style premium bold, paire bottom-top | **+41.6 %** |

**Figure** : `paper/figures/R6_endogeneity_tiers.png`  
**Table** : `paper/tables/table_r6_endogeneity_tiers.md`

**Takeaway** : les **meilleurs** formatent plus ; le biais style persiste partout,
**maximal en matchups déséquilibrés** (faible vs fort).

---

### Slide 9 — Synthèse & implications

**Ce qu'on a testé** : effondrement temporel → **non**  
**Ce qu'on a trouvé** : vulnérabilité stylistique **oui**, décomposée :

1. Causal : concision bat verbose (R3)
2. Mécanisme : style au-delà qualité + longueur (R5/R5bis)
3. Contexte : effet amplifié bottom-top (R6)

**Recommandations** :

- Style-control systématique (comme LMSYS)
- Contrôler longueur séparément du markdown structurel
- Auditer les matchups cross-tier

---

### Slide 10 — Limites & extensions

**Limites honnêtes** :

- AUC in-sample (pas de hold-out)
- R3 = juges LLM, pas votes humains Compar:IA
- Labels qualité = annotations post-vote, pas vérité terrain
- R4 forecast abandonné (pas de tendance à effondrer)

**Extension Zilinskas** : temporel + causal + longueur + tiers — pas une redécouverte.

**Questions ?**

---

## 3. Inventaire figures & tables (chemins repo)

| Asset | Chemin |
|---|---|
| R1 robustesse | `paper/figures/R1_convergence_robustness.png` |
| R1 brute | `paper/figures/R1_convergence.png` |
| R2bis | `paper/figures/R2bis_style_premium_longitudinal.png` |
| R3 N=100 | `paper/figures/R3_style_premium_n100.png` |
| R3 ensemble | `paper/figures/R3_ensemble_judge.png` |
| R5 | `paper/figures/R5_style_vs_quality.png` |
| R5bis | `paper/figures/R5bis_structure_vs_length.png` |
| R6 | `paper/figures/R6_endogeneity_tiers.png` |
| Table R3 | `paper/tables/table_r3_counterfactual.md` |
| Table R5 AUC | `paper/tables/table_r5_auc.md` |
| Table R5 coefs | `paper/tables/table_r5_style_coefficients.md` |
| Table R5bis | `paper/tables/table_r5bis_structure_vs_length.md` |
| Table R6 | `paper/tables/table_r6_endogeneity_tiers.md` |

---

## 4. Phrase d'accroche (slide 1 ou 2)

> « On est venus chercher l'effondrement de Compar:IA. On a trouvé quelque chose de
> plus utile : pas de convergence globale, mais une vulnérabilité décomposée — la
> verbosité est pénalisée, la structure markdown récompensée, et les deux passent
> au-delà des labels qualité. L'arène ne s'est pas mordu la queue, mais elle reste
> hackable. »

---

## 5. Références internes repo

| Document | Rôle |
|---|---|
| [`pitch_equipe_reference.md`](pitch_equipe_reference.md) | Définitions, chiffres détaillés, FAQ jury |
| [`project_status.md`](project_status.md) | État technique complet du projet |
| [`projet_arene_se_mord_la_queue.md`](../projet_arene_se_mord_la_queue.md) | Spec initiale |
| `style-control-analysis/paper_draft.md` | Papier Zilinskas de référence |

---

## 6. Instructions explicites pour Dust

1. Lire ce fichier + parcourir les figures listées en section 3.
2. Produire **10 slides** suivant la structure section 2 (ajuster à 12 max si besoin).
3. Chaque slide résultat (R3–R6) doit avoir **1 figure + 2–3 bullets chiffrés max**.
4. Slide 7 (R5bis) = **slide pivot** — la plus importante après R3.
5. Inclure une slide limites — crédibilité scientifique.
6. Ne pas inventer de chiffres : tout est dans les tables `paper/tables/`.
7. Titre deck : reprendre le message central section 1.
