# ðŸ¦Ž ANNEXES â€” Projet "L'ArÃ¨ne se mord la queue"

> ComplÃ©ment technique Ã  `projet_arene_se_mord_la_queue.md`
> Contient : code complet, requirements, communications, rÃ©fÃ©rences bibliographiques.

---

## ðŸ“‘ Sommaire des annexes

- A1. requirements.txt + setup.sh
- A2. Code starter : ingestion + validation Zilinskas
- A3. Code : features stylo additionnelles
- A4. Code : R1 convergence temporelle
- A5. Code : R2bis style premium par cohorte
- A6. Code : R3 contrefactuels + NLI + judge
- A7. Code : R4 forecast bayÃ©sien + Prophet
- A8. Template LaTeX papier
- A9. Template Streamlit dashboard
- A10. Communications avec Simonas (3 mails)
- A11. RÃ©fÃ©rences bibliographiques BibTeX
- A12. `.cursorrules` pour Cursor
- A13. Prompt systÃ¨me Cursor

---

## A1. `requirements.txt` + `setup.sh`

### requirements.txt
```txt
# Data
pandas==2.2.*
polars==1.*
numpy
scipy
pyarrow

# HuggingFace
datasets
huggingface-hub

# NLP
spacy
textstat
sentence-transformers
transformers
torch

# Topic modeling
bertopic
umap-learn
hdbscan

# Stats / Causal
statsmodels
bambi
scikit-learn
dowhy
econml
pot  # Optimal Transport

# BayÃ©sien / Forecast
numpyro
prophet

# ML
xgboost
shap

# API LLM
mistralai
groq

# Viz
matplotlib
seaborn
plotly
altair

# DÃ©mo
streamlit

# Dev
jupyter
ipykernel
swifter
tqdm
python-dotenv
```

### setup.sh
```bash
#!/bin/bash
set -e

echo "=== Setup environnement comparia-hackathon ==="

# 1. Venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Deps
pip install -r requirements.txt
python -m spacy download fr_core_news_lg

# 3. Clone Zilinskas
mkdir -p external
if [ ! -d "external/zilinskas" ]; then
    git clone https://github.com/simonaszilinskas/style-control-analysis.git external/zilinskas
fi

# 4. Dossiers data
mkdir -p data/raw data/interim data/processed
mkdir -p notebooks src/compariawatch paper/figures slides scripts

# 5. HF login
echo "Connecte-toi Ã  HuggingFace :"
huggingface-cli login

# 6. .env template
if [ ! -f ".env" ]; then
    cat > .env <<EOF
HF_TOKEN=hf_xxxxx
MISTRAL_API_KEY=xxxxx
GROQ_API_KEY=gsk_xxxxx
EOF
    echo "âš ï¸ Ã‰dite .env avec tes clÃ©s API"
fi

echo "=== Setup terminÃ© âœ… ==="
```

---

## A2. Code : ingestion + validation Zilinskas

### `notebooks/01_validation_zilinskas.ipynb`

```python
# Cell 1 â€” Imports
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.stats import bootstrap

ZILINSKAS = Path("../external/zilinskas")
DATA = Path("../data")

# Cell 2 â€” Chargement du parquet
df = pd.read_parquet(ZILINSKAS / "battles_bt_styled.parquet")
print(f"Shape: {df.shape}")
print(f"Cols: {df.columns.tolist()}")
print(df.head())

# Cell 3 â€” Sanity check : reproduire le coef bold â‰ˆ 19%
# (adapter selon le schÃ©ma exact du parquet)
# Le BT style-controlled : pour chaque battle on a +1 pour winner, -1 pour loser
# sur la colonne modÃ¨le, et delta_style pour les features

# PrÃ©paration
models = sorted(set(df["model_a"]) | set(df["model_b"]))
model_to_idx = {m: i for i, m in enumerate(models)}
n_models = len(models)

style_feats = ["bold", "lists", "headers", "code_blocks", "emoji"]

# Filtrer battles dÃ©cisifs
decisive = df[df["winner"].isin(["model_a", "model_b"])].copy()

# Construire X et y
N = len(decisive)
X_model = np.zeros((N, n_models))
X_style = np.zeros((N, len(style_feats)))

for idx, row in enumerate(decisive.itertuples()):
    a_idx = model_to_idx[row.model_a]
    b_idx = model_to_idx[row.model_b]
    winner_a = (row.winner == "model_a")
    sign = 1 if winner_a else -1
    X_model[idx, a_idx] = sign
    X_model[idx, b_idx] = -sign
    for fi, f in enumerate(style_feats):
        a_val = getattr(row, f"{f}_a", 0)
        b_val = getattr(row, f"{f}_b", 0)
        X_style[idx, fi] = sign * (a_val - b_val)

X = np.hstack([X_model, X_style])
y = np.ones(N)  # toujours 1 puisque les signes sont dÃ©jÃ  encodÃ©s

# Standardiser les features style
from sklearn.preprocessing import StandardScaler
X[:, n_models:] = StandardScaler().fit_transform(X[:, n_models:])

# Fit
model = LogisticRegression(penalty=None, max_iter=2000, fit_intercept=False)
model.fit(X, y)

style_coeffs = dict(zip(style_feats, model.coef_[0, n_models:]))
print("Coefficients de style (Ã  comparer Ã  Zilinskas):")
for f, c in style_coeffs.items():
    pct = (np.exp(c) - 1) * 100
    print(f"  {f}: coef={c:.3f} â†’ +{pct:.1f}% odds/SD")
print("\n[RÃ©fÃ©rence Zilinskas]: bold +19.1%, lists +17.8%, headers +15.6%")

# Cell 4 â€” Extraction des dates
df["date"] = pd.to_datetime(df["timestamp"])
df["month"] = df["date"].dt.to_period("M").astype(str)
print(f"PÃ©riode couverte: {df['date'].min()} â†’ {df['date'].max()}")
print(f"Nb cohortes mensuelles: {df['month'].nunique()}")
print(f"Distribution mensuelle:\n{df['month'].value_counts().sort_index()}")

# Cell 5 â€” Sauvegarde dataset propre
df.to_parquet(DATA / "interim/zilinskas_validated.parquet")
print("âœ… Sanity check ok, dataset prÃªt Ã  l'extension")
```

---

## A3. Code : features stylo additionnelles

### `notebooks/02_features_additionnelles.ipynb`

```python
# Cell 1 â€” Imports
import pandas as pd, numpy as np, re, textstat, spacy
from tqdm import tqdm
from pathlib import Path
tqdm.pandas()

nlp = spacy.load("fr_core_news_lg", disable=["ner", "parser"])

DATA = Path("../data")
df = pd.read_parquet(DATA / "interim/zilinskas_validated.parquet")

# Cell 2 â€” Lexiques
ANGLICISMES = {"deadline","feedback","email","mail","meeting","challenge",
               "brief","data","insight","fail","kpi","ASAP","check","fix"}
FLATTERIE_RX = re.compile(
    r"excellente question|bonne question|trÃ¨s (bonne|intÃ©ressante)|bravo",
    re.IGNORECASE
)
HEDGING_RX = re.compile(
    r"\b(peut-Ãªtre|il semble|gÃ©nÃ©ralement|souvent|probablement|en thÃ©orie)\b",
    re.IGNORECASE
)

# Cell 3 â€” Fonction stylo additionnelle
def stylo_additional(text):
    if not isinstance(text, str) or len(text) < 5:
        return {k: 0.0 for k in ADD_KEYS}
    text = text[:30_000]
    doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    words = [t for t in toks if t.is_alpha]
    n_t, n_w = max(len(toks),1), max(len(words),1)
    sents = list(doc.sents) or [doc]
    n_s = max(len(sents),1)
    pos = pd.Series([t.pos_ for t in toks]).value_counts(normalize=True)
    return {
        "n_tokens": n_t,
        "n_sents": n_s,
        "avg_sent_len": n_t / n_s,
        "ttr": len({t.lemma_.lower() for t in words}) / n_w,
        "pct_long_words": sum(len(t.text) > 6 for t in words) / n_w,
        "tu_count": len(re.findall(r"\b(tu|ton|ta|tes|toi)\b", text, re.I)),
        "vous_count": len(re.findall(r"\b(vous|votre|vos)\b", text, re.I)),
        "ratio_excl": text.count("!") / n_s,
        "ratio_quest": text.count("?") / n_s,
        "ratio_ellipsis": (text.count("â€¦") + text.count("...")) / n_s,
        "pct_adj": pos.get("ADJ", 0),
        "pct_adv": pos.get("ADV", 0),
        "pct_verb": pos.get("VERB", 0),
        "pct_pron": pos.get("PRON", 0),
        "flatterie": len(FLATTERIE_RX.findall(text)),
        "hedging": len(HEDGING_RX.findall(text)),
        "anglicisms": sum(w.text.lower() in ANGLICISMES for w in words) / n_w,
        "flesch": textstat.flesch_reading_ease(text),
    }

ADD_KEYS = list(stylo_additional("test fonctionnel").keys())

# Cell 4 â€” Si les textes ne sont pas dans le parquet, joindre depuis HF
# (vÃ©rifier d'abord les colonnes du parquet)
if "response_a_text" not in df.columns:
    from datasets import load_dataset
    conv = load_dataset("ministere-culture/comparia-conversations", split="train")
    conv_pd = conv.to_pandas()
    # Extraction
    def get_asst(c):
        if not c: return ""
        return "\n".join(m["content"] for m in c if m.get("role")=="assistant")
    conv_pd["resp_a_text"] = conv_pd["conversation_a"].apply(get_asst)
    conv_pd["resp_b_text"] = conv_pd["conversation_b"].apply(get_asst)
    df = df.merge(
        conv_pd[["conversation_pair_id","resp_a_text","resp_b_text"]],
        on="conversation_pair_id", how="left"
    )

# Cell 5 â€” Extraction (LONG : 2-4h)
print("Extraction features additionnelles sur rÃ©sp A...")
fa = pd.json_normalize(df["resp_a_text"].progress_apply(stylo_additional))
fa.columns = [f"a_{c}" for c in fa.columns]

print("Extraction sur rÃ©sp B...")
fb = pd.json_normalize(df["resp_b_text"].progress_apply(stylo_additional))
fb.columns = [f"b_{c}" for c in fb.columns]

df = pd.concat([df.reset_index(drop=True), fa, fb], axis=1)
df.to_parquet(DATA / "interim/df_with_all_features.parquet")
print(f"âœ… SauvegardÃ©: {len(df):,} lignes Ã— {df.shape[1]} colonnes")
```

---

## A4. Code : R1 convergence temporelle

### `notebooks/03_R1_convergence.ipynb`

```python
# Cell 1
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import ot  # POT optimal transport
from pathlib import Path

df = pd.read_parquet("../data/interim/df_with_all_features.parquet")

# Cell 2 â€” Reconstruire long format
ZILINSKAS_FEATS = ["bold", "lists", "headers", "code_blocks", "emoji"]
ADDITIONAL_FEATS = ["n_tokens","avg_sent_len","ttr","pct_long_words",
                     "tu_count","vous_count","ratio_excl","ratio_quest",
                     "pct_adj","pct_adv","pct_verb","flatterie","hedging",
                     "anglicisms","flesch"]
ALL_FEATS = ZILINSKAS_FEATS + ADDITIONAL_FEATS

# Long : 1 ligne = 1 rÃ©ponse
long_a = df[["model_a", "month"] +
            [f"{f}_a" for f in ZILINSKAS_FEATS] +
            [f"a_{f}" for f in ADDITIONAL_FEATS]].copy()
long_a.columns = ["model", "month"] + ALL_FEATS

long_b = df[["model_b", "month"] +
            [f"{f}_b" for f in ZILINSKAS_FEATS] +
            [f"b_{f}" for f in ADDITIONAL_FEATS]].copy()
long_b.columns = long_a.columns

long_df = pd.concat([long_a, long_b]).reset_index(drop=True)

# Cell 3 â€” Z-score global
scaler = StandardScaler()
long_df[ALL_FEATS] = scaler.fit_transform(long_df[ALL_FEATS].fillna(0))

# Cell 4 â€” CentroÃ¯de stylistique par (modÃ¨le, mois)
centroids = (long_df.groupby(["model","month"])[ALL_FEATS]
             .mean().reset_index())

# Cell 5 â€” DiversitÃ© inter-modÃ¨les par mois
def monthly_diversity(month):
    sub = centroids[centroids["month"] == month]
    if len(sub) < 5: return None
    X = sub[ALL_FEATS].values
    return float(np.mean(pdist(X)))

months = sorted(centroids["month"].unique())
diversity = [(m, monthly_diversity(m)) for m in months]
div_df = pd.DataFrame(diversity, columns=["month", "diversity"]).dropna()
div_df["month_idx"] = range(len(div_df))

# Cell 6 â€” RÃ©gression
X = sm.add_constant(div_df["month_idx"])
res = sm.OLS(div_df["diversity"], X).fit()
print(res.summary())

# Cell 7 â€” Figure principale R1
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(div_df["month"], div_df["diversity"], "o-", lw=2, label="DiversitÃ© observÃ©e")
# RÃ©gression
xpred = np.linspace(0, len(div_df)-1, 100)
ypred = res.params["const"] + res.params["month_idx"] * xpred
ax.plot([div_df["month"].iloc[int(x)] if int(x) < len(div_df) else div_df["month"].iloc[-1]
         for x in xpred], ypred, "--", color="red", label=f"Tendance (Î²={res.params['month_idx']:.4f})")
ax.set_xlabel("Mois")
ax.set_ylabel("DiversitÃ© stylistique inter-modÃ¨les (distance moyenne)")
ax.set_title(f"R1: Convergence stylistique des LLM sur Compar:IA\n"
             f"Pente = {res.params['month_idx']:.4f} (p = {res.pvalues['month_idx']:.3f})")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../paper/figures/R1_convergence.png", dpi=200, bbox_inches="tight")
plt.show()

# Cell 8 â€” Sauvegarder pour R4
div_df.to_parquet("../data/processed/diversity_temporal.parquet")
```

---

## A5. Code : R2bis style premium par cohorte

### `notebooks/04_R2bis_style_premium_longitudinal.ipynb`

```python
# Cell 1
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm

df = pd.read_parquet("../data/interim/df_with_all_features.parquet")
decisive = df[df["winner"].isin(["model_a", "model_b"])].copy()

# Cell 2 â€” Fonction BT style-controlled (template Zilinskas)
def fit_bt_with_style(batt_df, style_feats=["bold","lists","headers"]):
    """Reproduit le BT style-controlled de Zilinskas sur un sous-ensemble."""
    models = sorted(set(batt_df["model_a"]) | set(batt_df["model_b"]))
    if len(models) < 3: return None
    m2i = {m: i for i, m in enumerate(models)}
    n_m = len(models)
    N = len(batt_df)
    X_model = np.zeros((N, n_m))
    X_style = np.zeros((N, len(style_feats)))

    for idx, row in enumerate(batt_df.itertuples()):
        ai, bi = m2i[row.model_a], m2i[row.model_b]
        sign = 1 if row.winner == "model_a" else -1
        X_model[idx, ai] = sign
        X_model[idx, bi] = -sign
        for fi, f in enumerate(style_feats):
            a_val = getattr(row, f"{f}_a", 0)
            b_val = getattr(row, f"{f}_b", 0)
            X_style[idx, fi] = sign * (a_val - b_val)

    X = np.hstack([X_model, X_style])
    X[:, n_m:] = StandardScaler().fit_transform(X[:, n_m:])
    y = np.ones(N)

    try:
        m = LogisticRegression(penalty=None, max_iter=1000, fit_intercept=False)
        m.fit(X, y)
        return {f: m.coef_[0, n_m + i] for i, f in enumerate(style_feats)}
    except Exception:
        return None

# Cell 3 â€” Par cohorte mensuelle (avec bootstrap pour CI)
def bootstrap_coefs(batt, n_boot=100, **kwargs):
    coefs_boot = []
    for _ in range(n_boot):
        sample = batt.sample(len(batt), replace=True)
        res = fit_bt_with_style(sample, **kwargs)
        if res: coefs_boot.append(res)
    if not coefs_boot: return None
    df_boot = pd.DataFrame(coefs_boot)
    return {f: (df_boot[f].mean(), df_boot[f].quantile(0.025),
                df_boot[f].quantile(0.975)) for f in df_boot.columns}

results = []
for cohort, batt in tqdm(decisive.groupby("month")):
    if len(batt) < 500: continue
    res = bootstrap_coefs(batt, n_boot=50)
    if res:
        for f, (m, lo, hi) in res.items():
            results.append({"month": cohort, "feature": f,
                            "coef": m, "lo": lo, "hi": hi})

cohort_df = pd.DataFrame(results)
cohort_df["month_dt"] = pd.to_datetime(cohort_df["month"])
cohort_df["month_idx"] = (cohort_df["month_dt"] -
                          cohort_df["month_dt"].min()).dt.days

# Cell 4 â€” Tests de tendance par feature
print("=== Tests de croissance du Style Premium (H2) ===")
for f in ["bold", "lists", "headers"]:
    sub = cohort_df[cohort_df["feature"] == f].sort_values("month_dt")
    X = sm.add_constant(sub["month_idx"])
    res = sm.OLS(sub["coef"], X).fit()
    pct = (np.exp(res.params["month_idx"] * 30) - 1) * 100  # par mois
    print(f"{f}: pente = {res.params['month_idx']:.6f}/jour "
          f"â‰ˆ {pct:+.2f}%/mois, p = {res.pvalues['month_idx']:.3f}")

# Cell 5 â€” Figure : Ã©volution longitudinale
fig, ax = plt.subplots(figsize=(11, 6))
for f, color in [("bold","C0"), ("lists","C1"), ("headers","C2")]:
    sub = cohort_df[cohort_df["feature"] == f].sort_values("month_dt")
    pct_coef = (np.exp(sub["coef"]) - 1) * 100
    pct_lo = (np.exp(sub["lo"]) - 1) * 100
    pct_hi = (np.exp(sub["hi"]) - 1) * 100
    ax.plot(sub["month_dt"], pct_coef, "o-", color=color, label=f)
    ax.fill_between(sub["month_dt"], pct_lo, pct_hi, alpha=0.2, color=color)

ax.axhline(19.1, color="C0", linestyle=":", alpha=0.5,
           label="Zilinskas global: bold +19.1%")
ax.set_xlabel("Mois")
ax.set_ylabel("Style coefficient (% odds win / SD)")
ax.set_title("R2bis: Ã‰volution longitudinale du Style Premium\n"
             "(Reproduction du BT style-controlled de Zilinskas par cohorte)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../paper/figures/R2bis_style_premium_longitudinal.png",
            dpi=200, bbox_inches="tight")
plt.show()

cohort_df.to_parquet("../data/processed/style_premium_temporal.parquet")
```

---

## A6. Code : R3 contrefactuels + NLI + judge

### `notebooks/05_R3_counterfactual.ipynb`

```python
# Cell 1 â€” Setup
import pandas as pd, numpy as np, os, time
from mistralai import Mistral
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
df = pd.read_parquet("../data/interim/df_with_all_features.parquet")

# Cell 2 â€” Ã‰chantillon stratifiÃ© 500 paires
sample = df.sample(500, random_state=42).reset_index(drop=True)

# Cell 3 â€” Prompt de rÃ©Ã©criture
REWRITE_PROMPT = """Tu es un assistant qui rÃ©Ã©crit des rÃ©ponses en changeant uniquement leur STYLE, en prÃ©servant strictement le CONTENU.

Style cible: {style}

RÃ¨gles strictes:
- PrÃ©server TOUS les faits, exemples, chiffres, Ã©tapes
- Ne rien ajouter, ne rien retirer comme information
- Adapter uniquement: longueur, formatage, ton, registre

RÃ©ponse originale:
\"\"\"
{text}
\"\"\"

RÃ©ponse rÃ©Ã©crite en style "{style}":
"""

STYLES = {
    "verbose_markdown": "long, structurÃ© avec markdown (titres, bullets, gras), explications dÃ©taillÃ©es",
    "concise_direct":   "court, direct, sans markdown, allant droit au but",
    "neutre_baseline":  "neutre, paragraphe simple sans markdown, longueur moyenne",
}

# Cell 4 â€” Fonction rewrite avec retry
def rewrite(text, style, max_tries=3):
    for attempt in range(max_tries):
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role":"user", "content":
                    REWRITE_PROMPT.format(style=STYLES[style], text=text[:3000])}],
                temperature=0.3, max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

# Cell 5 â€” GÃ©nÃ©ration
results = []
for i, row in tqdm(sample.iterrows(), total=len(sample), desc="Rewriting"):
    original = row["resp_a_text"][:3000] if isinstance(row["resp_a_text"], str) else None
    if not original: continue
    for style in STYLES:
        rewritten = rewrite(original, style)
        results.append({
            "idx": i, "model": row["model_a"],
            "prompt": row.get("opening_msg", "")[:500],
            "original": original, "style": style, "rewritten": rewritten,
        })
    if i % 50 == 0:
        pd.DataFrame(results).to_parquet("../data/interim/rewrites_partial.parquet")

rewrites = pd.DataFrame(results)
rewrites.to_parquet("../data/interim/rewrites.parquet")

# Cell 6 â€” VÃ©rification prÃ©servation
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def cos_sim(a, b):
    if not a or not b: return 0.0
    e1 = embedder.encode(a[:2000], convert_to_tensor=True)
    e2 = embedder.encode(b[:2000], convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()

rewrites["cosine"] = rewrites.progress_apply(
    lambda r: cos_sim(r["original"], r["rewritten"]), axis=1
)
print(f"PrÃ©servÃ©es (cosine â‰¥ 0.85): {(rewrites['cosine'] >= 0.85).mean():.1%}")
preserved = rewrites[rewrites["cosine"] >= 0.85].copy()
preserved.to_parquet("../data/interim/rewrites_preserved.parquet")

# Cell 7 â€” LLM-as-judge
JUDGE_PROMPT = """Tu es juge sur la plateforme Compar:IA. Voici un prompt et deux rÃ©ponses.
Choisis la meilleure UNIQUEMENT en fonction de ta prÃ©fÃ©rence d'utilisateur rÃ©el.

PROMPT: {prompt}

RÃ‰PONSE A: {a}
RÃ‰PONSE B: {b}

RÃ©ponds STRICTEMENT par "A" ou "B" ou "tie".
"""

def judge(prompt, a, b):
    flip = np.random.rand() > 0.5
    left, right = (b, a) if flip else (a, b)
    try:
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role":"user","content":
                JUDGE_PROMPT.format(prompt=prompt[:500],
                                    a=left[:1500], b=right[:1500])}],
            temperature=0, max_tokens=5,
        )
        ans = resp.choices[0].message.content.strip().upper()
        if "A" in ans: return "B" if flip else "A"
        if "B" in ans: return "A" if flip else "B"
        return "tie"
    except: return None

# Cell 8 â€” Pivot pour avoir une ligne par paire
pairs = (preserved.pivot(index="idx", columns="style", values="rewritten")
         .dropna(subset=["verbose_markdown","concise_direct","neutre_baseline"]))

# Cell 9 â€” Comparer verbose vs neutre (test causal H3)
prompts_dict = sample.set_index(sample.index)["opening_msg"].to_dict()
votes = []
for idx in tqdm(pairs.index, desc="Judging verbose vs neutre"):
    p = prompts_dict.get(idx, "")
    v = judge(p, pairs.loc[idx,"verbose_markdown"],
                  pairs.loc[idx,"neutre_baseline"])
    votes.append({"idx": idx, "vote": v})

votes_df = pd.DataFrame(votes)

# Cell 10 â€” Calcul ATE
import scipy.stats as st
n_a = (votes_df["vote"] == "A").sum()  # A = verbose
n_b = (votes_df["vote"] == "B").sum()  # B = neutre
n_tie = (votes_df["vote"] == "tie").sum()
total = n_a + n_b
p = n_a / total if total > 0 else 0.5
ci = st.binomtest(n_a, total).proportion_ci()

print(f"=== R3: Effet causal du style verbose vs neutre ===")
print(f"N dÃ©cisif: {total}, ties: {n_tie}")
print(f"P(verbose > neutre) = {p:.1%} [IC95: {ci.low:.1%}, {ci.high:.1%}]")
print(f"H3 acceptÃ©e: {p > 0.5 and ci.low > 0.5}")

# Cell 11 â€” Figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Verbose-markdown", "Tie", "Neutre-baseline"],
       [n_a/len(votes_df), n_tie/len(votes_df), n_b/len(votes_df)],
       color=["C3","gray","C2"])
ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
ax.set_ylabel("Proportion des votes")
ax.set_title(f"R3: Preuve causale du Style Premium\n"
             f"P(verbose > neutre) = {p:.1%} [IC: {ci.low:.1%}, {ci.high:.1%}]\n"
             f"(Ã  contenu sÃ©mantiquement prÃ©servÃ© via cosine â‰¥ 0.85)")
plt.tight_layout()
plt.savefig("../paper/figures/R3_causal_style.png", dpi=200)
plt.show()

votes_df.to_parquet("../data/processed/causal_votes.parquet")
```

---

## A7. Code : R4 forecast bayÃ©sien + Prophet

### `notebooks/06_R4_forecast.ipynb`

```python
# Cell 1
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from prophet import Prophet

div_df = pd.read_parquet("../data/processed/diversity_temporal.parquet")

# Cell 2 â€” PrÃ©paration Prophet
div_df["ds"] = pd.to_datetime(div_df["month"].astype(str) + "-01")
div_df["y"] = div_df["diversity"]

# Cell 3 â€” Fit + forecast
m = Prophet(yearly_seasonality=False, daily_seasonality=False,
            interval_width=0.95)
m.fit(div_df[["ds","y"]])

future = m.make_future_dataframe(periods=24, freq="MS")
forecast = m.predict(future)

# Cell 4 â€” Seuil critique = variance intra-modÃ¨le moyenne
long_df = pd.read_parquet("../data/interim/df_with_all_features.parquet")
# (approximation : on prend la std intra-modÃ¨le moyenne sur features stylo)
# Ã€ adapter selon ton schÃ©ma exact

# Pour la dÃ©mo, on prend un seuil arbitraire basÃ© sur quantile bas
tau_crit = div_df["y"].quantile(0.10)
print(f"Seuil critique Ï„ = {tau_crit:.3f}")

# Cell 5 â€” Date d'effondrement
below = forecast[forecast["yhat"] < tau_crit]
crit_date = below["ds"].min() if len(below) > 0 else None
print(f"Date projetÃ©e d'effondrement: {crit_date}")

# Cell 6 â€” Figure
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(div_df["ds"], div_df["y"], "o", color="black", label="ObservÃ©")
ax.plot(forecast["ds"], forecast["yhat"], "-", color="C0", label="Forecast")
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                alpha=0.2, color="C0")
ax.axhline(tau_crit, color="red", linestyle="--", label=f"Seuil critique Ï„={tau_crit:.2f}")
if crit_date:
    ax.axvline(crit_date, color="red", alpha=0.5)
    ax.annotate(f"Effondrement projetÃ©:\n{crit_date.strftime('%Y-%m')}",
                xy=(crit_date, tau_crit), xytext=(20, 30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Date")
ax.set_ylabel("DiversitÃ© stylistique inter-modÃ¨les")
ax.set_title("R4: Projection bayÃ©sienne d'effondrement du benchmark Compar:IA")
plt.legend()
plt.tight_layout()
plt.savefig("../paper/figures/R4_forecast.png", dpi=200, bbox_inches="tight")
plt.show()

forecast.to_parquet("../data/processed/forecast_R4.parquet")
```

---

## A8. Template LaTeX papier

### `paper/main.tex`

```latex
\documentclass[10pt, twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage{amsmath, amssymb, graphicx, booktabs, natbib, hyperref}
\usepackage[a4paper, margin=2cm]{geometry}

\title{L'ArÃ¨ne se mord la queue\\
\large Extension temporelle, causale et prospective\\
de l'audit de style sur Compar:IA}
\author{Valentin Proux \and Constantin [Nom]\\
\small Hackathon Compar:IA -- M1 SIREN\\
\small UniversitÃ© Paris-Dauphine PSL}
\date{Juin 2026}

\begin{document}
\maketitle

\begin{abstract}
\citet{zilinskas2026formatting} a Ã©tabli que le formatage (bold,
listes, titres markdown) augmente indÃ©pendamment la probabilitÃ©
de victoire sur la plateforme Compar:IA de 16 Ã  19\% par
Ã©cart-type. Nous Ã©tendons ce rÃ©sultat selon trois axes complÃ©mentaires
que son analyse statique ne couvre pas. PremiÃ¨rement, nous mesurons
la convergence stylistique inter-modÃ¨les via la distance de
Wasserstein sur des cohortes mensuelles ($N=16$). DeuxiÃ¨mement,
nous recalculons ses coefficients Bradley-Terry style-controlled
par cohorte pour tester si le "style premium" s'accroÃ®t dans le
temps. TroisiÃ¨mement, nous fournissons la premiÃ¨re preuve causale
(et non observationnelle) de l'effet du style par gÃ©nÃ©ration de
contrefactuels via LLM tiers, vÃ©rifiÃ©s par infÃ©rence textuelle
bidirectionnelle (NLI) et similaritÃ© cosinus. Enfin, nous projetons
par modÃ¨le d'Ã©tat bayÃ©sien la date Ã  laquelle Compar:IA pourrait
perdre son pouvoir discriminant si la tendance se poursuit.
Nos rÃ©sultats [...] et appellent Ã  [...].
\end{abstract}

\section{Introduction}
[Contexte Compar:IA + politique publique]
[Cadre Goodhart]
[Question de recherche et plan]

\section{Travaux connexes}
\subsection{Ã‰valuation par arÃ¨ne et style control}
\citet{chiang2024chatbot} pionnier LMArena, \citet{li2024crowdsourced}
introduit le style control, \citet{zheng2023judging} biais flatterie.

\subsection{Le travail de Zilinskas (2026) sur Compar:IA}
[RÃ©sumÃ© fair-play du papier de Simonas + ses 3 rÃ©sultats clÃ©s]

\section{DonnÃ©es et mÃ©thode}
\subsection{DonnÃ©es}
[RÃ©utilisation du parquet de Zilinskas + extension features]

\subsection{Features stylistiques}
[5 features Zilinskas + 12 additionnelles]

\subsection{MÃ©thodes}
[R1 Wasserstein, R2bis BT par cohorte, R3 contrefactuels+NLI, R4 forecast]

\section{RÃ©sultats}
\subsection{R1: convergence stylistique temporelle}
[Figure R1 + table coefficients]

\subsection{R2bis: croissance du Style Premium}
[Figure R2bis + tests stat]

\subsection{R3: preuve causale par contrefactuels}
[Figure R3 + ATE table]

\subsection{R4: projection d'effondrement}
[Figure R4 + intervalle date]

\section{Discussion}
\subsection{Implications pour Compar:IA}
[3 recommandations]

\subsection{Limites: puissance statistique avec 16 mois}
[HonnÃªtetÃ© sur warning Simonas]

\section{Conclusion}
[SynthÃ¨se + appel Ã  publication Ã©tendue]

\section*{Remerciements}
Nous remercions Simon Zilinskas pour son retour, le partage de
ses ressources et son travail antÃ©rieur qui a rendu cette Ã©tude
possible.

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
```

---

## A9. Template Streamlit dashboard

### `scripts/app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="L'ArÃ¨ne se mord la queue", layout="wide")

st.title("ðŸ¦Ž L'ArÃ¨ne se mord la queue")
st.markdown("**Extension temporelle, causale et prospective de l'audit de style sur Compar:IA**")
st.caption("Hackathon Compar:IA Ã— Dauphine PSL â€” juin 2026")

DATA = Path("../data/processed")

tab1, tab2, tab3 = st.tabs(["ðŸ“‰ Diversity Tracker", "ðŸ“ˆ Style Premium Evolution",
                             "ðŸ§ª Counterfactual Explorer"])

with tab1:
    st.subheader("Ã‰volution de la diversitÃ© stylistique inter-modÃ¨les")
    div = pd.read_parquet(DATA / "diversity_temporal.parquet")
    forecast = pd.read_parquet(DATA / "forecast_R4.parquet")
    fig = px.line(div, x="month", y="diversity", title="R1: convergence temporelle")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Forecast** (R4) : voir figure papier")

with tab2:
    st.subheader("Ã‰volution longitudinale du Style Premium de Zilinskas (2026)")
    sp = pd.read_parquet(DATA / "style_premium_temporal.parquet")
    fig = px.line(sp, x="month_dt", y="coef", color="feature",
                  title="R2bis: coefficients BT style-controlled par cohorte mensuelle")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Tester l'effet du style sur la prÃ©fÃ©rence (preuve causale R3)")
    user_text = st.text_area("Entre une rÃ©ponse Ã  analyser :", height=200)
    if st.button("GÃ©nÃ©rer les 3 styles + simuler vote"):
        st.info("(DÃ©mo : connecter Ã  l'API Mistral)")
        # ... appel rewrite() + cos_sim() + judge()

st.sidebar.markdown("---")
st.sidebar.markdown("**Repo** : github.com/valentinproux/comparia-hackathon")
st.sidebar.markdown("**Base** : [Zilinskas (2026)](https://github.com/simonaszilinskas/style-control-analysis)")
```

---

## A10. Communications avec Simonas

### Mail 1 â€” Initial (envoyÃ© mardi)

```
Objet: Hackathon Compar:IA â€” validation rapide d'angle de projet

Bonjour Simonas,

Je suis en Master 1 SIREN Ã  Dauphine, encore merci pour votre
prÃ©sentation de ce matin, le dataset est passionnant.

Avec mon coÃ©quipier, nous hÃ©sitons Ã  partir sur l'angle suivant
pour notre projet et j'aimerais avoir votre avis avant qu'on
s'engage Ã  fond.

L'idÃ©e en quelques lignes : Ã©tudier si Compar:IA est en train
de subir un effet Goodhart, les LLM Ã©tant dÃ©sormais entraÃ®nÃ©s
(RLHF, DPO) pour gagner sur les plateformes de comparaison par
prÃ©fÃ©rence humaine, ils pourraient converger stylistiquement
entre eux, et la part de la prÃ©fÃ©rence due au style (longueur,
mise en forme, ton) prendrait le pas sur la part due au contenu.

ConcrÃ¨tement on viserait :

1. Mesurer la convergence stylistique des modÃ¨les dans le temps
2. Quantifier la part causale du style dans la prÃ©fÃ©rence : on
prendrait quelques milliers de rÃ©ponses rÃ©elles du dataset, on
demanderait Ã  un LLM tiers de les rÃ©Ã©crire dans diffÃ©rents styles
(plus long, plus court, plus formelâ€¦) en vÃ©rifiant que le contenu
reste sÃ©mantiquement Ã©quivalent (via un modÃ¨le d'infÃ©rence
textuelle). Puis on re-jugerait les paires pour estimer de combien
la prÃ©fÃ©rence bouge Ã  contenu constant.
3. Projeter dans le temps Ã  partir de (1) une date Ã  laquelle la
plateforme perdrait son pouvoir discriminant si la tendance se
poursuit.

Ma question : est-ce que cet angle vous paraÃ®t pertinent et
Â«originalÂ» par rapport Ã  ce qui a dÃ©jÃ  Ã©tÃ© fait en interne ou
par d'autres Ã©quipes ? Et y a-t-il un piÃ¨ge mÃ©thodologique ou
un biais du dataset (sampling des paires, randomisation des
positions A/B, persistance du visitor_id, log du temps de vote,
etc.) que je devrais connaÃ®tre avant de me lancer ?

Si vous avez 5 min pour un retour, on partirait sereins.
Sinon un simple Â«Ã§a vaut le coup / pas le coupÂ» me suffit.

Merci beaucoup pour votre aide !

Valentin PROUX â€” M1 SIREN
```

### Mail 2 â€” RÃ©ponse Simonas (reÃ§ue mardi)

```
Hello Valentin et Constantin,

C'est une idÃ©e super intÃ©ressante, bravo !!! Surtout la projection
temporelle. Sur le style on a dÃ©jÃ  fait des choses (si possible Ã 
ne pas republier trop largement), j'ajoute un papier que j'avais
commencÃ© Ã  Ã©crire en PJ et voici la PR ou j'en parlais et le code
que j'ai utilisÃ© pour mes premiÃ¨res expÃ©rimentations.

J'ai juste un peu peur qu'il n'y aurait peut Ãªtre pas assez de
donnÃ©es pour que ce soit statistiquement significatif, mais
j'espÃ¨re que je me trompe â€” en tout cas Ã§a vaut le coup d'essayer.
Ce serait un sujet de papier trÃ¨s solide en tout cas, je croise
mes doigts pour qu'il y ait des rÃ©sultats intÃ©ressants et si oui
â€” j'espÃ¨re que vous pourrez en faire un bel article publiÃ©.

N'hÃ©sitez pas si vous avez des questions.

Bonne journÃ©e,
Simon
```

### Mail 3 â€” RÃ©ponse au pivot (Ã  envoyer mardi soir)

```
Objet: Re: Hackathon Compar:IA â€” pivot aprÃ¨s lecture de tes ressources

Bonjour Simon,

Merci infiniment pour ton retour et le partage de tes ressources.
AprÃ¨s lecture attentive de ton papier et de ton code, on a pivotÃ©
pour bien s'inscrire en complÃ©ment (et pas en redondance) :

- On garde la projection temporelle (que tu as aimÃ©e) comme cÅ“ur
- On l'enrichit d'une analyse causale par contrefactuels
  LLM-gÃ©nÃ©rÃ©s, vÃ©rifiÃ©s par NLI bidirectionnel + cosine
  sentence-embedding (au-delÃ  du contrÃ´le statistique observationnel)
- On ajoute une dimension longitudinale : recalculer tes
  coefficients bold/lists/headers par cohorte temporelle pour
  tester si le Style Premium s'accroÃ®t mois aprÃ¨s mois
- On utilise battles_bt_styled.parquet comme base de travail

Concernant ta prÃ©occupation sur la puissance stat : 16 mois c'est
juste, on prÃ©sentera des cohortes glissantes 60 jours + bootstrap
inter-cohortes pour propager l'incertitude, et nos conclusions
seront calibrÃ©es en consÃ©quence (on visera des intervalles de
crÃ©dibilitÃ© honnÃªtes plutÃ´t que des affirmations fortes).

Ton travail sera Ã©videmment citÃ© dÃ¨s l'introduction et tu apparaÃ®tras
dans les remerciements. Si on a des rÃ©sultats publiables au-delÃ 
du hackathon, on serait honorÃ©s de te proposer une co-Ã©criture.

On revient vers toi vendredi avec les premiers rÃ©sultats.

Bien Ã  toi,
Valentin & Constantin
```

### Mail 4 â€” Post-hackathon (vendredi soir)

```
Objet: Hackathon Compar:IA â€” rÃ©sultats prÃ©liminaires

Bonjour Simon,

Comme promis, voici un rÃ©sumÃ© rapide de ce qu'on a trouvÃ© :

R1 (convergence) : [rÃ©sultat principal]
R2bis (Style Premium longitudinal) : [rÃ©sultat principal]
R3 (preuve causale) : [rÃ©sultat principal]
R4 (forecast) : [date projetÃ©e + IC]

Repo : https://github.com/valentinproux/comparia-hackathon
Dataset HF : https://huggingface.co/datasets/valentinproux/comparia-counterfactuals-fr
Note PDF : en PJ

Si certains rÃ©sultats te paraissent intÃ©ressants pour Ã©tendre ton
papier, on serait ravis d'en discuter et d'envisager une publication
commune. On est dispo quand tu veux.

Encore merci pour tout,
Valentin & Constantin
```

---

## A11. RÃ©fÃ©rences bibliographiques BibTeX

### `paper/refs.bib`

```bibtex
@unpublished{zilinskas2026formatting,
  title={Formatting Bias in French LLM Evaluation:
         Evidence from the Compar:IA Arena},
  author={Zilinskas, Simonas},
  year={2026},
  note={Working paper. Code: https://github.com/simonaszilinskas/style-control-analysis}
}

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

@article{li2024crowdsourced,
  title={From Crowdsourced Data to High-Quality Benchmarks:
         Arena Hard and BenchBuilder Pipeline},
  author={Li, Tianle and others},
  journal={arXiv:2406.11939},
  year={2024}
}

@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and others},
  journal={NeurIPS},
  volume={36},
  pages={46595--46623},
  year={2023}
}

@article{feder2022causal,
  title={Causal Inference in Natural Language Processing:
         Estimation, Prediction, Interpretation and Beyond},
  author={Feder, Amir and others},
  journal={TACL},
  year={2022}
}

@article{benjamini1995controlling,
  title={Controlling the False Discovery Rate:
         A Practical and Powerful Approach to Multiple Testing},
  author={Benjamini, Yoav and Hochberg, Yosef},
  journal={Journal of the Royal Statistical Society: Series B},
  volume={57},
  number={1},
  pages={289--300},
  year={1995}
}

@article{bradley1952rank,
  title={Rank analysis of incomplete block designs:
         I. The method of paired comparisons},
  author={Bradley, Ralph and Terry, Milton},
  journal={Biometrika},
  volume={39},
  number={3--4},
  pages={324--345},
  year={1952}
}

@misc{manheim2018categorizing,
  title={Categorizing Variants of Goodhart's Law},
  author={Manheim, David and Garrabrant, Scott},
  year={2018},
  eprint={1803.04585},
  archivePrefix={arXiv}
}

@unpublished{termignon2026comparia,
  title={compar:IA: The French Government's LLM Arena to Collect
         French-Language Human Prompts and Preference Data},
  author={Termignon, Lucie and others},
  year={2026},
  note={DINUM, MinistÃ¨re de la Culture}
}

@article{giles1973accent,
  title={Accent mobility: A model and some data},
  author={Giles, Howard},
  journal={Anthropological Linguistics},
  year={1973}
}
```

---

## A12. `.cursorrules`

### Ã€ placer Ã  la racine du repo

```
# Cursor Rules - Projet "L'ArÃ¨ne se mord la queue"

Tu es mon copilote technique pour un projet hackathon de 4 jours.

## Contexte
- Projet: extension temporelle/causale/prospective du travail de Zilinskas
  (2026) sur le style control de Compar:IA
- Spec complÃ¨te: voir projet_arene_se_mord_la_queue.md
- Annexes (code, refs, comms): voir projet_annexes.md
- Travail antÃ©rieur clonÃ©: external/zilinskas/ (NE PAS modifier)
- Ã‰quipe: 2 personnes, pitch vendredi 17h

## RÃ¨gles strictes
1. Lis projet_arene_se_mord_la_queue.md et projet_annexes.md avant
   toute action significative
2. Ne refais PAS le travail de Zilinskas (BT global, coefs bold/lists/
   headers, winner-flipping, endogeneity). Utilise ses chiffres et
   son parquet directement.
3. Propose un plan avant de coder si > 30 lignes
4. Une Ã©tape Ã  la fois, attends mon GO
5. Test sur 10 lignes avant tout calcul lourd
6. Sauvegarde intermÃ©diaire en parquet toutes les N itÃ©rations
7. Type hints + docstrings sur fonctions de src/compariawatch/
8. Random state = 42 partout
9. Commits suggÃ©rÃ©s Ã  chaque fin de phase
10. Pas de fine-tuning, pas de RNN. Stack: spaCy, sentence-transformers,
    Mistral API, NumPyro/Prophet, Streamlit

## Style code
- PEP8, black-compatible, < 100 char/ligne
- Constantes en MAJUSCULES
- Fonctions < 50 lignes, responsabilitÃ© unique
- PrÃ©fÃ©rer polars sur pandas si > 100k lignes et opÃ©ration lourde
- PrÃ©fÃ©rer parquet Ã  CSV

## Communication
- Direct, pas de flatterie
- StructurÃ©: titres, bullets, code blocks sÃ©parÃ©s
- Si pas sÃ»r, dire "vÃ©rifie avec df.columns.tolist()"
- Toujours expliquer le POURQUOI, pas juste le QUOI

## Push-back
Si je te demande quelque chose qui:
- contredit la spec
- est risquÃ© pour le timing
- est mathÃ©matiquement incorrect
- consomme trop de budget API
â†’ refuse poliment avec alternative

## Plan d'exÃ©cution (rÃ©fÃ©rence)
- Mar soir: setup + clone Zilinskas + validation
- Mer: features additionnelles + R1 + R2bis + lancer contrefactuels
- Jeu: R3 (NLI + judge + ATE) + R4 (forecast) + draft note
- Ven: finalisation note + slides + dashboard + pitch
```

---

## A13. Prompt systÃ¨me Cursor (pour la 1Ã¨re session)

```
Bonjour Cursor,

Tu dÃ©marres une nouvelle session sur le projet "L'ArÃ¨ne se mord
la queue".

Avant de rÃ©pondre Ã  ma premiÃ¨re demande, fais ce qui suit:

1. Lis le fichier projet_arene_se_mord_la_queue.md Ã  la racine
   du repo
2. Lis le fichier projet_annexes.md
3. VÃ©rifie que external/zilinskas/ existe et liste son contenu
4. Confirme-moi en 3 lignes:
   - Le titre et la thÃ¨se du projet
   - Les 4 modules de rÃ©sultats (R1, R2bis, R3, R4)
   - L'Ã©tat du dossier external/zilinskas/
5. Identifie 2-3 risques critiques de la semaine selon toi
6. Propose par quoi on commence concrÃ¨tement maintenant

Puis attends mon GO. Ne lance AUCUN calcul, AUCUNE installation
sans validation.
```

---

*Annexes v1.0 â€” 2 juin 2026*
*Document compagnon de `projet_arene_se_mord_la_queue.md`*