"""R5 / R5bis — Décomposition style, qualité et longueur dans Compar:IA.

R5 : le style prédit-il encore la victoire après contrôle des labels qualité ?
R5bis : le markdown structurel survit-il après contrôle qualité + longueur ?
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

VOTES_DATASET = "ministere-culture/comparia-votes"
JOIN_KEY = "conversation_pair_id"
RANDOM_STATE = 42

STYLE_FEATURES = ["headers", "lists", "bold", "code_blocks", "emoji"]

QUALITY_FEATURES = [
    "conv_complete",
    "conv_useful",
    "conv_clear_formatting",
    "conv_creative",
    "conv_incorrect",
    "conv_superficial",
    "conv_instructions_not_followed",
]

NEGATIVE_QUALITY_FEATURES = {
    "conv_incorrect",
    "conv_superficial",
    "conv_instructions_not_followed",
}


def load_votes_quality(cache_path: Path, local_votes_path: Path | None = None) -> pd.DataFrame:
    """Charge/cache les colonnes qualité de `comparia-votes`."""
    if cache_path.exists():
        print(f"Cache qualité chargé : {cache_path}")
        return pd.read_parquet(cache_path)

    load_dotenv(Path(".env"))
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        msg = "HF_TOKEN manquant dans .env"
        raise ValueError(msg)

    keep_cols = [
        JOIN_KEY,
        "chosen_model_name",
        "both_equal",
        "model_a_name",
        "model_b_name",
        "timestamp",
    ]
    for feature in QUALITY_FEATURES:
        keep_cols.extend([f"{feature}_a", f"{feature}_b"])

    if local_votes_path is not None and local_votes_path.exists():
        print(f"Lecture parquet local : {local_votes_path}")
        df = pd.read_parquet(local_votes_path, columns=[col for col in keep_cols])
        out = df.copy()
    else:
        print(f"Streaming {VOTES_DATASET} (colonnes qualité)")
        ds = load_dataset(VOTES_DATASET, split="train", streaming=True, token=token)
        rows: list[dict] = []
        for row in tqdm(ds, desc="votes-quality", unit=" rows"):
            rows.append({col: row.get(col) for col in keep_cols})
        out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"Cache qualité sauvegardé : {cache_path} ({len(out):,} lignes)")
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convertit bool/list/num en score numérique simple."""
    if series.dtype == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _assistant_char_count(conversation: object) -> int:
    """Compte les caractères des messages assistant d'une conversation HF."""
    if conversation is None:
        return 0
    messages = list(conversation)
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        total += len(content)
    return total


def load_votes_length(cache_path: Path, local_votes_path: Path | None = None) -> pd.DataFrame:
    """Charge/cache les longueurs assistant (caractères) depuis `comparia-votes`."""
    if cache_path.exists():
        print(f"Cache longueur chargé : {cache_path}")
        return pd.read_parquet(cache_path)

    if local_votes_path is None or not local_votes_path.exists():
        msg = f"Parquet votes introuvable pour longueur : {local_votes_path}"
        raise FileNotFoundError(msg)

    print(f"Extraction longueur depuis {local_votes_path}")
    raw = pd.read_parquet(
        local_votes_path,
        columns=[JOIN_KEY, "conversation_a", "conversation_b"],
    )
    raw["assistant_chars_a"] = raw["conversation_a"].map(_assistant_char_count)
    raw["assistant_chars_b"] = raw["conversation_b"].map(_assistant_char_count)
    out = raw[
        [JOIN_KEY, "assistant_chars_a", "assistant_chars_b"]
    ].drop_duplicates(subset=[JOIN_KEY], keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"Cache longueur sauvegardé : {cache_path} ({len(out):,} lignes)")
    return out


def build_style_quality_dataset(
    battles: pd.DataFrame,
    votes: pd.DataFrame,
    lengths: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Joint style + qualité (+ longueur) et construit les deltas A-B."""
    merged = battles.merge(votes, on=JOIN_KEY, how="inner", suffixes=("", "_vote"))
    if lengths is not None:
        merged = merged.merge(lengths, on=JOIN_KEY, how="left")
    decisive = merged[
        merged["winner"].isin(["model_a", "model_b"]) & merged["chosen_model_name"].notna()
    ].copy()
    decisive["y_model_a"] = (decisive["winner"] == "model_a").astype(int)

    for feature in STYLE_FEATURES:
        decisive[f"delta_style_{feature}"] = decisive[f"{feature}_a"] - decisive[f"{feature}_b"]

    for feature in QUALITY_FEATURES:
        col_a = f"{feature}_a"
        col_b = f"{feature}_b"
        if col_a not in decisive.columns or col_b not in decisive.columns:
            continue
        a = _to_numeric(decisive[col_a])
        b = _to_numeric(decisive[col_b])
        delta = a - b
        if feature in NEGATIVE_QUALITY_FEATURES:
            delta = -delta
        decisive[f"delta_quality_{feature}"] = delta

    if {"assistant_chars_a", "assistant_chars_b"}.issubset(decisive.columns):
        decisive["delta_assistant_chars"] = (
            decisive["assistant_chars_a"] - decisive["assistant_chars_b"]
        )
        decisive["delta_log_assistant_chars"] = np.log1p(decisive["assistant_chars_a"]) - np.log1p(
            decisive["assistant_chars_b"]
        )

    quality_cols = [col for col in decisive.columns if col.startswith("delta_quality_")]
    style_cols = [col for col in decisive.columns if col.startswith("delta_style_")]
    decisive["delta_quality_total"] = decisive[quality_cols].sum(axis=1)
    decisive["delta_style_total"] = decisive[style_cols].sum(axis=1)
    decisive["month_dt"] = pd.to_datetime(decisive["month"], errors="coerce")
    return decisive


def _component_cols(df: pd.DataFrame, prefix: str, *, exclude_total: bool = True) -> list[str]:
    """Colonnes composantes pour un préfixe donné."""
    cols = [col for col in df.columns if col.startswith(prefix)]
    if exclude_total:
        cols = [col for col in cols if not col.endswith("_total")]
    return cols


def _fit_specs(df: pd.DataFrame, specs: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit une collection de modèles logit standardisés."""
    y = df["y_model_a"].to_numpy()
    summary_rows: list[dict] = []
    coef_rows: list[dict] = []

    for name, cols in specs.items():
        cols = [col for col in cols if col in df.columns]
        if not cols:
            continue
        x = df[cols].fillna(0.0).to_numpy()
        x_scaled = StandardScaler().fit_transform(x)
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        model.fit(x_scaled, y)
        pred = model.predict_proba(x_scaled)[:, 1]
        auc = roc_auc_score(y, pred)

        summary_rows.append({"spec": name, "auc": auc, "n_features": len(cols), "n_rows": len(df)})
        for col, coef in zip(cols, model.coef_[0], strict=True):
            coef_rows.append(
                {
                    "spec": name,
                    "feature": col,
                    "coef": coef,
                    "odds_pct_per_sd": (np.exp(coef) - 1) * 100,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(coef_rows)


def fit_logit_specs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit modèles logit avec ablations sans colinéarité.

    Les specs principales excluent les totaux pour éviter d'inclure à la fois
    composantes et sommes (`delta_quality_total`, `delta_style_total`).
    Les specs `*_totals_only` sont gardées comme robustness check compact.
    """
    quality_components = _component_cols(df, "delta_quality_")
    style_components = _component_cols(df, "delta_style_")

    specs = {
        "quality_components_only": quality_components,
        "style_components_only": style_components,
        "quality_plus_style_components": quality_components + style_components,
        "quality_total_only": ["delta_quality_total"],
        "style_total_only": ["delta_style_total"],
        "quality_plus_style_totals": ["delta_quality_total", "delta_style_total"],
    }

    if "month_idx" not in df.columns:
        months = {month: idx for idx, month in enumerate(sorted(df["month"].dropna().unique()))}
        df = df.copy()
        df["month_idx"] = df["month"].map(months).fillna(0)

    for feature in STYLE_FEATURES:
        col = f"delta_style_{feature}"
        if col in df.columns:
            df[f"{col}_x_month"] = df[col] * df["month_idx"]

    specs["quality_style_time_interactions"] = specs["quality_plus_style_components"] + [
        col for col in df.columns if col.endswith("_x_month")
    ]

    return _fit_specs(df, specs)


def fit_r5bis_specs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit R5bis : structure markdown vs longueur, après contrôle qualité."""
    if "delta_assistant_chars" not in df.columns:
        msg = "delta_assistant_chars manquant : passer lengths à build_style_quality_dataset"
        raise ValueError(msg)

    quality_components = _component_cols(df, "delta_quality_")
    style_components = _component_cols(df, "delta_style_")
    length_features = ["delta_log_assistant_chars"]

    specs = {
        "length_log_only": length_features,
        "quality_plus_length": quality_components + length_features,
        "length_plus_style": length_features + style_components,
        "quality_plus_style": quality_components + style_components,
        "quality_length_style": quality_components + length_features + style_components,
    }
    return _fit_specs(df, specs)

