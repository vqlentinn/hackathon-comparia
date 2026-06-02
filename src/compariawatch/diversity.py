"""Mesures de diversité stylistique inter-modèles (module R1)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

STYLE_FEATURES = ["headers", "lists", "bold", "code_blocks", "emoji"]
MIN_MODELS_PER_COHORT = 5


def battles_to_long(
    battles: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Passe en format long : 1 ligne = 1 réponse (modèle × mois × features)."""
    feats = features or STYLE_FEATURES
    cols_a = [f"{f}_a" for f in feats]
    cols_b = [f"{f}_b" for f in feats]

    long_a = battles[["model_a_name", "month"] + cols_a].copy()
    long_a.columns = ["model", "month"] + feats

    long_b = battles[["model_b_name", "month"] + cols_b].copy()
    long_b.columns = long_a.columns

    return pd.concat([long_a, long_b], ignore_index=True)


def standardize_features(
    long_df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score global sur les features stylistiques."""
    feats = features or STYLE_FEATURES
    out = long_df.copy()
    scaler = StandardScaler()
    out[feats] = scaler.fit_transform(out[feats].fillna(0))
    return out, scaler


def mean_centroid_diversity(centroids: np.ndarray) -> float | None:
    """Diversité = distance euclidienne moyenne entre centroïdes de modèles."""
    if len(centroids) < MIN_MODELS_PER_COHORT:
        return None
    return float(np.mean(pdist(centroids)))


def wasserstein_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Wasserstein-2 empirique 1D par dimension, moyennée (fallback sans POT)."""
    from scipy.stats import wasserstein_distance

    dists = [wasserstein_distance(x[:, j], y[:, j]) for j in range(x.shape[1])]
    return float(np.mean(dists))


def wasserstein_diversity(distributions: dict[str, np.ndarray]) -> float | None:
    """W2 moyenne sur toutes les paires de modèles (distribution empirique)."""
    models = list(distributions.keys())
    if len(models) < MIN_MODELS_PER_COHORT:
        return None

    try:
        import ot  # noqa: PLC0415

        pairs = list(itertools.combinations(models, 2))
        dists = []
        for m1, m2 in pairs:
            a, b = distributions[m1], distributions[m2]
            n_a, n_b = len(a), len(b)
            a_w = np.ones(n_a) / n_a
            b_w = np.ones(n_b) / n_b
            m_cost = ot.dist(a, b)
            dists.append(ot.emd2(a_w, b_w, m_cost))
        return float(np.mean(dists))
    except ImportError:
        pairs = list(itertools.combinations(models, 2))
        dists = [wasserstein_pair(distributions[m1], distributions[m2]) for m1, m2 in pairs]
        return float(np.mean(dists))


def compute_monthly_diversity(
    battles: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "centroid",
) -> pd.DataFrame:
    """Calcule D_t par mois : diversité stylistique inter-modèles.

    Parameters
    ----------
    battles
        DataFrame avec ``model_a_name``, ``model_b_name``, ``month``, features.
    method
        ``centroid`` (rapide, défaut) ou ``wasserstein`` (POT si installé).
    """
    feats = features or STYLE_FEATURES
    sub = battles.dropna(subset=["month"]).copy()

    long_df, _ = standardize_features(battles_to_long(sub, feats), feats)
    centroids = long_df.groupby(["model", "month"])[feats].mean().reset_index()

    rows: list[dict] = []
    for month, grp in centroids.groupby("month"):
        if method == "wasserstein":
            dists: dict[str, np.ndarray] = {}
            for model in grp["model"].unique():
                mask = (long_df["month"] == month) & (long_df["model"] == model)
                dists[model] = long_df.loc[mask, feats].values
            div = wasserstein_diversity(dists)
        else:
            div = mean_centroid_diversity(grp[feats].values)

        if div is not None:
            rows.append(
                {
                    "month": month,
                    "diversity": div,
                    "n_models": grp["model"].nunique(),
                    "n_responses": len(long_df[long_df["month"] == month]),
                }
            )

    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    out["month_idx"] = range(len(out))
    return out
