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


def densify_features(
    battles: pd.DataFrame,
    lengths: pd.DataFrame,
    features: list[str] | None = None,
    *,
    drop_zero_length: bool = True,
    min_merge_rate: float = 0.85,
) -> pd.DataFrame:
    """Convertit les comptes bruts de features en densité par caractère.

    Merge ``battles`` ⨝ ``lengths`` sur ``conversation_pair_id`` puis divise
    ``{feature}_a`` par ``assistant_chars_a`` (idem côté b). Log le taux de
    jointure ; raise si < ``min_merge_rate``. Si ``drop_zero_length``, retire
    les lignes avec ``assistant_chars_{a,b}`` == 0 ou NaN.
    """
    feats = features or STYLE_FEATURES
    join_key = "conversation_pair_id"
    n_before = len(battles)
    cols = [join_key, "assistant_chars_a", "assistant_chars_b"]
    merged = battles.merge(lengths[cols], on=join_key, how="left", validate="m:1")
    n_matched = merged["assistant_chars_a"].notna().sum()
    rate = n_matched / n_before if n_before else 0.0
    print(
        f"[densify] merge votes_length : {n_matched:,}/{n_before:,} "
        f"({rate * 100:.2f}%)"
    )
    if rate < min_merge_rate:
        msg = (
            f"taux de jointure votes_length < {min_merge_rate * 100:.0f}% "
            f"(observé {rate * 100:.2f}%)"
        )
        raise ValueError(msg)

    if drop_zero_length:
        bad_a = (merged["assistant_chars_a"] == 0) | merged["assistant_chars_a"].isna()
        bad_b = (merged["assistant_chars_b"] == 0) | merged["assistant_chars_b"].isna()
        n_drop = int((bad_a | bad_b).sum())
        if n_drop:
            print(
                f"[densify] drop {n_drop:,} lignes avec assistant_chars==0/NaN "
                f"({n_drop / n_before * 100:.2f}%)"
            )
        merged = merged[~(bad_a | bad_b)].copy()

    for f in feats:
        merged[f"{f}_a"] = merged[f"{f}_a"] / merged["assistant_chars_a"]
        merged[f"{f}_b"] = merged[f"{f}_b"] / merged["assistant_chars_b"]
    return merged


def select_stable_cohort(
    battles: pd.DataFrame,
    min_battles_total: int = 100,
    min_months: int = 12,
    fallback_min_months: int = 10,
    min_cohort_size: int = 6,
) -> tuple[list[str], pd.DataFrame]:
    """Sélectionne la cohorte de modèles stables.

    Un modèle est retenu s'il apparaît (côté A ou B) dans ≥ ``min_battles_total``
    battles sur ≥ ``min_months`` mois distincts. Si la cohorte fait
    moins de ``min_cohort_size`` modèles, bascule sur ``fallback_min_months``.

    Returns
    -------
    cohort : list[str]
        Liste triée des modèles retenus.
    stats : pd.DataFrame
        Colonnes ``model | battles | months | min_months_used``.
    """
    long_a = battles[["model_a_name", "month"]].rename(columns={"model_a_name": "model"})
    long_b = battles[["model_b_name", "month"]].rename(columns={"model_b_name": "model"})
    long = pd.concat([long_a, long_b], ignore_index=True).dropna(subset=["month"])

    stats = (
        long.groupby("model")
        .agg(battles=("month", "size"), months=("month", "nunique"))
        .reset_index()
    )

    def _filter(min_m: int) -> list[str]:
        mask = (stats["battles"] >= min_battles_total) & (stats["months"] >= min_m)
        return stats.loc[mask, "model"].tolist()

    cohort = _filter(min_months)
    used = min_months
    if len(cohort) < min_cohort_size:
        print(
            f"[cohort] {len(cohort)} modèles avec ≥{min_months} mois (<{min_cohort_size}), "
            f"bascule à ≥{fallback_min_months} mois"
        )
        cohort = _filter(fallback_min_months)
        used = fallback_min_months

    cohort_stats = (
        stats[stats["model"].isin(cohort)]
        .sort_values("battles", ascending=False)
        .reset_index(drop=True)
    )
    cohort_stats["min_months_used"] = used
    return sorted(cohort), cohort_stats


def compute_dispersion_cohorte(
    battles: pd.DataFrame,
    cohort: list[str] | set[str],
    features: list[str] | None = None,
    min_battles_per_month: int = 100,
    min_models: int = MIN_MODELS_PER_COHORT,
) -> tuple[pd.DataFrame, list[dict]]:
    """Dispersion mensuelle globale + cohorte (centroid, z-score global).

    Pour chaque mois :
    * dispersion_globale = pdist moyen sur les modèles avec ≥ ``min_battles_per_month`` réponses
    * dispersion_cohorte = pdist moyen sur le sous-ensemble appartenant à ``cohort``
    * n_models_global / n_models_cohorte = effectifs correspondants

    Un mois est sauté (skipped) si l'une des deux populations a < ``min_models``.
    Métrique identique à ``compute_monthly_diversity(method='centroid')``.
    """
    feats = features or STYLE_FEATURES
    cohort_set = set(cohort)
    sub = battles.dropna(subset=["month"]).copy()

    long_df, _ = standardize_features(battles_to_long(sub, feats), feats)
    grouped = long_df.groupby(["model", "month"])
    centroids = grouped[feats].mean()
    counts = grouped.size().rename("n_responses")
    perfile = centroids.join(counts).reset_index()

    rows: list[dict] = []
    skipped: list[dict] = []
    for month, grp in perfile.groupby("month"):
        active = grp[grp["n_responses"] >= min_battles_per_month]
        cohort_active = active[active["model"].isin(cohort_set)]
        n_global = len(active)
        n_cohort = len(cohort_active)

        if n_global < min_models or n_cohort < min_models:
            skipped.append(
                {"month": month, "n_models_global": n_global, "n_models_cohorte": n_cohort}
            )
            continue

        rows.append(
            {
                "month": month,
                "dispersion_globale": float(np.mean(pdist(active[feats].values))),
                "dispersion_cohorte": float(np.mean(pdist(cohort_active[feats].values))),
                "n_models_global": n_global,
                "n_models_cohorte": n_cohort,
            }
        )

    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    if len(out):
        out["month_idx"] = range(len(out))
    return out, skipped


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
