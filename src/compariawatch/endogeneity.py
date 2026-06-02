"""R6 — Endogénéité : le style premium varie-t-il selon le tier des modèles ?

Extension de l'analyse Zilinskas (`endogeneity_analysis.py`) :
- tiers top/middle/bottom par rating BT standard ;
- coefficients style par type de paire (within-tier, cross-tier) ;
- modèle d'interaction tier × style (référence = bottom-bottom).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from compariawatch.style_premium import STYLE_FEATURES, fit_bt_style_coefficients

CORE_STYLE_FEATURES = ["bold", "lists", "headers"]
TIER_ORDER = ["bottom", "middle", "top"]
TIER_RANK = {"bottom": 1, "middle": 2, "top": 3}
RANDOM_STATE = 42
MIN_BATTLES_TIER = 300
MIN_MODELS_TIER = 3


def load_standard_ratings(json_path: Path) -> dict[str, float]:
    """Charge les ratings BT standard depuis les résultats Zilinskas."""
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rankings = payload["rankings"]["standard"]
    return {model: float(info["rating"]) for model, info in rankings.items()}


def assign_model_tiers(ratings: dict[str, float]) -> dict[str, str]:
    """Découpe les modèles en tiers égaux par rating BT standard."""
    ordered = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
    n = len(ordered)
    tier_size = max(n // 3, 1)
    tiers: dict[str, str] = {}
    for idx, (model, _rating) in enumerate(ordered):
        if idx < tier_size:
            tiers[model] = "top"
        elif idx < 2 * tier_size:
            tiers[model] = "middle"
        else:
            tiers[model] = "bottom"
    return tiers


def _pair_tier_label(tier_a: str, tier_b: str) -> str:
    """Label canonique pour une paire de tiers (ex. bottom-top)."""
    ranks = sorted([TIER_RANK[tier_a], TIER_RANK[tier_b]])
    inv = {value: name for name, value in TIER_RANK.items()}
    return f"{inv[ranks[0]]}-{inv[ranks[1]]}"


def prepare_battles_with_tiers(
    battles: pd.DataFrame,
    tiers: dict[str, str],
) -> pd.DataFrame:
    """Ajoute tier_a, tier_b et pair_tier aux battles."""
    out = battles.copy()
    out["tier_a"] = out["model_a_name"].map(tiers)
    out["tier_b"] = out["model_b_name"].map(tiers)
    out = out.dropna(subset=["tier_a", "tier_b"])
    out["pair_tier"] = [
        _pair_tier_label(a, b) for a, b in zip(out["tier_a"], out["tier_b"], strict=True)
    ]
    return out


def tier_style_coefficients(battles: pd.DataFrame) -> pd.DataFrame:
    """Coefficients style par type de paire (within-tier + cross-tier)."""
    rows: list[dict] = []
    for pair_tier, subset in battles.groupby("pair_tier"):
        coefs = fit_bt_style_coefficients(
            subset,
            style_features=STYLE_FEATURES,
        )
        if coefs is None:
            continue
        n_decisive = int(subset["winner"].isin(["model_a", "model_b"]).sum())
        n_models = len(set(subset["model_a_name"]) | set(subset["model_b_name"]))
        for feature, coef in coefs.items():
            rows.append(
                {
                    "pair_tier": pair_tier,
                    "feature": feature,
                    "coef": coef,
                    "odds_pct_per_sd": (np.exp(coef) - 1) * 100,
                    "n_battles": n_decisive,
                    "n_models": n_models,
                }
            )
    return pd.DataFrame(rows)


def fit_tier_interaction_model(
    battles: pd.DataFrame,
    min_battles: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Modèle logit : indicateurs modèle + style + style×tier (top/mid vs bottom)."""
    decisive = battles[battles["winner"].isin(["model_a", "model_b"])].copy()
    if len(decisive) < min_battles:
        msg = f"Trop peu de battles pour interaction tier×style : {len(decisive)}"
        raise ValueError(msg)

    models = sorted(set(decisive["model_a_name"]) | set(decisive["model_b_name"]))
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    n_rows = len(decisive)
    n_models = len(models)

    model_a_idx = decisive["model_a_name"].map(model_to_idx).to_numpy()
    model_b_idx = decisive["model_b_name"].map(model_to_idx).to_numpy()
    y = (decisive["winner"] == "model_a").astype(int).to_numpy()

    x_model = np.zeros((n_rows, n_models))
    x_model[np.arange(n_rows), model_a_idx] = 1
    x_model[np.arange(n_rows), model_b_idx] = -1

    x_style = (
        decisive[[f"{feat}_a" for feat in STYLE_FEATURES]].to_numpy(dtype=float)
        - decisive[[f"{feat}_b" for feat in STYLE_FEATURES]].to_numpy(dtype=float)
    )
    x_style = StandardScaler().fit_transform(x_style)

    is_top = (
        (decisive["tier_a"] == "top") & (decisive["tier_b"] == "top")
    ).astype(float).to_numpy()
    is_mid = (
        (decisive["tier_a"] == "middle") & (decisive["tier_b"] == "middle")
    ).astype(float).to_numpy()

    core_idx = [STYLE_FEATURES.index(feat) for feat in CORE_STYLE_FEATURES]
    x_inter_top = x_style[:, core_idx] * is_top[:, np.newaxis]
    x_inter_mid = x_style[:, core_idx] * is_mid[:, np.newaxis]

    x = np.hstack([x_model, x_style, x_inter_top, x_inter_mid])
    lr = LogisticRegression(fit_intercept=False, max_iter=5000, penalty=None, random_state=RANDOM_STATE)
    lr.fit(x, y)

    style_main = lr.coef_[0][n_models : n_models + len(STYLE_FEATURES)]
    interact_top = lr.coef_[0][n_models + len(STYLE_FEATURES) : n_models + len(STYLE_FEATURES) + len(CORE_STYLE_FEATURES)]
    interact_mid = lr.coef_[0][n_models + len(STYLE_FEATURES) + len(CORE_STYLE_FEATURES) :]

    summary_rows: list[dict] = []
    coef_rows: list[dict] = []

    for idx, feat in enumerate(STYLE_FEATURES):
        coef_rows.append(
            {
                "term": feat,
                "kind": "main",
                "coef": style_main[idx],
                "odds_pct_per_sd": (np.exp(style_main[idx]) - 1) * 100,
            }
        )

    for idx, feat in enumerate(CORE_STYLE_FEATURES):
        feat_idx = STYLE_FEATURES.index(feat)
        base = style_main[feat_idx]
        top_total = base + interact_top[idx]
        mid_total = base + interact_mid[idx]
        coef_rows.append(
            {
                "term": f"{feat}×top_top",
                "kind": "interaction",
                "coef": interact_top[idx],
                "odds_pct_per_sd": (np.exp(interact_top[idx]) - 1) * 100,
            }
        )
        coef_rows.append(
            {
                "term": f"{feat}×mid_mid",
                "kind": "interaction",
                "coef": interact_mid[idx],
                "odds_pct_per_sd": (np.exp(interact_mid[idx]) - 1) * 100,
            }
        )
        for label, total in [
            ("bottom-bottom", base),
            ("middle-middle", mid_total),
            ("top-top", top_total),
            ("cross-tier", base),
        ]:
            summary_rows.append(
                {
                    "feature": feat,
                    "pair_context": label,
                    "coef": total,
                    "odds_pct_per_sd": (np.exp(total) - 1) * 100,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(coef_rows)


def quality_formatting_correlation(
    battles: pd.DataFrame,
    ratings: dict[str, float],
) -> pd.DataFrame:
    """Corrélation rating BT vs intensité moyenne de formatage par modèle."""
    from scipy.stats import pearsonr, spearmanr

    rows: list[dict] = []
    models = sorted(set(battles["model_a_name"]) | set(battles["model_b_name"]))
    means: dict[str, dict[str, float]] = {}
    for model in models:
        as_a = battles[battles["model_a_name"] == model]
        as_b = battles[battles["model_b_name"] == model]
        means[model] = {}
        for feat in CORE_STYLE_FEATURES:
            vals = pd.concat([as_a[f"{feat}_a"], as_b[f"{feat}_b"]], ignore_index=True)
            means[model][feat] = float(vals.mean())
        means[model]["composite"] = sum(means[model][f] for f in CORE_STYLE_FEATURES)

    eligible = [m for m in models if m in ratings]
    rating_arr = np.array([ratings[m] for m in eligible])
    for feat in [*CORE_STYLE_FEATURES, "composite"]:
        feat_arr = np.array([means[m][feat] for m in eligible])
        r_pearson, p_pearson = pearsonr(rating_arr, feat_arr)
        r_spearman, p_spearman = spearmanr(rating_arr, feat_arr)
        rows.append(
            {
                "feature": feat,
                "pearson_r": r_pearson,
                "pearson_p": p_pearson,
                "spearman_rho": r_spearman,
                "spearman_p": p_spearman,
                "n_models": len(eligible),
            }
        )
    return pd.DataFrame(rows)


def formatting_intensity_by_tier(battles: pd.DataFrame) -> pd.DataFrame:
    """Intensité moyenne du formatage par tier (confounder vs mediator)."""
    rows: list[dict] = []
    for side in ["a", "b"]:
        tier_col = f"tier_{side}"
        for tier in TIER_ORDER:
            mask = battles[tier_col] == tier
            if not mask.any():
                continue
            for feat in CORE_STYLE_FEATURES:
                values = battles.loc[mask, f"{feat}_{side}"]
                rows.append(
                    {
                        "tier": tier,
                        "feature": feat,
                        "mean_count": float(values.mean()),
                        "side": side,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby(["tier", "feature"], as_index=False)["mean_count"]
        .mean()
        .sort_values(["feature", "tier"])
    )
