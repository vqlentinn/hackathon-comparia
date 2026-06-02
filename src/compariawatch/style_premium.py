"""Style Premium longitudinal (module R2bis).

Recalcule les coefficients Bradley-Terry style-controlled par cohorte
temporelle pour tester si les features de style gagnantes (headers, lists,
bold) prennent plus d'importance dans le temps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STYLE_FEATURES = ["headers", "lists", "bold", "code_blocks", "emoji"]
CORE_STYLE_FEATURES = ["headers", "lists", "bold"]
MIN_BATTLES_PER_COHORT = 500
RANDOM_STATE = 42


def fit_bt_style_coefficients(
    battles: pd.DataFrame,
    style_features: list[str] | None = None,
) -> dict[str, float] | None:
    """Fit Bradley-Terry avec covariables de style et retourne les coefficients.

    Le modèle reproduit la logique de Zilinskas :
    ``X_model = +1/-1``, ``y=1`` si `model_a` gagne, et ``X_style`` correspond
    aux deltas `style_a - style_b`, standardisés dans la cohorte.
    """
    feats = style_features or STYLE_FEATURES
    decisive = battles[battles["winner"].isin(["model_a", "model_b"])].copy()
    if len(decisive) < MIN_BATTLES_PER_COHORT:
        return None
    if decisive["winner"].nunique() < 2:
        return None

    models = sorted(set(decisive["model_a_name"]) | set(decisive["model_b_name"]))
    if len(models) < 3:
        return None

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
        decisive[[f"{feat}_a" for feat in feats]].to_numpy(dtype=float)
        - decisive[[f"{feat}_b" for feat in feats]].to_numpy(dtype=float)
    )
    x_style = StandardScaler().fit_transform(x_style)

    x = np.hstack([x_model, x_style])
    lr = LogisticRegression(
        fit_intercept=False,
        max_iter=5000,
        penalty=None,
        random_state=RANDOM_STATE,
    )
    lr.fit(x, y)

    coefs = lr.coef_[0][n_models:]
    return dict(zip(feats, coefs, strict=True))


def bootstrap_style_coefficients(
    battles: pd.DataFrame,
    n_boot: int = 40,
    style_features: list[str] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap des coefficients de style pour une cohorte."""
    feats = style_features or STYLE_FEATURES
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []

    point = fit_bt_style_coefficients(battles, feats)
    if point is None:
        return pd.DataFrame()

    for feat, coef in point.items():
        rows.append({"feature": feat, "coef": coef, "kind": "point", "boot_id": -1})

    n_rows = len(battles)
    for boot_id in range(n_boot):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        sample = battles.iloc[sample_idx]
        boot = fit_bt_style_coefficients(sample, feats)
        if boot is None:
            continue
        for feat, coef in boot.items():
            rows.append({"feature": feat, "coef": coef, "kind": "bootstrap", "boot_id": boot_id})

    return pd.DataFrame(rows)


def compute_monthly_style_premium(
    battles: pd.DataFrame,
    n_boot: int = 40,
    style_features: list[str] | None = None,
    min_battles: int = MIN_BATTLES_PER_COHORT,
) -> pd.DataFrame:
    """Calcule le Style Premium mensuel avec IC bootstrap 95 %."""
    feats = style_features or STYLE_FEATURES
    sub = battles.dropna(subset=["month"]).copy()
    rows: list[dict[str, float | str | int]] = []

    for month, month_df in sub.groupby("month"):
        decisive = month_df[month_df["winner"].isin(["model_a", "model_b"])]
        if len(decisive) < min_battles:
            continue

        boot_df = bootstrap_style_coefficients(decisive, n_boot=n_boot, style_features=feats)
        if boot_df.empty:
            continue

        for feat in feats:
            point = boot_df[(boot_df["feature"] == feat) & (boot_df["kind"] == "point")]
            boots = boot_df[(boot_df["feature"] == feat) & (boot_df["kind"] == "bootstrap")]
            if point.empty or boots.empty:
                continue

            coef = float(point["coef"].iloc[0])
            ci_low = float(boots["coef"].quantile(0.025))
            ci_high = float(boots["coef"].quantile(0.975))
            rows.append(
                {
                    "month": month,
                    "feature": feat,
                    "coef": coef,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "odds_pct": (np.exp(coef) - 1) * 100,
                    "odds_low": (np.exp(ci_low) - 1) * 100,
                    "odds_high": (np.exp(ci_high) - 1) * 100,
                    "n_battles": len(decisive),
                    "n_models": len(set(decisive["model_a_name"]) | set(decisive["model_b_name"])),
                    "n_boot": int((boots["boot_id"] >= 0).sum()),
                }
            )

    out = pd.DataFrame(rows).sort_values(["month", "feature"]).reset_index(drop=True)
    month_order = {month: idx for idx, month in enumerate(sorted(out["month"].unique()))}
    out["month_idx"] = out["month"].map(month_order)
    return out

