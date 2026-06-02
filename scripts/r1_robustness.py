"""Robustesses R1 : convergence stylistique brute vs contrôlée.

Produit :
- data/processed/diversity_temporal_robustness.parquet
- paper/figures/R1_convergence_robustness.png

Idée : R1 brute augmente. On teste si cette hausse est expliquée par
l'expansion de l'arène (nombre de modèles, volume) ou si une convergence
conditionnelle apparaît à composition plus stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.diversity import compute_monthly_diversity  # noqa: E402

INPUT = ROOT / "data" / "interim" / "battles_with_dates.parquet"
OUTPUT = ROOT / "data" / "processed" / "diversity_temporal_robustness.parquet"
FIGURE = ROOT / "paper" / "figures" / "R1_convergence_robustness.png"

MIN_MONTHS_RECURRENT = 6


def fit_ols(df: pd.DataFrame, formula_cols: list[str]) -> object:
    """Fit OLS ``diversity ~ formula_cols`` avec constante."""
    x = sm.add_constant(df[formula_cols])
    y = df["diversity"]
    return sm.OLS(y, x).fit()


def add_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une cohorte trimestrielle au format string."""
    out = df.copy()
    out["quarter"] = pd.to_datetime(out["date"]).dt.to_period("Q").astype(str)
    return out


def keep_recurrent_models(df: pd.DataFrame, min_months: int = MIN_MONTHS_RECURRENT) -> pd.DataFrame:
    """Filtre les battles aux modèles présents dans au moins ``min_months``."""
    long_a = df[["model_a_name", "month"]].rename(columns={"model_a_name": "model"})
    long_b = df[["model_b_name", "month"]].rename(columns={"model_b_name": "model"})
    long = pd.concat([long_a, long_b], ignore_index=True).drop_duplicates()
    model_months = long.groupby("model")["month"].nunique()
    recurrent = set(model_months[model_months >= min_months].index)

    out = df[df["model_a_name"].isin(recurrent) & df["model_b_name"].isin(recurrent)].copy()
    print(f"Modèles récurrents (>= {min_months} mois) : {len(recurrent)}")
    print(f"Battles après filtre récurrent : {len(out):,}/{len(df):,}")
    return out


def compute_quarterly_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Réutilise ``compute_monthly_diversity`` en remplaçant month par quarter."""
    qdf = add_quarter(df)
    qdf = qdf.drop(columns=["month"]).rename(columns={"quarter": "month"})
    out = compute_monthly_diversity(qdf, method="centroid")
    out = out.rename(columns={"month": "quarter"})
    out["quarter_idx"] = range(len(out))
    return out


def plot_results(
    monthly: pd.DataFrame,
    recurrent: pd.DataFrame,
    quarterly: pd.DataFrame,
    controlled_model: object,
) -> None:
    """Figure de synthèse robustesse R1."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(monthly["month"], monthly["diversity"], "o-", color="steelblue")
    ax.set_title("R1a — diversité brute mensuelle")
    ax.set_ylabel("Diversité")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    ax.scatter(monthly["n_models"], monthly["diversity"], color="darkorange")
    ax.set_title("Diversité vs nombre de modèles")
    ax.set_xlabel("n_models")
    ax.set_ylabel("Diversité")

    ax = axes[1, 0]
    ax.plot(recurrent["month"], recurrent["diversity"], "o-", color="seagreen")
    ax.set_title(f"R1b — modèles récurrents (>= {MIN_MONTHS_RECURRENT} mois)")
    ax.set_ylabel("Diversité")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 1]
    ax.plot(quarterly["quarter"], quarterly["diversity"], "o-", color="purple")
    ax.set_title("Robustesse — cohortes trimestrielles")
    ax.set_ylabel("Diversité")
    ax.tick_params(axis="x", rotation=45)

    beta = controlled_model.params.get("month_idx", float("nan"))
    pval = controlled_model.pvalues.get("month_idx", float("nan"))
    fig.suptitle(
        f"R1 robustesse : beta contrôlé month_idx={beta:+.4f} (p={pval:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def main() -> int:
    df = pd.read_parquet(INPUT)
    df = df[df["timestamp"].notna() & df["winner"].isin(["model_a", "model_b"])].copy()
    print(f"Battles R1 : {len(df):,}")

    monthly = compute_monthly_diversity(df, method="centroid")
    monthly["spec"] = "monthly_all"

    ols_raw = fit_ols(monthly, ["month_idx"])
    ols_ctrl = fit_ols(monthly, ["month_idx", "n_models", "n_responses"])

    corr_models = monthly[["diversity", "n_models"]].corr().iloc[0, 1]
    corr_responses = monthly[["diversity", "n_responses"]].corr().iloc[0, 1]

    recurrent_df = keep_recurrent_models(df)
    recurrent = compute_monthly_diversity(recurrent_df, method="centroid")
    recurrent["spec"] = "monthly_recurrent_models"
    ols_recurrent = fit_ols(recurrent, ["month_idx"])

    quarterly = compute_quarterly_diversity(df)
    quarterly["spec"] = "quarterly_all"
    quarterly_reg = quarterly.copy()
    quarterly_reg["month_idx"] = quarterly_reg["quarter_idx"]
    ols_quarterly = fit_ols(quarterly_reg, ["month_idx"])

    print("\n=== R1 brute ===")
    print(f"beta={ols_raw.params['month_idx']:+.4f}, p={ols_raw.pvalues['month_idx']:.4g}, R2={ols_raw.rsquared:.3f}")
    print("\n=== R1 contrôlée n_models + n_responses ===")
    print(ols_ctrl.summary())
    print("\n=== Corrélations ===")
    print(f"corr(diversity, n_models)={corr_models:+.3f}")
    print(f"corr(diversity, n_responses)={corr_responses:+.3f}")
    print("\n=== R1 modèles récurrents ===")
    print(
        f"beta={ols_recurrent.params['month_idx']:+.4f}, "
        f"p={ols_recurrent.pvalues['month_idx']:.4g}, R2={ols_recurrent.rsquared:.3f}"
    )
    print("\n=== R1 trimestrielle ===")
    print(
        f"beta={ols_quarterly.params['month_idx']:+.4f}, "
        f"p={ols_quarterly.pvalues['month_idx']:.4g}, R2={ols_quarterly.rsquared:.3f}"
    )

    summary_rows = [
        {
            "spec": "monthly_raw",
            "beta_month": ols_raw.params["month_idx"],
            "p_month": ols_raw.pvalues["month_idx"],
            "r2": ols_raw.rsquared,
            "n_periods": len(monthly),
            "corr_n_models": corr_models,
            "corr_n_responses": corr_responses,
        },
        {
            "spec": "monthly_controlled",
            "beta_month": ols_ctrl.params["month_idx"],
            "p_month": ols_ctrl.pvalues["month_idx"],
            "r2": ols_ctrl.rsquared,
            "n_periods": len(monthly),
            "corr_n_models": corr_models,
            "corr_n_responses": corr_responses,
        },
        {
            "spec": "monthly_recurrent_models",
            "beta_month": ols_recurrent.params["month_idx"],
            "p_month": ols_recurrent.pvalues["month_idx"],
            "r2": ols_recurrent.rsquared,
            "n_periods": len(recurrent),
            "corr_n_models": None,
            "corr_n_responses": None,
        },
        {
            "spec": "quarterly_raw",
            "beta_month": ols_quarterly.params["month_idx"],
            "p_month": ols_quarterly.pvalues["month_idx"],
            "r2": ols_quarterly.rsquared,
            "n_periods": len(quarterly),
            "corr_n_models": None,
            "corr_n_responses": None,
        },
    ]
    summary = pd.DataFrame(summary_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUTPUT, index=False)
    print(f"\nRésumé sauvegardé : {OUTPUT}")

    plot_results(monthly, recurrent, quarterly, ols_ctrl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
