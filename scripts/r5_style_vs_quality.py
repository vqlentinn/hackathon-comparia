"""R5 — Style vs qualité : contrôle des labels de votes Compar:IA."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.style_quality import (  # noqa: E402
    build_style_quality_dataset,
    fit_logit_specs,
    load_votes_quality,
)

BATTLES = ROOT / "data" / "interim" / "battles_with_dates.parquet"
QUALITY_CACHE = ROOT / "data" / "raw" / "votes_quality.parquet"
LOCAL_VOTES = ROOT / "data" / "raw" / "hf" / "comparia-votes" / "votes.parquet"
DATASET_OUT = ROOT / "data" / "processed" / "style_quality_dataset.parquet"
SUMMARY_OUT = ROOT / "data" / "processed" / "style_quality_model_summary.parquet"
COEFS_OUT = ROOT / "data" / "processed" / "style_quality_coefficients.parquet"
FIGURE = ROOT / "paper" / "figures" / "R5_style_vs_quality.png"


def plot_results(summary: pd.DataFrame, coefs: pd.DataFrame) -> None:
    """Figure R5 : AUC des specs + coefs style du modèle complet."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    summary_plot = summary.sort_values("auc")
    ax.barh(summary_plot["spec"], summary_plot["auc"], color="steelblue")
    ax.set_xlim(0.45, max(0.75, summary_plot["auc"].max() + 0.03))
    ax.set_xlabel("AUC in-sample")
    ax.set_title("Pouvoir prédictif : qualité vs style")

    ax = axes[1]
    full = coefs[coefs["spec"] == "quality_plus_style_components"].copy()
    style = full[full["feature"].str.startswith("delta_style_")].copy()
    style["feature_clean"] = style["feature"].str.replace("delta_style_", "", regex=False)
    style = style.sort_values("odds_pct_per_sd")
    colors = ["crimson" if value < 0 else "seagreen" for value in style["odds_pct_per_sd"]]
    ax.barh(style["feature_clean"], style["odds_pct_per_sd"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Effet style après contrôle qualité (% odds / SD)")
    ax.set_title("Style indépendant de la qualité")

    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def main() -> int:
    print("Chargement battles")
    battles = pd.read_parquet(BATTLES)
    battles = battles[battles["timestamp"].notna()].copy()

    local_votes = LOCAL_VOTES if LOCAL_VOTES.exists() else None
    votes = load_votes_quality(QUALITY_CACHE, local_votes_path=local_votes)
    dataset = build_style_quality_dataset(battles, votes)
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_OUT, index=False)
    print(f"Dataset R5 : {dataset.shape} -> {DATASET_OUT}")

    summary, coefs = fit_logit_specs(dataset)
    summary.to_parquet(SUMMARY_OUT, index=False)
    coefs.to_parquet(COEFS_OUT, index=False)

    print("\n=== AUC specs ===")
    print(summary.sort_values("auc").to_string(index=False))
    print("\n=== Coefficients style après contrôle qualité ===")
    style_full = coefs[
        (coefs["spec"] == "quality_plus_style_components")
        & (coefs["feature"].str.startswith("delta_style_"))
    ]
    print(style_full.sort_values("odds_pct_per_sd").to_string(index=False))

    print("\n=== Interactions style × mois ===")
    interactions = coefs[
        (coefs["spec"] == "quality_style_time_interactions")
        & (coefs["feature"].str.endswith("_x_month"))
    ]
    print(interactions.sort_values("odds_pct_per_sd").to_string(index=False))

    plot_results(summary, coefs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

