"""R6 — Style premium par tier de modèle (endogénéité / gaming du format)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.endogeneity import (  # noqa: E402
    CORE_STYLE_FEATURES,
    assign_model_tiers,
    fit_tier_interaction_model,
    formatting_intensity_by_tier,
    quality_formatting_correlation,
    load_standard_ratings,
    prepare_battles_with_tiers,
    tier_style_coefficients,
)

BATTLES = ROOT / "data" / "interim" / "battles_with_dates.parquet"
RATINGS_JSON = ROOT / "style-control-analysis" / "clean_analysis_results.json"
TIER_COEFS_OUT = ROOT / "data" / "processed" / "endogeneity_tier_coefficients.parquet"
INTERACT_SUMMARY_OUT = ROOT / "data" / "processed" / "endogeneity_interaction_summary.parquet"
INTERACT_COEFS_OUT = ROOT / "data" / "processed" / "endogeneity_interaction_coefficients.parquet"
CORR_OUT = ROOT / "data" / "processed" / "endogeneity_quality_formatting_corr.parquet"
INTENSITY_OUT = ROOT / "data" / "processed" / "endogeneity_formatting_intensity.parquet"
FIGURE = ROOT / "paper" / "figures" / "R6_endogeneity_tiers.png"
TABLE = ROOT / "paper" / "tables" / "table_r6_endogeneity_tiers.md"


def plot_tier_coefficients(tier_df: pd.DataFrame, interact_summary: pd.DataFrame) -> None:
    """Forest plot : effet style par contexte de paire."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    sub = tier_df[tier_df["feature"].isin(CORE_STYLE_FEATURES)].copy()
    pivot = sub.pivot(index="pair_tier", columns="feature", values="odds_pct_per_sd")
    order = [
        "bottom-bottom",
        "middle-middle",
        "top-top",
        "bottom-middle",
        "bottom-top",
        "middle-top",
    ]
    pivot = pivot.reindex([row for row in order if row in pivot.index])
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effet style (% odds / SD)")
    ax.set_title("Coefficients BT par type de paire")
    ax.legend(title="feature", loc="lower right")

    ax = axes[1]
    implied = interact_summary[interact_summary["feature"].isin(CORE_STYLE_FEATURES)].copy()
    contexts = ["bottom-bottom", "middle-middle", "top-top", "cross-tier"]
    plot_data = implied[implied["pair_context"].isin(contexts)]
    for feat in CORE_STYLE_FEATURES:
        feat_rows = plot_data[plot_data["feature"] == feat].set_index("pair_context").reindex(contexts)
        ax.plot(
            contexts,
            feat_rows["odds_pct_per_sd"],
            marker="o",
            label=feat,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Effet total implicite (% odds / SD)")
    ax.set_title("Interaction tier × style (modèle complet)")
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def write_table(
    tier_df: pd.DataFrame,
    interact_summary: pd.DataFrame,
    intensity: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    """Table markdown pour slides."""
    core = tier_df[tier_df["feature"].isin(CORE_STYLE_FEATURES)].copy()
    lines = [
        "## Coefficients par paire (bold / lists / headers)",
        "",
        "| pair_tier | feature | odds_pct_per_sd | n_battles |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in core.sort_values(["pair_tier", "feature"]).iterrows():
        lines.append(
            f"| {row['pair_tier']} | {row['feature']} | {row['odds_pct_per_sd']:.1f} | {row['n_battles']} |"
        )
    lines.extend(["", "## Effets implicites (interaction)", ""])
    implied = interact_summary[interact_summary["feature"].isin(CORE_STYLE_FEATURES)]
    lines.extend(["| context | feature | odds_pct_per_sd |", "| --- | --- | --- |"])
    for _, row in implied.sort_values(["feature", "pair_context"]).iterrows():
        lines.append(f"| {row['pair_context']} | {row['feature']} | {row['odds_pct_per_sd']:.1f} |")

    if not intensity.empty:
        lines.extend(["", "## Intensité formatage moyenne par tier", ""])
        lines.extend(["| tier | feature | mean_count |", "| --- | --- | --- |"])
        for _, row in intensity.iterrows():
            lines.append(f"| {row['tier']} | {row['feature']} | {row['mean_count']:.2f} |")

    if not corr.empty:
        lines.extend(["", "## Corrélation rating BT vs formatage", ""])
        lines.extend(
            ["| feature | pearson_r | spearman_rho |", "| --- | --- | --- |"]
        )
        for _, row in corr.iterrows():
            lines.append(
                f"| {row['feature']} | {row['pearson_r']:.3f} | {row['spearman_rho']:.3f} |"
            )

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    core.to_csv(TABLE.with_suffix(".csv"), index=False)
    print(f"Table sauvegardée : {TABLE}")


def main() -> int:
    print("Chargement battles")
    battles = pd.read_parquet(BATTLES)
    battles = battles[battles["timestamp"].notna()].copy()
    battles = battles[battles["source"] == "vote"].copy()
    print(f"Battles votes datés : {len(battles):,}")

    ratings = load_standard_ratings(RATINGS_JSON)
    tiers = assign_model_tiers(ratings)
    tier_counts = pd.Series(tiers).value_counts()
    print("Tiers modèles :", tier_counts.to_dict())

    battles = prepare_battles_with_tiers(battles, tiers)
    print("\nDistribution paires :")
    print(battles["pair_tier"].value_counts().to_string())

    tier_df = tier_style_coefficients(battles)
    intensity = formatting_intensity_by_tier(battles)
    corr = quality_formatting_correlation(battles, ratings)
    interact_summary, interact_coefs = fit_tier_interaction_model(battles)

    for path, df in [
        (TIER_COEFS_OUT, tier_df),
        (INTERACT_SUMMARY_OUT, interact_summary),
        (INTERACT_COEFS_OUT, interact_coefs),
        (INTENSITY_OUT, intensity),
        (CORR_OUT, corr),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    print("\n=== Coefficients par paire (core features) ===")
    show = tier_df[tier_df["feature"].isin(CORE_STYLE_FEATURES)].sort_values(
        ["pair_tier", "feature"]
    )
    print(show.to_string(index=False))

    print("\n=== Effets implicites interaction ===")
    print(
        interact_summary[interact_summary["feature"].isin(CORE_STYLE_FEATURES)].to_string(
            index=False
        )
    )

    print("\n=== Intensité formatage par tier ===")
    print(intensity.to_string(index=False))

    print("\n=== Corrélation rating vs formatage ===")
    print(corr.to_string(index=False))

    plot_tier_coefficients(tier_df, interact_summary)
    write_table(tier_df, interact_summary, intensity, corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
