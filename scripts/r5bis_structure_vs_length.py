"""R5bis — Structure markdown vs longueur après contrôle qualité."""

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
    fit_r5bis_specs,
    load_votes_length,
    load_votes_quality,
)

BATTLES = ROOT / "data" / "interim" / "battles_with_dates.parquet"
QUALITY_CACHE = ROOT / "data" / "raw" / "votes_quality.parquet"
LENGTH_CACHE = ROOT / "data" / "raw" / "votes_length.parquet"
LOCAL_VOTES = ROOT / "data" / "raw" / "hf" / "comparia-votes" / "votes.parquet"
DATASET_OUT = ROOT / "data" / "processed" / "style_length_quality_dataset.parquet"
SUMMARY_OUT = ROOT / "data" / "processed" / "style_length_quality_summary.parquet"
COEFS_OUT = ROOT / "data" / "processed" / "style_length_quality_coefficients.parquet"
FIGURE = ROOT / "paper" / "figures" / "R5bis_structure_vs_length.png"
TABLE = ROOT / "paper" / "tables" / "table_r5bis_structure_vs_length.md"

SPEC_LABELS = {
    "length_log_only": "Longueur (log) seule",
    "quality_plus_length": "Qualité + longueur",
    "length_plus_style": "Longueur + style structurel",
    "quality_plus_style": "Qualité + style (R5)",
    "quality_length_style": "Qualité + longueur + style (R5bis)",
}


def plot_results(summary: pd.DataFrame, coefs: pd.DataFrame) -> None:
    """Figure R5bis : AUC par spec + effets structurels après contrôle complet."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    plot_df = summary.copy()
    plot_df["label"] = plot_df["spec"].map(SPEC_LABELS).fillna(plot_df["spec"])
    plot_df = plot_df.sort_values("auc")
    colors = [
        "darkorange" if spec == "quality_length_style" else "steelblue"
        for spec in plot_df["spec"]
    ]
    ax.barh(plot_df["label"], plot_df["auc"], color=colors)
    ax.set_xlim(0.45, min(0.95, plot_df["auc"].max() + 0.03))
    ax.set_xlabel("AUC in-sample")
    ax.set_title("Structure vs longueur : ablations")

    ax = axes[1]
    main = coefs[coefs["spec"] == "quality_length_style"].copy()
    style = main[main["feature"].str.startswith("delta_style_")].copy()
    style["feature_clean"] = style["feature"].str.replace("delta_style_", "", regex=False)
    style = style.sort_values("odds_pct_per_sd")
    length_row = main[main["feature"] == "delta_log_assistant_chars"]
    if not length_row.empty:
        length_effect = float(length_row.iloc[0]["odds_pct_per_sd"])
        ax.axvline(length_effect, color="navy", linestyle=":", linewidth=1.5, label="longueur")
        ax.legend(loc="lower right")
    colors = ["crimson" if value < 0 else "seagreen" for value in style["odds_pct_per_sd"]]
    ax.barh(style["feature_clean"], style["odds_pct_per_sd"], color=colors)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Effet (% odds / SD)")
    ax.set_title("Style structurel après qualité + longueur")

    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def write_table(summary: pd.DataFrame, coefs: pd.DataFrame) -> None:
    """Table markdown pour slides / papier."""
    summary_out = summary.copy()
    summary_out["model"] = summary_out["spec"].map(SPEC_LABELS).fillna(summary_out["spec"])
    summary_out = summary_out[["model", "auc", "n_features", "n_rows"]].sort_values("auc")
    summary_out.to_csv(TABLE.with_suffix(".csv"), index=False)

    style_main = coefs[
        (coefs["spec"] == "quality_length_style")
        & coefs["feature"].str.startswith("delta_style_")
    ].copy()
    style_main["feature"] = style_main["feature"].str.replace("delta_style_", "", regex=False)

    lines = ["| model | auc | n_features | n_rows |", "| --- | --- | --- | --- |"]
    for _, row in summary_out.iterrows():
        lines.append(
            f"| {row['model']} | {row['auc']:.4f} | {row['n_features']} | {row['n_rows']} |"
        )
    lines.extend(["", "## Effets structurels (modèle complet)", ""])
    length_main = coefs[
        (coefs["spec"] == "quality_length_style")
        & (coefs["feature"] == "delta_log_assistant_chars")
    ]
    if not length_main.empty:
        val = length_main.iloc[0]["odds_pct_per_sd"]
        lines.append(f"- longueur log : **{val:.1f}%** odds / SD")
    lines.extend(["", "| feature | odds_pct_per_sd |", "| --- | --- |"])
    for _, row in style_main.sort_values("odds_pct_per_sd", ascending=False).iterrows():
        lines.append(f"| {row['feature']} | {row['odds_pct_per_sd']:.2f} |")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Table sauvegardée : {TABLE}")


def main() -> int:
    print("Chargement battles")
    battles = pd.read_parquet(BATTLES)
    battles = battles[battles["timestamp"].notna()].copy()

    local_votes = LOCAL_VOTES if LOCAL_VOTES.exists() else None
    votes = load_votes_quality(QUALITY_CACHE, local_votes_path=local_votes)
    lengths = load_votes_length(LENGTH_CACHE, local_votes_path=local_votes)
    dataset = build_style_quality_dataset(battles, votes, lengths=lengths)

    missing_length = dataset["delta_assistant_chars"].isna().mean()
    print(f"Dataset R5bis : {dataset.shape}, longueur manquante = {missing_length:.1%}")

    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_OUT, index=False)
    print(f"Sauvegardé : {DATASET_OUT}")

    r5_summary, _ = fit_logit_specs(dataset)
    summary, coefs = fit_r5bis_specs(dataset)
    summary = pd.concat([r5_summary, summary], ignore_index=True).drop_duplicates(
        subset=["spec"],
        keep="last",
    )
    summary.to_parquet(SUMMARY_OUT, index=False)
    coefs.to_parquet(COEFS_OUT, index=False)

    print("\n=== AUC R5bis ===")
    show = summary.copy()
    show["label"] = show["spec"].map(SPEC_LABELS).fillna(show["spec"])
    print(show.sort_values("auc")[["label", "auc", "n_features"]].to_string(index=False))

    print("\n=== Modèle principal : qualité + longueur + style ===")
    main = coefs[coefs["spec"] == "quality_length_style"].sort_values(
        "odds_pct_per_sd",
        ascending=False,
    )
    print(main.to_string(index=False))

    auc_r5 = float(
        summary.loc[summary["spec"] == "quality_plus_style", "auc"].iloc[0]
    )
    auc_r5bis = float(summary.loc[summary["spec"] == "quality_length_style", "auc"].iloc[0])
    print(f"\nΔAUC longueur contrôlée : {auc_r5:.4f} -> {auc_r5bis:.4f} ({auc_r5bis - auc_r5:+.4f})")

    plot_results(summary, coefs)
    write_table(summary, coefs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
