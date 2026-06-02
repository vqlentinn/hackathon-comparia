"""Génère les tables finales R3/R5 pour le papier et les slides."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper" / "tables"

R3_ENSEMBLE = ROOT / "data" / "processed" / "causal_style_votes_n100_ensemble.parquet"
R3_MISTRAL = ROOT / "data" / "processed" / "causal_style_votes_n100.parquet"
R5_SUMMARY = ROOT / "data" / "processed" / "style_quality_model_summary.parquet"
R5_COEFS = ROOT / "data" / "processed" / "style_quality_coefficients.parquet"


def wilson_pct(wins: int, n: int) -> tuple[float, float]:
    """Intervalle Wilson 95 % en pourcentage."""
    low, high = proportion_confint(wins, n, alpha=0.05, method="wilson")
    return 100 * low, 100 * high


def r3_table() -> pd.DataFrame:
    """Table R3 principale : Mistral + Llama70B + agreement 2/2."""
    ensemble = pd.read_parquet(R3_ENSEMBLE)
    rows: list[dict] = []
    for style, sub in ensemble.groupby("style_target"):
        n = len(sub)
        mistral_wins = int((sub["mistral_vote"] == "A").sum())
        llama_valid = sub[sub["llama_vote"].isin(["A", "B"])]
        llama_n = len(llama_valid)
        llama_wins = int((llama_valid["llama_vote"] == "A").sum())
        agreed = sub[
            sub["mistral_vote"].isin(["A", "B"])
            & sub["llama_vote"].isin(["A", "B"])
            & (sub["mistral_vote"] == sub["llama_vote"])
        ]
        agreed_n = len(agreed)
        agreed_wins = int((agreed["mistral_vote"] == "A").sum())

        m_low, m_high = wilson_pct(mistral_wins, n)
        l_low, l_high = wilson_pct(llama_wins, llama_n)
        a_low, a_high = wilson_pct(agreed_wins, agreed_n)
        rows.append(
            {
                "comparison": f"{style} vs neutral",
                "n_all": n,
                "mistral_win_pct": 100 * mistral_wins / n,
                "mistral_ci": f"[{m_low:.1f}, {m_high:.1f}]",
                "n_llama70b_valid": llama_n,
                "llama70b_win_pct": 100 * llama_wins / llama_n,
                "llama70b_ci": f"[{l_low:.1f}, {l_high:.1f}]",
                "n_agreement": agreed_n,
                "agreement_win_pct": 100 * agreed_wins / agreed_n,
                "agreement_ci": f"[{a_low:.1f}, {a_high:.1f}]",
            }
        )
    return pd.DataFrame(rows)


def r5_auc_table() -> pd.DataFrame:
    """Table R5 AUC par spécification."""
    summary = pd.read_parquet(R5_SUMMARY).copy()
    label_map = {
        "style_total_only": "Style total only",
        "style_components_only": "Style components only",
        "quality_total_only": "Quality total only",
        "quality_components_only": "Quality components only",
        "quality_plus_style_totals": "Quality total + style total",
        "quality_plus_style_components": "Quality components + style components",
        "quality_style_time_interactions": "Quality + style + style x month",
    }
    summary["model"] = summary["spec"].map(label_map).fillna(summary["spec"])
    return summary[["model", "auc", "n_features", "n_rows"]].sort_values("auc")


def r5_style_coef_table() -> pd.DataFrame:
    """Table R5 coefficients style après contrôle qualité, sans totaux."""
    coefs = pd.read_parquet(R5_COEFS)
    sub = coefs[
        (coefs["spec"] == "quality_plus_style_components")
        & coefs["feature"].str.startswith("delta_style_")
    ].copy()
    sub["feature"] = sub["feature"].str.replace("delta_style_", "", regex=False)
    sub["odds_ratio"] = 1 + sub["odds_pct_per_sd"] / 100
    return sub[["feature", "coef", "odds_ratio", "odds_pct_per_sd"]].sort_values(
        "odds_pct_per_sd",
        ascending=False,
    )


def write_table(df: pd.DataFrame, stem: str) -> None:
    """Écrit une table en CSV, Markdown et LaTeX."""
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    markdown = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        markdown.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    (TABLES / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (TABLES / f"{stem}.tex").write_text(
        df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )


def main() -> int:
    tables = {
        "table_r3_counterfactual": r3_table(),
        "table_r5_auc": r5_auc_table(),
        "table_r5_style_coefficients": r5_style_coef_table(),
    }
    for stem, df in tables.items():
        write_table(df, stem)
        print(f"\n=== {stem} ===")
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

