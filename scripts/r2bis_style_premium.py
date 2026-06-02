"""R2bis — Style Premium longitudinal.

Recalcule par mois les coefficients Bradley-Terry style-controlled de
Zilinskas et teste si `headers`, `lists`, `bold` augmentent dans le temps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.style_premium import (  # noqa: E402
    CORE_STYLE_FEATURES,
    STYLE_FEATURES,
    compute_monthly_style_premium,
)

INPUT = ROOT / "data" / "interim" / "battles_with_dates.parquet"
OUTPUT = ROOT / "data" / "processed" / "style_premium_temporal.parquet"
TRENDS_OUTPUT = ROOT / "data" / "processed" / "style_premium_trends.parquet"
FIGURE = ROOT / "paper" / "figures" / "R2bis_style_premium_longitudinal.png"

N_BOOT = 40


def fit_trends(style_df: pd.DataFrame) -> pd.DataFrame:
    """Fit `odds_pct ~ month_idx` pour chaque feature."""
    rows: list[dict[str, float | str | int]] = []
    for feature, sub in style_df.groupby("feature"):
        sub = sub.sort_values("month_idx")
        x = sm.add_constant(sub["month_idx"])
        y = sub["odds_pct"]
        model = sm.OLS(y, x).fit()
        rows.append(
            {
                "feature": feature,
                "beta_odds_pct_per_month": model.params["month_idx"],
                "p_value": model.pvalues["month_idx"],
                "r2": model.rsquared,
                "n_months": len(sub),
                "mean_odds_pct": sub["odds_pct"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def plot_style_premium(style_df: pd.DataFrame, trends: pd.DataFrame) -> None:
    """Figure longitudinale Style Premium."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)

    for ax, feature in zip(axes, CORE_STYLE_FEATURES, strict=True):
        sub = style_df[style_df["feature"] == feature].sort_values("month_idx")
        trend = trends[trends["feature"] == feature].iloc[0]

        ax.plot(sub["month"], sub["odds_pct"], "o-", color="steelblue", lw=2)
        ax.fill_between(
            sub["month"],
            sub["odds_low"],
            sub["odds_high"],
            color="steelblue",
            alpha=0.18,
            label="IC bootstrap 95 %",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"{feature}\n"
            f"β={trend['beta_odds_pct_per_month']:+.2f} pts/mois, "
            f"p={trend['p_value']:.3f}"
        )
        ax.set_xlabel("Mois")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Effet sur odds de victoire (% / SD)")

    fig.suptitle("R2bis — Évolution mensuelle du Style Premium", fontweight="bold")
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def main() -> int:
    battles = pd.read_parquet(INPUT)
    battles = battles[battles["timestamp"].notna()].copy()
    battles = battles[battles["winner"].isin(["model_a", "model_b"])].copy()

    print(f"Battles R2bis : {len(battles):,}")
    print(f"Mois disponibles : {battles['month'].nunique()}")
    print(f"Bootstrap par cohorte : n_boot={N_BOOT}")

    style_df = compute_monthly_style_premium(
        battles,
        n_boot=N_BOOT,
        style_features=STYLE_FEATURES,
    )
    trends = fit_trends(style_df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style_df.to_parquet(OUTPUT, index=False)
    trends.to_parquet(TRENDS_OUTPUT, index=False)

    print(f"\nRésultats sauvegardés : {OUTPUT}")
    print(f"Tendances sauvegardées : {TRENDS_OUTPUT}")
    print("\n=== Tendances Style Premium ===")
    print(trends.to_string(index=False))

    plot_style_premium(style_df, trends)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

