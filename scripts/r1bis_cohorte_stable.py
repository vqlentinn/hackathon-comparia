"""R1-bis — test de cohorte stable contre artefact de composition.

Réplique la mesure de dispersion stylistique inter-modèles de R1 sur une
COHORTE FIXE de modèles présents sur toute la période (≥100 battles totales,
≥12 mois distincts ; fallback à 10 mois si <6 modèles). Compare la courbe
globale et la courbe cohorte avec la même métrique (centroid sur features
densifiées), pour distinguer une vraie divergence d'un simple artefact
de composition de l'arène.

Pipeline
--------
1. battles_with_dates.parquet, filtre source=='vote' (handoff R1-bis), exclut
   ties (cohérence r1_robustness), drop NaT (log %), exclut mois en cours.
2. Densifie via votes_length (`feature / assistant_chars` par côté).
3. select_stable_cohort + log liste / nb mois actifs / battles totales.
4. compute_dispersion_cohorte → série mensuelle global + cohorte.
5. OLS sur les deux séries (seed=42, OLS déterministe).
6. Figure paper/figures/R1bis_cohorte_stable.png + CSV
   data/processed/dispersion_cohorte.csv + résumé console.

Pas d'interprétation Goodhart — livrables uniquement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch import RANDOM_STATE  # noqa: E402
from compariawatch.diversity import (  # noqa: E402
    STYLE_FEATURES,
    compute_dispersion_cohorte,
    densify_features,
    select_stable_cohort,
)

INPUT_BATTLES = ROOT / "data" / "interim" / "battles_with_dates.parquet"
INPUT_LENGTHS = ROOT / "data" / "raw" / "votes_length.parquet"
OUTPUT_CSV = ROOT / "data" / "processed" / "dispersion_cohorte.csv"
OUTPUT_FIG = ROOT / "paper" / "figures" / "R1bis_cohorte_stable.png"

NAT_HARD_LIMIT = 0.15
MIN_BATTLES_TOTAL = 100
MIN_MONTHS = 12
FALLBACK_MIN_MONTHS = 10
MIN_BATTLES_PER_MONTH = 100
MIN_MODELS = 5


def exclude_incomplete_current_month(df: pd.DataFrame) -> pd.DataFrame:
    """Exclut le mois calendaire courant (toujours incomplet par définition)."""
    current = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    n_before = len(df)
    out = df[df["month"] != current]
    n_drop = n_before - len(out)
    if n_drop:
        print(f"[scope] exclu mois courant {current!r} : -{n_drop:,} lignes")
    return out


def fit_ols(df: pd.DataFrame, y: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    x = sm.add_constant(df[["month_idx"]])
    return sm.OLS(df[y], x).fit()


def _ols_line(name: str, res: sm.regression.linear_model.RegressionResultsWrapper) -> str:
    beta = res.params["month_idx"]
    ci_low, ci_high = res.conf_int(alpha=0.05).loc["month_idx"]
    p = res.pvalues["month_idx"]
    return (
        f"  {name:<20s} pente={beta:+.5f}  "
        f"IC95=[{ci_low:+.5f}, {ci_high:+.5f}]  "
        f"p={p:.4g}  R²={res.rsquared:.3f}"
    )


def plot_results(
    monthly: pd.DataFrame,
    cohort_size: int,
    res_global: sm.regression.linear_model.RegressionResultsWrapper,
    res_cohort: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    _fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        monthly["month"],
        monthly["dispersion_globale"],
        "o-",
        color="steelblue",
        linewidth=1.8,
        label="Global (tous modèles ≥100 battles/mois)",
    )
    ax.plot(
        monthly["month"],
        monthly["dispersion_cohorte"],
        "o-",
        color="firebrick",
        linewidth=1.8,
        label=f"Cohorte stable ({cohort_size} modèles ≥12 mois, ≥100 battles)",
    )
    for _, row in monthly.iterrows():
        ax.annotate(
            f"{int(row['n_models_cohorte'])}",
            xy=(row["month"], row["dispersion_cohorte"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="firebrick",
        )

    beta_g = res_global.params["month_idx"]
    p_g = res_global.pvalues["month_idx"]
    beta_c = res_cohort.params["month_idx"]
    p_c = res_cohort.pvalues["month_idx"]
    ax.set_title(
        "R1-bis — dispersion stylistique mensuelle, cohorte stable vs global\n"
        f"global : β={beta_g:+.4f} (p={p_g:.3g})   ·   "
        f"cohorte : β={beta_c:+.4f} (p={p_c:.3g})",
        fontsize=12,
    )
    ax.set_xlabel("Mois")
    ax.set_ylabel("Dispersion (distance euclidienne moyenne entre centroïdes, features densifiées z-score)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    print(f"[out] figure : {OUTPUT_FIG}")


def main() -> int:
    np.random.seed(RANDOM_STATE)

    battles = pd.read_parquet(INPUT_BATTLES)
    print(f"[load] battles_with_dates : {len(battles):,} lignes")

    votes = battles[battles["source"] == "vote"].copy()
    print(
        f"[scope] source=='vote' : {len(votes):,}/{len(battles):,} "
        f"({len(votes) / len(battles) * 100:.1f}%)"
    )

    voted = votes[votes["winner"].isin(["model_a", "model_b"])].copy()
    print(f"[scope] exclus ties : -{len(votes) - len(voted):,} (reste {len(voted):,})")

    nat_rate = float(voted["timestamp"].isna().mean())
    voted = voted.dropna(subset=["timestamp"])
    print(f"[scope] drop NaT timestamps : {nat_rate * 100:.2f}%")
    if nat_rate > NAT_HARD_LIMIT:
        print(f"[STOP] NaT > {NAT_HARD_LIMIT * 100:.0f}% — jointure suspecte.")
        return 1

    voted = exclude_incomplete_current_month(voted)

    lengths = pd.read_parquet(INPUT_LENGTHS)
    print(f"[load] votes_length : {len(lengths):,} lignes")
    voted = densify_features(voted, lengths, features=STYLE_FEATURES)
    print(f"[densify] battles après densification : {len(voted):,}")

    cohort, cohort_stats = select_stable_cohort(
        voted,
        min_battles_total=MIN_BATTLES_TOTAL,
        min_months=MIN_MONTHS,
        fallback_min_months=FALLBACK_MIN_MONTHS,
    )
    print(f"\n=== COHORTE STABLE ({len(cohort)} modèles) ===")
    print(cohort_stats[["model", "battles", "months", "min_months_used"]].to_string(index=False))

    monthly, skipped = compute_dispersion_cohorte(
        voted,
        cohort=cohort,
        features=STYLE_FEATURES,
        min_battles_per_month=MIN_BATTLES_PER_MONTH,
        min_models=MIN_MODELS,
    )
    if skipped:
        print("\n[skip] mois écartés (n_models < min) :")
        for s in skipped:
            print(f"  {s['month']}  n_global={s['n_models_global']}  n_cohorte={s['n_models_cohorte']}")
    print(f"\n=== DISPERSION MENSUELLE ({len(monthly)} mois retenus) ===")
    cols = ["month", "dispersion_globale", "dispersion_cohorte", "n_models_global", "n_models_cohorte"]
    print(monthly[cols].to_string(index=False))

    res_global = fit_ols(monthly, "dispersion_globale")
    res_cohort = fit_ols(monthly, "dispersion_cohorte")
    print("\n=== OLS dispersion ~ month_idx (seed=42, OLS déterministe) ===")
    print(_ols_line("dispersion_globale", res_global))
    print(_ols_line("dispersion_cohorte", res_cohort))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    monthly[["month", "dispersion_globale", "dispersion_cohorte", "n_models_cohorte"]].to_csv(
        OUTPUT_CSV, index=False
    )
    print(f"\n[out] CSV : {OUTPUT_CSV}")
    plot_results(monthly, len(cohort), res_global, res_cohort)

    print("\n=== RÉSUMÉ ===")
    print(f"Cohorte ({len(cohort)} modèles, seuil utilisé ≥{cohort_stats['min_months_used'].iloc[0]} mois) :")
    print("  " + ", ".join(cohort))
    print(_ols_line("global", res_global))
    print(_ols_line("cohorte", res_cohort))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
