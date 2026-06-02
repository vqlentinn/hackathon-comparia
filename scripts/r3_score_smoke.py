"""R3 smoke scoring — cosine + LLM-as-judge.

Entrée :
- data/interim/rewrites_smoke.parquet

Sorties :
- data/interim/rewrites_smoke_scored.parquet
- data/processed/causal_style_votes_smoke.parquet
- paper/figures/R3_style_premium_smoke.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.counterfactual import COSINE_THRESHOLD, get_mistral_client  # noqa: E402
from compariawatch.judge import judge_pair  # noqa: E402

INPUT = ROOT / "data" / "interim" / "rewrites_smoke.parquet"
SCORED_OUTPUT = ROOT / "data" / "interim" / "rewrites_smoke_scored.parquet"
VOTES_OUTPUT = ROOT / "data" / "processed" / "causal_style_votes_smoke.parquet"
FIGURE = ROOT / "paper" / "figures" / "R3_style_premium_smoke.png"

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COMPARE_STYLES = ("verbose_markdown", "concise_direct")
BASELINE_STYLE = "neutre_baseline"


def add_cosine_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute cosine(original, rewritten) à chaque ligne."""
    print(f"Chargement embedder : {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    originals = df["original"].tolist()
    rewritten = df["rewritten"].tolist()
    emb_original = model.encode(originals, convert_to_tensor=True, show_progress_bar=True)
    emb_rewritten = model.encode(rewritten, convert_to_tensor=True, show_progress_bar=True)
    cosine = util.cos_sim(emb_original, emb_rewritten).diagonal().cpu().numpy()

    out = df.copy()
    out["cosine"] = cosine
    out["preserved_cosine"] = out["cosine"] >= COSINE_THRESHOLD
    return out


def judge_smoke(scored: pd.DataFrame) -> pd.DataFrame:
    """Juge style cible vs baseline neutre pour chaque conversation."""
    client = get_mistral_client()
    rows: list[dict] = []

    grouped = scored.pivot_table(
        index=["conversation_pair_id", "opening_msg", "original_model"],
        columns="style_target",
        values=["rewritten", "cosine", "preserved_cosine"],
        aggfunc="first",
    )
    grouped.columns = [f"{a}__{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()

    for idx, row in grouped.iterrows():
        baseline = row.get(f"rewritten__{BASELINE_STYLE}")
        if not isinstance(baseline, str) or not baseline:
            continue

        for style in COMPARE_STYLES:
            candidate = row.get(f"rewritten__{style}")
            if not isinstance(candidate, str) or not candidate:
                continue

            result = judge_pair(
                prompt=str(row["opening_msg"]),
                response_a=candidate,
                response_b=baseline,
                client=client,
                seed=42 + idx,
            )
            winner = result["winner"]
            rows.append(
                {
                    "conversation_pair_id": row["conversation_pair_id"],
                    "original_model": row["original_model"],
                    "style_target": style,
                    "baseline_style": BASELINE_STYLE,
                    "winner": winner,
                    "style_wins": winner == "a",
                    "baseline_wins": winner == "b",
                    "tie": winner == "tie",
                    "style_cosine": row.get(f"cosine__{style}"),
                    "baseline_cosine": row.get(f"cosine__{BASELINE_STYLE}"),
                    "style_preserved": row.get(f"preserved_cosine__{style}"),
                    "baseline_preserved": row.get(f"preserved_cosine__{BASELINE_STYLE}"),
                    "reason": result["reason"],
                    "raw": result["raw"],
                }
            )
            print(f"[judge] {idx + 1}/{len(grouped)} {style} -> {winner}")

    return pd.DataFrame(rows)


def plot_votes(votes: pd.DataFrame) -> None:
    """Figure smoke : part de victoires du style vs neutre."""
    summary = (
        votes.groupby("style_target")
        .agg(style_win_rate=("style_wins", "mean"), n=("style_wins", "size"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["style_target"], summary["style_win_rate"], color=["crimson", "steelblue"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(style cible > neutre)")
    ax.set_title("R3 smoke — préférence LLM judge vs neutre")
    for i, row in summary.iterrows():
        ax.text(i, row["style_win_rate"] + 0.03, f"n={int(row['n'])}", ha="center")
    plt.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {FIGURE}")


def main() -> int:
    df = pd.read_parquet(INPUT)
    print(f"Rewrites smoke : {df.shape}")

    scored = add_cosine_scores(df)
    SCORED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_OUTPUT, index=False)
    print(f"Scored sauvegardé : {SCORED_OUTPUT}")
    print("\nCosine par style :")
    print(scored.groupby("style_target")["cosine"].describe().round(3).to_string())
    print("\nPréservation cosine :")
    print(scored.groupby("style_target")["preserved_cosine"].mean().round(3).to_string())

    votes = judge_smoke(scored)
    VOTES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    votes.to_parquet(VOTES_OUTPUT, index=False)
    print(f"\nVotes sauvegardés : {VOTES_OUTPUT}")
    print(votes.groupby("style_target")[["style_wins", "baseline_wins", "tie"]].mean().round(3))

    plot_votes(votes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

