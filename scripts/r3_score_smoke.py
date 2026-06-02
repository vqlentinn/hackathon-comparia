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
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.counterfactual import COSINE_THRESHOLD, get_mistral_client  # noqa: E402
from compariawatch.judge import judge_pair  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "interim" / "rewrites_smoke.parquet"
DEFAULT_SCORED_OUTPUT = ROOT / "data" / "interim" / "rewrites_smoke_scored.parquet"
DEFAULT_VOTES_OUTPUT = ROOT / "data" / "processed" / "causal_style_votes_smoke.parquet"
DEFAULT_FIGURE = ROOT / "paper" / "figures" / "R3_style_premium_smoke.png"

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


def judge_smoke(
    scored: pd.DataFrame,
    votes_output: Path,
    require_cosine: bool = False,
    sleep_between_calls: float = 1.5,
) -> pd.DataFrame:
    """Juge style cible vs baseline neutre pour chaque conversation."""
    client = get_mistral_client()
    if votes_output.exists():
        existing = pd.read_parquet(votes_output)
        rows: list[dict] = existing.to_dict("records")
        done = set(zip(existing["conversation_pair_id"], existing["style_target"], strict=True))
        print(f"[resume votes] {votes_output} chargé ({len(rows)} jugements)")
    else:
        rows = []
        done = set()

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
            key = (row["conversation_pair_id"], style)
            if key in done:
                print(f"[skip judged] {idx + 1}/{len(grouped)} {style}")
                continue

            candidate = row.get(f"rewritten__{style}")
            if not isinstance(candidate, str) or not candidate:
                continue
            if require_cosine and not (
                bool(row.get(f"preserved_cosine__{style}"))
                and bool(row.get(f"preserved_cosine__{BASELINE_STYLE}"))
            ):
                print(f"[skip cosine] {idx + 1}/{len(grouped)} {style}")
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
            votes_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(votes_output, index=False)
            time.sleep(sleep_between_calls)
            print(f"[judge] {idx + 1}/{len(grouped)} {style} -> {winner}")

    return pd.DataFrame(rows)


def plot_votes(votes: pd.DataFrame, figure: Path) -> None:
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
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=200, bbox_inches="tight")
    print(f"Figure sauvegardée : {figure}")


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Score les contrefactuels R3")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--scored-output", type=Path, default=DEFAULT_SCORED_OUTPUT)
    parser.add_argument("--votes-output", type=Path, default=DEFAULT_VOTES_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--require-cosine",
        action="store_true",
        help="Juge seulement les paires où style et neutre passent le seuil cosine",
    )
    parser.add_argument("--sleep", type=float, default=1.5, help="Pause entre deux appels judge")
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    input_path = _rooted(args.input)
    scored_output = _rooted(args.scored_output)
    votes_output = _rooted(args.votes_output)
    figure = _rooted(args.figure)

    if scored_output.exists():
        scored = pd.read_parquet(scored_output)
        print(f"Scored existant chargé : {scored_output} {scored.shape}")
    else:
        df = pd.read_parquet(input_path)
        print(f"Rewrites smoke : {df.shape}")
        scored = add_cosine_scores(df)
        scored_output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(scored_output, index=False)
        print(f"Scored sauvegardé : {scored_output}")
    print("\nCosine par style :")
    print(scored.groupby("style_target")["cosine"].describe().round(3).to_string())
    print("\nPréservation cosine :")
    print(scored.groupby("style_target")["preserved_cosine"].mean().round(3).to_string())

    votes = judge_smoke(
        scored,
        votes_output=votes_output,
        require_cosine=args.require_cosine,
        sleep_between_calls=args.sleep,
    )
    votes_output.parent.mkdir(parents=True, exist_ok=True)
    votes.to_parquet(votes_output, index=False)
    print(f"\nVotes sauvegardés : {votes_output}")
    print(votes.groupby("style_target")[["style_wins", "baseline_wins", "tie"]].mean().round(3))

    plot_votes(votes, figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

