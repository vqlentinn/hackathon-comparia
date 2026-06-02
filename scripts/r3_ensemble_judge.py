"""R3 — Ensemble judge multi-modèles sur contrefactuels N=100.

Juge les paires filtrées par cosine avec :
- Mistral Small ;
- Groq Llama 3.3 70B ;
- Groq Mixtral 8x7B si disponible.

Le script est reprenable : il recharge l'output existant et saute les paires
déjà jugées.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.judge import (  # noqa: E402
    GROQ_LLAMA_MODEL,
    GROQ_MIXTRAL_MODEL,
    get_groq_client,
    groq_judge,
)

INPUT = ROOT / "data" / "interim" / "rewrites_n100_scored.parquet"
MISTRAL_VOTES = ROOT / "data" / "processed" / "causal_style_votes_n100.parquet"
OUTPUT = ROOT / "data" / "processed" / "causal_style_votes_n100_ensemble.parquet"
BASELINE_STYLE = "neutre_baseline"
COMPARE_STYLES = ("concise_direct", "verbose_markdown")
SAVE_EVERY = 1


def majority_vote(votes: list[str | None]) -> tuple[str, int, bool]:
    """Calcule majorité, consensus et accord total sur votes disponibles."""
    valid = [vote for vote in votes if vote in {"A", "B", "tie"}]
    if not valid:
        return "tie", 0, False
    counts = {vote: valid.count(vote) for vote in ["A", "B", "tie"]}
    top_count = max(counts.values())
    top_votes = [vote for vote, count in counts.items() if count == top_count]
    majority = top_votes[0] if len(top_votes) == 1 else "tie"
    return majority, top_count, top_count == 3


def mistral_votes_map(path: Path) -> dict[tuple[str, str], str]:
    """Mappe les votes Mistral existants vers A=style, B=baseline."""
    votes = pd.read_parquet(path)
    mapped: dict[tuple[str, str], str] = {}
    for _, row in votes.iterrows():
        key = (row["conversation_pair_id"], row["style_target"])
        if bool(row["style_wins"]):
            mapped[key] = "A"
        elif bool(row["baseline_wins"]):
            mapped[key] = "B"
        else:
            mapped[key] = "tie"
    return mapped


def build_filtered_pairs(scored: pd.DataFrame) -> pd.DataFrame:
    """Construit les comparaisons style cible vs neutre après filtre cosine."""
    grouped = scored.pivot_table(
        index=["conversation_pair_id", "opening_msg", "original_model"],
        columns="style_target",
        values=["rewritten", "cosine", "preserved_cosine"],
        aggfunc="first",
    )
    grouped.columns = [f"{left}__{right}" for left, right in grouped.columns]
    grouped = grouped.reset_index()

    rows: list[dict] = []
    for _, row in grouped.iterrows():
        baseline = row.get(f"rewritten__{BASELINE_STYLE}")
        baseline_ok = bool(row.get(f"preserved_cosine__{BASELINE_STYLE}"))
        if not isinstance(baseline, str) or not baseline_ok:
            continue

        for style in COMPARE_STYLES:
            candidate = row.get(f"rewritten__{style}")
            style_ok = bool(row.get(f"preserved_cosine__{style}"))
            if not isinstance(candidate, str) or not style_ok:
                continue
            rows.append(
                {
                    "conversation_pair_id": row["conversation_pair_id"],
                    "opening_msg": row["opening_msg"],
                    "original_model": row["original_model"],
                    "style_target": style,
                    "baseline_style": BASELINE_STYLE,
                    "response_style": candidate,
                    "response_baseline": baseline,
                    "style_cosine": row.get(f"cosine__{style}"),
                    "baseline_cosine": row.get(f"cosine__{BASELINE_STYLE}"),
                }
            )
    return pd.DataFrame(rows)


def load_existing(output: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """Charge les votes déjà produits."""
    if not output.exists():
        return [], set()
    existing = pd.read_parquet(output)
    rows = existing.to_dict("records")
    done = set(zip(existing["conversation_pair_id"], existing["style_target"], strict=True))
    print(f"[resume] {len(rows)} jugements existants chargés")
    return rows, done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3 ensemble judge")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--mistral-votes", type=Path, default=MISTRAL_VOTES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Limite smoke test")
    parser.add_argument("--sleep", type=float, default=1.0, help="Pause entre paires")
    parser.add_argument(
        "--groq-llama-model",
        type=str,
        default=GROQ_LLAMA_MODEL,
        help="Modèle Groq juge #2 (fallback si quota 70B épuisé : compound-beta-mini)",
    )
    parser.add_argument(
        "--groq-third-model",
        type=str,
        default=GROQ_MIXTRAL_MODEL,
        help="Modèle Groq juge #3",
    )
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    input_path = _rooted(args.input)
    mistral_path = _rooted(args.mistral_votes)
    output_path = _rooted(args.output)

    scored = pd.read_parquet(input_path)
    pairs = build_filtered_pairs(scored)
    if args.limit is not None:
        pairs = pairs.head(args.limit).copy()
    print(f"Paires à juger : {len(pairs)}")
    print(pairs["style_target"].value_counts().to_string())
    print(f"Groq juge #2 : {args.groq_llama_model}")
    print(f"Groq juge #3 : {args.groq_third_model}")
    mistral_lookup = mistral_votes_map(mistral_path)

    rows, done = load_existing(output_path)
    groq_client = get_groq_client()
    for idx, row in pairs.reset_index(drop=True).iterrows():
        key = (row["conversation_pair_id"], row["style_target"])
        if key in done:
            print(f"[skip] {idx + 1}/{len(pairs)} {row['style_target']}")
            continue

        mistral_vote = mistral_lookup.get(key)
        llama = groq_judge(
            prompt=str(row["opening_msg"]),
            response_a=str(row["response_style"]),
            response_b=str(row["response_baseline"]),
            model=args.groq_llama_model,
            seed=42 + idx + 10_000,
            client=groq_client,
        )
        mixtral = groq_judge(
            prompt=str(row["opening_msg"]),
            response_a=str(row["response_style"]),
            response_b=str(row["response_baseline"]),
            model=args.groq_third_model,
            seed=42 + idx + 20_000,
            client=groq_client,
        )
        majority, consensus, all_agree = majority_vote(
            [mistral_vote, str(llama["vote"]), str(mixtral["vote"])]
        )
        out = {
            "conversation_pair_id": row["conversation_pair_id"],
            "original_model": row["original_model"],
            "style_target": row["style_target"],
            "baseline_style": row["baseline_style"],
            "style_cosine": row["style_cosine"],
            "baseline_cosine": row["baseline_cosine"],
            "mistral_vote": mistral_vote,
            "llama_vote": llama["vote"],
            "mixtral_vote": mixtral["vote"],
            "majority": majority,
            "consensus": consensus,
            "all_agree": all_agree,
            "raw_mistral": None,
            "raw_llama": llama["raw"],
            "raw_mixtral": mixtral["raw"],
            "flipped_mistral": None,
            "flipped_llama": llama["position_flipped"],
            "flipped_mixtral": mixtral["position_flipped"],
            "llama_groq_model": args.groq_llama_model,
            "third_groq_model": args.groq_third_model,
        }
        rows.append(out)
        done.add(key)
        print(
            f"[ok] {idx + 1}/{len(pairs)} {row['style_target']} "
            f"M={out['mistral_vote']} L={out['llama_vote']} X={out['mixtral_vote']} "
            f"maj={out['majority']}"
        )

        if len(rows) % SAVE_EVERY == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            print(f"[save] {output_path} ({len(rows)} lignes)")
        time.sleep(args.sleep)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Terminé : {output_path} ({len(rows)} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

