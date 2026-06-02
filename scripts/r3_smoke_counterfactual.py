"""R3 smoke test — génération de contrefactuels stylistiques.

Smoke test volontairement petit :
- 10 réponses gagnantes issues de comparia-votes ;
- 3 styles : verbose_markdown, concise_direct, neutre_baseline ;
- 30 appels Mistral max ;
- sauvegarde incrémentale dans data/interim/rewrites_smoke.parquet.

Ce script ne fait pas encore NLI/cosine/judge. Il valide d'abord l'extraction
texte, les prompts, l'API Mistral et le format de sortie.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.counterfactual import (  # noqa: E402
    STYLE_TARGETS,
    build_rewrite_rows,
    choose_original_response,
)

DATASET = "ministere-culture/comparia-votes"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "rewrites_smoke.parquet"
N_SMOKE = 10
MIN_RESPONSE_CHARS = 300


def hf_token() -> str:
    """Retourne le token HF depuis `.env`."""
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        msg = "HF_TOKEN manquant dans .env"
        raise ValueError(msg)
    return token


def collect_smoke_rows(n_rows: int = N_SMOKE) -> list[dict[str, Any]]:
    """Collecte `n_rows` paires décisives avec une réponse exploitable."""
    ds = load_dataset(DATASET, split="train", streaming=True, token=hf_token())
    rows: list[dict[str, Any]] = []

    for row in tqdm(ds, desc="collect votes", unit=" rows"):
        if row.get("both_equal"):
            continue
        if not row.get("chosen_model_name"):
            continue

        _, _, original = choose_original_response(row)
        if len(original) < MIN_RESPONSE_CHARS:
            continue

        rows.append(row)
        if len(rows) >= n_rows:
            break

    if len(rows) < n_rows:
        msg = f"Seulement {len(rows)} lignes collectées sur {n_rows}"
        raise RuntimeError(msg)
    return rows


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(description="Génère des contrefactuels R3")
    parser.add_argument("--n", type=int, default=N_SMOKE, help="Nombre de réponses source")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Parquet de sortie",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output

    print(f"Collecte {args.n} réponses depuis {DATASET}")
    rows = collect_smoke_rows(args.n)
    print(f"Lignes collectées : {len(rows)}")
    print(f"Styles : {STYLE_TARGETS}")

    rewrites = build_rewrite_rows(rows, output)
    print(f"\nSauvegardé : {output}")
    print(f"Shape : {rewrites.shape}")
    print(rewrites[["style_target", "original_n_chars", "rewritten_n_chars"]].head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

