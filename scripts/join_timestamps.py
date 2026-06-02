"""Jointure timestamps HF votes → parquet Zilinskas.

Usage
-----
    # Smoke test (10 battles, 100 votes streamés)
    python scripts/join_timestamps.py --smoke

    # Run complet (stream votes → cache → jointure)
    python scripts/join_timestamps.py

Sortie : data/interim/battles_with_dates.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compariawatch.data import (  # noqa: E402
    fetch_vote_timestamps,
    join_battles_with_dates,
    load_battles_zilinskas,
)

CACHE = ROOT / "data" / "raw" / "votes_timestamps.parquet"
OUTPUT = ROOT / "data" / "interim" / "battles_with_dates.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Jointure timestamps sur battles Zilinskas")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test : 10 battles, 100 votes streamés",
    )
    args = parser.parse_args()

    battles = load_battles_zilinskas(ROOT)
    print(f"Battles Zilinskas : {len(battles):,} lignes")

    if args.smoke:
        battles = battles.head(10)
        timestamps = fetch_vote_timestamps(limit=100)
    else:
        timestamps = fetch_vote_timestamps(cache_path=CACHE)

    result = join_battles_with_dates(battles, timestamps)

    if args.smoke:
        print("\n--- Smoke test (10 premières lignes) ---")
        cols = ["conversation_pair_id", "model_a_name", "winner", "timestamp", "month"]
        print(result[cols].head(10).to_string())
        print("\nSmoke test OK.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    print(f"\nSauvegardé : {OUTPUT} ({len(result):,} lignes × {result.shape[1]} cols)")

    if result["month"].notna().any():
        print("\nDistribution mensuelle (top 15) :")
        print(result["month"].value_counts().sort_index().head(15).to_string())
        print(f"\nCohortes mensuelles : {result['month'].nunique()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
