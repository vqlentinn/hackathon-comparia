"""Ingestion et préparation des données Compar:IA.

Schéma validé via ``scripts/check_access.py`` (juin 2026).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

CONV_DATASET = "ministere-culture/comparia-conversations"
VOTES_DATASET = "ministere-culture/comparia-votes"

JOIN_KEY = "conversation_pair_id"
ZILINSKAS_PARQUET = "style-control-analysis/battles_bt_styled.parquet"

TIMESTAMP_COLS = (JOIN_KEY, "timestamp")


def _hf_token() -> str:
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "")
    if not token or token.startswith("hf_xxx"):
        msg = "HF_TOKEN manquant dans .env"
        raise ValueError(msg)
    return token


def load_battles_zilinskas(root: Path | str = ".") -> pd.DataFrame:
    """Charge le parquet battles de Zilinskas (142k battles, features style)."""
    path = Path(root) / ZILINSKAS_PARQUET
    if not path.exists():
        msg = f"Parquet introuvable : {path}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def fetch_vote_timestamps(
    limit: int | None = None,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """Stream ``comparia-votes`` et extrait ``conversation_pair_id`` + ``timestamp``.

    Parameters
    ----------
    limit
        Si fourni, arrête après N lignes (smoke test).
    cache_path
        Si fourni et existant, charge le cache parquet au lieu de streamer.

    Returns
    -------
    pd.DataFrame
        Une ligne par vote, dédupliquée sur ``conversation_pair_id``.
    """
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        print(f"Cache timestamps : {cache}")
        return pd.read_parquet(cache)

    from datasets import load_dataset

    token = _hf_token()
    print(f"Streaming {VOTES_DATASET} (colonnes {TIMESTAMP_COLS}) …")
    ds = load_dataset(
        VOTES_DATASET,
        split="train",
        streaming=True,
        token=token,
    )

    rows: list[dict] = []
    for i, row in enumerate(tqdm(ds, desc="votes", unit=" rows")):
        rows.append({JOIN_KEY: row[JOIN_KEY], "timestamp": row["timestamp"]})
        if limit is not None and i + 1 >= limit:
            break

    df = pd.DataFrame(rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=[JOIN_KEY], keep="first")
    if n_before != len(df):
        print(f"Déduplication : {n_before:,} → {len(df):,} paires uniques")

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        print(f"Cache sauvegardé : {cache}")

    return df


def join_battles_with_dates(
    battles: pd.DataFrame,
    timestamps: pd.DataFrame,
) -> pd.DataFrame:
    """Joint battles Zilinskas avec timestamps HF sur ``conversation_pair_id``."""
    ts = timestamps.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True)
    ts = ts.drop_duplicates(subset=[JOIN_KEY], keep="first")

    merged = battles.merge(ts, on=JOIN_KEY, how="left", validate="m:1")
    merged["date"] = merged["timestamp"].dt.tz_convert(None)
    merged["month"] = merged["date"].dt.to_period("M").astype(str)

    n_total = len(merged)
    n_matched = merged["timestamp"].notna().sum()
    pct = 100 * n_matched / n_total if n_total else 0
    print(f"Jointure : {n_matched:,}/{n_total:,} battles avec timestamp ({pct:.1f}%)")

    return merged
