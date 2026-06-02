"""Validation d'accès : clés API + schéma réel des datasets HF gated.

Usage
-----
    .venv/bin/python scripts/check_access.py

Vérifie (sans rien télécharger de lourd) :
1. présence du `.env` et des 3 clés (HF_TOKEN, MISTRAL_API_KEY, GROQ_API_KEY) ;
2. accès HF + dump des colonnes réelles de comparia-conversations / -votes
   via streaming (10 lignes) ;
3. 1 appel minimal Mistral et Groq pour valider les clés.

Tout est défensif : un échec sur un bloc n'empêche pas les autres.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

CONV_DATASET = "ministere-culture/comparia-conversations"
VOTES_DATASET = "ministere-culture/comparia-votes"
N_PEEK = 10


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _ko(msg: str) -> None:
    print(f"  [KO]   {msg}")


def _info(msg: str) -> None:
    print(f"  [..]   {msg}")


def load_env() -> dict[str, str]:
    """Charge .env et retourne les clés attendues (valeurs masquées au log)."""
    print("\n=== 1. Fichier .env et clés ===")
    if not ENV_PATH.exists():
        _ko(f".env absent ({ENV_PATH}). Copier .env.example -> .env et remplir.")
        return {}
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
    except ImportError:
        _info("python-dotenv non installé ; lecture manuelle du .env.")
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    keys = {}
    for name in ("HF_TOKEN", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        val = os.environ.get(name, "")
        if val and not val.startswith(("hf_xxx", "xxx", "gsk_xxx")):
            _ok(f"{name} présent ({val[:6]}…{val[-3:]}).")
            keys[name] = val
        else:
            _ko(f"{name} manquant ou encore au placeholder.")
    return keys


def check_hf(token: str) -> None:
    """Dump le schéma réel des datasets en streaming (10 lignes)."""
    print("\n=== 2. Accès HuggingFace + schéma réel ===")
    if not token:
        _ko("Pas de HF_TOKEN : section sautée.")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        _ko("`datasets` non installé : pip install -r requirements.txt.")
        return

    for name in (CONV_DATASET, VOTES_DATASET):
        try:
            _info(f"Streaming {name} …")
            ds = load_dataset(name, split="train", streaming=True, token=token)
            first = next(iter(ds))
            _ok(f"{name} : {len(first)} colonnes")
            print(f"         colonnes = {sorted(first.keys())}")
        except Exception as exc:  # noqa: BLE001
            _ko(f"{name} : {type(exc).__name__}: {exc}")


def check_mistral(key: str) -> None:
    print("\n=== 3. Clé Mistral ===")
    if not key:
        _ko("Pas de MISTRAL_API_KEY : section sautée.")
        return
    try:
        # mistralai >= 2.x : Mistral n'est plus exporté depuis le package racine
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral  # fallback mistralai 1.x

        client = Mistral(api_key=key)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Réponds juste: OK"}],
            max_tokens=5,
        )
        _ok(f"Mistral répond : {resp.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        _ko(f"Mistral : {type(exc).__name__}: {exc}")


def check_groq(key: str) -> None:
    print("\n=== 4. Clé Groq (fallback) ===")
    if not key:
        _ko("Pas de GROQ_API_KEY : section sautée.")
        return
    try:
        from groq import Groq

        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Réponds juste: OK"}],
            max_tokens=5,
        )
        _ok(f"Groq répond : {resp.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        _ko(f"Groq : {type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 60)
    print(" check_access.py — validation accès Compar:IA hackathon")
    print("=" * 60)
    keys = load_env()
    check_hf(keys.get("HF_TOKEN", ""))
    check_mistral(keys.get("MISTRAL_API_KEY", ""))
    check_groq(keys.get("GROQ_API_KEY", ""))
    print("\nTerminé.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
