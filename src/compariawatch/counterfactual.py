"""Génération et vérification de contrefactuels stylistiques (module R3).

Pipeline (cf. spec 7.4.1) :
1. réécrire une réponse ``r`` vers un style cible (verbose, concis, neutre…)
   via l'API LLM (Mistral principal, Groq fallback) ;
2. vérifier la préservation sémantique par double critère :
   - NLI bidirectionnel (r |= r' ET r' |= r), score > seuil ;
   - cosine sur sentence-embeddings > seuil ;
3. filtrer les paires qui échouent.

Code défensif obligatoire : try/except sur appels API, retry backoff
exponentiel, sauvegarde parquet toutes les N itérations (règle 4).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

STYLE_TARGETS = ("verbose_markdown", "concise_direct", "neutre_baseline")

STYLE_DESCRIPTIONS = {
    "verbose_markdown": (
        "style long, structuré, avec titres markdown, listes, gras, "
        "explications détaillées et ton pédagogique"
    ),
    "concise_direct": (
        "style court, direct, sans markdown décoratif, sans flatterie, "
        "allant droit au but"
    ),
    "neutre_baseline": (
        "style neutre, paragraphe simple, longueur moyenne, sans emphase, "
        "sans titres et sans listes"
    ),
}

REWRITE_PROMPT = """Tu réécris une réponse en changeant UNIQUEMENT son style.

Style cible : {style_description}

Règles strictes :
- Préserve tous les faits, exemples, chiffres, conditions et nuances.
- N'ajoute aucune information nouvelle.
- Ne retire aucune information importante.
- Ne dis pas que tu as réécrit le texte.
- Réponds uniquement avec la réponse réécrite.

Réponse originale :
\"\"\"
{text}
\"\"\"
"""

NLI_THRESHOLD = 0.70
COSINE_THRESHOLD = 0.85
SAVE_EVERY = 5
MISTRAL_MODEL = "mistral-small-latest"
MAX_INPUT_CHARS = 3_000
MAX_REWRITE_TOKENS = 1_200


def get_mistral_client() -> Any:
    """Construit un client Mistral compatible avec `mistralai >= 2.x`."""
    load_dotenv(Path(".env"))
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        msg = "MISTRAL_API_KEY manquant dans .env"
        raise ValueError(msg)

    try:
        from mistralai.client import Mistral
    except ImportError:
        from mistralai import Mistral  # type: ignore

    return Mistral(api_key=api_key)


def extract_assistant_text(conversation: list[dict[str, Any]] | None) -> str:
    """Extrait la dernière réponse assistant d'une conversation Compar:IA."""
    if not conversation:
        return ""
    assistant_messages = [
        message.get("content", "")
        for message in conversation
        if message.get("role") == "assistant" and message.get("content")
    ]
    return "\n\n".join(assistant_messages).strip()


def choose_original_response(row: dict[str, Any]) -> tuple[str, str, str]:
    """Choisit la réponse gagnante si disponible, sinon réponse A.

    Returns
    -------
    tuple[str, str, str]
        `(side, model_name, response_text)`.
    """
    chosen = row.get("chosen_model_name")
    model_a = row.get("model_a_name")
    model_b = row.get("model_b_name")

    if chosen == model_b:
        return "b", str(model_b), extract_assistant_text(row.get("conversation_b"))
    return "a", str(model_a), extract_assistant_text(row.get("conversation_a"))


def rewrite_response(
    text: str,
    style_target: str,
    client: Any | None = None,
    max_tries: int = 3,
) -> str:
    """Réécrit ``text`` dans le style cible via l'API LLM.

    Parameters
    ----------
    text
        Réponse originale à réécrire.
    style_target
        Style cible, l'une des valeurs de ``STYLE_TARGETS``.

    Returns
    -------
    str
        Réponse réécrite ``r'``.

    Raises
    ------
    ValueError
        Si le style cible est inconnu.
    """
    if style_target not in STYLE_DESCRIPTIONS:
        msg = f"Style cible inconnu : {style_target}"
        raise ValueError(msg)

    mistral = client or get_mistral_client()
    prompt = REWRITE_PROMPT.format(
        style_description=STYLE_DESCRIPTIONS[style_target],
        text=text[:MAX_INPUT_CHARS],
    )

    last_error: Exception | None = None
    for attempt in range(max_tries):
        try:
            response = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=MAX_REWRITE_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_s = 2**attempt
            print(f"[retry] Mistral {style_target}, tentative {attempt + 1}: {exc}")
            time.sleep(sleep_s)

    msg = f"Échec rewrite après {max_tries} tentatives : {last_error}"
    raise RuntimeError(msg)


def build_rewrite_rows(
    source_rows: list[dict[str, Any]],
    output_path: Path,
    styles: tuple[str, ...] = STYLE_TARGETS,
) -> pd.DataFrame:
    """Génère les réécritures pour un petit batch et sauvegarde régulièrement."""
    client = get_mistral_client()
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        rows: list[dict[str, Any]] = existing.to_dict("records")
        completed = (
            existing.groupby("conversation_pair_id")["style_target"]
            .apply(lambda values: set(values) >= set(styles))
            .to_dict()
        )
        print(f"[resume] {output_path} chargé ({len(rows)} lignes existantes)")
    else:
        rows = []
        completed = {}

    for row_idx, row in enumerate(source_rows):
        pair_id = row.get("conversation_pair_id")
        if completed.get(pair_id, False):
            print(f"[skip] {row_idx + 1}/{len(source_rows)} déjà complet")
            continue

        side, model_name, original = choose_original_response(row)
        if not original:
            print(f"[skip] ligne {row_idx}: réponse assistant vide")
            continue

        existing_styles = {
            item["style_target"]
            for item in rows
            if item.get("conversation_pair_id") == pair_id
        }
        for style in styles:
            if style in existing_styles:
                continue
            rewritten = rewrite_response(original, style, client=client)
            rows.append(
                {
                    "conversation_pair_id": pair_id,
                    "opening_msg": row.get("opening_msg"),
                    "timestamp": row.get("timestamp"),
                    "chosen_model_name": row.get("chosen_model_name"),
                    "original_side": side,
                    "original_model": model_name,
                    "style_target": style,
                    "original": original,
                    "rewritten": rewritten,
                    "original_n_chars": len(original),
                    "rewritten_n_chars": len(rewritten),
                }
            )
            print(f"[ok] {row_idx + 1}/{len(source_rows)} {style}")

        if rows and (row_idx + 1) % SAVE_EVERY == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            print(f"[save] {output_path} ({len(rows)} lignes)")

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out
