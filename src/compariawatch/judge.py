"""LLM-as-judge : jugement de préférence sur paires de réponses (module R3).

Reproduit le vote humain de Compar:IA via un LLM juge, appliqué aux paires
contrefactuelles (r vs r') pour estimer l'effet causal du style sur la
préférence. Contrôle du biais de position : ordre A/B randomisé.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from compariawatch.counterfactual import get_mistral_client

JUDGE_MODEL = "mistral-small-latest"
GROQ_LLAMA_MODEL = "llama-3.3-70b-versatile"
GROQ_MIXTRAL_MODEL = "llama-3.1-8b-instant"
POSITION_RANDOMIZED = True
RANDOM_STATE = 42
MAX_JUDGE_CHARS = 2_500

JUDGE_PROMPT = """Tu es juge sur une arène de comparaison de modèles IA.

Tu dois choisir la meilleure réponse pour l'utilisateur.

Critères :
- utilité et exactitude ;
- clarté ;
- respect de la demande ;
- ne favorise pas automatiquement la réponse la plus longue.

Question utilisateur :
\"\"\"
{prompt}
\"\"\"

Réponse A :
\"\"\"
{response_a}
\"\"\"

Réponse B :
\"\"\"
{response_b}
\"\"\"

Réponds STRICTEMENT en JSON valide :
{{"winner": "A" ou "B" ou "tie", "reason": "raison très courte"}}
"""


def _parse_winner(raw: str) -> tuple[str, str]:
    """Parse robuste d'une réponse juge JSON ou quasi-JSON."""
    try:
        data = json.loads(raw)
        winner = str(data.get("winner", "tie")).lower()
        reason = str(data.get("reason", ""))
    except json.JSONDecodeError:
        match = re.search(r"\b(A|B|tie)\b", raw, flags=re.IGNORECASE)
        winner = match.group(1).lower() if match else "tie"
        reason = raw[:200]

    if winner == "a":
        return "a", reason
    if winner == "b":
        return "b", reason
    return "tie", reason


def _to_public_vote(winner: str) -> str:
    """Convertit `a`/`b`/`tie` en `A`/`B`/`tie`."""
    if winner == "a":
        return "A"
    if winner == "b":
        return "B"
    return "tie"


def _from_public_vote(vote: str) -> str:
    """Convertit `A`/`B`/`tie` en `a`/`b`/`tie`."""
    if vote == "A":
        return "a"
    if vote == "B":
        return "b"
    return "tie"


def _retry_sleep_seconds(error: Exception, attempt: int) -> float:
    """Calcule un backoff, en respectant les délais Groq si fournis."""
    text = str(error)
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text, flags=re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 5
    return min(45, 2 ** (attempt + 2))


def get_groq_client() -> Groq:
    """Construit un client Groq depuis `GROQ_API_KEY` ou `.env`."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        msg = "GROQ_API_KEY manquant dans .env"
        raise ValueError(msg)
    return Groq(api_key=api_key)


def judge_pair(
    prompt: str,
    response_a: str,
    response_b: str,
    client: Any | None = None,
    randomize_position: bool = POSITION_RANDOMIZED,
    seed: int = RANDOM_STATE,
    max_tries: int = 4,
) -> dict[str, str | bool]:
    """Juge la paire (response_a, response_b) pour un prompt donné.

    Parameters
    ----------
    prompt
        Question / consigne initiale.
    response_a, response_b
        Réponses à comparer.

    Returns
    -------
    dict
        ``{"winner": "a"|"b"|"tie", "raw": <texte brut du juge>}``.

    L'ordre A/B est randomisé par défaut, puis le gagnant est remappé vers
    les labels originaux (`a`, `b`, `tie`).
    """
    rng = random.Random(seed)
    flipped = randomize_position and rng.random() > 0.5
    left, right = (response_b, response_a) if flipped else (response_a, response_b)

    mistral = client or get_mistral_client()
    content = JUDGE_PROMPT.format(
        prompt=prompt[:MAX_JUDGE_CHARS],
        response_a=left[:MAX_JUDGE_CHARS],
        response_b=right[:MAX_JUDGE_CHARS],
    )

    last_error: Exception | None = None
    for attempt in range(max_tries):
        try:
            response = mistral.chat.complete(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=120,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_s = min(60, 2 ** (attempt + 2))
            print(f"[retry judge] tentative {attempt + 1}/{max_tries}: {exc}; sleep={sleep_s}s")
            time.sleep(sleep_s)
    else:
        msg = f"Échec judge après {max_tries} tentatives : {last_error}"
        raise RuntimeError(msg)

    raw = response.choices[0].message.content.strip()
    winner, reason = _parse_winner(raw)

    if flipped:
        if winner == "a":
            winner = "b"
        elif winner == "b":
            winner = "a"

    return {"winner": winner, "reason": reason, "raw": raw, "flipped": flipped}


def groq_judge(
    prompt: str,
    response_a: str,
    response_b: str,
    *,
    model: str = GROQ_LLAMA_MODEL,
    randomize_position: bool = True,
    seed: int = RANDOM_STATE,
    client: Groq | None = None,
    max_tries: int = 3,
) -> dict[str, str | bool]:
    """Juge via Groq avec randomisation de position et retry backoff.

    Modèles supportés :
    - ``llama-3.3-70b-versatile`` ;
    - ``llama-3.1-8b-instant``.

    Returns
    -------
    dict
        ``{"vote": "A"|"B"|"tie", "raw": str, "position_flipped": bool}``,
        où A/B sont remappés sur l'ordre original fourni à la fonction.
    """
    rng = random.Random(seed)
    flipped = randomize_position and rng.random() > 0.5
    left, right = (response_b, response_a) if flipped else (response_a, response_b)
    groq_client = client or get_groq_client()
    content = JUDGE_PROMPT.format(
        prompt=prompt[:MAX_JUDGE_CHARS],
        response_a=left[:MAX_JUDGE_CHARS],
        response_b=right[:MAX_JUDGE_CHARS],
    )

    last_error: Exception | None = None
    for attempt in range(max_tries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            winner, _ = _parse_winner(raw)
            vote = _to_public_vote(winner)
            if flipped:
                if vote == "A":
                    vote = "B"
                elif vote == "B":
                    vote = "A"
            return {"vote": vote, "raw": raw, "position_flipped": flipped}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_s = _retry_sleep_seconds(exc, attempt)
            print(f"[retry groq {model}] {attempt + 1}/{max_tries}: {exc}; sleep={sleep_s}s")
            time.sleep(sleep_s)

    msg = f"Échec Groq judge {model} après {max_tries} tentatives : {last_error}"
    raise RuntimeError(msg)


def ensemble_judge(
    prompt: str,
    response_a: str,
    response_b: str,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Juge avec Mistral, Groq Llama et Groq Mixtral, puis agrège.

    Tolère les timeouts/erreurs : un vote indisponible vaut ``None``. La
    majorité est calculée sur les votes disponibles ; s'il y a égalité, elle
    vaut ``tie``.
    """
    votes: dict[str, str | None] = {"mistral": None, "llama": None, "mixtral": None}
    raw: dict[str, str | None] = {"mistral": None, "llama": None, "mixtral": None}
    flipped: dict[str, bool | None] = {"mistral": None, "llama": None, "mixtral": None}

    try:
        mistral = judge_pair(
            prompt,
            response_a,
            response_b,
            randomize_position=True,
            seed=seed,
        )
        votes["mistral"] = _to_public_vote(str(mistral["winner"]))
        raw["mistral"] = str(mistral["raw"])
        flipped["mistral"] = bool(mistral["flipped"])
    except Exception as exc:  # noqa: BLE001
        raw["mistral"] = f"ERROR: {exc}"

    groq_client = get_groq_client()
    for judge_name, model_name, offset in [
        ("llama", GROQ_LLAMA_MODEL, 10_000),
        ("mixtral", GROQ_MIXTRAL_MODEL, 20_000),
    ]:
        try:
            result = groq_judge(
                prompt,
                response_a,
                response_b,
                model=model_name,
                seed=seed + offset,
                client=groq_client,
            )
            votes[judge_name] = str(result["vote"])
            raw[judge_name] = str(result["raw"])
            flipped[judge_name] = bool(result["position_flipped"])
        except Exception as exc:  # noqa: BLE001
            raw[judge_name] = f"ERROR: {exc}"

    valid_votes = [vote for vote in votes.values() if vote in {"A", "B", "tie"}]
    counts = {vote: valid_votes.count(vote) for vote in ["A", "B", "tie"]}
    if not valid_votes:
        majority = "tie"
        consensus = 0
    else:
        top_count = max(counts.values())
        top_votes = [vote for vote, count in counts.items() if count == top_count]
        majority = top_votes[0] if len(top_votes) == 1 else "tie"
        consensus = top_count

    return {
        "votes": votes,
        "majority": majority,
        "consensus": consensus,
        "all_agree": consensus == 3,
        "raw": raw,
        "position_flipped": flipped,
    }
