"""LLM-as-judge : jugement de préférence sur paires de réponses (module R3).

Reproduit le vote humain de Compar:IA via un LLM juge, appliqué aux paires
contrefactuelles (r vs r') pour estimer l'effet causal du style sur la
préférence. Contrôle du biais de position : ordre A/B randomisé.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from compariawatch.counterfactual import get_mistral_client

JUDGE_MODEL = "mistral-small-latest"
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


def judge_pair(
    prompt: str,
    response_a: str,
    response_b: str,
    client: Any | None = None,
    randomize_position: bool = POSITION_RANDOMIZED,
    seed: int = RANDOM_STATE,
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
    response = mistral.chat.complete(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    prompt=prompt[:MAX_JUDGE_CHARS],
                    response_a=left[:MAX_JUDGE_CHARS],
                    response_b=right[:MAX_JUDGE_CHARS],
                ),
            }
        ],
        temperature=0,
        max_tokens=120,
    )
    raw = response.choices[0].message.content.strip()
    winner, reason = _parse_winner(raw)

    if flipped:
        if winner == "a":
            winner = "b"
        elif winner == "b":
            winner = "a"

    return {"winner": winner, "reason": reason, "raw": raw, "flipped": flipped}
