"""LLM-as-judge : jugement de préférence sur paires de réponses (module R3).

Reproduit le vote humain de Compar:IA via un LLM juge, appliqué aux paires
contrefactuelles (r vs r') pour estimer l'effet causal du style sur la
préférence. Contrôle du biais de position : ordre A/B randomisé.
"""

from __future__ import annotations

JUDGE_MODEL = "mistral-small-latest"
POSITION_RANDOMIZED = True


def judge_pair(prompt: str, response_a: str, response_b: str) -> dict:
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

    Notes
    -----
    Implémenté en Phase 4. Position A/B randomisée pour neutraliser le
    biais de position.
    """
    raise NotImplementedError("Implémenté en Phase 4.")
