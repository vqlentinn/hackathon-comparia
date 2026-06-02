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

STYLE_TARGETS = ("verbose_markdown", "concise_direct", "flatteur_poli", "neutre_baseline")

NLI_THRESHOLD = 0.70
COSINE_THRESHOLD = 0.85
SAVE_EVERY = 50


def rewrite_response(text: str, style_target: str) -> str:
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

    Notes
    -----
    Implémenté en Phase 4. Smoke test sur 10 paires avant lancement.
    """
    raise NotImplementedError("Implémenté en Phase 4.")
