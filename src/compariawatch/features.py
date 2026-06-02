"""Extraction des features stylistiques (NLP symbolique).

~25 dimensions interprétables par texte (cf. spec 7.2.1) : longueur,
complexité syntaxique, formalité, densité lexicale, registre (tu/vous),
ponctuation, markdown, ratios POS, hedging, flatterie, anglicismes,
lisibilité (Flesch FR), sentiment.

Calculées sur 3 textes par paire : prompt, réponse A, réponse B.
"""

from __future__ import annotations

import pandas as pd

SPACY_MODEL = "fr_core_news_lg"


def extract_style_features(texts: list[str]) -> pd.DataFrame:
    """Extrait les features stylistiques pour une liste de textes.

    Parameters
    ----------
    texts
        Liste de textes bruts (réponses ou prompts).

    Returns
    -------
    pd.DataFrame
        Une ligne par texte, une colonne par dimension stylistique.

    Notes
    -----
    Implémenté en Phase 2 (notebook ``02_features_stylo``). Smoke test
    sur 10 textes obligatoire avant passage à l'échelle (règle 5).
    """
    raise NotImplementedError("Implémenté en Phase 2.")
