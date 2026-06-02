"""Mesures de diversité stylistique inter-modèles (module R1).

Quantifie la dispersion stylistique entre modèles au sein d'une cohorte
temporelle. Mesure principale : distance de Wasserstein-2 (POT) entre les
distributions empiriques de features standardisées. Mesures alternatives
(robustness) : variance pooled, coefficient de Gini, entropie de clusters.
"""

from __future__ import annotations

import numpy as np


def wasserstein_diversity(distributions: dict[str, np.ndarray]) -> float:
    """Diversité inter-modèles = W2 moyenne sur toutes les paires de modèles.

    Parameters
    ----------
    distributions
        Dict ``{model_name: array (n_samples, n_features)}`` des features
        stylistiques standardisées par modèle.

    Returns
    -------
    float
        Distance de Wasserstein-2 moyenne sur les paires (m, m'), m != m'.

    Notes
    -----
    Implémenté en Phase 3 (notebook ``03_R1_convergence``).
    """
    raise NotImplementedError("Implémenté en Phase 3.")
