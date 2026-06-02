"""Forecast d'effondrement du benchmark (module R4).

Modèle d'état bayésien (random walk avec dérive) sur la série temporelle
de diversité stylistique D_t, estimé par NumPyro (NUTS). Projette la date
à laquelle D_t passe sous le seuil critique tau_crit. Fallback : Prophet
ou lissage exponentiel si NumPyro est indisponible/trop lent.

Modèle (cf. spec 7.6.1) :
    D_t = mu_t + eps_t
    mu_t = mu_{t-1} + lambda + nu_t
"""

from __future__ import annotations

import numpy as np

N_SAMPLES = 2000
N_CHAINS = 4
PRIOR_TREND_SCALE = 0.1


def forecast_collapse(d_observed: np.ndarray, horizon: int) -> dict:
    """Projette la diversité future et la date d'effondrement.

    Parameters
    ----------
    d_observed
        Série temporelle observée de la diversité inter-modèles D_t.
    horizon
        Nombre de pas de temps futurs à projeter.

    Returns
    -------
    dict
        Trajectoire postérieure, P(D_t < tau_crit) et date espérée
        d'effondrement avec intervalle de crédibilité 95 %.

    Notes
    -----
    Implémenté en Phase 5. Fallback Prophet si NumPyro plante (règle survie).
    """
    raise NotImplementedError("Implémenté en Phase 5.")
