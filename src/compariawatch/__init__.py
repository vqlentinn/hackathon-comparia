"""compariawatch -- audit causal de Compar:IA (loi de Goodhart).

Librairie support du projet « L'Arène se mord la queue ». Regroupe le
pipeline d'ingestion, l'extraction de features stylistiques, les mesures
de diversité (transport optimal), la génération/jugement de contrefactuels
et le forecast d'effondrement du benchmark.

Modules
-------
data            Ingestion + jointure conversations ⨝ votes, filtre FR.
features        Extraction des ~25 features stylistiques (NLP symbolique).
diversity       Diversité inter-modèles : Wasserstein, Gini, entropie.
counterfactual  Réécriture stylistique (LLM) + vérification NLI/cosine.
judge           LLM-as-judge sur paires de réponses.
forecast        Modèle d'état bayésien (NumPyro) / fallback Prophet.
"""

__version__ = "0.1.0"

RANDOM_STATE = 42
