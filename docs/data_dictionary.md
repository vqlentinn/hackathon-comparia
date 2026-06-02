# Dictionnaire des données

> ⚠️ Schéma issu de la **spec** (section 9). À **confirmer** via
> `scripts/check_access.py` avant tout usage en code (règle : ne pas
> inventer les noms de colonnes). Cette page sera mise à jour avec le
> schéma réel après validation.

## `comparia-conversations` (gated, ~8.7 Go)

| Colonne | Type attendu | Usage |
|---|---|---|
| `conversation_pair_id` | str | clé de jointure avec votes |
| `conversation_a` / `conversation_b` | list[{role, content}] | extraction prompt + réponses |
| `model_a_name` / `model_b_name` | str | identité des modèles comparés |
| `system_prompt_a` / `_b` | str | contrôle (identiques ?) |
| `total_conv_a_output_tokens` / `_b` | int | longueur (covariable confusion) |
| `total_conv_a_kwh` / `_b` | float | empreinte énergétique |
| `model_a_total_params` / `_active_params` (+ b) | int | stratification taille |
| `opening_msg` | str | prompt initial (topic modeling) |
| `short_summary`, `keywords`, `categories` | str/list | topics |
| `languages` | list[str] | filtre FR |
| `visitor_id` | str | dédup / power users |
| `timestamp` | datetime | cohorte temporelle |

## `comparia-votes` (gated, ~50 Mo) — **schéma réel validé**

| Colonne | Usage |
|---|---|
| `conversation_pair_id` | clé de jointure avec parquet Zilinskas |
| `chosen_model_name` | modèle gagnant (pas `preference` !) |
| `both_equal` | égalité |
| `timestamp` | datation du vote → cohortes R1/R2bis/R4 |
| `model_a_name` / `model_b_name` | identité modèles |
| `conv_*_a/b` | attributs qualité (complétude, créativité, etc.) |

**Note** : pas de colonne `preference`. Le gagnant est `chosen_model_name`.
Les battles `source=reaction` du parquet Zilinskas (~19 %) n'ont pas de
timestamp dans votes → filtrer sur `timestamp.notna()` pour analyses temporelles.

## `models.json` (public, betagouv/ComparIA)

Métadonnées des 110+ modèles : nom, éditeur, date de release, params,
licence. Sert R2 (âge des modèles) et stratifications.
