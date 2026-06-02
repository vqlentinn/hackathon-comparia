| model | auc | n_features | n_rows |
| --- | --- | --- | --- |
| Longueur (log) seule | 0.6614 | 1 | 79073 |
| style_total_only | 0.6671 | 1 | 79073 |
| style_components_only | 0.6686 | 5 | 79073 |
| Longueur + style structurel | 0.6780 | 6 | 79073 |
| quality_total_only | 0.8111 | 1 | 79073 |
| quality_components_only | 0.8206 | 7 | 79073 |
| quality_plus_style_totals | 0.8583 | 2 | 79073 |
| Qualité + longueur | 0.8598 | 8 | 79073 |
| quality_plus_style_components | 0.8624 | 12 | 79073 |
| Qualité + style (R5) | 0.8624 | 12 | 79073 |
| quality_style_time_interactions | 0.8632 | 17 | 79073 |
| Qualité + longueur + style (R5bis) | 0.8650 | 13 | 79073 |

## Effets structurels (modèle complet)

- longueur log : **40.5%** odds / SD

| feature | odds_pct_per_sd |
| --- | --- |
| bold | 14.53 |
| headers | 13.77 |
| emoji | 6.18 |
| lists | 5.25 |
| code_blocks | 1.07 |
