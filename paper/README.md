# Article LaTeX — L'Arène se mord la queue

## Compilation (recommandé : Tectonic)

MacTeX demande un mot de passe admin. **Utilise Tectonic** à la place :

```bash
brew install tectonic   # une seule fois, pas de sudo
cd paper
make                    # ou : tectonic main.tex
open main.pdf
```

Le PDF est généré : `paper/main.pdf`.

## Alternative MacTeX

Si tu as installé MacTeX **jusqu'au bout** (avec mot de passe admin) :

```bash
eval "$(/usr/libexec/path_helper)"   # ajoute /Library/TeX/texbin au PATH
cd paper
make pdflatex
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `main.tex` | Article scientifique complet (FR) |
| `main.pdf` | PDF compilé |
| `refs.bib` | Bibliographie |
| `figures/` | Figures PNG (R1–R6) |
| `tables/` | Tables exportées |

## Figures attendues

- `R1_convergence_robustness.png`
- `R2bis_style_premium_longitudinal.png`
- `R3_ensemble_judge.png`
- `R5_style_vs_quality.png`
- `R5bis_structure_vs_length.png`
- `R6_endogeneity_tiers.png`

Regénérer si besoin :

```bash
source ../.venv/bin/activate
python ../scripts/r1_robustness.py
python ../scripts/r2bis_style_premium.py
python ../scripts/r5_style_vs_quality.py
python ../scripts/r5bis_structure_vs_length.py
python ../scripts/r6_endogeneity_tiers.py
```
