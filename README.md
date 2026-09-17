# Dual-driver-framework-data

Code and anonymised survey data supporting the analyses in *A Dual-Driver Framework for Scaling Nature-Based Solutions in Agriculture: Evidence from the UK*.

The repository includes data processing, multiple correspondence analysis, clustering, logistic regression, robustness outputs, and figure files.

## Repository contents

- `code/NbS_TUSSIE.R` — R script for data recoding, correspondence analysis, clustering, and logistic models
- `data/` — Qualtrics survey exports used by the script
- `outputs/tables/` — descriptive statistics, marginal effects, and robustness summaries
- `outputs/figures/` — selected SVG/PNG figures
- `docs/DATA_DICTIONARY.md` — Qualtrics item IDs

The original Qualtrics CSV is also retained at the repository root as `NBS perspectives survey data.csv`.

## How to run

Requires R 4.2+ and the packages listed in `requirements-r.txt`.

```r
install.packages(c(
  "readxl", "dplyr", "ggplot2", "forcats", "FactoMineR",
  "factoextra", "gplots", "tidyverse", "stargazer", "corrplot"
))

source("code/NbS_TUSSIE.R")
```

The script reads files from `data/` relative to the repository root.

## Data

| File | Description |
| --- | --- |
| `data/Results_text.xlsx` | Text-coded survey responses |
| `data/Results_sep.xlsx` | Split-coded multiple-response items |
| `data/Results_values.xlsx` | Numeric-coded Qualtrics export |
| `data/Perspectives on Nature Based Solutions_November 12, 2025_03.16.csv` | Full Qualtrics CSV export |

Please cite the associated paper if you reuse these materials.
