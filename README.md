# UK Nature-based Solutions Farmer Survey — Article I

This folder is a GitHub-ready package for the Article I analysis:

**A Dual-Driver Framework for Scaling Nature-based Solutions in Agriculture: Evidence from the UK**

It keeps the original exploratory R analysis, the Qualtrics export files needed to rerun it, and selected manuscript tables/figures. Large Illustrator/TIFF figure masters and Word manuscripts are **not** included, because they are too large for GitHub and are not required to rerun the code.

## What this repository contains

```text
github-repo/
  README.md
  LICENSE
  .gitignore
  CITATION.cff
  requirements-r.txt
  code/
    NbS_TUSSIE.R
  data/
    Results_text.xlsx
    Results_values.xlsx
    Results_sep.xlsx
    Perspectives on Nature Based Solutions_November 12, 2025_03.16.csv
  outputs/
    tables/
    figures/
  docs/
    DATA_DICTIONARY.md
    NOTES.md
```

## Analysis overview

The survey asked UK farmers about:

- farm and household characteristics
- extreme weather experience
- climate change attribution
- funding and collaboration
- Nature-based Solutions (NbS) adoption at farm and landscape scales

The R script is the original dissertation/exploratory workflow. It:

1. Merges Qualtrics text labels (`Results_text.xlsx`) with split-coded items (`Results_sep.xlsx`).
2. Recodes funding, weather events, NbS adoption, neighbour effects, and perception items.
3. Produces descriptive plots of NbS adoption by exposure, attribution, income, education, and funding.
4. Runs correspondence analysis (CA) and multiple correspondence analysis (MCA).
5. Clusters farmers with k-means and hierarchical clustering on principal components (HCPC).
6. Fits logistic models for overall, individual, and collaborative NbS adoption.

The later manuscript also reports Python-based MCA/HCPC/SHAP figures. Those Python scripts were **not found** in `Article I` or the parent `NBS survey` folder. Only the R script and derived outputs are available here.

## How to run

### Requirements

- R 4.2+ (4.3/4.4 should also work)
- Packages listed in `requirements-r.txt`

Install packages in R:

```r
install.packages(c(
  "readxl", "dplyr", "ggplot2", "forcats", "FactoMineR",
  "factoextra", "gplots", "tidyverse", "stargazer", "corrplot"
))

# Optional, used for some older average-marginal-effect calls:
# install.packages("fastlogitME")
# install.packages("rstudioapi")
```

`fastlogitME` is not always on CRAN. The script still loads it at the top. If installation fails, comment out `library(fastlogitME)` unless you need those specific calls.

### Run the script

From the `github-repo` folder:

```r
source("code/NbS_TUSSIE.R")
```

Or in RStudio, open `code/NbS_TUSSIE.R` and source it. The script now looks for `data/` relative to the repository root. You do **not** need the original macOS `setwd()` path.

The script is exploratory: it prints tables and draws plots in the R graphics device. It does not currently write a complete set of manuscript figures to disk.

## Data files

| File | Role |
| --- | --- |
| `data/Results_text.xlsx` | Qualtrics text-coded responses. 212 rows, 65 columns. Used as the main labelled survey file. |
| `data/Results_sep.xlsx` | Split-coded multiple-response items (funding, farm activities, challenges). Merged by `ResponseId`. |
| `data/Results_values.xlsx` | Numeric-coded Qualtrics export. Loaded in the script but not used in the opening merge. |
| `data/Perspectives on Nature Based Solutions_November 12, 2025_03.16.csv` | Raw Qualtrics CSV export, including two metadata rows. Useful as the original backup export. |

Completed responses in `Results_text.xlsx`: **131** (`Finished == TRUE`). Incomplete/opened-only records remain in the file and are filtered in the script.

## Key recoded variables in `NbS_TUSSIE.R`

| Variable | Meaning |
| --- | --- |
| `NbSyes` | 1 if the respondent implements any NbS |
| `NbSindiv` | 1 if NbS is implemented individually |
| `NbScollab` | 1 if NbS is implemented with other farmers |
| `events` | 1 if any extreme weather event is reported |
| `EvFreq` | frequency of extreme weather, including "Never" |
| `CCattr` | attribution of events to climate change: Not at all / Partly / Mainly |
| `fundingPub` / `fundingPriv` | public vs private funding |
| `neighNbS` | neighbour implements NbS |
| `partner` | belongs to a community/farm partnership |

See `docs/DATA_DICTIONARY.md` for the original Qualtrics item IDs.

## Outputs included here

These are copied from the local Article I working folders so GitHub has the manuscript-facing tables and vector figures:

- descriptive statistics and average marginal effects CSVs
- MCA clustering summary
- robustness/sensitivity summaries
- SVG/PNG figures used in later manuscript drafts

Very large TIFF/AI files (some >100 MB, Figure 4 TIFF is ~858 MB) stay in the original `Article I/Figures` folder and are not copied here.

## Notes for GitHub

- Survey data include open-text answers. They do not include names or emails in the columns inspected here, but they are still human-subjects data. Prefer a **private** GitHub repository unless you have ethics approval to share the raw responses.
- If you later recover the Python MCA/HCPC/SHAP scripts, put them in `code/python/` and add a short note in this README.
- The original local R script remains unchanged at `Article I/Code/NbS_TUSSIE.R`.

## Suggested git commands

```bash
cd "D:/Work at KCL/NBS survey/Article I/github-repo"
git init
git add .
git commit -m "Add Article I survey analysis package"
git branch -M main
git remote add origin git@github.com:<ORG-OR-USER>/<REPO>.git
git push -u origin main
```

Replace `<ORG-OR-USER>/<REPO>` with your GitHub repository.
