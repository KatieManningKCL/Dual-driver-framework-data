# Packaging notes

## Local source folders

This GitHub package was assembled on 17 September 2026 from:

`D:\Work at KCL\NBS survey\Article I`

Original local folders that were **not** copied wholesale:

- `Code/` — original R script remains there; a path-fixed copy is in `code/NbS_TUSSIE.R`
- `Data/` — source Qualtrics exports; copies are in `data/`
- `Output_Manuscript_Analysis/`
- `Output_Results/`
- `Updated_Analysis_Results/`
- `Figures/` — includes very large TIFF masters, not suitable for GitHub
- Word manuscripts and Illustrator files

## Code inventory

Searched under `D:\Work at KCL\NBS survey` for `.R`, `.py`, `.ipynb`, `.Rmd`.

Found:

- `Article I/Code/NbS_TUSSIE.R` (this package)
- `Article I/Code/.Rhistory` (empty, not copied)

Not found:

- Python scripts for MCA/HCPC, logistic AMEs, LightGBM/XGBoost, SHAP, or robustness figures
- Decision-tree model code (only SVG/CSV outputs exist in `NBS survey/Decision tree`)

If those scripts are recovered from another machine or chat export, add them under `code/python/` or `code/decision-tree/`.

## Privacy

No email or name columns were found in the Excel exports. Open-text answers can still identify farms, clusters, or people. Keep the GitHub repository private unless sharing is approved.
