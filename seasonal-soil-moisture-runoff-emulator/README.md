# Seasonal Soil Moisture and Runoff Emulator for Great Britain

This repository contains the code used to build, run, validate and visualise a seasonal 1 km emulator for soil moisture and runoff-related variables across Great Britain under future RCP climate scenarios.

The work forms part of the **Joint Landscapes** project, an interdisciplinary project focused on multifunctional landscape futures, climate adaptation and nature-based solutions in the UK uplands and connected landscapes. More information about the wider project is available here: [Joint Landscapes project website](https://blogs.city.ac.uk/joineduplandscapes/).

## Project Overview

The emulator links historical hydrological behaviour with seasonal climate, land-use, soil, terrain and hydroclimatic predictors. It was trained using CHESS-land hydrological outputs and driven by harmonised CHESS-met, CHESS-SCAPE, PFT/land-use, PET, radiation/pressure, antecedent precipitation, terrain and soil-property datasets.

Five model families were tested, including linear, tree-based, neural-network and process-informed approaches. The preferred product is based on a **process-informed hybrid emulator**, selected because it provided the strongest balance between predictive performance, hydrological consistency and physically plausible seasonal responses.

The final outputs provide seasonal projections for:

- soil moisture
- surface runoff
- subsurface runoff
- total runoff

for 2020-2079 under RCP2.6, RCP4.5 and RCP8.5.

## Data Availability

The processed emulator outputs, key validation products, summary figures and metadata are available on Zenodo:

[https://doi.org/10.5281/zenodo.20788083](https://doi.org/10.5281/zenodo.20788083)

Large NetCDF outputs are not stored in this GitHub repository. This repository contains the reproducible code, workflow scripts and documentation needed to prepare inputs, run the emulator, generate figures and perform validation checks.

## Validation Summary

The workflow includes three complementary validation and plausibility checks:

1. **Scenario consistency validation**  
   Internal physical-consistency checks across RCP2.6, RCP4.5 and RCP8.5. These assess whether projected soil moisture and runoff changes follow plausible scenario ordering, seasonal behaviour and hydrological accounting.

2. **G2G validation**  
   External comparison with UKCEH Grid-to-Grid hydrological model outputs driven by UKCP18 Regional RCM simulations. This is strongest for RCP8.5 soil moisture, where both G2G and the emulator show late-century drying. River-flow comparison is interpreted as process-level hydrological evidence rather than direct local runoff validation.

3. **CDS/EURO-CORDEX hydrology consistency check**  
   External comparison with Copernicus CDS hydrology-related climate impact indicators derived from bias-adjusted EURO-CORDEX projections. This provides an additional runoff and soil-moisture consistency check across RCP2.6, RCP4.5 and RCP8.5 using independent 5 km hydrological indicators.

Together, these checks support the broad scenario ordering and high-emission signal of the emulator outputs, while also documenting uncertainties caused by differences in spatial resolution, hydrological model structure, variable definitions and comparison periods.

## Repository Structure

```text
scripts/
  data_preprocessing/   Data download and preprocessing scripts
  modeling/             Model input checks, model fitting and prediction scripts
  visualization/        Figure generation and summary plotting scripts
  validation/           Scenario, G2G and CDS/EURO-CORDEX validation scripts

slurm/                  Example KCL CREATE HPC Slurm job scripts
docs/                   Method notes and data-source documentation
```

## Main Workflow

1. Prepare seasonal 1 km predictor datasets from CHESS-met, CHESS-SCAPE, PFT/land-use, soil, terrain and hydroclimatic sources.
2. Prepare seasonal CHESS-land hydrological targets for soil moisture and runoff.
3. Train and compare five emulator families.
4. Run the selected process-informed hybrid emulator for 2020-2079 under RCP2.6, RCP4.5 and RCP8.5.
5. Generate annual, seasonal and spatial summary figures.
6. Validate the outputs using scenario consistency checks, G2G outputs and CDS/EURO-CORDEX hydrology indicators.

## Main Scripts

- `scripts/modeling/run_model_all_scenarios.py`  
  Runs model fitting and future prediction for all selected scenarios and members.

- `scripts/modeling/check_inputs_v2.py`  
  Checks the required seasonal NetCDF inputs before model execution.

- `scripts/modeling/check_reasonableness_v2.py`  
  Performs post-processing checks on predicted trends and hydrological consistency.

- `scripts/visualization/make_ppt_figures_all_rcps_by_model.py`  
  Generates publication/PPT-style figures comparing RCP trajectories and model outputs.

- `scripts/validation/run_scenario_consistency_validation.py`  
  Performs internal scenario-ordering, seasonal-response and runoff-accounting checks.

- `scripts/validation/process_g2g_validation_full_hpc.py`  
  Processes G2G soil moisture and river-flow validation NetCDF files into seasonal validation products.

- `scripts/validation/run_cds_eurocordex_validation.py`  
  Compares emulator outputs with CDS/EURO-CORDEX hydrology indicators for runoff, river discharge and soil moisture.

## HPC Usage

The Slurm scripts in `slurm/` are written for KCL CREATE HPC. Paths may need to be adapted for another environment.

Example model run:

```bash
cd /scratch/users/USER/emulator_seasonal/scripts/HPC_code_v2_5models
sbatch run_all_5models_array.slurm
```

Example validation runs:

```bash
sbatch slurm/run_g2g_validation_full_hpc.slurm
sbatch slurm/run_cds_eurocordex_validation.slurm
```

## Python Environment

The code was run with Python 3.11. Key packages include:

- numpy
- pandas
- scipy
- scikit-learn
- lightgbm
- xgboost
- netCDF4
- xarray
- matplotlib
- seaborn
- joblib
- tqdm
- psutil
- pygam
- shap

See `requirements.txt` for a draft environment specification.

## Important Notes

- Raw source datasets are not redistributed in this repository.
- Users should obtain raw datasets from the original providers and follow their licensing terms.
- Validation outputs should be interpreted according to variable definition and spatial-scale differences.
- G2G river flow and CDS river discharge are not identical to local grid-cell total runoff.

## License

The code is released under the MIT License. Check upstream data licences before redistributing processed datasets.
