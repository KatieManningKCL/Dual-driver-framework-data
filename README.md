# Dual-driver-framework-data

This repository contains code, anonymised survey data, and modelling resources supporting the analyses presented in **“A Dual-Driver Framework for Scaling Nature-Based Solutions in Agriculture: Evidence from the UK.”** It brings together the empirical and computational components used in the study, including survey-data processing, Multiple Correspondence Analysis, clustering, logistic regression, robustness analyses, figure generation, and environmental modelling.

The file **`NBS perspectives survey data.csv`** contains the anonymised survey dataset used to examine stakeholder perspectives, behavioural patterns, and potential drivers influencing the uptake and scaling of nature-based solutions (NBS) in agricultural landscapes.

The **`seasonal-soil-moisture-runoff-emulator/`** folder contains the reproducible workflow for a seasonal 1-km soil-moisture and runoff emulator developed for Great Britain. The emulator links hydrological behaviour with seasonal climate, land-use, soil, terrain, and hydroclimatic predictors, and is used to explore future changes under RCP2.6, RCP4.5, and RCP8.5 climate scenarios.

The emulator workflow includes data-preprocessing scripts, model-fitting and prediction procedures, validation routines, visualisation scripts, Slurm job files for HPC execution, and supporting documentation. It also includes comparisons with independent hydrological datasets and scenario-consistency checks to assess the physical plausibility of projected soil-moisture and runoff responses.

Together, these resources support the broader dual-driver framework by combining evidence on stakeholder and behavioural dimensions of NBS adoption with quantitative modelling of environmental and hydrological change. The repository is intended to improve transparency, reproducibility, and reuse of the analytical workflows associated with the study.
