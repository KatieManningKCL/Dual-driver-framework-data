# Data Sources

This project uses processed derivatives of several UK hydroclimate and environmental datasets. Raw datasets should be obtained from the original providers.

## Historical Climate

- CHESS-met historical climate data, processed to seasonal 1 km predictors.

## Historical Hydrological Targets

- CHESS-land / Hydro-ecological outputs, processed to seasonal soil moisture and runoff targets.

## Future Climate

- CHESS-SCAPE future climate data, processed to seasonal 1 km predictors for RCP2.6, RCP4.5 and RCP8.5.

## Land Use / PFT

- Historical and future land-use information converted to Plant Functional Type fractions.

## Additional Predictors

- Hydro-PE potential evapotranspiration.
- CEH-GEAR precipitation-derived seasonal/antecedent metrics.
- OS Terrain 50 terrain features resampled to the CHESS 1 km grid.
- SoilGrids soil property predictors aggregated to the CHESS 1 km grid.

## External Validation

- UKCEH/EIDC G2G soil moisture driven by UKCP18 Regional RCM.
- UKCEH/EIDC G2G river flow driven by UKCP18 Regional RCM.
- Copernicus CDS hydrology-related climate impact indicators derived from bias-adjusted EURO-CORDEX projections.

G2G river flow and CDS river discharge are not identical to grid-cell total runoff and should be interpreted as process-level hydrological consistency evidence.

## Processed Output Archive

Processed emulator outputs, validation products, figures and metadata are archived on Zenodo:

https://doi.org/10.5281/zenodo.20788083
