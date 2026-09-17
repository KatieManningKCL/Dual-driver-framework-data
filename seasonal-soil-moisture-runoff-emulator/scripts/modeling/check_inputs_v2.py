from __future__ import annotations

from pathlib import Path
import argparse

from netCDF4 import Dataset


REQUIRED = [
    "01_historical_climate/CHESS-met_gb_1km_seasonal_1961-2019.nc",
    "02_historical_targets/CHESS-land_HydEn_gb_1km_seasonal_targets_1961-2015.nc",
    "03_historical_pft/CHESSLand_PFT_1961_2015_1km_annual.nc",
    "04_future_climate/CHESS-SCAPE_uk_1km_seasonal_1980-2080_by_scenario_member.nc",
    "05_future_aft_to_pft/CRAFTY_AFT_to_PFT_RCP2_6_SSP1_2020_2079_1km.nc",
    "05_future_aft_to_pft/CRAFTY_AFT_to_PFT_RCP4_5_SSP2_2020_2079_1km.nc",
    "05_future_aft_to_pft/CRAFTY_AFT_to_PFT_RCP8_5_SSP5_2020_2079_1km.nc",
    "06_historical_pet/HydroPE_CHESS_GB_1km_seasonal_PET_PETI_1961_2019.nc",
    "07_future_radiation_pressure/CHESS-SCAPE_uk_1km_seasonal_rsds_psurf_1980-2080_by_scenario_member.nc",
    "08_historical_precip_antecedent/CEH-GEAR_GB_1km_seasonal_precip_from_monthly_1961_2019.nc",
    "09_static_terrain/OS_Terrain50_CHESS_1km_terrain_features.nc",
    "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
]

EXPECTED_VARIABLES = {
    "01_historical_climate/CHESS-met_gb_1km_seasonal_1961-2019.nc": [
        "season_year", "season_code", "precip", "tas", "huss", "hurs", "tasmax", "tasmin", "rlds", "sfcWind", "psurf", "dtr"
    ],
    "02_historical_targets/CHESS-land_HydEn_gb_1km_seasonal_targets_1961-2015.nc": [
        "season_year", "season_code", "landpoint", "latitude", "longitude", "soil_moisture", "surface_runoff", "subsurface_runoff"
    ],
    "03_historical_pft/CHESSLand_PFT_1961_2015_1km_annual.nc": [
        "year", "landpoint", "eastings", "northings", "le", "gpp"
    ],
    "04_future_climate/CHESS-SCAPE_uk_1km_seasonal_1980-2080_by_scenario_member.nc": [
        "hurs", "huss", "pr", "rlds", "sfcWind", "tas", "tasmax", "tasmin"
    ],
    "06_historical_pet/HydroPE_CHESS_GB_1km_seasonal_PET_PETI_1961_2019.nc": [
        "season_year", "season_code", "pet_seasonal_total", "peti_seasonal_total"
    ],
    "07_future_radiation_pressure/CHESS-SCAPE_uk_1km_seasonal_rsds_psurf_1980-2080_by_scenario_member.nc": [
        "rsds", "psurf"
    ],
    "08_historical_precip_antecedent/CEH-GEAR_GB_1km_seasonal_precip_from_monthly_1961_2019.nc": [
        "season_year", "season_code", "precip_total", "precip_mean_monthly", "precip_max_monthly", "antecedent_precip_total", "antecedent_precip_max_monthly"
    ],
    "09_static_terrain/OS_Terrain50_CHESS_1km_terrain_features.nc": [
        "elevation_mean", "elevation_sd", "elevation_range"
    ],
    "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc": [
        "clay_0_30cm", "silt_0_30cm", "sand_0_30cm", "bulk_density_0_30cm", "coarse_fragments_0_30cm", "soil_organic_carbon_0_30cm", "soil_water_holding_proxy"
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/scratch/users/k2585234/emulator_seasonal")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data = root / "HPC_input_data"
    print(f"Checking input folder: {data}")
    if not data.exists():
        raise SystemExit(f"Missing input folder: {data}")

    for rel in REQUIRED:
        path = data / rel
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")
        size_gb = path.stat().st_size / 1024**3
        with Dataset(path) as ds:
            dims = {k: len(v) for k, v in ds.dimensions.items()}
            variables = list(ds.variables)[:15]
            missing = [v for v in EXPECTED_VARIABLES.get(rel, []) if v not in ds.variables]
            if missing:
                raise SystemExit(f"Missing variables in {path}: {missing}")
        print(f"OK: {rel} ({size_gb:.3f} GB)")
        print(f"    dims: {dims}")
        print(f"    first variables: {variables}")

    print("Input check passed.")


if __name__ == "__main__":
    main()
