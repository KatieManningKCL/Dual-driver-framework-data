from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", message="X does not have valid feature names")


SCENARIOS = ["rcp26", "rcp45", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
SEASON_CODES = np.array([1, 2, 3, 4], dtype=np.int8)
SEASON_NAMES = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}
SCENARIO_INDEX = {"rcp26": 0, "rcp45": 1, "rcp85": 2}
MEMBER_INDEX = {"01": 0, "04": 1, "06": 2, "15": 3}

PFT_FILE_BY_SCENARIO = {
    "rcp26": "CRAFTY_AFT_to_PFT_RCP2_6_SSP1_2020_2079_1km.nc",
    "rcp45": "CRAFTY_AFT_to_PFT_RCP4_5_SSP2_2020_2079_1km.nc",
    "rcp85": "CRAFTY_AFT_to_PFT_RCP8_5_SSP5_2020_2079_1km.nc",
}

PFT_NAMES = [
    "pft_broad_leaf_tree",
    "pft_needle_leaf_tree",
    "pft_grass",
    "pft_crops",
    "pft_shrubs",
    "pft_bare_soil",
]

STATIC_GRID_VARS = {
    "elevation_mean": "09_static_terrain/OS_Terrain50_CHESS_1km_terrain_features.nc",
    "elevation_sd": "09_static_terrain/OS_Terrain50_CHESS_1km_terrain_features.nc",
    "elevation_range": "09_static_terrain/OS_Terrain50_CHESS_1km_terrain_features.nc",
    "clay_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "silt_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "sand_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "bulk_density_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "coarse_fragments_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "soil_organic_carbon_0_30cm": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
    "soil_water_holding_proxy": "10_static_soil_properties/SoilGrids_CHESS_1km_soil_properties_0_30cm.nc",
}

STATIC_NAMES = [
    "eastings",
    "northings",
    "latitude",
    "longitude",
    *STATIC_GRID_VARS.keys(),
    "season_sin",
    "season_cos",
    "year_norm",
]

DYNAMIC_NAMES = [
    "precip_flux",
    "tas",
    "huss",
    "hurs",
    "tasmax",
    "tasmin",
    "rlds",
    "sfcWind",
    "psurf",
    "dtr",
    "pet_total",
    "peti_total",
    "precip_total",
    "precip_mean_monthly",
    "precip_max_monthly",
    "antecedent_precip_total",
    "antecedent_precip_max_monthly",
    "aridity_pet_minus_precip",
]

LOG_RATIO_FEATURES = {
    "precip_flux",
    "huss",
    "sfcWind",
    "pet_total",
    "peti_total",
    "precip_total",
    "precip_mean_monthly",
    "precip_max_monthly",
    "antecedent_precip_total",
    "antecedent_precip_max_monthly",
}

TARGETS = ["soil_moisture", "surface_runoff", "subsurface_runoff"]
OUTPUT_TARGETS = ["soil_moisture", "surface_runoff", "subsurface_runoff", "total_runoff"]
TARGET_SCALE = {"soil_moisture": 1.0, "surface_runoff": 86400.0, "subsurface_runoff": 86400.0}
TARGET_UNITS = {
    "soil_moisture": "kg m-2",
    "surface_runoff": "kg m-2 s-1",
    "subsurface_runoff": "kg m-2 s-1",
    "total_runoff": "kg m-2 s-1",
}


def log(message: str) -> None:
    print(message, flush=True)


def clean_values(values) -> np.ndarray:
    arr = np.ma.filled(values, np.nan).astype(np.float32)
    arr = np.where(arr > 9.0e19, np.nan, arr)
    arr = np.where(arr < -9.0e19, np.nan, arr)
    return arr.astype(np.float32)


def season_index(code: int) -> int:
    return int(code) - 1


def future_time_index(year: int, season_code: int) -> int:
    return int((int(year) - 1981) * 4 + season_index(season_code))


def days_in_season(year: int, season_code: int) -> int:
    if season_code == 1:
        return 91 if (year + 1) % 4 == 0 else 90
    if season_code == 2:
        return 92
    if season_code == 3:
        return 92
    return 91


def season_features(code: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * season_index(code) / 4.0
    return (
        np.full(n, np.sin(angle), dtype=np.float32),
        np.full(n, np.cos(angle), dtype=np.float32),
    )


def time_lookup(ds: Dataset) -> dict[tuple[int, int], int]:
    years = np.asarray(ds.variables["season_year"][:], dtype=np.int16)
    codes = np.asarray(ds.variables["season_code"][:], dtype=np.int8)
    return {(int(y), int(c)): i for i, (y, c) in enumerate(zip(years, codes))}


def load_landpoints(target_ds: Dataset, climate_ds: Dataset) -> dict[str, np.ndarray]:
    landpoint = np.asarray(target_ds.variables["landpoint"][:], dtype=np.int64)
    x_coords = np.asarray(climate_ds.variables["x"][:], dtype=np.float32)
    y_coords = np.asarray(climate_ds.variables["y"][:], dtype=np.float32)
    flat0 = landpoint - 1
    yi = (flat0 // len(x_coords)).astype(np.int32)
    xi = (flat0 % len(x_coords)).astype(np.int32)
    return {
        "landpoint": landpoint.astype(np.int32),
        "yi": yi,
        "xi": xi,
        "eastings": x_coords[xi].astype(np.float32),
        "northings": y_coords[yi].astype(np.float32),
        "latitude": np.asarray(target_ds.variables["latitude"][:], dtype=np.float32),
        "longitude": np.asarray(target_ds.variables["longitude"][:], dtype=np.float32),
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


def grid_to_land(ds: Dataset, var_name: str, land: dict[str, np.ndarray]) -> np.ndarray:
    grid = clean_values(ds.variables[var_name][:])
    return grid[land["yi"], land["xi"]].astype(np.float32)


def read_grid_points(ds: Dataset, var_name: str, time_idx: int, points: np.ndarray, land: dict[str, np.ndarray]) -> np.ndarray:
    grid = clean_values(ds.variables[var_name][time_idx, :, :])
    return grid[land["yi"][points], land["xi"][points]].astype(np.float32)


def nearest_indices(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(source_coords, dtype=np.float64)
    targets = np.asarray(target_coords, dtype=np.float64)
    if coords[0] > coords[-1]:
        rev = coords[::-1]
        idx = np.searchsorted(rev, targets)
        idx = np.clip(idx, 1, len(rev) - 1)
        left = rev[idx - 1]
        right = rev[idx]
        nearest = np.where(np.abs(targets - left) <= np.abs(targets - right), idx - 1, idx)
        return (len(coords) - 1 - nearest).astype(np.int64)
    idx = np.searchsorted(coords, targets)
    idx = np.clip(idx, 1, len(coords) - 1)
    left = coords[idx - 1]
    right = coords[idx]
    nearest = np.where(np.abs(targets - left) <= np.abs(targets - right), idx - 1, idx)
    return nearest.astype(np.int64)


def build_grid_index(ds: Dataset, land: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map CHESS landpoint coordinates onto an external regular 1 km grid."""
    x_idx = nearest_indices(np.asarray(ds.variables["x"][:]), land["eastings"])
    y_idx = nearest_indices(np.asarray(ds.variables["y"][:]), land["northings"])
    return {"xi": x_idx.astype(np.int64), "yi": y_idx.astype(np.int64)}


def read_grid_points_indexed(ds: Dataset, var_name: str, time_idx: int, points: np.ndarray, grid_index: dict[str, np.ndarray]) -> np.ndarray:
    grid = clean_values(ds.variables[var_name][time_idx, :, :])
    return grid[grid_index["yi"][points], grid_index["xi"][points]].astype(np.float32)


def read_future_grid_points(
    ds: Dataset,
    var_name: str,
    scenario: str,
    member: str,
    time_idx: int,
    points: np.ndarray,
    land: dict[str, np.ndarray],
) -> np.ndarray:
    grid = clean_values(ds.variables[var_name][SCENARIO_INDEX[scenario], MEMBER_INDEX[member], time_idx, :, :])
    return grid[land["yi"][points], land["xi"][points]].astype(np.float32)


def read_future_derived(
    ctx: dict,
    name: str,
    scenario: str,
    member: str,
    year: int,
    code: int,
    points: np.ndarray,
) -> np.ndarray:
    t = future_time_index(year, code)
    fut = ctx["future_climate"]
    extra = ctx["future_extra"]
    land = ctx["land"]
    if name == "precip_flux":
        return read_future_grid_points(fut, "pr", scenario, member, t, points, land)
    if name in {"tas", "huss", "hurs", "tasmax", "tasmin", "rlds", "sfcWind"}:
        return read_future_grid_points(fut, name, scenario, member, t, points, land)
    if name == "psurf":
        return read_future_grid_points(extra, "psurf", scenario, member, t, points, land)
    if name == "dtr":
        return (
            read_future_grid_points(fut, "tasmax", scenario, member, t, points, land)
            - read_future_grid_points(fut, "tasmin", scenario, member, t, points, land)
        ).astype(np.float32)
    if name in {"precip_total", "precip_mean_monthly", "precip_max_monthly", "antecedent_precip_total", "antecedent_precip_max_monthly"}:
        pr = read_future_grid_points(fut, "pr", scenario, member, t, points, land)
        total = pr * float(days_in_season(year, code) * 86400.0)
        if name == "precip_total":
            return total.astype(np.float32)
        if name == "precip_mean_monthly":
            return (total / 3.0).astype(np.float32)
        if name == "precip_max_monthly":
            return (total / 3.0).astype(np.float32)
        prev_year, prev_code = previous_season(year, code)
        if prev_year < 2020:
            base = ctx["dynamic_baseline"][season_index(code), DYNAMIC_NAMES.index("antecedent_precip_total"), points]
            return base.astype(np.float32)
        prev_t = future_time_index(prev_year, prev_code)
        prev_pr = read_future_grid_points(fut, "pr", scenario, member, prev_t, points, land)
        prev_total = prev_pr * float(days_in_season(prev_year, prev_code) * 86400.0)
        if name == "antecedent_precip_total":
            return prev_total.astype(np.float32)
        return (prev_total / 3.0).astype(np.float32)
    if name in {"pet_total", "peti_total"}:
        rsds = read_future_grid_points(extra, "rsds", scenario, member, t, points, land)
        tas = read_future_grid_points(fut, "tas", scenario, member, t, points, land) - 273.15
        dtr = np.maximum(
            read_future_grid_points(fut, "tasmax", scenario, member, t, points, land)
            - read_future_grid_points(fut, "tasmin", scenario, member, t, points, land),
            0.0,
        )
        radiation_mj_day = np.maximum(rsds, 0.0) * 0.0864
        et0_day = 0.0023 * np.maximum(tas + 17.8, 0.0) * np.sqrt(dtr + 1.0e-6) * radiation_mj_day
        pet = et0_day * float(days_in_season(year, code))
        if name == "peti_total":
            pet = pet * 0.95
        return pet.astype(np.float32)
    if name == "aridity_pet_minus_precip":
        return (
            read_future_derived(ctx, "peti_total", scenario, member, year, code, points)
            - read_future_derived(ctx, "precip_total", scenario, member, year, code, points)
        ).astype(np.float32)
    raise KeyError(name)


def previous_season(year: int, code: int) -> tuple[int, int]:
    if code == 1:
        return year - 1, 4
    return year, code - 1


def read_hist_dynamic(ctx: dict, name: str, year: int, code: int, points: np.ndarray) -> np.ndarray:
    land = ctx["land"]
    if name in {"precip_flux", "tas", "huss", "hurs", "tasmax", "tasmin", "rlds", "sfcWind", "psurf", "dtr"}:
        var = "precip" if name == "precip_flux" else name
        t = ctx["hist_climate_lookup"][(year, code)]
        return read_grid_points(ctx["hist_climate"], var, t, points, land)
    if name == "pet_total":
        t = ctx["hist_pet_lookup"][(year, code)]
        return read_grid_points(ctx["hist_pet"], "pet_seasonal_total", t, points, land)
    if name == "peti_total":
        t = ctx["hist_pet_lookup"][(year, code)]
        return read_grid_points(ctx["hist_pet"], "peti_seasonal_total", t, points, land)
    if name in {"precip_total", "precip_mean_monthly", "precip_max_monthly", "antecedent_precip_total", "antecedent_precip_max_monthly"}:
        t = ctx["hist_precip_lookup"][(year, code)]
        return read_grid_points_indexed(ctx["hist_precip"], name, t, points, ctx["hist_precip_grid_index"])
    if name == "aridity_pet_minus_precip":
        return (
            read_hist_dynamic(ctx, "peti_total", year, code, points)
            - read_hist_dynamic(ctx, "precip_total", year, code, points)
        ).astype(np.float32)
    raise KeyError(name)


def validate_historical_pft_alignment(pft_ds: Dataset, land: dict[str, np.ndarray]) -> np.ndarray:
    pft_landpoint = np.asarray(pft_ds.variables["landpoint"][:], dtype=np.int64)
    target_landpoint = land["landpoint"].astype(np.int64)
    if len(pft_landpoint) == len(target_landpoint) and np.array_equal(pft_landpoint, target_landpoint):
        return np.arange(len(target_landpoint), dtype=np.int64)
    lookup = {int(lp): i for i, lp in enumerate(pft_landpoint)}
    idx = np.array([lookup.get(int(lp), -1) for lp in target_landpoint], dtype=np.int64)
    if np.any(idx < 0):
        raise RuntimeError(f"Historical PFT file is missing {int(np.sum(idx < 0))} target landpoints.")
    return idx


def read_pft_share_from_vars(le_var, gpp_var, year_idx: int, pft_idx: np.ndarray) -> np.ndarray:
    le = clean_values(le_var[year_idx, :, pft_idx])
    gpp = clean_values(gpp_var[year_idx, :, pft_idx])
    activity = np.where(np.isfinite(le) & (le > 0.0), le, 0.0)
    denom = np.sum(activity, axis=0)
    shares = np.full(activity.shape, np.nan, dtype=np.float32)
    np.divide(activity, denom, out=shares, where=denom > 0.0)
    missing = ~(denom > 0.0)
    if np.any(missing):
        gpp_activity = np.where(np.isfinite(gpp[:, missing]) & (gpp[:, missing] > 0.0), gpp[:, missing], 0.0)
        gpp_denom = np.sum(gpp_activity, axis=0)
        fallback = np.zeros((6, int(np.sum(missing))), dtype=np.float32)
        np.divide(gpp_activity, gpp_denom, out=fallback, where=gpp_denom > 0.0)
        fallback[5, gpp_denom <= 0.0] = 1.0
        shares[:, missing] = fallback
    shares = np.nan_to_num(shares, nan=0.0, posinf=0.0, neginf=0.0)
    sums = np.sum(shares, axis=0)
    np.divide(shares, sums, out=shares, where=sums > 0.0)
    shares[5, sums <= 0.0] = 1.0
    return shares.T.astype(np.float32)


def hist_pft_for_year(ctx: dict, year: int, points: np.ndarray) -> np.ndarray:
    pft_ds = ctx["hist_pft"]
    years = np.asarray(pft_ds.variables["year"][:], dtype=np.int16)
    yidx = int(np.where(years == year)[0][0])
    pft_idx = ctx["hist_pft_index"][points]
    return read_pft_share_from_vars(pft_ds.variables["le"], pft_ds.variables["gpp"], yidx, pft_idx)


def compute_hist_pft_baseline(ctx: dict, years: np.ndarray) -> np.ndarray:
    log("Computing historical PFT baseline...")
    n = len(ctx["land"]["landpoint"])
    acc = np.zeros((n, 6), dtype=np.float64)
    for year in years:
        pts = np.arange(n, dtype=np.int64)
        acc += hist_pft_for_year(ctx, int(year), pts).astype(np.float64)
    return (acc / float(len(years))).astype(np.float32)


def build_future_pft_index(pft_ds: Dataset, land: dict[str, np.ndarray]) -> np.ndarray:
    east = np.asarray(pft_ds.variables["eastings"][:], dtype=np.int64)
    north = np.asarray(pft_ds.variables["northings"][:], dtype=np.int64)
    lookup = {(int(e), int(n)): i for i, (e, n) in enumerate(zip(east, north))}
    idx = np.full(len(land["landpoint"]), -1, dtype=np.int64)
    for i, (e, n) in enumerate(zip(land["eastings"], land["northings"])):
        idx[i] = lookup.get((int(e), int(n)), -1)
    return idx


def future_pft_for_year(pft_ds: Dataset, pft_index: np.ndarray, pft_baseline: np.ndarray, year: int, points: np.ndarray) -> np.ndarray:
    years = np.asarray(pft_ds.variables["year"][:], dtype=np.int16)
    yidx = int(np.where(years == year)[0][0])
    src_idx = pft_index[points]
    out = pft_baseline[points, :].copy()
    valid = src_idx >= 0
    if np.any(valid):
        vals = clean_values(pft_ds.variables["pft_proportion"][yidx, :, src_idx[valid]]).T
        vals = np.nan_to_num(vals, nan=0.0)
        sums = vals.sum(axis=1)
        vals[sums <= 0.0, 5] = 1.0
        sums = vals.sum(axis=1)
        vals = vals / np.maximum(sums[:, None], 1.0e-6)
        out[valid, :] = vals.astype(np.float32)
    return out.astype(np.float32)


def compute_dynamic_baseline(ctx: dict, baseline_years: np.ndarray) -> np.ndarray:
    log("Computing dynamic historical baselines...")
    n = len(ctx["land"]["landpoint"])
    baseline = np.full((4, len(DYNAMIC_NAMES), n), np.nan, dtype=np.float32)
    full_points = np.arange(n, dtype=np.int64)
    for code in SEASON_CODES:
        for f_idx, name in enumerate(DYNAMIC_NAMES):
            acc = np.zeros(n, dtype=np.float64)
            count = np.zeros(n, dtype=np.int16)
            for year in baseline_years:
                key = (int(year), int(code))
                if key not in ctx["hist_climate_lookup"] or key not in ctx["hist_pet_lookup"] or key not in ctx["hist_precip_lookup"]:
                    continue
                vals = read_hist_dynamic(ctx, name, int(year), int(code), full_points).astype(np.float64)
                ok = np.isfinite(vals)
                acc[ok] += vals[ok]
                count[ok] += 1
            np.divide(acc, count, out=baseline[season_index(code), f_idx, :], where=count > 0)
    return baseline


def compute_target_baseline(ctx: dict, baseline_years: np.ndarray) -> dict[str, np.ndarray]:
    log("Computing target baselines...")
    target_ds = ctx["hist_target"]
    years = np.asarray(target_ds.variables["season_year"][:], dtype=np.int16)
    codes = np.asarray(target_ds.variables["season_code"][:], dtype=np.int8)
    out = {}
    for target in TARGETS:
        arr = np.full((4, len(ctx["land"]["landpoint"])), np.nan, dtype=np.float32)
        var = target_ds.variables[target]
        for code in SEASON_CODES:
            mask = np.isin(years, baseline_years) & (codes == code)
            time_indices = np.where(mask)[0]
            vals = clean_values(var[time_indices, :])
            arr[season_index(code), :] = np.nanmean(vals, axis=0).astype(np.float32)
        out[target] = arr
    return out


def build_static_matrix(ctx: dict) -> np.ndarray:
    land = ctx["land"]
    cols = [
        land["eastings"].astype(np.float32),
        land["northings"].astype(np.float32),
        land["latitude"].astype(np.float32),
        land["longitude"].astype(np.float32),
    ]
    opened = {}
    try:
        for name, rel in STATIC_GRID_VARS.items():
            if rel not in opened:
                opened[rel] = Dataset(ctx["data_dir"] / rel)
            cols.append(grid_to_land(opened[rel], name, land))
    finally:
        for ds in opened.values():
            ds.close()
    return np.column_stack(cols).astype(np.float32)


def dynamic_anomaly(raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw, dtype=np.float32)
    for i, name in enumerate(DYNAMIC_NAMES):
        vals = raw[:, i].astype(np.float32)
        ref = baseline[:, i].astype(np.float32)
        if name in LOG_RATIO_FEATURES:
            eps = 1.0e-6
            out[:, i] = np.log((np.maximum(vals, 0.0) + eps) / (np.maximum(ref, 0.0) + eps)).astype(np.float32)
        else:
            out[:, i] = (vals - ref).astype(np.float32)
    return out


def feature_names() -> list[str]:
    return STATIC_NAMES + [f"delta_{n}" for n in DYNAMIC_NAMES] + [f"delta_{n}" for n in PFT_NAMES]


def build_features_for_group(
    ctx: dict,
    points: np.ndarray,
    year: int,
    code: int,
    source: str,
    scenario: str | None = None,
    member: str | None = None,
) -> np.ndarray:
    n = len(points)
    season_sin, season_cos = season_features(code, n)
    static = ctx["static_matrix"][points, :]
    time_cols = np.column_stack([
        season_sin,
        season_cos,
        np.full(n, (year - 1990) / 50.0, dtype=np.float32),
    ])
    dyn_raw = np.column_stack([
        read_hist_dynamic(ctx, name, year, code, points)
        if source == "historical"
        else read_future_derived(ctx, name, scenario or "rcp85", member or "01", year, code, points)
        for name in DYNAMIC_NAMES
    ]).astype(np.float32)
    dyn_base = ctx["dynamic_baseline"][season_index(code), :, points]
    if dyn_base.shape[0] == len(DYNAMIC_NAMES):
        dyn_base = dyn_base.T
    dyn_base = dyn_base.astype(np.float32)
    dyn_delta = dynamic_anomaly(dyn_raw, dyn_base)
    if source == "historical":
        pft = hist_pft_for_year(ctx, year, points)
    else:
        pft_ds = ctx["future_pft"][scenario or "rcp85"]
        pft_idx = ctx["future_pft_index"][scenario or "rcp85"]
        pft = future_pft_for_year(pft_ds, pft_idx, ctx["pft_baseline"], year, points)
    pft_delta = (pft - ctx["pft_baseline"][points, :]).astype(np.float32)
    return np.column_stack([static, time_cols, dyn_delta, pft_delta]).astype(np.float32)


def sample_rows(ctx: dict, time_indices: np.ndarray, n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    target_ds = ctx["hist_target"]
    years = np.asarray(target_ds.variables["season_year"][:], dtype=np.int16)
    codes = np.asarray(target_ds.variables["season_code"][:], dtype=np.int8)
    n_land = len(ctx["land"]["landpoint"])
    rows_time = rng.choice(time_indices, size=n_rows, replace=True)
    rows_point = rng.integers(0, n_land, size=n_rows, dtype=np.int64)

    X_parts = []
    order_parts = []
    for t in np.unique(rows_time):
        mask = rows_time == t
        pts = rows_point[mask]
        X_parts.append(build_features_for_group(ctx, pts, int(years[t]), int(codes[t]), "historical"))
        order_parts.append(np.where(mask)[0])
    X = np.empty((n_rows, len(feature_names())), dtype=np.float32)
    for part, idx in zip(X_parts, order_parts):
        X[idx, :] = part

    y = {}
    for target in TARGETS:
        vals = np.empty(n_rows, dtype=np.float32)
        var = target_ds.variables[target]
        for t in np.unique(rows_time):
            mask = rows_time == t
            vals[mask] = clean_values(var[int(t), rows_point[mask]])
        base = ctx["target_baseline"][target][np.array([season_index(c) for c in codes[rows_time]]), rows_point]
        y[target] = ((vals - base) * TARGET_SCALE[target]).astype(np.float32)
    return X, rows_point, y


def impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return med


def impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(X)
    if np.any(bad):
        X = X.copy()
        X[bad] = np.take(med, np.where(bad)[1])
    return X.astype(np.float32)


def finite_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask, :], y[mask]


def make_estimator(model_name: str, target: str, seed: int, threads: int):
    if model_name == "elastic_net":
        alpha = 0.002 if target == "soil_moisture" else 0.0008
        return make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=0.15, max_iter=4000, random_state=seed))
    if model_name == "forest":
        return ExtraTreesRegressor(
            n_estimators=360,
            max_features=0.72,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=threads,
        )
    if model_name == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=63 if target == "soil_moisture" else 47,
            min_child_samples=35,
            subsample=0.86,
            subsample_freq=1,
            colsample_bytree=0.86,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        )
    if model_name == "mlp_stack":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1.0e-4,
                batch_size=4096,
                learning_rate_init=8.0e-4,
                max_iter=90,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=8,
                random_state=seed,
                verbose=False,
            ),
        )
    raise ValueError(model_name)


def fit_process_hybrid(X_train, y_train, X_val, y_val, target: str, seed: int, threads: int) -> dict:
    lgbm = make_estimator("lightgbm", target, seed, threads)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=900,
        max_depth=7 if target == "soil_moisture" else 6,
        learning_rate=0.025,
        subsample=0.86,
        colsample_bytree=0.86,
        reg_lambda=1.0,
        min_child_weight=8,
        random_state=seed,
        n_jobs=threads,
        tree_method="hist",
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=60.0 if target == "soil_moisture" else 20.0))
    lgbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    pred_l = lgbm.predict(X_val).astype(np.float32)
    pred_x = xgb.predict(X_val).astype(np.float32)
    pred_r = ridge.predict(X_val).astype(np.float32)
    best = (math.inf, 0.5, 0.45, 0.05)
    for wl in np.linspace(0.35, 0.7, 8):
        for wx in np.linspace(0.2, 0.6, 9):
            wr = 1.0 - wl - wx
            if wr < 0.0 or wr > 0.25:
                continue
            pred = wl * pred_l + wx * pred_x + wr * pred_r
            rmse = math.sqrt(mean_squared_error(y_val, pred))
            if rmse < best[0]:
                best = (rmse, float(wl), float(wx), float(wr))
    return {"lgbm": lgbm, "xgb": xgb, "ridge": ridge, "weights": best[1:]}


def predict_model(model_name: str, model, X: np.ndarray) -> np.ndarray:
    if model_name == "process_hybrid":
        wl, wx, wr = model["weights"]
        return (
            wl * model["lgbm"].predict(X).astype(np.float32)
            + wx * model["xgb"].predict(X).astype(np.float32)
            + wr * model["ridge"].predict(X).astype(np.float32)
        ).astype(np.float32)
    return model.predict(X).astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def create_output_nc(path: Path, ctx: dict, model_name: str, start_year: int, end_year: int) -> None:
    if path.exists():
        path.unlink()
    n_time = (end_year - start_year + 1) * 4
    land = ctx["land"]
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("scenario", len(SCENARIOS))
        ds.createDimension("member", len(MEMBERS))
        ds.createDimension("time", n_time)
        ds.createDimension("landpoint", len(land["landpoint"]))
        scen = ds.createVariable("scenario", "i2", ("scenario",))
        scen[:] = [26, 45, 85]
        scen.labels = "rcp26,rcp45,rcp85"
        mem = ds.createVariable("member", "i2", ("member",))
        mem[:] = [1, 4, 6, 15]
        mem.labels = "01,04,06,15"
        tv = ds.createVariable("time", "i4", ("time",))
        yv = ds.createVariable("year", "i2", ("time",))
        sv = ds.createVariable("season_code", "i1", ("time",))
        years = []
        codes = []
        for year in range(start_year, end_year + 1):
            for code in SEASON_CODES:
                years.append(year)
                codes.append(int(code))
        tv[:] = np.arange(n_time, dtype=np.int32)
        yv[:] = years
        sv[:] = codes
        lp = ds.createVariable("landpoint", "i4", ("landpoint",))
        lp[:] = land["landpoint"]
        lat = ds.createVariable("latitude", "f4", ("landpoint",), zlib=True, complevel=4)
        lon = ds.createVariable("longitude", "f4", ("landpoint",), zlib=True, complevel=4)
        lat[:] = land["latitude"]
        lon[:] = land["longitude"]
        chunk = (1, 1, 1, min(50000, len(land["landpoint"])))
        for target in OUTPUT_TARGETS:
            v = ds.createVariable(
                target,
                "f4",
                ("scenario", "member", "time", "landpoint"),
                zlib=True,
                complevel=4,
                chunksizes=chunk,
                fill_value=np.float32(np.nan),
            )
            v.units = TARGET_UNITS[target]
            v.long_name = f"Predicted seasonal {target} from {model_name} emulator"
        ds.model_name = model_name
        ds.processing = "Predictions produced by process-informed seasonal emulator using climate, PET, PFT, precipitation-memory, terrain and soil predictors."


def reconstruct(change: np.ndarray, target: str, base: np.ndarray) -> np.ndarray:
    vals = base.astype(np.float32) + change.astype(np.float32) / TARGET_SCALE[target]
    return np.maximum(vals, 0.0).astype(np.float32)


def train_models(ctx: dict, args) -> tuple[dict, dict, np.ndarray]:
    target_ds = ctx["hist_target"]
    years = np.asarray(target_ds.variables["season_year"][:], dtype=np.int16)
    train_times = np.where((years >= args.train_start) & (years <= args.train_end))[0]
    val_times = np.where((years >= args.val_start) & (years <= args.val_end))[0]

    log(f"Sampling training rows: {args.train_rows}")
    X_train, _, y_train_all = sample_rows(ctx, train_times, args.train_rows, args.seed)
    log(f"Sampling validation rows: {args.val_rows}")
    X_val, _, y_val_all = sample_rows(ctx, val_times, args.val_rows, args.seed + 100)
    med = impute_fit(X_train)
    X_train = impute_apply(X_train, med)
    X_val = impute_apply(X_val, med)

    models = {}
    validation = {}
    for target in TARGETS:
        log(f"Training {args.model}: {target}")
        yt = y_train_all[target]
        yv = y_val_all[target]
        Xtr, ytr = finite_y(X_train, yt)
        Xva, yva = finite_y(X_val, yv)
        if args.model == "process_hybrid":
            model = fit_process_hybrid(Xtr, ytr, Xva, yva, target, args.seed, args.threads)
        else:
            model = make_estimator(args.model, target, args.seed, args.threads)
            model.fit(Xtr, ytr)
        pred = predict_model(args.model, model, Xva)
        validation[target] = metrics(yva, pred)
        log(f"  validation {target}: {validation[target]}")
        models[target] = model

    total_pred = (
        predict_model(args.model, models["surface_runoff"], X_val)
        + predict_model(args.model, models["subsurface_runoff"], X_val)
    ) / 86400.0
    total_obs = (
        y_val_all["surface_runoff"] + y_val_all["subsurface_runoff"]
    ) / 86400.0
    ok = np.isfinite(total_obs) & np.isfinite(total_pred)
    validation["total_runoff_change_proxy"] = metrics(total_obs[ok], total_pred[ok])
    return models, validation, med


def predict_all(ctx: dict, args, models: dict, med: np.ndarray, out_nc: Path, out_csv: Path) -> None:
    create_output_nc(out_nc, ctx, args.model, args.start_year, args.end_year)
    land = ctx["land"]
    n = len(land["landpoint"])
    rows = []
    with Dataset(out_nc, "a") as ds:
        for s_idx, scenario in enumerate(SCENARIOS):
            for m_idx, member in enumerate(MEMBERS):
                log(f"Predicting {args.model}: {scenario} member {member}")
                time_pos = 0
                for year in range(args.start_year, args.end_year + 1):
                    for code in SEASON_CODES:
                        accum = {target: [] for target in OUTPUT_TARGETS}
                        for start in range(0, n, args.predict_chunk):
                            stop = min(start + args.predict_chunk, n)
                            pts = np.arange(start, stop, dtype=np.int64)
                            X = build_features_for_group(ctx, pts, year, int(code), "future", scenario, member)
                            X = impute_apply(X, med)
                            base_idx = season_index(int(code))
                            pred = {}
                            for target in TARGETS:
                                change = predict_model(args.model, models[target], X)
                                pred[target] = reconstruct(change, target, ctx["target_baseline"][target][base_idx, pts])
                            pred["total_runoff"] = np.maximum(pred["surface_runoff"] + pred["subsurface_runoff"], 0.0).astype(np.float32)
                            if args.model == "process_hybrid":
                                # Explicit hydrological post-processing.
                                pred["surface_runoff"] = np.maximum(pred["surface_runoff"], 0.0)
                                pred["subsurface_runoff"] = np.maximum(pred["subsurface_runoff"], 0.0)
                                pred["total_runoff"] = (pred["surface_runoff"] + pred["subsurface_runoff"]).astype(np.float32)
                            for target in OUTPUT_TARGETS:
                                ds.variables[target][s_idx, m_idx, time_pos, start:stop] = pred[target]
                                accum[target].append(float(np.nanmean(pred[target])))
                        for target in OUTPUT_TARGETS:
                            rows.append({
                                "model": args.model,
                                "scenario": scenario,
                                "member": member,
                                "year": year,
                                "season": SEASON_NAMES[int(code)],
                                "season_code": int(code),
                                "target": target,
                                "spatial_mean": float(np.mean(accum[target])),
                            })
                        log(f"  predicted {scenario} member {member} {year} {SEASON_NAMES[int(code)]}")
                        time_pos += 1

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def open_context(root: Path) -> dict:
    data = root / "HPC_input_data"
    ctx = {"root": root, "data_dir": data}
    ctx["hist_climate"] = Dataset(data / "01_historical_climate/CHESS-met_gb_1km_seasonal_1961-2019.nc")
    ctx["hist_target"] = Dataset(data / "02_historical_targets/CHESS-land_HydEn_gb_1km_seasonal_targets_1961-2015.nc")
    ctx["hist_pft"] = Dataset(data / "03_historical_pft/CHESSLand_PFT_1961_2015_1km_annual.nc")
    ctx["future_climate"] = Dataset(data / "04_future_climate/CHESS-SCAPE_uk_1km_seasonal_1980-2080_by_scenario_member.nc")
    ctx["hist_pet"] = Dataset(data / "06_historical_pet/HydroPE_CHESS_GB_1km_seasonal_PET_PETI_1961_2019.nc")
    ctx["future_extra"] = Dataset(data / "07_future_radiation_pressure/CHESS-SCAPE_uk_1km_seasonal_rsds_psurf_1980-2080_by_scenario_member.nc")
    ctx["hist_precip"] = Dataset(data / "08_historical_precip_antecedent/CEH-GEAR_GB_1km_seasonal_precip_from_monthly_1961_2019.nc")
    ctx["land"] = load_landpoints(ctx["hist_target"], ctx["hist_climate"])
    ctx["hist_climate_lookup"] = time_lookup(ctx["hist_climate"])
    ctx["hist_pet_lookup"] = time_lookup(ctx["hist_pet"])
    ctx["hist_precip_lookup"] = time_lookup(ctx["hist_precip"])
    ctx["hist_precip_grid_index"] = build_grid_index(ctx["hist_precip"], ctx["land"])
    ctx["hist_pft_index"] = validate_historical_pft_alignment(ctx["hist_pft"], ctx["land"])
    ctx["static_matrix"] = build_static_matrix(ctx)
    baseline_years = np.arange(1971, 2001, dtype=np.int16)
    ctx["dynamic_baseline"] = compute_dynamic_baseline(ctx, baseline_years)
    ctx["target_baseline"] = compute_target_baseline(ctx, baseline_years)
    ctx["pft_baseline"] = compute_hist_pft_baseline(ctx, baseline_years)
    ctx["future_pft"] = {}
    ctx["future_pft_index"] = {}
    for scenario, fname in PFT_FILE_BY_SCENARIO.items():
        ds = Dataset(data / "05_future_aft_to_pft" / fname)
        ctx["future_pft"][scenario] = ds
        ctx["future_pft_index"][scenario] = build_future_pft_index(ds, ctx["land"])
    return ctx


def close_context(ctx: dict) -> None:
    for key, value in list(ctx.items()):
        if isinstance(value, Dataset):
            value.close()
    for ds in ctx.get("future_pft", {}).values():
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one 5-model seasonal emulator over all scenarios and members.")
    parser.add_argument("--root", default="/scratch/users/k2585234/emulator_seasonal")
    parser.add_argument("--model", required=True, choices=["elastic_net", "forest", "lightgbm", "process_hybrid", "mlp_stack"])
    parser.add_argument("--train-rows", type=int, default=900000)
    parser.add_argument("--val-rows", type=int, default=220000)
    parser.add_argument("--train-start", type=int, default=1961)
    parser.add_argument("--train-end", type=int, default=2005)
    parser.add_argument("--val-start", type=int, default=2006)
    parser.add_argument("--val-end", type=int, default=2015)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2079)
    parser.add_argument("--predict-chunk", type=int, default=60000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "outputs_v2_5models" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output folder: {out_dir}")
    log(f"Model: {args.model}")

    ctx = open_context(root)
    try:
        models, validation, med = train_models(ctx, args)
        metrics_path = out_dir / "validation_metrics.json"
        metrics_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        (out_dir / "feature_names.txt").write_text("\n".join(feature_names()), encoding="utf-8")
        np.save(out_dir / "feature_imputation_medians.npy", med)
        predict_all(
            ctx,
            args,
            models,
            med,
            out_dir / f"seasonal_soil_moisture_runoff_predictions_1km_2020_2079_{args.model}.nc",
            out_dir / "seasonal_spatial_mean_trends.csv",
        )
        report = [
            f"Model: {args.model}",
            "Purpose: seasonal 1 km process-informed emulator for soil moisture and runoff.",
            f"Training rows: {args.train_rows}",
            f"Validation rows: {args.val_rows}",
            f"Future period: {args.start_year}-{args.end_year}",
            "Scenarios: rcp26, rcp45, rcp85",
            "Members: 01, 04, 06, 15",
            "",
            "Validation metrics:",
            json.dumps(validation, indent=2),
            "",
            "Hydrological constraints:",
            "- Soil moisture and runoff are clipped to non-negative values.",
            "- total_runoff is calculated as surface_runoff + subsurface_runoff.",
            "- process_hybrid combines LightGBM, XGBoost and Ridge and applies explicit runoff consistency.",
        ]
        (out_dir / "run_report.txt").write_text("\n".join(report), encoding="utf-8")
        log("Finished.")
    finally:
        close_context(ctx)


if __name__ == "__main__":
    main()
