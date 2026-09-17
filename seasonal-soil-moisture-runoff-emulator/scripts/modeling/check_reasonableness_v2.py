from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


MODELS = ["elastic_net", "forest", "lightgbm", "process_hybrid", "mlp_stack"]
SCENARIOS = ["rcp26", "rcp45", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
TARGETS = ["soil_moisture", "surface_runoff", "subsurface_runoff", "total_runoff"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def target_scale(target: str) -> float:
    return 86400.0 if "runoff" in target else 1.0


def load_rows(root: Path) -> list[dict]:
    rows = []
    for model in MODELS:
        path = root / "outputs_v2_5models" / model / "seasonal_spatial_mean_trends.csv"
        if not path.exists():
            print(f"Missing trend CSV, skipped: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["year"] = int(row["year"])
                row["season_code"] = int(row["season_code"])
                row["spatial_mean"] = float(row["spatial_mean"])
                row["model"] = model
                rows.append(row)
    if not rows:
        raise RuntimeError("No trend CSV files were found. Run the model jobs first.")
    return rows


def mean_value(rows: list[dict], model: str, scenario: str, target: str, years: range, season: str | None = None) -> float:
    vals = [
        r["spatial_mean"] * target_scale(target)
        for r in rows
        if r["model"] == model
        and r["scenario"] == scenario
        and r["target"] == target
        and r["year"] in years
        and (season is None or r["season"] == season)
    ]
    return float(np.mean(vals)) if vals else float("nan")


def percent_change(early: float, late: float) -> float:
    if not np.isfinite(early) or abs(early) < 1.0e-12:
        return float("nan")
    return float((late - early) / early * 100.0)


def annual_member_series(rows: list[dict], model: str, scenario: str, target: str) -> dict[str, dict[int, float]]:
    grouped = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["scenario"] == scenario and r["target"] == target:
            grouped[(r["member"], r["year"])].append(r["spatial_mean"] * target_scale(target))
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (member, year), vals in grouped.items():
        out[member][year] = float(np.mean(vals))
    return dict(out)


def trend_diagnostics(rows: list[dict]) -> dict:
    early_years = range(2020, 2030)
    late_years = range(2070, 2080)
    out: dict[str, object] = {
        "percent_change_2020s_to_2070s": [],
        "seasonal_contrast_process_hybrid_rcp85": {},
        "ensemble_spread_change": [],
    }

    for model in MODELS:
        for scenario in SCENARIOS:
            for target in TARGETS:
                early = mean_value(rows, model, scenario, target, early_years)
                late = mean_value(rows, model, scenario, target, late_years)
                out["percent_change_2020s_to_2070s"].append({
                    "model": model,
                    "scenario": scenario,
                    "target": target,
                    "early_2020s": early,
                    "late_2070s": late,
                    "percent_change": percent_change(early, late),
                })

    model = "process_hybrid"
    scenario = "rcp85"
    for target in ["soil_moisture", "total_runoff"]:
        early_season = {
            season: mean_value(rows, model, scenario, target, early_years, season)
            for season in SEASONS
        }
        late_season = {
            season: mean_value(rows, model, scenario, target, late_years, season)
            for season in SEASONS
        }
        annual_early = float(np.nanmean(list(early_season.values())))
        annual_late = float(np.nanmean(list(late_season.values())))
        out["seasonal_contrast_process_hybrid_rcp85"][target] = {
            "early_seasonal_means": early_season,
            "late_seasonal_means": late_season,
            "early_range_percent_of_mean": percent_change(annual_early, annual_early + (max(early_season.values()) - min(early_season.values()))),
            "late_range_percent_of_mean": percent_change(annual_late, annual_late + (max(late_season.values()) - min(late_season.values()))),
            "jja_percent_change": percent_change(early_season["JJA"], late_season["JJA"]),
        }

    for model in MODELS:
        for scenario in SCENARIOS:
            for target in TARGETS:
                series = annual_member_series(rows, model, scenario, target)
                early_member_means = []
                late_member_means = []
                for member, by_year in series.items():
                    early_vals = [by_year[y] for y in early_years if y in by_year]
                    late_vals = [by_year[y] for y in late_years if y in by_year]
                    if early_vals and late_vals:
                        early_member_means.append(float(np.mean(early_vals)))
                        late_member_means.append(float(np.mean(late_vals)))
                if len(early_member_means) >= 2:
                    early_spread = float(np.max(early_member_means) - np.min(early_member_means))
                    late_spread = float(np.max(late_member_means) - np.min(late_member_means))
                    out["ensemble_spread_change"].append({
                        "model": model,
                        "scenario": scenario,
                        "target": target,
                        "early_member_range": early_spread,
                        "late_member_range": late_spread,
                        "late_minus_early_range": late_spread - early_spread,
                    })
    return out


def sample_nc_checks(root: Path) -> dict:
    checks = {}
    for model in MODELS:
        nc_path = root / "outputs_v2_5models" / model / f"seasonal_soil_moisture_runoff_predictions_1km_2020_2079_{model}.nc"
        if not nc_path.exists():
            continue
        with Dataset(nc_path) as ds:
            n_land = len(ds.dimensions["landpoint"])
            sample_idx = np.linspace(0, n_land - 1, min(5000, n_land), dtype=np.int64)
            years = np.asarray(ds.variables["year"][:])
            season_code = np.asarray(ds.variables["season_code"][:])
            selected_time = np.where(((years <= 2024) | (years >= 2075)) & np.isin(season_code, [1, 3]))[0]
            model_check = {"negative_minima": {}, "total_runoff_max_abs_error": None}
            for target in TARGETS:
                arr = np.stack([
                    np.asarray(ds.variables[target][:, :, int(t), sample_idx])
                    for t in selected_time
                ], axis=2)
                model_check["negative_minima"][target] = float(np.nanmin(arr))
            total = np.stack([
                np.asarray(ds.variables["total_runoff"][:, :, int(t), sample_idx])
                for t in selected_time
            ], axis=2)
            surface = np.stack([
                np.asarray(ds.variables["surface_runoff"][:, :, int(t), sample_idx])
                for t in selected_time
            ], axis=2)
            subsurface = np.stack([
                np.asarray(ds.variables["subsurface_runoff"][:, :, int(t), sample_idx])
                for t in selected_time
            ], axis=2)
            model_check["total_runoff_max_abs_error"] = float(np.nanmax(np.abs(total - surface - subsurface)))
            checks[model] = model_check
    return checks


def regional_process_hybrid_check(root: Path) -> dict:
    model = "process_hybrid"
    nc_path = root / "outputs_v2_5models" / model / f"seasonal_soil_moisture_runoff_predictions_1km_2020_2079_{model}.nc"
    if not nc_path.exists():
        return {"note": "process_hybrid NetCDF not found."}
    rng = np.random.default_rng(42)
    with Dataset(nc_path) as ds:
        lat = np.asarray(ds.variables["latitude"][:])
        lon = np.asarray(ds.variables["longitude"][:])
        south_east = np.where((lat < 53.0) & (lon > -1.0))[0]
        west_north = np.where((lat > 54.0) & (lon < -2.5))[0]
        if len(south_east) > 25000:
            south_east = rng.choice(south_east, 25000, replace=False)
        if len(west_north) > 25000:
            west_north = rng.choice(west_north, 25000, replace=False)
        years = np.asarray(ds.variables["year"][:])
        season_code = np.asarray(ds.variables["season_code"][:])
        early_jja = np.where((years >= 2020) & (years <= 2029) & (season_code == 3))[0]
        late_jja = np.where((years >= 2070) & (years <= 2079) & (season_code == 3))[0]

        def region_mean(target: str, points: np.ndarray, times: np.ndarray) -> float:
            vals = []
            for t in times:
                arr = np.asarray(ds.variables[target][2, :, int(t), points]) * target_scale(target)
                vals.append(float(np.nanmean(arr)))
            return float(np.mean(vals))

        out = {}
        for target in ["soil_moisture", "total_runoff"]:
            se_early = region_mean(target, south_east, early_jja)
            se_late = region_mean(target, south_east, late_jja)
            wn_early = region_mean(target, west_north, early_jja)
            wn_late = region_mean(target, west_north, late_jja)
            out[target] = {
                "south_east_jja_percent_change": percent_change(se_early, se_late),
                "west_north_jja_percent_change": percent_change(wn_early, wn_late),
                "south_east_points": int(len(south_east)),
                "west_north_points": int(len(west_north)),
            }
        return out


def write_text_report(path: Path, diagnostics: dict, nc_checks: dict, regional: dict) -> None:
    lines = [
        "Seasonal 1 km soil moisture/runoff emulator reasonableness report",
        "",
        "Purpose",
        "This report checks whether the 5-model outputs satisfy basic hydrological and geospatial expectations.",
        "These are diagnostic checks only; they are not manual trend adjustments.",
        "",
        "Hard consistency checks",
    ]
    for model, chk in nc_checks.items():
        min_text = ", ".join(f"{k} min={v:.4g}" for k, v in chk["negative_minima"].items())
        lines.append(f"- {model}: {min_text}; max |total - surface - subsurface| = {chk['total_runoff_max_abs_error']:.4g}")

    lines.extend(["", "Key trend diagnostics"])
    pct_rows = diagnostics["percent_change_2020s_to_2070s"]
    for model in MODELS:
        rows = [r for r in pct_rows if r["model"] == model and r["scenario"] == "rcp85" and r["target"] in ["soil_moisture", "total_runoff"]]
        if not rows:
            continue
        bits = [f"{r['target']} {r['percent_change']:.2f}%" for r in rows]
        lines.append(f"- RCP8.5 annual mean change, {model}: " + "; ".join(bits))

    lines.extend(["", "Seasonal RCP8.5 signal in the process-informed hybrid"])
    for target, info in diagnostics["seasonal_contrast_process_hybrid_rcp85"].items():
        lines.append(
            f"- {target}: JJA change = {info['jja_percent_change']:.2f}%; "
            f"early seasonal range = {info['early_range_percent_of_mean']:.2f}% of mean; "
            f"late seasonal range = {info['late_range_percent_of_mean']:.2f}% of mean."
        )

    lines.extend(["", "Regional RCP8.5 JJA check in the process-informed hybrid"])
    if "note" in regional:
        lines.append(f"- {regional['note']}")
    else:
        for target, info in regional.items():
            lines.append(
                f"- {target}: south/east JJA change = {info['south_east_jja_percent_change']:.2f}%; "
                f"west/north JJA change = {info['west_north_jja_percent_change']:.2f}%."
            )

    lines.extend([
        "",
        "How to interpret",
        "- A strong result should show clear seasonal differences, especially for JJA soil moisture.",
        "- RCP8.5 should usually have a clearer late-century signal than RCP2.6, but the exact direction depends on target and season.",
        "- Soil moisture is expected to show summer drying in many regions; annual GB means can hide this if winter/autumn wetting offsets summer drying.",
        "- Runoff should remain hydrologically consistent: total_runoff = surface_runoff + subsurface_runoff and values should not be negative.",
        "- Ensemble spread should generally become larger after mid-century, especially under RCP8.5.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/scratch/users/k2585234/emulator_seasonal")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "outputs_v2_5models" / "reasonableness_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root)
    diagnostics = trend_diagnostics(rows)
    nc_checks = sample_nc_checks(root)
    regional = regional_process_hybrid_check(root)
    combined = {
        "trend_diagnostics": diagnostics,
        "netcdf_sample_checks": nc_checks,
        "regional_process_hybrid_rcp85_jja": regional,
    }
    (out_dir / "reasonableness_checks.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_text_report(out_dir / "reasonableness_report.txt", diagnostics, nc_checks, regional)
    print(f"Saved reasonableness checks to {out_dir}")


if __name__ == "__main__":
    main()
