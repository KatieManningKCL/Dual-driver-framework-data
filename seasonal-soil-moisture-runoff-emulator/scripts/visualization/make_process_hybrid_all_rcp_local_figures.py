from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


ROOT = Path(r"E:\Emulator seasonal")
NC_PATH = ROOT / "seasonal_soil_moisture_runoff_predictions_1km_2020_2079_process_hybrid.nc"
TREND_CSV = ROOT / "outputs_v2_5models_png_reports" / "outputs_v2_5models" / "process_hybrid" / "seasonal_spatial_mean_trends.csv"
OUT_DIR = ROOT / "process_hybrid_all_rcp_local_figures"

SCENARIOS = ["rcp26", "rcp45", "rcp85"]
SCENARIO_INDEX = {"rcp26": 0, "rcp45": 1, "rcp85": 2}
SCEN_LABEL = {"rcp26": "RCP2.6", "rcp45": "RCP4.5", "rcp85": "RCP8.5"}
SCEN_COLORS = {"rcp26": "#0072B2", "rcp45": "#009E73", "rcp85": "#D55E00"}
MEMBERS = ["01", "04", "06", "15"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_CODE = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
TARGETS = ["soil_moisture", "total_runoff"]
TARGET_LABEL = {"soil_moisture": "Soil moisture", "total_runoff": "Total runoff"}
TARGET_UNITS = {"soil_moisture": "kg m$^{-2}$", "total_runoff": "mm day$^{-1}$"}


def target_scale(target: str) -> float:
    return 86400.0 if "runoff" in target else 1.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_rows() -> list[dict]:
    rows = []
    with TREND_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["year"] = int(row["year"])
            row["season_code"] = int(row["season_code"])
            row["spatial_mean"] = float(row["spatial_mean"])
            rows.append(row)
    return rows


def annual_series(rows: list[dict], target: str) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["target"] == target:
            grouped[(row["scenario"], row["member"], row["year"])].append(row["spatial_mean"] * target_scale(target))
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (scenario, member, year), vals in grouped.items():
        out[(scenario, member)][year] = float(np.nanmean(vals))
    return dict(out)


def seasonal_series(rows: list[dict], target: str, season: str) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["target"] == target and row["season"] == season:
            grouped[(row["scenario"], row["member"], row["year"])].append(row["spatial_mean"] * target_scale(target))
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (scenario, member, year), vals in grouped.items():
        out[(scenario, member)][year] = float(np.nanmean(vals))
    return dict(out)


def plot_annual(rows: list[dict], target: str) -> Path:
    data = annual_series(rows, target)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for scenario in SCENARIOS:
        series = []
        years = None
        for member in MEMBERS:
            d = data.get((scenario, member))
            if not d:
                continue
            years = sorted(d) if years is None else years
            series.append([d[y] for y in years])
        if not series:
            continue
        arr = np.asarray(series)
        ax.plot(years, arr.mean(axis=0), color=SCEN_COLORS[scenario], lw=2.6, label=f"{SCEN_LABEL[scenario]} mean")
        ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=SCEN_COLORS[scenario], alpha=0.16, lw=0)
    ax.set_title(f"Process-informed Hybrid: annual spatial mean {TARGET_LABEL[target]}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{TARGET_LABEL[target]} ({TARGET_UNITS[target]})")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=3)
    path = OUT_DIR / f"process_hybrid_01_annual_all_rcps_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seasonal_grid(rows: list[dict], target: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5), sharex=True, constrained_layout=True)
    for ax, season in zip(axes.ravel(), SEASONS):
        data = seasonal_series(rows, target, season)
        for scenario in SCENARIOS:
            series = []
            years = None
            for member in MEMBERS:
                d = data.get((scenario, member))
                if not d:
                    continue
                years = sorted(d) if years is None else years
                series.append([d[y] for y in years])
            if not series:
                continue
            arr = np.asarray(series)
            ax.plot(years, arr.mean(axis=0), color=SCEN_COLORS[scenario], lw=2.1, label=SCEN_LABEL[scenario])
            ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=SCEN_COLORS[scenario], alpha=0.11, lw=0)
        ax.set_title(season)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_xlabel("Year")
        ax.set_ylabel(TARGET_UNITS[target])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Process-informed Hybrid: seasonal {TARGET_LABEL[target]} across RCPs", fontsize=14, y=1.10)
    path = OUT_DIR / f"process_hybrid_02_seasonal_grid_all_rcps_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def land_to_grid(values: np.ndarray, landpoint: np.ndarray, ny: int = 1057, nx: int = 656) -> np.ndarray:
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    flat = landpoint.astype(np.int64) - 1
    yi = flat // nx
    xi = flat % nx
    ok = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
    grid[yi[ok], xi[ok]] = values[ok].astype(np.float32)
    return grid


def read_period_mean(ds: Dataset, target: str, scenario: str, season_code: int, years: range) -> np.ndarray:
    all_years = np.asarray(ds.variables["year"][:], dtype=np.int16)
    all_codes = np.asarray(ds.variables["season_code"][:], dtype=np.int8)
    time_idx = np.where(np.isin(all_years, list(years)) & (all_codes == season_code))[0]
    scenario_idx = SCENARIO_INDEX[scenario]
    member_means = []
    for member_idx in range(len(MEMBERS)):
        time_means = []
        for t in time_idx:
            arr = np.asarray(ds.variables[target][scenario_idx, member_idx, int(t), :], dtype=np.float32)
            time_means.append(arr * target_scale(target))
        member_means.append(np.nanmean(np.stack(time_means, axis=0), axis=0))
    return np.nanmean(np.stack(member_means, axis=0), axis=0).astype(np.float32)


def diverging_limits(grids: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([g[np.isfinite(g)].ravel() for g in grids if np.any(np.isfinite(g))])
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(vals, [2, 98])
    vmax = float(max(abs(lo), abs(hi)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def plot_spatial(target: str) -> Path:
    print(f"Reading spatial data for {target}...", flush=True)
    grids = []
    with Dataset(NC_PATH) as ds:
        landpoint = np.asarray(ds.variables["landpoint"][:], dtype=np.int64)
        for scenario in SCENARIOS:
            row = []
            for season in SEASONS:
                early = read_period_mean(ds, target, scenario, SEASON_CODE[season], range(2020, 2030))
                late = read_period_mean(ds, target, scenario, SEASON_CODE[season], range(2070, 2080))
                row.append(land_to_grid(late - early, landpoint))
                print(f"  {target}: {SCEN_LABEL[scenario]} {season} ready", flush=True)
            grids.append(row)

    flat = [g for row in grids for g in row]
    vmin, vmax = diverging_limits(flat)
    cmap = "BrBG" if target == "soil_moisture" else "PuOr"
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 12.8), constrained_layout=True)
    image = None
    for i, scenario in enumerate(SCENARIOS):
        for j, season in enumerate(SEASONS):
            ax = axes[i, j]
            image = ax.imshow(grids[i][j], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f2f2f2")
            if i == 0:
                ax.set_title(season, fontsize=12)
            if j == 0:
                ax.set_ylabel(SCEN_LABEL[scenario], fontsize=12)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.012)
    cbar.set_label(f"Change in {TARGET_LABEL[target]} ({TARGET_UNITS[target]})")
    fig.suptitle(
        f"Process-informed Hybrid: seasonal spatial change across RCPs, 2070s minus 2020s\n"
        f"{TARGET_LABEL[target]} ensemble mean across members",
        fontsize=14,
        y=1.02,
    )
    path = OUT_DIR / f"process_hybrid_03_spatial_3rcp_4season_2070s_minus_2020s_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(rows: list[dict]) -> Path:
    path = OUT_DIR / "process_hybrid_all_rcps_summary_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "target", "early_2020s", "late_2070s", "percent_change"])
        writer.writeheader()
        for scenario in SCENARIOS:
            for target in TARGETS:
                early_vals = [
                    r["spatial_mean"] * target_scale(target)
                    for r in rows
                    if r["scenario"] == scenario and r["target"] == target and 2020 <= r["year"] <= 2029
                ]
                late_vals = [
                    r["spatial_mean"] * target_scale(target)
                    for r in rows
                    if r["scenario"] == scenario and r["target"] == target and 2070 <= r["year"] <= 2079
                ]
                early = float(np.nanmean(early_vals))
                late = float(np.nanmean(late_vals))
                pct = float((late - early) / early * 100.0) if abs(early) > 1.0e-12 else np.nan
                writer.writerow({"scenario": scenario, "target": target, "early_2020s": early, "late_2070s": late, "percent_change": pct})
    return path


def main() -> None:
    if not NC_PATH.exists() or NC_PATH.stat().st_size == 0:
        raise SystemExit(f"Missing or empty NetCDF: {NC_PATH}")
    if not TREND_CSV.exists():
        raise SystemExit(f"Missing trend CSV: {TREND_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = load_rows()
    outputs = []
    for target in TARGETS:
        outputs.append(plot_annual(rows, target))
        outputs.append(plot_seasonal_grid(rows, target))
        outputs.append(plot_spatial(target))
    outputs.append(write_summary(rows))
    (OUT_DIR / "README.txt").write_text(
        "Process-informed Hybrid all-RCP local figures.\n"
        "Spatial maps show 2070s minus 2020s for RCP2.6, RCP4.5 and RCP8.5 across DJF/MAM/JJA/SON.\n",
        encoding="utf-8",
    )
    for path in outputs:
        print(f"Saved: {path}", flush=True)
    print(f"Finished. Output folder: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
