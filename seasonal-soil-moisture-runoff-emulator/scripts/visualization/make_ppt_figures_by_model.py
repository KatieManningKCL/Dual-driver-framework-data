from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


MODELS = ["elastic_net", "forest", "lightgbm", "process_hybrid", "mlp_stack"]
SCENARIOS = ["rcp26", "rcp45", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
TARGETS = ["soil_moisture", "total_runoff"]

MODEL_LABEL = {
    "elastic_net": "Elastic Net",
    "forest": "ExtraTrees Forest",
    "lightgbm": "LightGBM",
    "process_hybrid": "Process-informed Hybrid",
    "mlp_stack": "MLP Stack",
}
SCEN_LABEL = {"rcp26": "RCP2.6", "rcp45": "RCP4.5", "rcp85": "RCP8.5"}
TARGET_LABEL = {"soil_moisture": "Soil moisture", "total_runoff": "Total runoff"}
TARGET_UNITS = {"soil_moisture": "kg m$^{-2}$", "total_runoff": "mm day$^{-1}$"}
SCEN_COLORS = {"rcp26": "#0072B2", "rcp45": "#009E73", "rcp85": "#D55E00"}
SEASON_COLORS = {"DJF": "#0072B2", "MAM": "#009E73", "JJA": "#D55E00", "SON": "#CC79A7"}


def target_scale(target: str) -> float:
    return 86400.0 if "runoff" in target else 1.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_trend_rows(model_dir: Path) -> list[dict]:
    path = model_dir / "seasonal_spatial_mean_trends.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["year"] = int(row["year"])
            row["season_code"] = int(row["season_code"])
            row["spatial_mean"] = float(row["spatial_mean"])
            rows.append(row)
    return rows


def annual_series(rows: list[dict], target: str) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["target"] != target:
            continue
        grouped[(row["scenario"], row["member"], row["year"])].append(row["spatial_mean"] * target_scale(target))
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (scenario, member, year), vals in grouped.items():
        out[(scenario, member)][year] = float(np.nanmean(vals))
    return dict(out)


def seasonal_series(rows: list[dict], target: str, scenario: str) -> dict[tuple[str, str], dict[int, float]]:
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["target"] == target and row["scenario"] == scenario:
            grouped[(row["season"], row["member"], row["year"])].append(row["spatial_mean"] * target_scale(target))
    for (season, member, year), vals in grouped.items():
        out[(season, member)][year] = float(np.nanmean(vals))
    return dict(out)


def plot_annual_trend(model: str, rows: list[dict], target: str, out_dir: Path) -> Path:
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
        arr = np.asarray(series, dtype=np.float64)
        ax.plot(years, arr.mean(axis=0), color=SCEN_COLORS[scenario], lw=2.6, label=f"{SCEN_LABEL[scenario]} ensemble mean")
        ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=SCEN_COLORS[scenario], alpha=0.17, lw=0)
    ax.set_title(f"{MODEL_LABEL[model]}: annual spatial mean {TARGET_LABEL[target]}")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{TARGET_LABEL[target]} ({TARGET_UNITS[target]})")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="best")
    path = out_dir / f"{model}_01_annual_trend_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seasonal_rcp85(model: str, rows: list[dict], target: str, out_dir: Path) -> Path:
    data = seasonal_series(rows, target, "rcp85")
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for season in SEASONS:
        series = []
        years = None
        for member in MEMBERS:
            d = data.get((season, member))
            if not d:
                continue
            years = sorted(d) if years is None else years
            series.append([d[y] for y in years])
        if not series:
            continue
        arr = np.asarray(series, dtype=np.float64)
        ax.plot(years, arr.mean(axis=0), color=SEASON_COLORS[season], lw=2.5, label=season)
        ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=SEASON_COLORS[season], alpha=0.14, lw=0)
    ax.set_title(f"{MODEL_LABEL[model]}: seasonal {TARGET_LABEL[target]} under RCP8.5")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{TARGET_LABEL[target]} ({TARGET_UNITS[target]})")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=4, loc="best")
    path = out_dir / f"{model}_02_seasonal_rcp85_{target}.png"
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


def read_period_mean(ds: Dataset, target: str, scenario_idx: int, season_code: int, years: range) -> np.ndarray:
    all_years = np.asarray(ds.variables["year"][:], dtype=np.int16)
    all_codes = np.asarray(ds.variables["season_code"][:], dtype=np.int8)
    time_idx = np.where(np.isin(all_years, list(years)) & (all_codes == season_code))[0]
    member_means = []
    for member_idx in range(len(MEMBERS)):
        time_means = []
        for t in time_idx:
            arr = np.asarray(ds.variables[target][scenario_idx, member_idx, int(t), :], dtype=np.float32)
            time_means.append(arr * target_scale(target))
        member_means.append(np.nanmean(np.stack(time_means, axis=0), axis=0))
    return np.nanmean(np.stack(member_means, axis=0), axis=0).astype(np.float32)


def symmetric_limits(grids: list[np.ndarray], lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    vals = np.concatenate([g[np.isfinite(g)].ravel() for g in grids if np.any(np.isfinite(g))])
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(vals, [lower, upper])
    vmax = float(max(abs(lo), abs(hi)))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0
    return -vmax, vmax


def plot_spatial_change(model: str, nc_path: Path, target: str, out_dir: Path) -> Path:
    with Dataset(nc_path) as ds:
        landpoint = np.asarray(ds.variables["landpoint"][:], dtype=np.int64)
        grids = []
        for code in [1, 2, 3, 4]:
            early = read_period_mean(ds, target, 2, code, range(2020, 2030))
            late = read_period_mean(ds, target, 2, code, range(2070, 2080))
            grids.append(land_to_grid(late - early, landpoint))

    vmin, vmax = symmetric_limits(grids)
    cmap = "BrBG" if target == "soil_moisture" else "PuOr"
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 11.4), constrained_layout=True)
    images = []
    for ax, season, grid in zip(axes.ravel(), SEASONS, grids):
        img = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        images.append(img)
        ax.set_title(season)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f2f2f2")
    label = f"Change in {TARGET_LABEL[target]} ({TARGET_UNITS[target]})"
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label(label)
    fig.suptitle(
        f"{MODEL_LABEL[model]}: RCP8.5 seasonal spatial change, 2070s minus 2020s\n"
        f"{TARGET_LABEL[target]} ensemble mean across members",
        fontsize=14,
        y=1.02,
    )
    path = out_dir / f"{model}_03_spatial_rcp85_2070s_minus_2020s_{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_model_summary(model: str, rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / f"{model}_summary_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "scenario", "target", "early_2020s", "late_2070s", "percent_change"],
        )
        writer.writeheader()
        for scenario in SCENARIOS:
            for target in TARGETS:
                vals_early = [
                    r["spatial_mean"] * target_scale(target)
                    for r in rows
                    if r["scenario"] == scenario and r["target"] == target and 2020 <= r["year"] <= 2029
                ]
                vals_late = [
                    r["spatial_mean"] * target_scale(target)
                    for r in rows
                    if r["scenario"] == scenario and r["target"] == target and 2070 <= r["year"] <= 2079
                ]
                early = float(np.nanmean(vals_early))
                late = float(np.nanmean(vals_late))
                pct = float((late - early) / early * 100.0) if abs(early) > 1.0e-12 else np.nan
                writer.writerow(
                    {
                        "model": model,
                        "scenario": scenario,
                        "target": target,
                        "early_2020s": early,
                        "late_2070s": late,
                        "percent_change": pct,
                    }
                )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create English PPT-ready figures for each 5-model emulator output.")
    parser.add_argument("--root", default="/scratch/users/k2585234/emulator_seasonal")
    parser.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    args = parser.parse_args()

    configure_style()
    root = Path(args.root).resolve()
    base = root / "outputs_v2_5models"
    out_root = base / "ppt_figures_by_model"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for model in args.models:
        model_dir = base / model
        nc_path = model_dir / f"seasonal_soil_moisture_runoff_predictions_1km_2020_2079_{model}.nc"
        if not model_dir.exists():
            print(f"Skipping missing model directory: {model_dir}")
            continue
        if not nc_path.exists():
            print(f"Skipping spatial maps for missing NetCDF: {nc_path}")
        rows = load_trend_rows(model_dir)
        out_dir = out_root / model
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = [
            plot_annual_trend(model, rows, "soil_moisture", out_dir),
            plot_annual_trend(model, rows, "total_runoff", out_dir),
            plot_seasonal_rcp85(model, rows, "soil_moisture", out_dir),
            plot_seasonal_rcp85(model, rows, "total_runoff", out_dir),
        ]
        if nc_path.exists():
            paths.extend(
                [
                    plot_spatial_change(model, nc_path, "soil_moisture", out_dir),
                    plot_spatial_change(model, nc_path, "total_runoff", out_dir),
                ]
            )
        paths.append(write_model_summary(model, rows, out_dir))
        for p in paths:
            manifest_rows.append({"model": model, "file": str(p.relative_to(out_root)), "bytes": p.stat().st_size})
        print(f"Created PPT figures for {model}: {out_dir}")

    with (out_root / "ppt_figure_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "file", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = [
        "PPT-ready seasonal emulator figures",
        "",
        "Each model has the same figure set:",
        "1. Annual spatial mean soil moisture across RCP2.6, RCP4.5 and RCP8.5.",
        "2. Annual spatial mean total runoff across RCP2.6, RCP4.5 and RCP8.5.",
        "3. Seasonal RCP8.5 soil moisture trend.",
        "4. Seasonal RCP8.5 total runoff trend.",
        "5. RCP8.5 seasonal spatial change in soil moisture, 2070s minus 2020s.",
        "6. RCP8.5 seasonal spatial change in total runoff, 2070s minus 2020s.",
        "",
        "Line shading shows the ensemble member range. Spatial maps use the ensemble mean across members.",
        "Runoff is plotted in mm day-1; soil moisture is plotted in kg m-2.",
    ]
    (out_root / "README_PPT_FIGURES.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Saved all PPT figures to {out_root}")


if __name__ == "__main__":
    main()
