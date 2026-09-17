from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ["elastic_net", "forest", "lightgbm", "process_hybrid", "mlp_stack"]
SCENARIOS = ["rcp26", "rcp45", "rcp85"]
MEMBERS = ["01", "04", "06", "15"]
TARGETS = ["soil_moisture", "surface_runoff", "subsurface_runoff", "total_runoff"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]

MODEL_LABEL = {
    "elastic_net": "Elastic Net",
    "forest": "ExtraTrees",
    "lightgbm": "LightGBM",
    "process_hybrid": "Process-informed hybrid",
    "mlp_stack": "MLP stack",
}
SCEN_LABEL = {"rcp26": "RCP2.6", "rcp45": "RCP4.5", "rcp85": "RCP8.5"}
TARGET_LABEL = {
    "soil_moisture": "Soil moisture",
    "surface_runoff": "Surface runoff",
    "subsurface_runoff": "Subsurface runoff",
    "total_runoff": "Total runoff",
}


def target_scale(target: str) -> float:
    return 86400.0 if "runoff" in target else 1.0


def target_units(target: str) -> str:
    return "mm day$^{-1}$" if "runoff" in target else "kg m$^{-2}$"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_rows(root: Path) -> list[dict]:
    rows = []
    for model in MODELS:
        path = root / "outputs_v2_5models" / model / "seasonal_spatial_mean_trends.csv"
        if not path.exists():
            print(f"Missing trend CSV, skipped: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["model"] = model
                r["year"] = int(r["year"])
                r["season_code"] = int(r["season_code"])
                r["spatial_mean"] = float(r["spatial_mean"])
                rows.append(r)
    if not rows:
        raise RuntimeError("No model trend CSVs found.")
    return rows


def annual_model_scenario(rows: list[dict], target: str):
    grouped = defaultdict(list)
    for r in rows:
        if r["target"] == target:
            grouped[(r["model"], r["scenario"], r["member"], r["year"])].append(r["spatial_mean"])
    out = defaultdict(dict)
    for (model, scen, member, year), vals in grouped.items():
        out[(model, scen, member)][year] = float(np.mean(vals))
    return out


def plot_annual_by_model(rows: list[dict], out_dir: Path) -> None:
    colors = {"rcp26": "#2b8cbe", "rcp45": "#41ab5d", "rcp85": "#de2d26"}
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        for target in TARGETS:
            data = annual_model_scenario(model_rows, target)
            fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
            for scen in SCENARIOS:
                series = []
                years = None
                for member in MEMBERS:
                    d = data.get((model, scen, member))
                    if not d:
                        continue
                    years = sorted(d) if years is None else years
                    series.append([d[y] * target_scale(target) for y in years])
                if not series:
                    continue
                arr = np.asarray(series)
                ax.plot(years, arr.mean(axis=0), color=colors[scen], lw=2.4, label=f"{SCEN_LABEL[scen]} mean")
                ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=colors[scen], alpha=0.16, lw=0)
            ax.set_title(f"{MODEL_LABEL[model]}: annual spatial mean {TARGET_LABEL[target]}")
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{TARGET_LABEL[target]} ({target_units(target)})")
            ax.grid(True, color="#dddddd", linewidth=0.7)
            ax.legend(frameon=False, ncol=3)
            fig.savefig(out_dir / f"annual_{model}_{target}.png")
            plt.close(fig)


def plot_model_comparison_rcp85(rows: list[dict], out_dir: Path) -> None:
    colors = {
        "elastic_net": "#8c8c8c",
        "forest": "#756bb1",
        "lightgbm": "#3182bd",
        "process_hybrid": "#de2d26",
        "mlp_stack": "#31a354",
    }
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
        data = annual_model_scenario(rows, target)
        for model in MODELS:
            member_series = []
            years = None
            for member in MEMBERS:
                d = data.get((model, "rcp85", member))
                if not d:
                    continue
                years = sorted(d) if years is None else years
                member_series.append([d[y] * target_scale(target) for y in years])
            if not member_series:
                continue
            arr = np.asarray(member_series)
            ax.plot(years, arr.mean(axis=0), color=colors[model], lw=2.2, label=MODEL_LABEL[model])
            ax.fill_between(years, arr.min(axis=0), arr.max(axis=0), color=colors[model], alpha=0.10, lw=0)
        ax.set_title(f"Model comparison under RCP8.5: {TARGET_LABEL[target]}")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{TARGET_LABEL[target]} ({target_units(target)})")
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.legend(frameon=False, ncol=2)
        fig.savefig(out_dir / f"model_comparison_rcp85_{target}.png")
        plt.close(fig)


def plot_seasonal_rcp85_process(rows: list[dict], out_dir: Path) -> None:
    colors = {"DJF": "#08519c", "MAM": "#31a354", "JJA": "#de2d26", "SON": "#756bb1"}
    model = "process_hybrid"
    for target in ["soil_moisture", "total_runoff"]:
        fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
        for season in SEASONS:
            grouped = defaultdict(list)
            for r in rows:
                if r["model"] == model and r["scenario"] == "rcp85" and r["target"] == target and r["season"] == season:
                    grouped[r["year"]].append(r["spatial_mean"] * target_scale(target))
            years = sorted(grouped)
            if not years:
                continue
            mean = [float(np.mean(grouped[y])) for y in years]
            lo = [float(np.min(grouped[y])) for y in years]
            hi = [float(np.max(grouped[y])) for y in years]
            ax.plot(years, mean, color=colors[season], lw=2.3, label=season)
            ax.fill_between(years, lo, hi, color=colors[season], alpha=0.13, lw=0)
        ax.set_title(f"Process-informed hybrid seasonal signal under RCP8.5: {TARGET_LABEL[target]}")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{TARGET_LABEL[target]} ({target_units(target)})")
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.legend(frameon=False, ncol=4)
        fig.savefig(out_dir / f"seasonal_rcp85_process_hybrid_{target}.png")
        plt.close(fig)


def write_percent_change(rows: list[dict], out_dir: Path) -> None:
    path = out_dir / "percent_change_2020_2079_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "scenario", "target", "early_2020_2029", "late_2070_2079", "percent_change"])
        writer.writeheader()
        for model in MODELS:
            for scen in SCENARIOS:
                for target in TARGETS:
                    early = [
                        r["spatial_mean"] * target_scale(target)
                        for r in rows
                        if r["model"] == model and r["scenario"] == scen and r["target"] == target and 2020 <= r["year"] <= 2029
                    ]
                    late = [
                        r["spatial_mean"] * target_scale(target)
                        for r in rows
                        if r["model"] == model and r["scenario"] == scen and r["target"] == target and 2070 <= r["year"] <= 2079
                    ]
                    if not early or not late:
                        continue
                    e = float(np.mean(early))
                    l = float(np.mean(late))
                    pct = float((l - e) / e * 100.0) if abs(e) > 1.0e-12 else np.nan
                    writer.writerow({
                        "model": model,
                        "scenario": scen,
                        "target": target,
                        "early_2020_2029": e,
                        "late_2070_2079": l,
                        "percent_change": pct,
                    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/scratch/users/k2585234/emulator_seasonal")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "outputs_v2_5models" / "model_comparison_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows = load_rows(root)
    plot_annual_by_model(rows, out_dir)
    plot_model_comparison_rcp85(rows, out_dir)
    plot_seasonal_rcp85_process(rows, out_dir)
    write_percent_change(rows, out_dir)
    (out_dir / "visualization_manifest.txt").write_text(
        "Generated annual ensemble-range plots, RCP8.5 model comparisons, process-hybrid seasonal plots, and percent-change summary.\n",
        encoding="utf-8",
    )
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
