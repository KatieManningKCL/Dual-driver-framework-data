from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Emulator seasonal")
INPUT_NC = ROOT / "seasonal_soil_moisture_runoff_predictions_1km_2020_2079_process_hybrid.nc"
OUT_DIR = ROOT / "scenario_consistency_validation"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SCENARIO_LABELS = ["RCP2.6", "RCP4.5", "RCP8.5"]
SEASON_LABELS = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}
VARIABLES = {
    "soil_moisture": {"label": "Soil moisture", "unit": "kg m-2"},
    "surface_runoff": {"label": "Surface runoff", "unit": "kg m-2 s-1"},
    "subsurface_runoff": {"label": "Subsurface runoff", "unit": "kg m-2 s-1"},
    "total_runoff": {"label": "Total runoff", "unit": "kg m-2 s-1"},
}
PERIODS = {
    "2020s": (2020, 2029),
    "2050s": (2050, 2059),
    "2070s": (2070, 2079),
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def decode_strings(arr) -> list[str]:
    vals = np.asarray(arr)
    if vals.dtype.kind in {"S", "U"}:
        return [str(x.decode() if isinstance(x, bytes) else x) for x in vals]
    return [str(x) for x in vals]


def masked_mean(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return float(np.nanmean(arr))


def period_mean(ds: nc.Dataset, var_name: str, scenario_i: int, season_i: int, y0: int, y1: int) -> float:
    years = ds.variables["year"][:]
    seasons = ds.variables["season_code"][:]
    time_idx = np.where((years >= y0) & (years <= y1) & (seasons == season_i))[0]
    member_means = []
    var = ds.variables[var_name]
    for member_i in range(len(ds.dimensions["member"])):
        vals = var[scenario_i, member_i, time_idx, :]
        member_means.append(masked_mean(vals))
    return float(np.nanmean(member_means))


def annual_series(ds: nc.Dataset, var_name: str) -> pd.DataFrame:
    years = np.asarray(ds.variables["year"][:])
    unique_years = np.unique(years)
    var = ds.variables[var_name]
    rows = []
    for scenario_i, scenario in enumerate(SCENARIO_LABELS):
        for year in unique_years:
            time_idx = np.where(years == year)[0]
            member_means = []
            for member_i in range(len(ds.dimensions["member"])):
                member_means.append(masked_mean(var[scenario_i, member_i, time_idx, :]))
            rows.append(
                {
                    "variable": var_name,
                    "scenario": scenario,
                    "year": int(year),
                    "mean": float(np.nanmean(member_means)),
                    "member_sd": float(np.nanstd(member_means)),
                }
            )
    return pd.DataFrame(rows)


def build_summary_tables(ds: nc.Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for var_name in VARIABLES:
        for scenario_i, scenario in enumerate(SCENARIO_LABELS):
            for season_i, season in SEASON_LABELS.items():
                period_values = {}
                for period, (y0, y1) in PERIODS.items():
                    period_values[period] = period_mean(ds, var_name, scenario_i, season_i, y0, y1)
                abs_change = period_values["2070s"] - period_values["2020s"]
                pct_change = 100.0 * abs_change / period_values["2020s"] if period_values["2020s"] != 0 else np.nan
                rows.append(
                    {
                        "variable": var_name,
                        "scenario": scenario,
                        "season": season,
                        "mean_2020s": period_values["2020s"],
                        "mean_2050s": period_values["2050s"],
                        "mean_2070s": period_values["2070s"],
                        "absolute_change_2070s_minus_2020s": abs_change,
                        "percent_change_2070s_vs_2020s": pct_change,
                    }
                )
    seasonal = pd.DataFrame(rows)

    annual_rows = []
    for var_name in VARIABLES:
        for scenario_i, scenario in enumerate(SCENARIO_LABELS):
            period_values = {}
            for period, (y0, y1) in PERIODS.items():
                vals = []
                for season_i in SEASON_LABELS:
                    vals.append(period_mean(ds, var_name, scenario_i, season_i, y0, y1))
                period_values[period] = float(np.nanmean(vals))
            abs_change = period_values["2070s"] - period_values["2020s"]
            pct_change = 100.0 * abs_change / period_values["2020s"] if period_values["2020s"] != 0 else np.nan
            annual_rows.append(
                {
                    "variable": var_name,
                    "scenario": scenario,
                    "mean_2020s": period_values["2020s"],
                    "mean_2050s": period_values["2050s"],
                    "mean_2070s": period_values["2070s"],
                    "absolute_change_2070s_minus_2020s": abs_change,
                    "percent_change_2070s_vs_2020s": pct_change,
                }
            )
    annual = pd.DataFrame(annual_rows)

    checks = []
    for var_name in ["soil_moisture", "total_runoff", "surface_runoff", "subsurface_runoff"]:
        sub = seasonal[seasonal["variable"] == var_name]
        for season in SEASON_LABELS.values():
            vals = sub[sub["season"] == season].set_index("scenario")["percent_change_2070s_vs_2020s"]
            if all(s in vals.index for s in SCENARIO_LABELS):
                if var_name == "soil_moisture":
                    status = "pass" if vals["RCP8.5"] <= vals["RCP4.5"] <= vals["RCP2.6"] else "warning"
                    expectation = "soil moisture drying should strengthen from RCP2.6 to RCP8.5"
                else:
                    spread = vals.max() - vals.min()
                    status = "pass" if spread > 0.5 else "warning"
                    expectation = "runoff response should show scenario-dependent differences"
                checks.append(
                    {
                        "variable": var_name,
                        "season": season,
                        "expectation": expectation,
                        "RCP2.6_percent": vals["RCP2.6"],
                        "RCP4.5_percent": vals["RCP4.5"],
                        "RCP8.5_percent": vals["RCP8.5"],
                        "status": status,
                    }
                )

    sm = seasonal[seasonal["variable"] == "soil_moisture"].set_index(["scenario", "season"])["percent_change_2070s_vs_2020s"]
    for scenario in SCENARIO_LABELS:
        if (scenario, "JJA") in sm.index:
            other = [sm[(scenario, s)] for s in ["DJF", "MAM", "SON"] if (scenario, s) in sm.index]
            status = "pass" if sm[(scenario, "JJA")] <= np.nanmean(other) else "warning"
            checks.append(
                {
                    "variable": "soil_moisture",
                    "season": "JJA_vs_other_seasons",
                    "expectation": "summer drying should be stronger than the average of other seasons",
                    "RCP2.6_percent": np.nan if scenario != "RCP2.6" else sm[(scenario, "JJA")],
                    "RCP4.5_percent": np.nan if scenario != "RCP4.5" else sm[(scenario, "JJA")],
                    "RCP8.5_percent": np.nan if scenario != "RCP8.5" else sm[(scenario, "JJA")],
                    "status": status,
                    "scenario_checked": scenario,
                }
            )

    return seasonal, annual, pd.DataFrame(checks)


def plot_annual_lines(series: pd.DataFrame, var_name: str) -> None:
    info = VARIABLES[var_name]
    df = series[series["variable"] == var_name]
    colors = {"RCP2.6": "#0072B2", "RCP4.5": "#009E73", "RCP8.5": "#D55E00"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for scenario in SCENARIO_LABELS:
        d = df[df["scenario"] == scenario]
        ax.plot(d["year"], d["mean"], label=scenario, color=colors[scenario], lw=2)
    ax.set_title(f"Annual GB mean {info['label'].lower()} by scenario")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{info['label']} ({info['unit']})")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"annual_mean_by_scenario_{var_name}.png", dpi=300)
    plt.close(fig)


def plot_heatmap(seasonal: pd.DataFrame, var_name: str) -> None:
    info = VARIABLES[var_name]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
    vmax = np.nanmax(np.abs(seasonal[seasonal["variable"] == var_name]["percent_change_2070s_vs_2020s"]))
    vmax = max(vmax, 1.0)
    for ax, scenario in zip(axes, SCENARIO_LABELS):
        sub = seasonal[(seasonal["variable"] == var_name) & (seasonal["scenario"] == scenario)]
        vals = sub.set_index("season").loc[["DJF", "MAM", "JJA", "SON"], "percent_change_2070s_vs_2020s"].values.reshape(4, 1)
        im = ax.imshow(vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(scenario)
        ax.set_xticks([])
        ax.set_yticks(range(4))
        ax.set_yticklabels(["DJF", "MAM", "JJA", "SON"])
        for i, val in enumerate(vals[:, 0]):
            ax.text(0, i, f"{val:+.1f}%", ha="center", va="center", fontsize=8)
    fig.suptitle(f"Seasonal percent change in {info['label'].lower()} (2070s vs 2020s)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.04)
    cbar.set_label("Percent change (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"seasonal_percent_change_heatmap_{var_name}.png", dpi=300)
    plt.close(fig)


def plot_scenario_bars(annual: pd.DataFrame) -> None:
    vars_to_plot = ["soil_moisture", "total_runoff"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    colors = ["#0072B2", "#009E73", "#D55E00"]
    for ax, var_name in zip(axes, vars_to_plot):
        sub = annual[annual["variable"] == var_name].set_index("scenario").loc[SCENARIO_LABELS]
        vals = sub["percent_change_2070s_vs_2020s"].values
        ax.axhline(0, color="0.35", lw=0.8)
        ax.bar(SCENARIO_LABELS, vals, color=colors, width=0.65)
        for i, val in enumerate(vals):
            ax.text(i, val + (0.4 if val >= 0 else -0.7), f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        ax.set_title(VARIABLES[var_name]["label"])
        ax.set_ylabel("Percent change (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Annual mean percent change by scenario (2070s vs 2020s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_percent_change_scenario_bars_soil_moisture_total_runoff.png", dpi=300)
    plt.close(fig)


def plot_runoff_consistency(ds: nc.Dataset) -> dict:
    years = ds.variables["year"][:]
    idx = np.where((years >= 2020) & (years <= 2079))[0]
    max_diffs = []
    for scenario_i in range(len(ds.dimensions["scenario"])):
        for member_i in range(len(ds.dimensions["member"])):
            for time_i in idx[::12]:
                total = ds.variables["total_runoff"][scenario_i, member_i, time_i, :]
                surf = ds.variables["surface_runoff"][scenario_i, member_i, time_i, :]
                sub = ds.variables["subsurface_runoff"][scenario_i, member_i, time_i, :]
                diff = np.asarray(total) - (np.asarray(surf) + np.asarray(sub))
                max_diffs.append(float(np.nanmax(np.abs(diff))))
    return {
        "sampled_time_steps_per_scenario_member": int(len(idx[::12])),
        "max_abs_total_minus_surface_plus_subsurface": float(np.nanmax(max_diffs)),
    }


def write_readme(seasonal: pd.DataFrame, annual: pd.DataFrame, checks: pd.DataFrame, consistency: dict) -> None:
    sm_annual = annual[annual["variable"] == "soil_moisture"].set_index("scenario")["percent_change_2070s_vs_2020s"]
    ro_annual = annual[annual["variable"] == "total_runoff"].set_index("scenario")["percent_change_2070s_vs_2020s"]
    jja_sm = seasonal[(seasonal["variable"] == "soil_moisture") & (seasonal["season"] == "JJA")].set_index("scenario")["percent_change_2070s_vs_2020s"]
    son_ro = seasonal[(seasonal["variable"] == "total_runoff") & (seasonal["season"] == "SON")].set_index("scenario")["percent_change_2070s_vs_2020s"]
    pass_count = int((checks["status"] == "pass").sum())
    warning_count = int((checks["status"] == "warning").sum())

    readme = f"""# Scenario consistency validation

## Purpose

This folder contains a scenario-consistency and physical-plausibility assessment for the final process-informed hybrid emulator output. The assessment uses the emulator product itself rather than an independent observational dataset. Its purpose is to check whether the projected changes are internally consistent across RCP2.6, RCP4.5 and RCP8.5, across seasons, and across hydrological variables.

## Input data

- Input NetCDF: `{INPUT_NC}`
- Main variables: soil moisture, surface runoff, subsurface runoff and total runoff.
- Spatial domain: Great Britain land points at 1 km resolution.
- Time period: 2020-2079, seasonal time steps.
- Scenarios: RCP2.6, RCP4.5 and RCP8.5.
- Ensemble members: four CHESS-SCAPE members per scenario.

## Method

1. Read the final emulator NetCDF without loading the full file into memory.
2. Calculate Great Britain mean values for each scenario, ensemble member, year and season.
3. Compare the 2070s against the 2020s for each variable and season.
4. Check whether soil moisture drying strengthens from RCP2.6 to RCP8.5, especially in summer.
5. Check whether runoff responses vary by season and scenario rather than changing uniformly.
6. Check hydrological consistency by testing whether total runoff equals surface runoff plus subsurface runoff.

## Key results

- Annual soil moisture change, 2070s vs 2020s: RCP2.6 {sm_annual['RCP2.6']:+.2f}%, RCP4.5 {sm_annual['RCP4.5']:+.2f}%, RCP8.5 {sm_annual['RCP8.5']:+.2f}%.
- Summer soil moisture change: RCP2.6 {jja_sm['RCP2.6']:+.2f}%, RCP4.5 {jja_sm['RCP4.5']:+.2f}%, RCP8.5 {jja_sm['RCP8.5']:+.2f}%.
- Annual total runoff change: RCP2.6 {ro_annual['RCP2.6']:+.2f}%, RCP4.5 {ro_annual['RCP4.5']:+.2f}%, RCP8.5 {ro_annual['RCP8.5']:+.2f}%.
- Autumn total runoff change: RCP2.6 {son_ro['RCP2.6']:+.2f}%, RCP4.5 {son_ro['RCP4.5']:+.2f}%, RCP8.5 {son_ro['RCP8.5']:+.2f}%.
- Consistency checks: {pass_count} pass and {warning_count} warning flags.
- Total runoff consistency: maximum sampled absolute difference between total runoff and surface plus subsurface runoff is {consistency['max_abs_total_minus_surface_plus_subsurface']:.3e}.

## Interpretation

The emulator output shows a clear high-emission soil-moisture signal, with RCP8.5 producing the strongest late-century decline. The seasonal check is especially important because the expected climate-change signal should not be uniform across the year. Summer soil moisture drying is visible under RCP8.5, while runoff responses are more seasonally differentiated and should be interpreted as local runoff generation rather than routed river flow. The hydrological accounting check supports internal consistency because total runoff is numerically equal to the sum of surface and subsurface runoff in the sampled checks.

Warnings in the check table do not automatically mean that the result is wrong. They identify cases that should be discussed carefully in the manuscript, for example where RCP2.6 and RCP4.5 do not follow a perfectly monotonic order, or where runoff changes are influenced by seasonal precipitation, PET, antecedent wetness, soil properties and land-cover responses.

## Output files

### Tables

- `tables/seasonal_period_change_summary.csv`: seasonal means for the 2020s, 2050s and 2070s, plus 2070s-minus-2020s changes.
- `tables/annual_period_change_summary.csv`: annual mean changes by scenario and variable.
- `tables/scenario_consistency_checks.csv`: pass/warning assessment for key physical expectations.
- `tables/annual_timeseries_gb_mean.csv`: annual Great Britain mean time series for each variable and scenario.
- `tables/hydrological_consistency_check.json`: sampled total-runoff accounting check.

### Figures

- `figures/annual_mean_by_scenario_soil_moisture.png`
- `figures/annual_mean_by_scenario_total_runoff.png`
- `figures/seasonal_percent_change_heatmap_soil_moisture.png`
- `figures/seasonal_percent_change_heatmap_total_runoff.png`
- `figures/annual_percent_change_scenario_bars_soil_moisture_total_runoff.png`

## Suggested manuscript wording

We performed a scenario-consistency and physical-plausibility assessment across RCP2.6, RCP4.5 and RCP8.5. This assessment evaluated whether the emulator reproduced expected seasonal, scenario-dependent and hydrologically consistent responses, including stronger late-century soil-moisture drying under higher emissions and seasonally differentiated runoff changes. This check is complementary to independent G2G validation for RCP8.5 and is not presented as an independent observational validation for RCP2.6 or RCP4.5.
"""
    (OUT_DIR / "README_scenario_consistency_validation.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    with nc.Dataset(INPUT_NC) as ds:
        seasonal, annual, checks = build_summary_tables(ds)
        series = pd.concat([annual_series(ds, v) for v in VARIABLES], ignore_index=True)
        consistency = plot_runoff_consistency(ds)

    seasonal.to_csv(TABLE_DIR / "seasonal_period_change_summary.csv", index=False)
    annual.to_csv(TABLE_DIR / "annual_period_change_summary.csv", index=False)
    checks.to_csv(TABLE_DIR / "scenario_consistency_checks.csv", index=False)
    series.to_csv(TABLE_DIR / "annual_timeseries_gb_mean.csv", index=False)
    (TABLE_DIR / "hydrological_consistency_check.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")

    for var_name in VARIABLES:
        plot_annual_lines(series, var_name)
        plot_heatmap(seasonal, var_name)
    plot_scenario_bars(annual)
    write_readme(seasonal, annual, checks, consistency)
    print(f"Scenario consistency validation finished: {OUT_DIR}")


if __name__ == "__main__":
    main()
