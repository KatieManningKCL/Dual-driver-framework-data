from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


ROOT = Path("/scratch/users/k2585234/emulator_seasonal/validation_cds_eurocordex")
RAW_DIR = ROOT / "raw_nc"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
EMULATOR_NC = Path(
    "/scratch/users/k2585234/emulator_seasonal/outputs_v2_5models/process_hybrid/"
    "seasonal_soil_moisture_runoff_predictions_1km_2020_2079_process_hybrid.nc"
)

SCENARIO_LABELS = {"rcp26": "RCP2.6", "rcp45": "RCP4.5", "rcp85": "RCP8.5"}
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
CDS_VAR_LABELS = {
    "run": "Mean runoff",
    "rdis": "River discharge",
    "smoist": "Mean soil moisture",
}
CDS_VAR_TO_NC = {
    "run": "run_ymonmean",
    "rdis": "rdis_ymonmean",
    "smoist": "smoist_ymonmean",
}
EMULATOR_VAR_MAP = {
    "run": "total_runoff",
    "smoist": "soil_moisture",
}
PERIOD_ORDER = ["2011-2040", "2041-2070", "2071-2100"]
RCP_ORDER = ["RCP2.6", "RCP4.5", "RCP8.5"]


@dataclass(frozen=True)
class CdsFile:
    path: Path
    variable: str
    hydrological_model: str
    rcp: str
    period: str


def parse_file(path: Path) -> CdsFile:
    pattern = re.compile(
        r"^(?P<var>[^_]+)_ymonmean_abs_(?P<hmodel>.+?)-EUR-11_MOHC-HadGEM2-ES_"
        r"(?P<rcp>rcp\d+)_(?P<member>r\di\dp\d)_SMHI-RCA4-v1_na_"
        r"(?P<period>\d{4}-\d{4})_grid5km_v1\.nc$"
    )
    match = pattern.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse CDS filename: {path.name}")
    return CdsFile(
        path=path,
        variable=match.group("var"),
        hydrological_model=match.group("hmodel"),
        rcp=SCENARIO_LABELS[match.group("rcp")],
        period=match.group("period"),
    )


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def gb_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (lat >= 49.0) & (lat <= 61.5) & (lon >= -8.8) & (lon <= 2.2)


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    values = np.asanyarray(arr)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    else:
        values = np.asarray(values, dtype="float64")
    values = np.where(values >= 1.0e19, np.nan, values)
    values = np.where(mask, values, np.nan)
    return float(np.nanmean(values))


def weighted_seasonal_mean(month_values: dict[int, float], season: str) -> float:
    return float(np.nanmean([month_values[m] for m in SEASON_MONTHS[season]]))


def process_cds_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metadata_rows = []
    for path in sorted(RAW_DIR.glob("*.nc")):
        meta = parse_file(path)
        var_name = CDS_VAR_TO_NC[meta.variable]
        with nc.Dataset(path) as ds:
            lat = np.asarray(ds.variables["lat"][:])
            lon = np.asarray(ds.variables["lon"][:])
            mask = gb_mask(lat, lon)
            data = ds.variables[var_name]
            units = getattr(data, "units", "")
            month_values = {}
            for month_idx in range(12):
                month_values[month_idx + 1] = safe_mean(data[month_idx, :, :], mask)
            for season in ["DJF", "MAM", "JJA", "SON"]:
                rows.append(
                    {
                        "source": "CDS_EUROCORDEX",
                        "variable": meta.variable,
                        "variable_label": CDS_VAR_LABELS[meta.variable],
                        "hydrological_model": meta.hydrological_model,
                        "scenario": meta.rcp,
                        "period": meta.period,
                        "season": season,
                        "mean_value": weighted_seasonal_mean(month_values, season),
                        "units": units,
                    }
                )
            rows.append(
                {
                    "source": "CDS_EUROCORDEX",
                    "variable": meta.variable,
                    "variable_label": CDS_VAR_LABELS[meta.variable],
                    "hydrological_model": meta.hydrological_model,
                    "scenario": meta.rcp,
                    "period": meta.period,
                    "season": "Annual",
                    "mean_value": float(np.nanmean(list(month_values.values()))),
                    "units": units,
                }
            )
            metadata_rows.append(
                {
                    "file": path.name,
                    "variable": meta.variable,
                    "hydrological_model": meta.hydrological_model,
                    "scenario": meta.rcp,
                    "period": meta.period,
                    "units": units,
                    "n_gb_window_cells": int(np.sum(mask)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(metadata_rows)


def process_emulator() -> pd.DataFrame:
    rows = []
    scenario_names = ["RCP2.6", "RCP4.5", "RCP8.5"]
    season_names = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}
    periods = {
        "2020-2040": (2020, 2040),
        "2041-2070": (2041, 2070),
        "2071-2079": (2071, 2079),
    }
    with nc.Dataset(EMULATOR_NC) as ds:
        years = np.asarray(ds.variables["year"][:])
        seasons = np.asarray(ds.variables["season_code"][:])
        for cds_var, emulator_var in EMULATOR_VAR_MAP.items():
            var = ds.variables[emulator_var]
            units = getattr(var, "units", "")
            for scenario_i, scenario in enumerate(scenario_names):
                for period, (start, end) in periods.items():
                    season_values = []
                    for season_code, season_label in season_names.items():
                        idx = np.where((years >= start) & (years <= end) & (seasons == season_code))[0]
                        member_means = []
                        for member_i in range(len(ds.dimensions["member"])):
                            values = var[scenario_i, member_i, idx, :]
                            if np.ma.isMaskedArray(values):
                                values = values.filled(np.nan)
                            member_means.append(float(np.nanmean(values)))
                        mean_value = float(np.nanmean(member_means))
                        season_values.append(mean_value)
                        rows.append(
                            {
                                "source": "emulator",
                                "variable": cds_var,
                                "emulator_variable": emulator_var,
                                "scenario": scenario,
                                "period": period,
                                "season": season_label,
                                "mean_value": mean_value,
                                "units": units,
                            }
                        )
                    rows.append(
                        {
                            "source": "emulator",
                            "variable": cds_var,
                            "emulator_variable": emulator_var,
                            "scenario": scenario,
                            "period": period,
                            "season": "Annual",
                            "mean_value": float(np.nanmean(season_values)),
                            "units": units,
                        }
                    )
    return pd.DataFrame(rows)


def change_table_cds(cds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in cds.groupby(["variable", "variable_label", "hydrological_model", "scenario", "season", "units"]):
        variable, label, hmodel, scenario, season, units = keys
        values = group.set_index("period")["mean_value"]
        if "2011-2040" not in values.index or "2071-2100" not in values.index:
            continue
        early = float(values["2011-2040"])
        late = float(values["2071-2100"])
        rows.append(
            {
                "source": "CDS_EUROCORDEX",
                "variable": variable,
                "variable_label": label,
                "hydrological_model": hmodel,
                "scenario": scenario,
                "season": season,
                "early_period": "2011-2040",
                "late_period": "2071-2100",
                "early_mean": early,
                "late_mean": late,
                "absolute_change": late - early,
                "percent_change": 100.0 * (late - early) / early if early != 0 else np.nan,
                "units": units,
            }
        )
    return pd.DataFrame(rows)


def change_table_emulator(emu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in emu.groupby(["variable", "emulator_variable", "scenario", "season", "units"]):
        variable, emulator_variable, scenario, season, units = keys
        values = group.set_index("period")["mean_value"]
        if "2020-2040" not in values.index or "2071-2079" not in values.index:
            continue
        early = float(values["2020-2040"])
        late = float(values["2071-2079"])
        rows.append(
            {
                "source": "emulator",
                "variable": variable,
                "emulator_variable": emulator_variable,
                "scenario": scenario,
                "season": season,
                "early_period": "2020-2040",
                "late_period": "2071-2079",
                "early_mean": early,
                "late_mean": late,
                "absolute_change": late - early,
                "percent_change": 100.0 * (late - early) / early if early != 0 else np.nan,
                "units": units,
            }
        )
    return pd.DataFrame(rows)


def comparison_table(cds_change: pd.DataFrame, emu_change: pd.DataFrame) -> pd.DataFrame:
    cds_ens = (
        cds_change.groupby(["variable", "scenario", "season"], as_index=False)
        .agg(
            cds_percent_mean=("percent_change", "mean"),
            cds_percent_min=("percent_change", "min"),
            cds_percent_max=("percent_change", "max"),
            n_cds_models=("hydrological_model", "nunique"),
        )
    )
    emu = emu_change[["variable", "scenario", "season", "percent_change"]].rename(
        columns={"percent_change": "emulator_percent_change"}
    )
    comp = cds_ens.merge(emu, on=["variable", "scenario", "season"], how="left")
    comp["same_direction"] = np.sign(comp["cds_percent_mean"]) == np.sign(comp["emulator_percent_change"])
    comp["absolute_difference_percentage_points"] = (
        comp["emulator_percent_change"] - comp["cds_percent_mean"]
    ).abs()
    return comp


def ranking_table(change: pd.DataFrame, source_name: str, value_col: str = "percent_change") -> pd.DataFrame:
    rows = []
    data = change.copy()
    if source_name == "CDS_EUROCORDEX":
        data = data.groupby(["variable", "scenario", "season"], as_index=False).agg(percent_change=("percent_change", "mean"))
    for keys, group in data.groupby(["variable", "season"]):
        variable, season = keys
        vals = group.set_index("scenario")[value_col if value_col in group.columns else "percent_change"]
        if all(rcp in vals.index for rcp in RCP_ORDER):
            ordered = " > ".join(vals.sort_values(ascending=False).index.tolist())
            rows.append(
                {
                    "source": source_name,
                    "variable": variable,
                    "season": season,
                    "RCP2.6": vals["RCP2.6"],
                    "RCP4.5": vals["RCP4.5"],
                    "RCP8.5": vals["RCP8.5"],
                    "ranking_high_to_low": ordered,
                }
            )
    return pd.DataFrame(rows)


def plot_annual_bars(comp: pd.DataFrame) -> None:
    plot_vars = [("run", "Runoff"), ("smoist", "Soil moisture")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    colors = {"CDS": "#0072B2", "Emulator": "#D55E00"}
    for ax, (var, label) in zip(axes, plot_vars):
        sub = comp[(comp["variable"] == var) & (comp["season"] == "Annual")].set_index("scenario").loc[RCP_ORDER]
        x = np.arange(len(RCP_ORDER))
        ax.axhline(0, color="0.35", lw=0.8)
        ax.bar(x - 0.18, sub["cds_percent_mean"], width=0.35, color=colors["CDS"], label="CDS/EURO-CORDEX")
        ax.bar(x + 0.18, sub["emulator_percent_change"], width=0.35, color=colors["Emulator"], label="Emulator")
        ax.set_xticks(x)
        ax.set_xticklabels(RCP_ORDER)
        ax.set_ylabel("Percent change (%)")
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Annual percent change: late vs early future")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_percent_change_emulator_vs_cds.png", dpi=300)
    plt.close(fig)


def plot_heatmap(comp: pd.DataFrame, variable: str, title: str) -> None:
    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    vals_all = []
    for source_col in ["cds_percent_mean", "emulator_percent_change"]:
        pivot = comp[(comp["variable"] == variable) & (comp["season"].isin(seasons))].pivot(
            index="season", columns="scenario", values=source_col
        ).loc[seasons, RCP_ORDER]
        vals_all.extend(pivot.values.ravel().tolist())
    vmax = max(1.0, float(np.nanmax(np.abs(vals_all))))
    for ax, source_col, source_label in zip(axes, ["cds_percent_mean", "emulator_percent_change"], ["CDS/EURO-CORDEX", "Emulator"]):
        pivot = comp[(comp["variable"] == variable) & (comp["season"].isin(seasons))].pivot(
            index="season", columns="scenario", values=source_col
        ).loc[seasons, RCP_ORDER]
        im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(RCP_ORDER)))
        ax.set_xticklabels(RCP_ORDER, rotation=30, ha="right")
        ax.set_yticks(range(len(seasons)))
        ax.set_yticklabels(seasons)
        ax.set_title(source_label)
        for i in range(len(seasons)):
            for j in range(len(RCP_ORDER)):
                ax.text(j, i, f"{pivot.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04)
    cbar.set_label("Percent change (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"seasonal_heatmap_{variable}_emulator_vs_cds.png", dpi=300)
    plt.close(fig)


def plot_river_discharge(cds_change: pd.DataFrame) -> None:
    sub = cds_change[(cds_change["variable"] == "rdis") & (cds_change["season"] == "Annual")]
    summary = sub.groupby("scenario", as_index=False).agg(mean=("percent_change", "mean"), min=("percent_change", "min"), max=("percent_change", "max"))
    summary = summary.set_index("scenario").loc[RCP_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(summary))
    ax.axhline(0, color="0.35", lw=0.8)
    ax.bar(x, summary["mean"], color="#009E73", width=0.6)
    ax.errorbar(
        x,
        summary["mean"],
        yerr=[summary["mean"] - summary["min"], summary["max"] - summary["mean"]],
        fmt="none",
        ecolor="0.2",
        capsize=4,
        lw=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"])
    ax.set_ylabel("Percent change (%)")
    ax.set_title("CDS/EURO-CORDEX river discharge change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_percent_change_cds_river_discharge.png", dpi=300)
    plt.close(fig)


def write_reports(
    cds_change: pd.DataFrame,
    emu_change: pd.DataFrame,
    comp: pd.DataFrame,
    rank: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    annual = comp[comp["season"] == "Annual"].copy()
    agreement_rate = float(comp["same_direction"].mean() * 100.0)
    annual_agreement = float(annual["same_direction"].mean() * 100.0)
    lines = [
        "CDS/EURO-CORDEX hydrology validation summary",
        "",
        f"Raw CDS files processed: {len(metadata)}",
        "Variables: mean runoff (run), river discharge (rdis), mean soil moisture (smoist)",
        "Scenarios: RCP2.6, RCP4.5, RCP8.5",
        "CDS periods: 2011-2040, 2041-2070, 2071-2100",
        "Emulator periods: 2020-2040, 2041-2070, 2071-2079",
        "",
        f"Direction agreement across runoff and soil-moisture comparison rows: {agreement_rate:.1f}%",
        f"Annual direction agreement: {annual_agreement:.1f}%",
        "",
        "Annual comparison:",
    ]
    for _, row in annual.sort_values(["variable", "scenario"]).iterrows():
        lines.append(
            f"- {row['variable']} {row['scenario']}: CDS {row['cds_percent_mean']:+.2f}% "
            f"(range {row['cds_percent_min']:+.2f} to {row['cds_percent_max']:+.2f}%), "
            f"emulator {row['emulator_percent_change']:+.2f}%, same direction={row['same_direction']}"
        )
    (OUTPUT_DIR / "validation_summary_report.txt").write_text("\n".join(lines), encoding="utf-8")

    readme = f"""# CDS/EURO-CORDEX Hydrology Validation

## Purpose

This workflow uses the Copernicus Climate Data Store hydrology-related climate impact indicators as an external hydrology benchmark for the seasonal emulator. The comparison is designed as an external consistency check, not a direct pixel-level validation, because the CDS products are 5 km time-slice climatologies while the emulator outputs are 1 km seasonal projections for Great Britain.

## Input Data

- CDS dataset: Hydrology-related climate impact indicators from 1970 to 2100 derived from bias-adjusted European climate projections.
- CDS DOI: 10.24381/cds.73237ad6.
- CDS variables used: mean runoff (`run`), river discharge (`rdis`) and mean soil moisture (`smoist`).
- Hydrological models: E-HYPEgrid and VIC-WUR.
- Climate chain: MOHC-HadGEM2-ES driven by SMHI-RCA4, ensemble member r1i1p1.
- Scenarios: RCP2.6, RCP4.5 and RCP8.5.
- CDS periods: 2011-2040, 2041-2070 and 2071-2100.
- Emulator file: `{EMULATOR_NC}`.

## Method

1. The CDS 5 km grids were clipped to a broad Great Britain window using latitude and longitude bounds.
2. Monthly mean values were aggregated to DJF, MAM, JJA, SON and annual means.
3. CDS late-future change was calculated as 2071-2100 relative to 2011-2040.
4. Emulator late-future change was calculated as 2071-2079 relative to 2020-2040.
5. The comparison focuses on percentage change, direction of change, RCP ranking and seasonal patterns.

## Key Results

- Overall direction agreement across runoff and soil-moisture comparison rows: {agreement_rate:.1f}%.
- Annual direction agreement: {annual_agreement:.1f}%.
- The river-discharge product is used as an additional hydrological response benchmark, but it is not compared one-to-one with local emulator runoff.
- Any mismatch should be interpreted in light of differences in spatial resolution, time-slice definition, hydrological model structure and variable definition.

## Important Interpretation

This validation is strongest for broad hydroclimatic consistency: whether the emulator gives plausible RCP ordering and seasonal direction of change. It is weaker for absolute magnitude because local runoff generation, gridded runoff, river discharge and soil moisture are not identical quantities.

## Outputs

- `tables/cds_monthly_to_seasonal_means.csv`
- `tables/cds_period_change_summary.csv`
- `tables/emulator_period_change_summary.csv`
- `tables/emulator_vs_cds_agreement_table.csv`
- `tables/rcp_ranking_comparison.csv`
- `tables/cds_file_manifest.csv`
- `figures/annual_percent_change_emulator_vs_cds.png`
- `figures/seasonal_heatmap_run_emulator_vs_cds.png`
- `figures/seasonal_heatmap_smoist_emulator_vs_cds.png`
- `figures/annual_percent_change_cds_river_discharge.png`
- `validation_summary_report.txt`
"""
    (OUTPUT_DIR / "README_CDS_EUROCORDEX_validation.md").write_text(readme, encoding="utf-8")

    methods = """# Methods and Limitations

Suggested manuscript wording:

We used hydrology-related climate impact indicators from the Copernicus Climate Data Store as an external hydrological consistency benchmark. Monthly 5 km CDS indicators were aggregated to seasonal and annual means over a broad Great Britain window. Changes between 2011-2040 and 2071-2100 were compared with emulator changes between the early future and late-century periods. The comparison focused on direction of change, scenario ordering and seasonal behaviour rather than exact magnitudes.

Limitations:

1. The CDS products are 5 km time-slice climatologies, while the emulator product is a 1 km seasonal time series.
2. CDS runoff and river discharge are generated by independent hydrological models, whereas the emulator predicts local hydrological variables trained from CHESS-land.
3. River discharge is routed flow and should not be interpreted as the same quantity as local total runoff.
4. The downloaded CDS subset uses one GCM-RCM chain and two hydrological models, so it supports an external consistency check rather than a full multi-model uncertainty analysis.
"""
    (OUTPUT_DIR / "METHODS_AND_LIMITATIONS.md").write_text(methods, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cds, metadata = process_cds_files()
    emu = process_emulator()
    cds_change = change_table_cds(cds)
    emu_change = change_table_emulator(emu)
    comp = comparison_table(cds_change, emu_change)
    rank = pd.concat(
        [
            ranking_table(cds_change, "CDS_EUROCORDEX"),
            ranking_table(emu_change, "emulator"),
        ],
        ignore_index=True,
    )

    cds.to_csv(TABLE_DIR / "cds_monthly_to_seasonal_means.csv", index=False)
    metadata.to_csv(TABLE_DIR / "cds_file_manifest.csv", index=False)
    cds_change.to_csv(TABLE_DIR / "cds_period_change_summary.csv", index=False)
    emu_change.to_csv(TABLE_DIR / "emulator_period_change_summary.csv", index=False)
    comp.to_csv(TABLE_DIR / "emulator_vs_cds_agreement_table.csv", index=False)
    rank.to_csv(TABLE_DIR / "rcp_ranking_comparison.csv", index=False)

    plot_annual_bars(comp)
    plot_heatmap(comp, "run", "Seasonal runoff percent change: CDS/EURO-CORDEX vs emulator")
    plot_heatmap(comp, "smoist", "Seasonal soil moisture percent change: CDS/EURO-CORDEX vs emulator")
    plot_river_discharge(cds_change)
    write_reports(cds_change, emu_change, comp, rank, metadata)

    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "raw_files": len(metadata),
                "output_tables": len(list(TABLE_DIR.glob("*.csv"))),
                "output_figures": len(list(FIG_DIR.glob("*.png"))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished CDS/EURO-CORDEX validation: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
