from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


MEMBERS = ["01", "04", "06", "15"]
YEARS = np.arange(1980, 2081, dtype=np.int32)
SEASONS = ["DJF", "MAM", "JJA", "SON"]

KINDS = {
    "soil_moisture": {
        "pattern": "g2g_gb_mmsoil_ukcp18rcm_{member}_1980_2080.nc",
        "source_var": "mmsoil",
        "units": "m water/m soil",
        "long_name": "G2G monthly mean soil moisture",
    },
    "river_flow": {
        "pattern": "g2g_gb_mmflow_ukcp18rcm_{member}_1980_2080.nc",
        "source_var": "mmflow",
        "units": "m3 s-1",
        "long_name": "G2G monthly mean river flow",
    },
}


def configure(root: str):
    root_path = Path(root)
    return {
        "root": root_path,
        "raw": root_path / "raw_tmp",
        "out": root_path / "G2G_validation_full_seasonal_grid_RCP85_1980_2080.nc",
        "txt": root_path / "G2G_validation_full_processing_notes.txt",
        "manifest": root_path / "G2G_validation_full_manifest.json",
        "log": root_path / "G2G_validation_full_processing_log.txt",
    }


def log(paths, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with paths["log"].open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def to_float(arr) -> np.ndarray:
    if np.ma.isMaskedArray(arr):
        out = arr.astype("float32").filled(np.nan)
    else:
        out = np.asarray(arr, dtype="float32")
    return np.where(np.abs(out) > 1.0e30, np.nan, out)


def read_year_months(ds: Dataset):
    t = ds.variables["time"]
    dates = num2date(
        t[:],
        units=t.units,
        calendar=getattr(t, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return [(int(d.year), int(d.month)) for d in dates]


def season_indices(lookup, year: int, season: str):
    if season == "DJF":
        keys = [(year - 1, 12), (year, 1), (year, 2)]
    elif season == "MAM":
        keys = [(year, 3), (year, 4), (year, 5)]
    elif season == "JJA":
        keys = [(year, 6), (year, 7), (year, 8)]
    else:
        keys = [(year, 9), (year, 10), (year, 11)]
    if all(k in lookup for k in keys):
        return [lookup[k] for k in keys]
    return None


def nanmean_grids(grids):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(grids, axis=0), axis=0).astype("float32")


def copy_coordinates(src: Dataset, dst: Dataset):
    for dim in ("y", "x"):
        if dim in src.dimensions and dim not in dst.dimensions:
            dst.createDimension(dim, len(src.dimensions[dim]))

    for name in ("x", "y", "lat", "lon", "x_bnds", "y_bnds", "crsOSGB"):
        if name not in src.variables or name in dst.variables:
            continue
        vsrc = src.variables[name]
        dims = vsrc.dimensions
        for d in dims:
            if d not in dst.dimensions:
                dst.createDimension(d, len(src.dimensions[d]))
        kwargs = {}
        if hasattr(vsrc, "_FillValue"):
            kwargs["fill_value"] = getattr(vsrc, "_FillValue")
        compress = len(dims) >= 2 and vsrc.dtype.kind in "fiu"
        vdst = dst.createVariable(name, vsrc.dtype, dims, zlib=compress, complevel=4, **kwargs)
        vdst[:] = vsrc[:]
        for attr in vsrc.ncattrs():
            if attr != "_FillValue":
                setattr(vdst, attr, getattr(vsrc, attr))


def create_output(paths, template_file: Path):
    if paths["out"].exists():
        paths["out"].unlink()

    with Dataset(template_file) as src, Dataset(paths["out"], "w", format="NETCDF4") as dst:
        dst.createDimension("member", len(MEMBERS))
        dst.createDimension("year", len(YEARS))
        dst.createDimension("season", len(SEASONS))
        dst.createDimension("period", 2)
        copy_coordinates(src, dst)

        dst.title = "Full G2G validation seasonal 1 km grid for emulator comparison"
        dst.summary = (
            "Full seasonal 1 km G2G validation grids from UKCEH/EIDC G2G products driven by "
            "UKCP18 Regional RCM RCP8.5, members 01, 04, 06 and 15. This product is intended "
            "for external validation of seasonal emulator soil moisture and runoff/river-flow behaviour."
        )
        dst.created = datetime.now().isoformat(timespec="seconds")
        dst.scenario = "UKCP18 Regional RCM RCP8.5"
        dst.members = ",".join(MEMBERS)
        dst.season_definition = "DJF, MAM, JJA, SON; DJF labelled by January/February year."

        v = dst.createVariable("member_id", "i4", ("member",))
        v[:] = np.array([int(m) for m in MEMBERS], dtype=np.int32)
        v.long_name = "UKCP18 Regional RCM member identifier"

        v = dst.createVariable("year", "i4", ("year",))
        v[:] = YEARS

        v = dst.createVariable("season_code", "i4", ("season",))
        v[:] = np.arange(1, 5, dtype=np.int32)
        v.long_name = "1=DJF, 2=MAM, 3=JJA, 4=SON"

        v = dst.createVariable("period_code", "i4", ("period",))
        v[:] = np.array([1, 2], dtype=np.int32)
        v.long_name = "1=2020s (2020-2029), 2=2070s (2070-2079)"

        chunk_grid = (1, 1, 1, 100, 100)
        chunk_map = (1, 1, 100, 100)
        for kind, meta in KINDS.items():
            grid = dst.createVariable(
                f"{kind}_seasonal_grid",
                "f4",
                ("member", "year", "season", "y", "x"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=chunk_grid,
                fill_value=np.nan,
            )
            grid.units = meta["units"]
            grid.long_name = f"{meta['long_name']} aggregated to seasonal 1 km grids"
            grid.source_variable = meta["source_var"]

            sm = dst.createVariable(
                f"{kind}_seasonal_gb_mean",
                "f4",
                ("member", "year", "season"),
                zlib=True,
                complevel=4,
                fill_value=np.nan,
            )
            sm.units = meta["units"]
            sm.long_name = f"Spatial mean of {kind} over valid GB grid cells"

            am = dst.createVariable(
                f"{kind}_annual_gb_mean",
                "f4",
                ("member", "year"),
                zlib=True,
                complevel=4,
                fill_value=np.nan,
            )
            am.units = meta["units"]
            am.long_name = f"Annual mean calculated from four seasonal GB means for {kind}"

            for suffix, units in (
                ("period_mean_map", meta["units"]),
                ("change_2070s_minus_2020s", meta["units"]),
                ("percent_change_2070s_vs_2020s", "%"),
            ):
                mv = dst.createVariable(
                    f"{kind}_{suffix}",
                    "f4",
                    ("member", "season", "y", "x") if suffix != "period_mean_map" else ("member", "period", "season", "y", "x"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=chunk_map if suffix != "period_mean_map" else (1, 1, 1, 100, 100),
                    fill_value=np.nan,
                )
                mv.units = units


def process_one(paths, kind: str, member: str):
    meta = KINDS[kind]
    source_path = paths["raw"] / meta["pattern"].format(member=member)
    mi = MEMBERS.index(member)
    log(paths, f"Processing {source_path.name}")

    with Dataset(source_path) as src, Dataset(paths["out"], "a") as dst:
        var = src.variables[meta["source_var"]]
        times = read_year_months(src)
        lookup = {ym: i for i, ym in enumerate(times)}
        seasonal_means = np.full((len(YEARS), len(SEASONS)), np.nan, dtype="float32")
        annual_means = np.full(len(YEARS), np.nan, dtype="float32")

        ydim, xdim = var.shape[1], var.shape[2]
        sum_2020 = np.zeros((4, ydim, xdim), dtype="float64")
        sum_2070 = np.zeros((4, ydim, xdim), dtype="float64")
        cnt_2020 = np.zeros((4, ydim, xdim), dtype="uint16")
        cnt_2070 = np.zeros((4, ydim, xdim), dtype="uint16")

        grid_out = dst.variables[f"{kind}_seasonal_grid"]
        for yi, year in enumerate(YEARS):
            for si, season in enumerate(SEASONS):
                idxs = season_indices(lookup, int(year), season)
                if idxs is None:
                    continue
                grid = nanmean_grids([to_float(var[i, :, :]) for i in idxs])
                grid_out[mi, yi, si, :, :] = grid
                seasonal_means[yi, si] = np.nanmean(grid)

                if 2020 <= int(year) <= 2029:
                    valid = np.isfinite(grid)
                    sum_2020[si][valid] += grid[valid]
                    cnt_2020[si][valid] += 1
                elif 2070 <= int(year) <= 2079:
                    valid = np.isfinite(grid)
                    sum_2070[si][valid] += grid[valid]
                    cnt_2070[si][valid] += 1

            if yi % 5 == 0 or yi == len(YEARS) - 1:
                log(paths, f"  {kind} member {member}: wrote through year {int(year)}")

        with np.errstate(invalid="ignore"):
            annual_means[:] = np.nanmean(seasonal_means, axis=1)
        dst.variables[f"{kind}_seasonal_gb_mean"][mi, :, :] = seasonal_means
        dst.variables[f"{kind}_annual_gb_mean"][mi, :] = annual_means

        with np.errstate(invalid="ignore", divide="ignore"):
            map_2020 = np.where(cnt_2020 > 0, sum_2020 / cnt_2020, np.nan).astype("float32")
            map_2070 = np.where(cnt_2070 > 0, sum_2070 / cnt_2070, np.nan).astype("float32")
            change = (map_2070 - map_2020).astype("float32")
            pct = ((map_2070 - map_2020) / np.where(np.abs(map_2020) > 1.0e-12, map_2020, np.nan) * 100).astype("float32")

        dst.variables[f"{kind}_period_mean_map"][mi, 0, :, :, :] = map_2020
        dst.variables[f"{kind}_period_mean_map"][mi, 1, :, :, :] = map_2070
        dst.variables[f"{kind}_change_2070s_minus_2020s"][mi, :, :, :] = change
        dst.variables[f"{kind}_percent_change_2070s_vs_2020s"][mi, :, :, :] = pct

    return {
        "file": source_path.name,
        "bytes": source_path.stat().st_size,
        "kind": kind,
        "member": member,
    }


def make_figures(paths):
    if plt is None:
        log(paths, "matplotlib unavailable; skipping PNG figures")
        return
    with Dataset(paths["out"]) as ds:
        years = ds.variables["year"][:]
        members = ds.variables["member_id"][:]
        for kind in KINDS:
            annual = np.asarray(ds.variables[f"{kind}_annual_gb_mean"][:], dtype="float32")
            fig, ax = plt.subplots(figsize=(8.8, 5), dpi=180)
            for mi, mem in enumerate(members):
                ax.plot(years, annual[mi], lw=1.3, label=f"member {int(mem):02d}")
            ax.set_title(f"G2G validation annual GB mean: {kind.replace('_', ' ')}")
            ax.set_xlabel("Year")
            ax.set_ylabel(ds.variables[f"{kind}_annual_gb_mean"].units)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(paths["root"] / f"G2G_full_{kind}_annual_members.png")
            plt.close(fig)

            seasonal = np.asarray(ds.variables[f"{kind}_seasonal_gb_mean"][:], dtype="float32")
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180, sharex=True)
            for si, ax in enumerate(axes.ravel()):
                for mi, mem in enumerate(members):
                    ax.plot(years, seasonal[mi, :, si], lw=1.0, label=f"{int(mem):02d}")
                ax.set_title(SEASONS[si])
                ax.grid(alpha=0.25)
            axes.ravel()[0].legend(title="member", ncol=2, fontsize=7)
            fig.suptitle(f"G2G validation seasonal GB mean: {kind.replace('_', ' ')}")
            fig.tight_layout()
            fig.savefig(paths["root"] / f"G2G_full_{kind}_seasonal_members.png")
            plt.close(fig)

            pct = np.asarray(ds.variables[f"{kind}_percent_change_2070s_vs_2020s"][:], dtype="float32")
            ens = np.nanmean(pct, axis=0)
            vmax = np.nanpercentile(np.abs(ens), 98)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1
            fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.5), dpi=180)
            for si, ax in enumerate(axes.ravel()):
                im = ax.imshow(ens[si], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(SEASONS[si])
                ax.set_axis_off()
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
            cbar.set_label("% change, 2070s vs 2020s")
            fig.suptitle(f"G2G validation ensemble mean spatial change: {kind.replace('_', ' ')}")
            fig.savefig(paths["root"] / f"G2G_full_{kind}_spatial_percent_change_2070s_vs_2020s.png", bbox_inches="tight")
            plt.close(fig)


def write_notes(paths, manifest):
    paths["txt"].write_text(
        "G2G full validation processing notes\n"
        "====================================\n\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Output NetCDF: {paths['out'].name}\n\n"
        "Data sources:\n"
        "- Soil moisture: UKCEH/EIDC Grid-to-Grid monthly mean soil moisture driven by UKCP18 Regional RCM RCP8.5, GB, members 01/04/06/15.\n"
        "- River flow: UKCEH/EIDC Grid-to-Grid monthly mean river flow driven by UKCP18 Regional RCM RCP8.5, GB, members 01/04/06/15.\n\n"
        "Processing:\n"
        "- Monthly means were aggregated into seasonal means: DJF, MAM, JJA and SON.\n"
        "- DJF is labelled by the January/February year.\n"
        "- The full NetCDF preserves the 1 km seasonal grid for each member, year and season.\n"
        "- It also includes GB mean seasonal and annual time series, 2020s and 2070s mean maps, absolute changes and percentage changes.\n\n"
        "Interpretation:\n"
        "- G2G soil moisture is a suitable external process-level comparator for emulator soil moisture change.\n"
        "- G2G river flow is not identical to grid-cell total runoff, so it should be used as hydrological consistency validation rather than direct one-to-one runoff validation.\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Full seasonal 1 km G2G validation processing for HPC.")
    parser.add_argument("--root", default=os.environ.get("G2G_VALIDATION_ROOT", "/scratch/users/k2585234/emulator_seasonal/validation_g2g"))
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw input NetCDF files after successful processing.")
    args = parser.parse_args()

    paths = configure(args.root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    (paths["root"] / "logs").mkdir(exist_ok=True)
    if paths["log"].exists():
        paths["log"].unlink()

    missing = []
    for kind, meta in KINDS.items():
        for member in MEMBERS:
            p = paths["raw"] / meta["pattern"].format(member=member)
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise SystemExit("Missing raw files:\n" + "\n".join(missing))

    first_template = paths["raw"] / KINDS["soil_moisture"]["pattern"].format(member=MEMBERS[0])
    log(paths, "Creating full output NetCDF")
    create_output(paths, first_template)

    manifest = []
    for kind in ("soil_moisture", "river_flow"):
        for member in MEMBERS:
            manifest.append(process_one(paths, kind, member))

    log(paths, "Creating PNG figures")
    make_figures(paths)
    write_notes(paths, manifest)
    log(paths, f"Full G2G validation completed: {paths['out']}")

    if args.keep_raw:
        log(paths, "Keeping raw files because --keep-raw was requested")
    else:
        for item in manifest:
            p = paths["raw"] / item["file"]
            if p.exists():
                p.unlink()
                log(paths, f"Deleted raw file {p.name}")


if __name__ == "__main__":
    main()
