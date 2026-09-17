from __future__ import annotations

import json
import argparse
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


DEFAULT_ROOT = Path(os.environ.get("G2G_VALIDATION_ROOT", r"E:\Emulator seasonal\validation"))
ROOT = DEFAULT_ROOT
RAW = ROOT / "raw_tmp"
OUT = ROOT / "G2G_validation_compact_fast_RCP85_1980_2080.nc"
TXT = ROOT / "G2G_validation_fast_processing_notes.txt"
LOG = ROOT / "G2G_validation_fast_processing_log.txt"

MEMBERS = ["01", "04", "06", "15"]
KINDS = {
    "soil_moisture": {
        "pattern": "g2g_gb_mmsoil_ukcp18rcm_{member}_1980_2080.nc",
        "var": "mmsoil",
        "units": "m water/m soil",
        "long": "Monthly mean soil moisture",
    },
    "river_flow": {
        "pattern": "g2g_gb_mmflow_ukcp18rcm_{member}_1980_2080.nc",
        "var": "mmflow",
        "units": "m3 s-1",
        "long": "Monthly mean river flow",
    },
}
SEASONS = ["DJF", "MAM", "JJA", "SON"]
YEARS = np.arange(1980, 2081, dtype=np.int32)


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ym_from_time(ds: Dataset):
    t = ds.variables["time"]
    dates = num2date(t[:], units=t.units, calendar=getattr(t, "calendar", "standard"),
                     only_use_cftime_datetimes=False, only_use_python_datetimes=False)
    return [(int(d.year), int(d.month)) for d in dates]


def idxs_for_season(lookup, year: int, season: str):
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


def as_float(a):
    if np.ma.isMaskedArray(a):
        x = a.astype("float32").filled(np.nan)
    else:
        x = np.asarray(a, dtype="float32")
    x = np.where(np.abs(x) > 1e30, np.nan, x)
    return x


def nanmean_stack(arrs):
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(arrs, axis=0), axis=0).astype("float32")


def process_file(path: Path, varname: str):
    log(f"Open {path.name}")
    with Dataset(path) as ds:
        var = ds.variables[varname]
        times = ym_from_time(ds)
        lookup = {ym: i for i, ym in enumerate(times)}
        ydim, xdim = var.shape[1], var.shape[2]
        seasonal_mean = np.full((len(YEARS), 4), np.nan, dtype="float32")
        p2020_sum = np.zeros((4, ydim, xdim), dtype="float64")
        p2070_sum = np.zeros((4, ydim, xdim), dtype="float64")
        p2020_count = np.zeros((4, ydim, xdim), dtype="uint16")
        p2070_count = np.zeros((4, ydim, xdim), dtype="uint16")

        for yi, year in enumerate(YEARS):
            for si, season in enumerate(SEASONS):
                idxs = idxs_for_season(lookup, int(year), season)
                if idxs is None:
                    continue
                grid = nanmean_stack([as_float(var[i, :, :]) for i in idxs])
                seasonal_mean[yi, si] = float(np.nanmean(grid))
                if 2020 <= year <= 2029:
                    valid = np.isfinite(grid)
                    p2020_sum[si][valid] += grid[valid]
                    p2020_count[si][valid] += 1
                if 2070 <= year <= 2079:
                    valid = np.isfinite(grid)
                    p2070_sum[si][valid] += grid[valid]
                    p2070_count[si][valid] += 1
            if yi % 10 == 0:
                log(f"  {path.name}: processed to year {int(year)}")

        with np.errstate(invalid="ignore", divide="ignore"):
            map2020 = np.where(p2020_count > 0, p2020_sum / p2020_count, np.nan).astype("float32")
            map2070 = np.where(p2070_count > 0, p2070_sum / p2070_count, np.nan).astype("float32")
            change = (map2070 - map2020).astype("float32")
            pct = ((map2070 - map2020) / np.where(np.abs(map2020) > 1e-12, map2020, np.nan) * 100).astype("float32")

        coords = {}
        for cname in ("x", "y", "lat", "lon"):
            if cname in ds.variables:
                coords[cname] = np.asarray(ds.variables[cname][:])
        attrs = {k: getattr(var, k, "") for k in ("units", "long_name")}
    return seasonal_mean, map2020, map2070, change, pct, coords, attrs


def create_output(shape, coords):
    if OUT.exists():
        OUT.unlink()
    ydim, xdim = shape
    with Dataset(OUT, "w", format="NETCDF4") as ds:
        ds.createDimension("member", len(MEMBERS))
        ds.createDimension("year", len(YEARS))
        ds.createDimension("season", 4)
        ds.createDimension("y", ydim)
        ds.createDimension("x", xdim)
        ds.title = "Compact G2G validation dataset for seasonal emulator comparison"
        ds.summary = "GB seasonal/annual means and 2020s vs 2070s spatial changes from G2G UKCP18 RCP8.5 soil moisture and river flow."
        ds.created = datetime.now().isoformat(timespec="seconds")
        ds.scenario = "UKCP18 Regional RCM RCP8.5"
        ds.members = ",".join(MEMBERS)
        ds.createVariable("member_id", "i4", ("member",))[:] = np.array([int(m) for m in MEMBERS])
        ds.createVariable("year", "i4", ("year",))[:] = YEARS
        ds.createVariable("season_code", "i4", ("season",))[:] = np.arange(1, 5)
        for cname, arr in coords.items():
            if arr.ndim == 1 and cname in ("x", "y"):
                v = ds.createVariable(cname, arr.dtype, (cname,))
                v[:] = arr
            elif arr.ndim == 2:
                v = ds.createVariable(cname, "f4", ("y", "x"), zlib=True, complevel=4)
                v[:] = arr.astype("float32")
        for kind, meta in KINDS.items():
            ds.createVariable(f"{kind}_seasonal_gb_mean", "f4", ("member", "year", "season"), zlib=True, complevel=4, fill_value=np.nan)
            ds.createVariable(f"{kind}_annual_gb_mean", "f4", ("member", "year"), zlib=True, complevel=4, fill_value=np.nan)
            for suffix in ["map_2020s", "map_2070s", "change_2070s_minus_2020s", "percent_change_2070s_vs_2020s"]:
                ds.createVariable(f"{kind}_{suffix}", "f4", ("member", "season", "y", "x"), zlib=True, complevel=4, fill_value=np.nan)


def fill_attrs(ds, kind, attrs):
    units = attrs.get("units") or KINDS[kind]["units"]
    for name in ds.variables:
        if name.startswith(kind + "_"):
            ds.variables[name].units = "%" if "percent_change" in name else units
            ds.variables[name].source = "UKCEH/EIDC G2G UKCP18 Regional RCM RCP8.5"


def make_plots():
    if plt is None or not OUT.exists():
        return
    with Dataset(OUT) as ds:
        years = ds.variables["year"][:]
        members = ds.variables["member_id"][:]
        for kind in KINDS:
            annual = ds.variables[f"{kind}_annual_gb_mean"][:]
            fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
            for mi, mem in enumerate(members):
                ax.plot(years, annual[mi], lw=1.2, label=f"member {int(mem):02d}")
            ax.set_title(f"G2G validation annual GB mean: {kind.replace('_', ' ')}")
            ax.set_xlabel("Year")
            ax.set_ylabel(ds.variables[f"{kind}_annual_gb_mean"].units)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(ROOT / f"G2G_validation_{kind}_annual_members.png")
            plt.close(fig)

            pct = np.asarray(ds.variables[f"{kind}_percent_change_2070s_vs_2020s"][:], dtype="float32")
            ens = np.nanmean(pct, axis=0)
            vmax = np.nanpercentile(np.abs(ens), 98)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1
            fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=180)
            for si, ax in enumerate(axes.ravel()):
                im = ax.imshow(ens[si], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(SEASONS[si])
                ax.set_axis_off()
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
            cbar.set_label("% change, 2070s vs 2020s")
            fig.suptitle(f"G2G validation spatial change: {kind.replace('_', ' ')}")
            fig.savefig(ROOT / f"G2G_validation_{kind}_spatial_percent_change.png", bbox_inches="tight")
            plt.close(fig)


def main():
    global ROOT, RAW, OUT, TXT, LOG
    parser = argparse.ArgumentParser(description="Fast local/HPC processing for G2G validation NetCDF files.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Validation working folder containing raw_tmp.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw files after successful processing.")
    args = parser.parse_args()
    ROOT = Path(args.root)
    RAW = ROOT / "raw_tmp"
    OUT = ROOT / "G2G_validation_compact_fast_RCP85_1980_2080.nc"
    TXT = ROOT / "G2G_validation_fast_processing_notes.txt"
    LOG = ROOT / "G2G_validation_fast_processing_log.txt"
    ROOT.mkdir(parents=True, exist_ok=True)

    if LOG.exists():
        LOG.unlink()
    files = sorted(RAW.glob("g2g_gb_mm*flow*_1980_2080.nc")) + sorted(RAW.glob("g2g_gb_mmsoil*_1980_2080.nc"))
    log(f"Found {len(files)} raw NC files")
    missing = []
    for kind, meta in KINDS.items():
        for m in MEMBERS:
            p = RAW / meta["pattern"].format(member=m)
            if not p.exists():
                missing.append(p.name)
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))

    created = False
    manifest = []
    for kind, meta in KINDS.items():
        for m in MEMBERS:
            path = RAW / meta["pattern"].format(member=m)
            result = process_file(path, meta["var"])
            seasonal, m2020, m2070, change, pct, coords, attrs = result
            if not created:
                create_output(m2020.shape[1:], coords)
                created = True
            mi = MEMBERS.index(m)
            with Dataset(OUT, "a") as ds:
                ds.variables[f"{kind}_seasonal_gb_mean"][mi, :, :] = seasonal
                ds.variables[f"{kind}_annual_gb_mean"][mi, :] = np.nanmean(seasonal, axis=1)
                ds.variables[f"{kind}_map_2020s"][mi, :, :, :] = m2020
                ds.variables[f"{kind}_map_2070s"][mi, :, :, :] = m2070
                ds.variables[f"{kind}_change_2070s_minus_2020s"][mi, :, :, :] = change
                ds.variables[f"{kind}_percent_change_2070s_vs_2020s"][mi, :, :, :] = pct
                fill_attrs(ds, kind, attrs)
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "kind": kind, "member": m})
            log(f"Written compact data for {path.name}")

    make_plots()
    TXT.write_text(
        "G2G validation compact processing notes\n"
        "======================================\n\n"
        f"Created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Output NetCDF: {OUT.name}\n\n"
        "Sources:\n"
        "- Soil moisture: UKCEH/EIDC G2G monthly mean soil moisture driven by UKCP18 Regional RCM RCP8.5, GB, members 01/04/06/15.\n"
        "- River flow: UKCEH/EIDC G2G monthly mean river flow driven by UKCP18 Regional RCM RCP8.5, GB, members 01/04/06/15.\n\n"
        "Processing:\n"
        "- Monthly means were converted to seasonal means (DJF, MAM, JJA, SON). DJF is labelled by the January/February year.\n"
        "- The compact file stores GB seasonal and annual mean time series for 1980-2080.\n"
        "- It also stores 1 km seasonal maps for the 2020s and 2070s, plus absolute and percentage changes.\n"
        "- This is for external validation only; it was not used to train the emulator.\n\n"
        "Important limitation:\n"
        "- G2G river flow is not identical to local grid-cell total runoff, so it should be interpreted as hydrological consistency validation rather than direct one-to-one runoff validation.\n",
        encoding="utf-8",
    )
    (ROOT / "G2G_validation_fast_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Compact processing completed successfully")
    if args.keep_raw:
        log("Keeping raw files because --keep-raw was used")
    else:
        for item in manifest:
            p = RAW / item["file"]
            if p.exists():
                p.unlink()
                log(f"Deleted raw file {p.name}")
        for p in RAW.glob("*.part"):
            p.unlink()
            log(f"Deleted partial file {p.name}")


if __name__ == "__main__":
    main()
