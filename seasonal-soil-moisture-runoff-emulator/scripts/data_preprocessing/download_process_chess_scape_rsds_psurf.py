"""
Download and compact CHESS-SCAPE seasonal rsds and psurf for the seasonal emulator.

What this script does
---------------------
1. Downloads one source NetCDF file at a time from the CEDA CHESS-SCAPE archive.
2. Keeps rsds from the bias-corrected seasonal archive.
3. Keeps psurf from the non-bias-corrected seasonal archive, because psurf is not
   present in the bias-corrected seasonal directory.
4. Writes one compact NetCDF:
      CHESS-SCAPE_uk_1km_seasonal_rsds_psurf_1980-2080_by_scenario_member.nc
5. Deletes each raw downloaded source file immediately after it is merged.

Run from a normal Windows cmd/PowerShell, not inside Codex:
    python "E:\\Emulator seasonal\\download_process_chess_scape_rsds_psurf.py"

If CEDA asks for authentication, enter your CEDA username/password when prompted.
"""

from __future__ import annotations

from getpass import getpass
from pathlib import Path
from urllib.parse import urljoin, urlparse
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import requests

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    print("Missing required Python packages.")
    print("Please run this once in Anaconda Prompt or cmd:")
    print("    pip install netCDF4 requests tqdm")
    print(f"Original import error: {exc}")
    raise SystemExit(1)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -------------------------
# User settings
# -------------------------

BASE = "https://data.ceda.ac.uk/badc/deposited2021/chess-scape/data/"

OUT_ROOT = Path(r"E:\Emulator seasonal")
OUT_DIR = OUT_ROOT / "CHESS-SCAPE future climate rsds_psurf seasonal, 1980-2080"
TMP_DIR = OUT_DIR / "raw_tmp"
OUT_NC = OUT_DIR / "CHESS-SCAPE_uk_1km_seasonal_rsds_psurf_1980-2080_by_scenario_member.nc"
REPORT_TXT = OUT_DIR / "CHESS-SCAPE_rsds_psurf_processing_notes.txt"

SCENARIOS = {
    "rcp26": 26,
    "rcp45": 45,
    "rcp85": 85,
}
MEMBERS = ["01", "04", "06", "15"]

VARIABLES = {
    # rsds exists in the bias-corrected seasonal directory.
    "rsds": {
        "subdir_kind": "bias-corrected",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "Surface downwelling shortwave radiation",
        "units": "W m-2",
    },
    # psurf is present in the non-bias-corrected seasonal directory.
    "psurf": {
        "subdir_kind": "raw",
        "standard_name": "surface_air_pressure",
        "long_name": "Surface air pressure",
        "units": "Pa",
    },
}

START_FRESH = True


# -------------------------
# Authentication and HTTP
# -------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    print("CEDA authentication:")
    print("  If the data download works without login, just press Enter for username.")
    username = input("CEDA username/email (optional): ").strip()
    if username:
        password = getpass("CEDA password: ")
        session.auth = (username, password)
    session.headers.update({"User-Agent": "seasonal-emulator-chess-scape-downloader/1.0"})
    return session


def get_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=120)
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Access denied for {url}\n"
            "Please make sure your CEDA account has accepted the dataset licence, "
            "then rerun the script and enter your CEDA username/password."
        )
    response.raise_for_status()
    return response.text


def download_file(session: requests.Session, url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with session.get(url, stream=True, timeout=300) as response:
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Access denied for {url}\n"
                "Please check CEDA credentials and dataset licence acceptance."
            )
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=out_path.name[:45],
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    tmp_path.rename(out_path)
    return out_path


def filename_from_url(url: str) -> str:
    """Return a Windows-safe filename from a CEDA URL that may include ?download=1."""
    return Path(urlparse(url).path).name


# -------------------------
# CEDA path discovery
# -------------------------

def source_directory(scenario: str, member: str, var: str) -> str:
    kind = VARIABLES[var]["subdir_kind"]
    if kind == "bias-corrected":
        return urljoin(BASE, f"{scenario}_bias-corrected/{member}/seasonal/")
    return urljoin(BASE, f"{scenario}/{member}/seasonal/")


def find_source_url(session: requests.Session, scenario: str, member: str, var: str) -> str:
    directory = source_directory(scenario, member, var)
    html = get_html(session, directory)
    links = sorted(set(re.findall(r"https://dap\.ceda\.ac\.uk/badc/[^\"'<>]+\.nc(?:\?download=1)?", html)))
    if not links:
        links = sorted(set(re.findall(r"[^\"'<>\\s]+\.nc(?:\?download=1)?", html)))
    matches = [
        link for link in links
        if f"_{member}_{var}_" in link or f"_{var}_" in link
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find variable {var} in {directory}\n"
            f"Available NetCDF links: {links[:20]}"
        )
    if len(matches) > 1:
        # Prefer the file whose name contains the expected scenario/member/var pattern.
        exact = [m for m in matches if scenario in m and member in m and f"_{var}_" in m]
        matches = exact or matches
    return urljoin(directory, matches[0])


# -------------------------
# NetCDF writing
# -------------------------

def first_download_for_template(session: requests.Session) -> tuple[str, Path]:
    first_var = "rsds"
    first_scenario = "rcp26"
    first_member = "01"
    url = find_source_url(session, first_scenario, first_member, first_var)
    local = TMP_DIR / filename_from_url(url)
    print(f"Downloading template file: {url}")
    download_file(session, url, local)
    return first_var, local


def find_data_var(ds: xr.Dataset, expected: str) -> str:
    if expected in ds.variables:
        return expected
    candidates = [v for v in ds.variables if expected.lower() in v.lower()]
    if candidates:
        return candidates[0]
    for v in ds.variables:
        dims = ds.variables[v].dimensions
        if "time" in dims and len(dims) >= 3:
            return v
    raise ValueError(f"Cannot identify data variable for {expected}. Variables: {list(ds.variables)}")


def create_output_from_template(template_path: Path) -> tuple[str, str]:
    if OUT_NC.exists() and START_FRESH:
        OUT_NC.unlink()

    ds = Dataset(template_path, "r")
    data_var = find_data_var(ds, "rsds")
    src_var = ds.variables[data_var]
    dims = src_var.dimensions

    if "time" not in dims:
        raise ValueError(f"Template variable {data_var} has no time dimension: {dims}")
    spatial_dims = [d for d in dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dims, got {spatial_dims} from {dims}")
    ydim, xdim = spatial_dims

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nc = Dataset(OUT_NC, "w", format="NETCDF4")

    nc.createDimension("scenario", len(SCENARIOS))
    nc.createDimension("member", len(MEMBERS))
    for dim_name, dim in ds.dimensions.items():
        if dim_name in nc.dimensions:
            continue
        nc.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

    scen = nc.createVariable("scenario", "i2", ("scenario",))
    scen[:] = list(SCENARIOS.values())
    scen.long_name = "Representative Concentration Pathway code"
    scen.labels = ",".join(SCENARIOS.keys())

    mem = nc.createVariable("member", "i2", ("member",))
    mem[:] = [int(m) for m in MEMBERS]
    mem.long_name = "CHESS-SCAPE ensemble member"
    mem.labels = ",".join(MEMBERS)

    for coord in ["time", "time_bnds", xdim, ydim, "x_bnds", "y_bnds", "lat", "lon"]:
        if coord not in ds.variables:
            continue
        copy_variable(ds, nc, coord, compress=True)

    # CRS/grid mapping variables, if present.
    for coord in ["crsOSGB", "crs"]:
        if coord in ds.variables and coord not in nc.variables:
            copy_variable(ds, nc, coord, compress=False)

    chunks = (
        1,
        1,
        1,
        min(256, len(ds.dimensions[ydim])),
        min(256, len(ds.dimensions[xdim])),
    )
    for var, meta in VARIABLES.items():
        out = nc.createVariable(
            var,
            "f4",
            ("scenario", "member", "time", ydim, xdim),
            zlib=True,
            complevel=4,
            chunksizes=chunks,
            fill_value=np.float32(np.nan),
        )
        out.standard_name = meta["standard_name"]
        out.long_name = meta["long_name"]
        out.units = meta["units"]
        out.description = (
            f"Merged seasonal {var} from CHESS-SCAPE. "
            "Dimensions are scenario, member, time, y, x."
        )

    nc.title = "CHESS-SCAPE seasonal rsds and psurf for Great Britain, 1980-2080"
    nc.source = "CEDA CHESS-SCAPE archive: https://data.ceda.ac.uk/badc/deposited2021/chess-scape/data"
    nc.processing = (
        "Downloaded one source NetCDF at a time; rsds from bias-corrected seasonal files, "
        "psurf from non-bias-corrected seasonal files; each source file was deleted after merging."
    )
    nc.scenarios = ",".join(SCENARIOS.keys())
    nc.members = ",".join(MEMBERS)
    nc.close()
    ds.close()
    return ydim, xdim


def copy_attrs(src, dst, skip_fill_value: bool = True) -> None:
    for attr in src.ncattrs():
        if skip_fill_value and attr == "_FillValue":
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            setattr(dst, attr, str(getattr(src, attr)))


def copy_variable(src_ds: Dataset, dst_ds: Dataset, name: str, compress: bool) -> None:
    src = src_ds.variables[name]
    fill_value = getattr(src, "_FillValue", None)
    kwargs = {}
    if compress and src.ndim > 0:
        kwargs.update({"zlib": True, "complevel": 4})
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    dst = dst_ds.createVariable(name, src.datatype, src.dimensions, **kwargs)
    copy_attrs(src, dst)
    if src.ndim == 0:
        try:
            dst.assignValue(src.getValue())
        except Exception:
            pass
    else:
        dst[:] = src[:]


def merge_one_file(local_path: Path, var: str, scenario_index: int, member_index: int) -> None:
    src = Dataset(local_path, "r")
    data_var = find_data_var(src, var)
    src_data = src.variables[data_var]

    if "time" not in src_data.dimensions:
        src.close()
        raise ValueError(f"Variable {data_var} in {local_path} has no time dimension")

    time_axis = src_data.dimensions.index("time")
    if time_axis != 0:
        src.close()
        raise ValueError(
            f"Expected time to be the first dimension for {data_var}, got {src_data.dimensions}"
        )

    dst = Dataset(OUT_NC, "a")
    dst_var = dst.variables[var]
    nt = len(src.dimensions["time"])

    for t in tqdm(range(nt), desc=f"writing {var} t-slices", leave=False):
        arr = src_data[t, :, :]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        dst_var[scenario_index, member_index, t, :, :] = np.asarray(arr, dtype="float32")

    dst.close()
    src.close()


def write_report(processed: list[tuple[str, str, str, str]]) -> None:
    lines = [
        "CHESS-SCAPE rsds and psurf seasonal processing report",
        "",
        "Purpose:",
        "This file adds shortwave radiation (rsds) and surface pressure (psurf) to the existing seasonal emulator inputs.",
        "",
        "Source:",
        "CEDA CHESS-SCAPE archive: https://data.ceda.ac.uk/badc/deposited2021/chess-scape/data",
        "",
        "Processing:",
        "- Variables retained: rsds and psurf only.",
        "- Scenarios retained: rcp26, rcp45, rcp85.",
        "- Ensemble members retained: 01, 04, 06, 15.",
        "- rsds was taken from bias-corrected seasonal files.",
        "- psurf was taken from non-bias-corrected seasonal files because psurf is not present in the bias-corrected seasonal directory.",
        "- Each raw NetCDF source file was downloaded, merged into the compact output NetCDF, and deleted immediately.",
        "",
        "Output:",
        str(OUT_NC),
        "",
        "Processed files:",
    ]
    for var, scenario, member, url in processed:
        lines.append(f"- {var}, {scenario}, member {member}: {url}")
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Output folder:", OUT_DIR)
    print("Final NetCDF:", OUT_NC)
    print("Temporary raw files:", TMP_DIR)
    print()

    session = make_session()
    processed = []

    first_var, template_path = first_download_for_template(session)
    print("Creating compact output NetCDF...")
    create_output_from_template(template_path)
    template_path.unlink(missing_ok=True)

    tasks = []
    for var in VARIABLES:
        for scenario in SCENARIOS:
            for member in MEMBERS:
                tasks.append((var, scenario, member))

    for var, scenario, member in tasks:
        scen_i = list(SCENARIOS.keys()).index(scenario)
        mem_i = MEMBERS.index(member)
        print()
        print(f"Processing {var} | {scenario} | member {member}")
        url = find_source_url(session, scenario, member, var)
        local = TMP_DIR / filename_from_url(url)
        try:
            download_file(session, url, local)
            merge_one_file(local, var, scen_i, mem_i)
            processed.append((var, scenario, member, url))
            print(f"Finished and merged: {local.name}")
        finally:
            local.unlink(missing_ok=True)

    # Clean temp folder if empty.
    try:
        TMP_DIR.rmdir()
    except OSError:
        pass

    write_report(processed)

    print()
    print("All done.")
    print("Saved NetCDF:", OUT_NC)
    print("Saved report:", REPORT_TXT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as exc:
        print("\nERROR:", exc)
        print("The partially downloaded raw file, if any, is in:", TMP_DIR)
        raise
