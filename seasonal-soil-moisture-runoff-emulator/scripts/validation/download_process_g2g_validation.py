from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
from netCDF4 import Dataset, num2date

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - figures are optional
    plt = None

try:
    import requests
except Exception as exc:
    raise SystemExit("This script needs the 'requests' package. Install it with: python -m pip install requests") from exc


OUT_DIR = Path(r"E:\Emulator seasonal\validation")
RAW_DIR = OUT_DIR / "raw_tmp"
OUT_NC = OUT_DIR / "G2G_UKCP18_RCP85_validation_compact_seasonal_1980_2080.nc"
NOTES_TXT = OUT_DIR / "G2G_validation_processing_notes.txt"
MANIFEST_JSON = OUT_DIR / "G2G_validation_manifest.json"
LOG_TXT = OUT_DIR / "G2G_validation_processing_log.txt"
STATUS_JSON = OUT_DIR / "G2G_validation_status.json"

SOIL_ROOT = "https://catalogue.ceh.ac.uk/datastore/eidchub/f7142ced-f6ff-486b-af33-44fb8f763cde/"
FLOW_ROOT = "https://catalogue.ceh.ac.uk/datastore/eidchub/18be3704-0a6d-4917-aa2e-bf38927321c5/"

# These four UKCP18 RCM members match the members most commonly used in our emulator workflow.
# The script can be extended to all 12 members, but that is intentionally not the default.
MEMBERS = ["01", "04", "06", "15"]

SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
SEASON_NAMES = list(SEASONS)
YEARS = np.arange(1980, 2081, dtype=np.int32)
PERIODS = {
    "2020s": (2020, 2029),
    "2070s": (2070, 2079),
}
PERIOD_NAMES = list(PERIODS)

COORD_NAMES = {
    "time",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "x",
    "y",
    "east",
    "north",
    "easting",
    "northing",
    "projection_x_coordinate",
    "projection_y_coordinate",
    "crs",
    "crsOSGB",
    "transverse_mercator",
}


@dataclass
class SourceSpec:
    label: str
    root: str
    filename_template: str
    preferred_name_tokens: Tuple[str, ...]
    long_name: str
    compact_var_prefix: str
    catalogue: str
    doi: str


SOURCES = [
    SourceSpec(
        label="soil_moisture",
        root=SOIL_ROOT,
        filename_template="g2g_gb_mmsoil_ukcp18rcm_{member}_1980_2080.nc",
        preferred_name_tokens=("mmsoil", "soil", "moist"),
        long_name="G2G monthly mean soil moisture for Great Britain driven by UKCP18 Regional RCM",
        compact_var_prefix="g2g_soil_moisture",
        catalogue="https://catalogue.ceh.ac.uk/id/f7142ced-f6ff-486b-af33-44fb8f763cde",
        doi="https://doi.org/10.5285/f7142ced-f6ff-486b-af33-44fb8f763cde",
    ),
    SourceSpec(
        label="river_flow",
        root=FLOW_ROOT,
        filename_template="g2g_gb_mmflow_ukcp18rcm_{member}_1980_2080.nc",
        preferred_name_tokens=("mmflow", "flow"),
        long_name="G2G monthly mean river flow for Great Britain driven by UKCP18 Regional RCM",
        compact_var_prefix="g2g_river_flow",
        catalogue="https://catalogue.ceh.ac.uk/id/18be3704-0a6d-4917-aa2e-bf38927321c5",
        doi="https://doi.org/10.5285/18be3704-0a6d-4917-aa2e-bf38927321c5",
    ),
]


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_TXT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kwargs) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }
    tmp = STATUS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATUS_JSON)


def progress_bar(percent: Optional[float], width: int = 30) -> str:
    if percent is None or not np.isfinite(percent):
        return "[" + "." * width + "]"
    pct = max(0.0, min(100.0, float(percent)))
    filled = int(round(width * pct / 100.0))
    return "[" + "#" * filled + "." * (width - filled) + f"] {pct:5.1f}%"


def safe_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num_bytes)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{num_bytes} B"


def build_session(no_login: bool = False) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "seasonal-emulator-validation/1.0"})

    if no_login:
        log("Running in no-login mode.")
        return session

    print("\nUKCEH/EIDC authentication")
    print("If downloads work without login, press Enter twice.")
    username = input("Username/email (optional, default jin.rui@kcl.ac.uk): ").strip()
    if username:
        password = getpass("Password: ")
        session.auth = (username, password)
    return session


def download_file(session: requests.Session, url: str, target: Path) -> Dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if target.exists():
        log(f"Using already downloaded file: {target.name} ({safe_size(target.stat().st_size)})")
        return {"url": url, "filename": target.name, "bytes": target.stat().st_size}

    head = session.head(url, allow_redirects=True, timeout=60)
    if head.status_code in (401, 403) and session.auth is None:
        raise RuntimeError(f"Authentication required for {url}. Re-run and enter your UKCEH/EIDC credentials.")
    head.raise_for_status()
    expected = int(head.headers.get("content-length") or 0)
    accept_ranges = str(head.headers.get("accept-ranges", "")).lower() == "bytes"
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing and expected and existing >= expected:
        tmp.rename(target)
        log(f"Recovered completed partial download: {target.name}")
        return {"url": url, "filename": target.name, "bytes": target.stat().st_size}

    if existing:
        log(f"Resuming {target.name} from {safe_size(existing)} of {safe_size(expected)}")
    else:
        log(f"Downloading {target.name} ({safe_size(expected)})")
    write_status(
        phase="downloading",
        current_file=target.name,
        downloaded_bytes=existing,
        total_bytes=expected,
        percent=(existing / expected * 100.0) if expected else None,
        message=f"Downloading {target.name}",
        finished=False,
    )

    max_attempts = 50
    done = existing
    system_curl = Path(r"C:\Windows\System32\curl.exe")
    curl = str(system_curl) if system_curl.exists() else (shutil.which("curl") or shutil.which("curl.exe"))
    if curl is None:
        raise RuntimeError("curl.exe was not found on this Windows system.")

    for attempt in range(1, max_attempts + 1):
        if done and not accept_ranges:
            log("Server did not advertise byte ranges; restarting this file from zero.")
            tmp.unlink(missing_ok=True)
            done = 0

        cmd = [
            curl,
            "--location",
            "--fail",
            "--continue-at",
            "-",
            "--retry",
            "20",
            "--retry-all-errors",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "60",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--output",
            str(tmp),
            url,
        ]
        if session.auth is not None:
            username, password = session.auth
            cmd[1:1] = ["--user", f"{username}:{password}"]

        log(f"  curl attempt {attempt}/{max_attempts} for {target.name}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        last_print = 0.0
        stderr_tail: List[str] = []
        while proc.poll() is None:
            now = time.time()
            if now - last_print >= 10:
                done = tmp.stat().st_size if tmp.exists() else 0
                percent = (done / expected * 100.0) if expected else None
                bar = progress_bar(percent)
                if expected:
                    log(f"  {target.name} {bar} {safe_size(done)} / {safe_size(expected)}")
                else:
                    log(f"  {target.name} {bar} {safe_size(done)}")
                write_status(
                    phase="downloading",
                    current_file=target.name,
                    downloaded_bytes=done,
                    total_bytes=expected,
                    percent=percent,
                    message=f"Downloading with curl; attempt {attempt}/{max_attempts}",
                    finished=False,
                )
                last_print = now
            time.sleep(2)

        stderr = proc.stderr.read() if proc.stderr else ""
        if stderr:
            stderr_tail = stderr.strip().splitlines()[-5:]
        done = tmp.stat().st_size if tmp.exists() else 0
        if proc.returncode == 0 and (expected == 0 or done >= expected):
            break
        if attempt >= max_attempts:
            detail = " | ".join(stderr_tail)
            raise RuntimeError(f"Download failed after {max_attempts} attempts for {target.name}. Last curl output: {detail}")
        wait = min(120, 10 * attempt)
        detail = " | ".join(stderr_tail)
        log(f"WARNING: curl attempt {attempt}/{max_attempts} ended at {safe_size(done)}. Retrying in {wait}s. {detail}")
        write_status(
            phase="retry_wait",
            current_file=target.name,
            downloaded_bytes=done,
            total_bytes=expected,
            percent=(done / expected * 100.0) if expected else None,
            message=f"Retrying curl download. Last output: {detail}",
            finished=False,
        )
        time.sleep(wait)

    if expected and done < expected:
        raise RuntimeError(f"Download incomplete after retries: {safe_size(done)} of {safe_size(expected)} for {target.name}")
    tmp.rename(target)
    write_status(
        phase="downloaded",
        current_file=target.name,
        downloaded_bytes=target.stat().st_size,
        total_bytes=expected,
        percent=100.0 if expected else None,
        message=f"Downloaded {target.name}",
        finished=False,
    )
    return {"url": url, "filename": target.name, "bytes": target.stat().st_size}


def identify_data_variable(ds: Dataset, spec: SourceSpec) -> str:
    candidates: List[Tuple[int, str]] = []
    for name, var in ds.variables.items():
        lname = name.lower()
        if lname in {n.lower() for n in COORD_NAMES}:
            continue
        if not getattr(var, "dimensions", None):
            continue
        if "time" not in [d.lower() for d in var.dimensions]:
            continue
        if len(var.dimensions) < 3:
            continue
        if not np.issubdtype(var.dtype, np.number):
            continue
        score = 0
        attrs = " ".join(str(getattr(var, a, "")) for a in ("long_name", "standard_name", "description", "units")).lower()
        haystack = f"{lname} {attrs}"
        for token in spec.preferred_name_tokens:
            if token.lower() in haystack:
                score += 10
        candidates.append((score, name))

    if not candidates:
        raise RuntimeError(f"Could not identify a time-varying data variable in {ds.filepath()}.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def get_time_values(ds: Dataset) -> List[Tuple[int, int]]:
    if "time" not in ds.variables:
        raise RuntimeError(f"No 'time' variable found in {ds.filepath()}.")
    t = ds.variables["time"]
    vals = t[:]
    units = getattr(t, "units", None)
    calendar = getattr(t, "calendar", "standard")
    if units:
        dates = num2date(vals, units=units, calendar=calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=False)
        return [(int(d.year), int(d.month)) for d in dates]

    # Last-resort fallback for monthly data starting in Jan 1980.
    log("WARNING: time units missing; assuming monthly data starting at 1980-01.")
    out = []
    for i in range(len(vals)):
        y = 1980 + i // 12
        m = 1 + i % 12
        out.append((y, m))
    return out


def season_for_month(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 0
    if month in (1, 2):
        return year, 0
    if month in (3, 4, 5):
        return year, 1
    if month in (6, 7, 8):
        return year, 2
    return year, 3


def to_float_array(a) -> np.ndarray:
    if np.ma.isMaskedArray(a):
        out = a.astype(np.float32).filled(np.nan)
    else:
        out = np.asarray(a, dtype=np.float32)
    # Some NetCDF files use very large fill values without a mask.
    out = np.where(np.abs(out) > 1.0e30, np.nan, out)
    return out


def month_indices_for_year_month(times: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    return {(y, m): i for i, (y, m) in enumerate(times)}


def complete_month_indices(times: List[Tuple[int, int]], season_year: int, season_idx: int) -> Optional[List[int]]:
    lookup = month_indices_for_year_month(times)
    if season_idx == 0:
        needed = [(season_year - 1, 12), (season_year, 1), (season_year, 2)]
    elif season_idx == 1:
        needed = [(season_year, 3), (season_year, 4), (season_year, 5)]
    elif season_idx == 2:
        needed = [(season_year, 6), (season_year, 7), (season_year, 8)]
    else:
        needed = [(season_year, 9), (season_year, 10), (season_year, 11)]
    if all(k in lookup for k in needed):
        return [lookup[k] for k in needed]
    return None


def read_season_grid(var, indices: List[int], spatial_slices=None) -> np.ndarray:
    grids = []
    for idx in indices:
        if spatial_slices is None:
            grids.append(to_float_array(var[idx, ...]))
        else:
            grids.append(to_float_array(var[(idx, *spatial_slices)]))
    return np.nanmean(np.stack(grids, axis=0), axis=0).astype(np.float32)


def finite_mean(arr: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        val = np.nanmean(arr)
    if np.isnan(val):
        return np.nan
    return float(val)


def make_period_sum_arrays(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((len(PERIODS), len(SEASONS), *shape), dtype=np.float64)
    counts = np.zeros((len(PERIODS), len(SEASONS), *shape), dtype=np.uint16)
    return sums, counts


def process_one_source_file(path: Path, spec: SourceSpec) -> Dict[str, object]:
    log(f"Processing {path.name}")
    write_status(phase="processing", current_file=path.name, percent=None, message=f"Processing {path.name}", finished=False)
    with Dataset(path) as ds:
        data_name = identify_data_variable(ds, spec)
        var = ds.variables[data_name]
        times = get_time_values(ds)
        dims = list(var.dimensions)
        time_pos = [d.lower() for d in dims].index("time")
        if time_pos != 0:
            raise RuntimeError(f"Expected time as first dimension for {data_name}; got {dims}.")
        shape = tuple(int(var.shape[i]) for i in range(1, len(var.shape)))
        if len(shape) != 2:
            raise RuntimeError(f"Expected 2D spatial grid after time for {data_name}; got shape {var.shape}.")

        units = getattr(var, "units", "unknown")
        long_name = getattr(var, "long_name", spec.long_name)
        spatial_dim_names = tuple(dims[1:])

        seasonal_mean = np.full((len(YEARS), len(SEASONS)), np.nan, dtype=np.float32)
        period_sums, period_counts = make_period_sum_arrays(shape)

        for yi, season_year in enumerate(YEARS):
            for si, _season_name in enumerate(SEASON_NAMES):
                idxs = complete_month_indices(times, int(season_year), si)
                if idxs is None:
                    continue
                grid = read_season_grid(var, idxs)
                seasonal_mean[yi, si] = finite_mean(grid)
                for pi, (_period_name, (start, end)) in enumerate(PERIODS.items()):
                    if start <= season_year <= end:
                        valid = np.isfinite(grid)
                        period_sums[pi, si][valid] += grid[valid]
                        period_counts[pi, si][valid] += 1
            if yi % 10 == 0 or yi == len(YEARS) - 1:
                pct = (yi + 1) / len(YEARS) * 100.0
                log(f"  processing {path.name}: {progress_bar(pct)} year {int(season_year)}")
                write_status(
                    phase="processing",
                    current_file=path.name,
                    percent=pct,
                    message=f"Processing {path.name}: year {int(season_year)}",
                    finished=False,
                )

        with np.errstate(invalid="ignore", divide="ignore"):
            period_maps = np.where(period_counts > 0, period_sums / period_counts, np.nan).astype(np.float32)
        change_map = (period_maps[1] - period_maps[0]).astype(np.float32)
        pct_change_map = ((period_maps[1] - period_maps[0]) / np.where(np.abs(period_maps[0]) > 1.0e-12, period_maps[0], np.nan) * 100.0).astype(np.float32)
        annual_mean = np.nanmean(seasonal_mean, axis=1).astype(np.float32)

        coord_payload = {}
        for cname in ("x", "y", "east", "north", "easting", "northing", "projection_x_coordinate", "projection_y_coordinate", "lat", "latitude", "lon", "longitude"):
            if cname in ds.variables:
                cv = ds.variables[cname]
                try:
                    arr = np.asarray(cv[:])
                except Exception:
                    continue
                if arr.ndim in (1, 2):
                    coord_payload[cname] = {
                        "data": arr,
                        "dimensions": tuple(cv.dimensions),
                        "attrs": {a: getattr(cv, a) for a in cv.ncattrs()},
                    }

        source_attrs = {
            "source_variable_name": data_name,
            "source_variable_units": units,
            "source_variable_long_name": long_name,
            "source_spatial_dimensions": spatial_dim_names,
            "source_shape": var.shape,
        }

    return {
        "shape": shape,
        "spatial_dim_names": spatial_dim_names,
        "seasonal_mean": seasonal_mean,
        "annual_mean": annual_mean,
        "period_maps": period_maps,
        "change_map": change_map,
        "pct_change_map": pct_change_map,
        "coord_payload": coord_payload,
        "source_attrs": source_attrs,
    }


def create_output_nc(first_result: Dict[str, object]) -> None:
    if OUT_NC.exists():
        OUT_NC.unlink()
    ydim, xdim = first_result["spatial_dim_names"]
    ny, nx = first_result["shape"]

    with Dataset(OUT_NC, "w", format="NETCDF4") as out:
        out.createDimension("member", len(MEMBERS))
        out.createDimension("season_year", len(YEARS))
        out.createDimension("season", len(SEASONS))
        out.createDimension("period", len(PERIODS))
        out.createDimension(ydim, ny)
        out.createDimension(xdim, nx)

        out.title = "Compact G2G validation data for seasonal soil moisture and river flow"
        out.summary = (
            "Compact validation NetCDF derived from UKCEH/EIDC G2G UKCP18 RCM products. "
            "It stores GB mean seasonal/annual time series plus 1 km period mean and change maps "
            "for 2020s and 2070s. Raw monthly source files are not retained."
        )
        out.scenario = "UKCP18 Regional RCM, RCP8.5"
        out.created = datetime.now().isoformat(timespec="seconds")
        out.processing_periods = "2020s=2020-2029; 2070s=2070-2079"
        out.members = ",".join(MEMBERS)

        v = out.createVariable("member_id", "i4", ("member",))
        v[:] = np.array([int(m) for m in MEMBERS], dtype=np.int32)
        v.long_name = "UKCP18 Regional RCM ensemble member identifier"

        v = out.createVariable("season_year", "i4", ("season_year",))
        v[:] = YEARS
        v.long_name = "Season year; DJF is labelled by January/February year"

        v = out.createVariable("season_code", "i4", ("season",))
        v[:] = np.arange(1, len(SEASONS) + 1, dtype=np.int32)
        v.long_name = "Season code: 1=DJF, 2=MAM, 3=JJA, 4=SON"

        v = out.createVariable("period_code", "i4", ("period",))
        v[:] = np.arange(1, len(PERIODS) + 1, dtype=np.int32)
        v.long_name = "Period code: 1=2020s, 2=2070s"

        # Copy useful coordinates when dimensions match the compact output.
        coord_payload = first_result.get("coord_payload", {})
        for cname, item in coord_payload.items():
            arr = item["data"]
            dims = tuple(item["dimensions"])
            if all(d in out.dimensions for d in dims):
                try:
                    cv = out.createVariable(cname, arr.dtype, dims, zlib=True, complevel=4)
                    cv[:] = arr
                    for attr, val in item["attrs"].items():
                        if attr != "_FillValue":
                            setattr(cv, attr, val)
                except Exception as exc:
                    log(f"WARNING: could not copy coordinate {cname}: {exc}")

        for spec in SOURCES:
            prefix = spec.compact_var_prefix
            ts = out.createVariable(f"{prefix}_seasonal_gb_mean", "f4", ("member", "season_year", "season"), zlib=True, complevel=4, fill_value=np.nan)
            ts.long_name = f"{spec.long_name}; seasonal spatial mean over valid GB grid cells"
            ts.units = "set_from_source_file"
            ts.comment = "Seasonal mean from monthly mean G2G source data. This is a compact validation time series, not a full monthly grid."

            ann = out.createVariable(f"{prefix}_annual_gb_mean", "f4", ("member", "season_year"), zlib=True, complevel=4, fill_value=np.nan)
            ann.long_name = f"{spec.long_name}; annual mean over four seasonal GB means"
            ann.units = "set_from_source_file"

            pm = out.createVariable(f"{prefix}_period_mean_map", "f4", ("member", "period", "season", ydim, xdim), zlib=True, complevel=4, fill_value=np.nan)
            pm.long_name = f"{spec.long_name}; 1 km seasonal mean maps for 2020s and 2070s"
            pm.units = "set_from_source_file"

            cm = out.createVariable(f"{prefix}_change_map_2070s_minus_2020s", "f4", ("member", "season", ydim, xdim), zlib=True, complevel=4, fill_value=np.nan)
            cm.long_name = f"{spec.long_name}; 1 km change map, 2070s minus 2020s"
            cm.units = "set_from_source_file"

            pc = out.createVariable(f"{prefix}_percent_change_map_2070s_vs_2020s", "f4", ("member", "season", ydim, xdim), zlib=True, complevel=4, fill_value=np.nan)
            pc.long_name = f"{spec.long_name}; 1 km percentage change map, 2070s relative to 2020s"
            pc.units = "%"


def write_result_to_nc(result: Dict[str, object], spec: SourceSpec, member: str) -> None:
    mi = MEMBERS.index(member)
    prefix = spec.compact_var_prefix
    with Dataset(OUT_NC, "a") as out:
        out.variables[f"{prefix}_seasonal_gb_mean"][mi, :, :] = result["seasonal_mean"]
        out.variables[f"{prefix}_annual_gb_mean"][mi, :] = result["annual_mean"]
        out.variables[f"{prefix}_period_mean_map"][mi, :, :, :, :] = result["period_maps"]
        out.variables[f"{prefix}_change_map_2070s_minus_2020s"][mi, :, :, :] = result["change_map"]
        out.variables[f"{prefix}_percent_change_map_2070s_vs_2020s"][mi, :, :, :] = result["pct_change_map"]
        for suffix in ("seasonal_gb_mean", "annual_gb_mean", "period_mean_map", "change_map_2070s_minus_2020s"):
            v = out.variables[f"{prefix}_{suffix}"]
            v.units = result["source_attrs"]["source_variable_units"]
            v.source_variable_name = result["source_attrs"]["source_variable_name"]
            v.source_variable_long_name = result["source_attrs"]["source_variable_long_name"]


def make_figures() -> None:
    if plt is None:
        log("matplotlib is not available; skipping PNG figures.")
        return
    if not OUT_NC.exists():
        return
    log("Creating compact validation PNG figures")
    with Dataset(OUT_NC) as ds:
        years = ds.variables["season_year"][:]
        members = ds.variables["member_id"][:]
        for spec in SOURCES:
            prefix = spec.compact_var_prefix
            annual = ds.variables[f"{prefix}_annual_gb_mean"][:]
            seasonal = ds.variables[f"{prefix}_seasonal_gb_mean"][:]

            fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
            for i, m in enumerate(members):
                ax.plot(years, annual[i], lw=1.4, label=f"member {int(m):02d}")
            ax.set_title(f"G2G RCP8.5 validation: {spec.label.replace('_', ' ')} annual GB mean")
            ax.set_xlabel("Year")
            ax.set_ylabel(getattr(ds.variables[f"{prefix}_annual_gb_mean"], "units", "value"))
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{prefix}_annual_gb_mean_RCP85_members.png")
            plt.close(fig)

            fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180, sharex=True)
            axes = axes.ravel()
            for si, season in enumerate(SEASON_NAMES):
                ax = axes[si]
                for i, m in enumerate(members):
                    ax.plot(years, seasonal[i, :, si], lw=1.0, label=f"{int(m):02d}")
                ax.set_title(season)
                ax.grid(alpha=0.25)
            axes[0].legend(title="member", ncol=2, fontsize=7)
            fig.suptitle(f"G2G RCP8.5 validation: seasonal {spec.label.replace('_', ' ')}")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{prefix}_seasonal_gb_mean_RCP85_members.png")
            plt.close(fig)

            chg = ds.variables[f"{prefix}_percent_change_map_2070s_vs_2020s"][:]
            ens = np.nanmean(chg, axis=0)
            fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=180)
            axes = axes.ravel()
            vmax = np.nanpercentile(np.abs(ens), 98)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1
            for si, season in enumerate(SEASON_NAMES):
                im = axes[si].imshow(ens[si], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                axes[si].set_title(season)
                axes[si].set_axis_off()
            cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
            cbar.set_label("% change, 2070s vs 2020s")
            fig.suptitle(f"G2G ensemble mean spatial change: {spec.label.replace('_', ' ')}")
            fig.savefig(OUT_DIR / f"{prefix}_spatial_percent_change_2070s_vs_2020s.png", bbox_inches="tight")
            plt.close(fig)


def write_notes(manifest: List[Dict[str, object]]) -> None:
    lines = []
    lines.append("G2G validation data processing notes")
    lines.append("=" * 38)
    lines.append("")
    lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Output NetCDF: {OUT_NC.name}")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This compact dataset is prepared for external validation of the seasonal 1 km soil moisture and runoff emulator. "
        "It is not used to train the emulator. It is intended for comparing broad future trends, seasonal differences and spatial change patterns."
    )
    lines.append("")
    lines.append("Data sources")
    lines.append("------------")
    for spec in SOURCES:
        lines.append(f"- {spec.long_name}")
        lines.append(f"  Catalogue: {spec.catalogue}")
        lines.append(f"  DOI: {spec.doi}")
    lines.append("")
    lines.append("Downloaded files")
    lines.append("----------------")
    for item in manifest:
        lines.append(f"- {item['filename']} ({safe_size(int(item.get('bytes', 0)))})")
    lines.append("")
    lines.append("Processing")
    lines.append("----------")
    lines.append("- Only Great Britain files were used; Northern Ireland files were not downloaded.")
    lines.append(f"- UKCP18 Regional RCM members used: {', '.join(MEMBERS)}.")
    lines.append("- Monthly mean G2G variables were aggregated to seasonal means: DJF, MAM, JJA and SON.")
    lines.append("- DJF is labelled by the January/February year, e.g. December 2019 belongs to DJF 2020.")
    lines.append("- The compact NetCDF stores GB mean seasonal and annual time series for 1980-2080.")
    lines.append("- The compact NetCDF also stores 1 km seasonal mean maps for the 2020s and 2070s, plus 2070s-minus-2020s absolute and percentage change maps.")
    lines.append("- Raw monthly NetCDF files were deleted after successful processing to save disk space.")
    lines.append("")
    lines.append("Important limitation")
    lines.append("--------------------")
    lines.append(
        "The G2G river-flow product is a hydrological river-flow estimate, not the same variable as local grid-cell total runoff. "
        "It should therefore be used as a process-consistency validation, not as a one-to-one numeric validation of emulator total runoff."
    )
    lines.append(
        "The G2G products are driven by UKCP18 Regional RCM RCP8.5, so they mainly validate the RCP8.5 direction, seasonality and spatial pattern."
    )
    NOTES_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and compact G2G validation data.")
    parser.add_argument("--no-login", action="store_true", help="Run without UKCEH/EIDC username/password prompts.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_TXT.exists():
        LOG_TXT.unlink()

    log("Starting compact G2G validation processing")
    log(f"Output folder: {OUT_DIR}")
    log(f"Temporary raw folder: {RAW_DIR}")
    write_status(phase="starting", message="Starting compact G2G validation processing", finished=False)
    session = build_session(no_login=args.no_login)

    manifest: List[Dict[str, object]] = []
    output_created = False

    try:
        for spec in SOURCES:
            for member in MEMBERS:
                filename = spec.filename_template.format(member=member)
                url = urljoin(spec.root, filename)
                raw_path = RAW_DIR / filename

                item = download_file(session, url, raw_path)
                item.update({"source": spec.label, "member": member})
                manifest.append(item)

                result = process_one_source_file(raw_path, spec)
                if not output_created:
                    create_output_nc(result)
                    output_created = True
                write_result_to_nc(result, spec, member)

                log(f"Deleting raw file after successful processing: {raw_path.name}")
                raw_path.unlink(missing_ok=True)
                write_status(
                    phase="completed_file",
                    current_file=filename,
                    percent=100.0,
                    message=f"Completed and deleted raw file: {filename}",
                    finished=False,
                )

                MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                write_notes(manifest)

        make_figures()
        write_notes(manifest)
        MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"Finished successfully. Compact NetCDF: {OUT_NC}")
        write_status(
            phase="finished",
            current_file=None,
            percent=100.0,
            message=f"Finished successfully. Compact NetCDF: {OUT_NC}",
            finished=True,
        )
        try:
            shutil.rmtree(RAW_DIR)
            RAW_DIR.mkdir(exist_ok=True)
            log("Cleaned temporary raw folder.")
        except Exception as exc:
            log(f"WARNING: could not fully clean raw_tmp: {exc}")
    except KeyboardInterrupt:
        log("Stopped by user. Partial raw file(s), if any, remain in raw_tmp.")
        write_status(phase="stopped", message="Stopped by user", finished=False)
        raise
    except Exception as exc:
        log(f"ERROR: {exc}")
        log("The script stopped before deleting the current raw file so it can be inspected or resumed manually.")
        write_status(phase="error", message=str(exc), finished=False)
        raise


if __name__ == "__main__":
    main()
