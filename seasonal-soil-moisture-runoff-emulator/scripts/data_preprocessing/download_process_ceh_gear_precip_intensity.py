"""
Download and compact CEH-GEAR daily rainfall into seasonal rainfall-intensity features.

This is the "third dataset" for the seasonal emulator:
precipitation intensity and antecedent wetness features.

The script downloads one annual daily NetCDF file at a time, derives seasonal
1 km indicators, writes them into one compact NetCDF, and deletes the raw
annual file immediately.

Output variables:
- precip_total: seasonal total rainfall, mm season-1
- precip_max_1day: maximum daily rainfall within the season, mm day-1
- wet_day_count: number of days with rainfall >= 1 mm day-1
- heavy_day_count: number of days with rainfall >= 10 mm day-1
- antecedent_precip_total: previous-season precip_total, mm season-1
- antecedent_precip_max_1day: previous-season precip_max_1day, mm day-1

Run from Windows cmd/PowerShell:
    python "E:\\Emulator seasonal\\download_process_ceh_gear_precip_intensity.py"
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
import re

import numpy as np
import requests

try:
    from netCDF4 import Dataset, date2num, num2date
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


BASE_URL = "https://catalogue.ceh.ac.uk/datastore/eidchub/dbf13dd5-90cd-457a-a986-f2f9dd97e93c/GB/daily/"
OUT_ROOT = Path(r"E:\Emulator seasonal")
OUT_DIR = OUT_ROOT / "CEH-GEAR_precip_intensity_seasonal_1961_2019"
TMP_DIR = OUT_DIR / "raw_tmp"
OUT_NC = OUT_DIR / "CEH-GEAR_GB_1km_seasonal_precip_intensity_1961_2019.nc"
REPORT_TXT = OUT_DIR / "CEH-GEAR_precip_intensity_processing_notes.txt"

START_YEAR = 1961
END_YEAR = 2019
START_FRESH = True

WET_DAY_THRESHOLD_MM = 1.0
HEAVY_DAY_THRESHOLD_MM = 10.0


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name


def download_file(session: requests.Session, url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")
    if part.exists():
        part.unlink()

    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(part, "wb") as f:
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
    part.rename(out_path)
    return out_path


def get_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def find_year_url(session: requests.Session, year: int) -> str:
    html = get_html(session, BASE_URL)
    target = f"CEH_GEAR_daily_GB_{year}.nc"
    links = sorted(set(re.findall(r'href="([^"]+\.nc)"', html)))
    if target not in links:
        raise FileNotFoundError(f"Could not find {target} at {BASE_URL}")
    return urljoin(BASE_URL, target)


def find_rainfall_variable(ds: Dataset) -> str:
    preferred = ["rainfall_amount", "precip", "rainfall", "pr", "rain"]
    names = list(ds.variables.keys())
    lower = {name.lower(): name for name in names}
    for p in preferred:
        for lname, original in lower.items():
            if p in lname and original not in ds.dimensions:
                return original
    for name in names:
        var = ds.variables[name]
        dims = var.dimensions
        if "time" in dims and len(dims) >= 3 and name not in ["lat", "lon"]:
            return name
    raise ValueError(f"Cannot identify rainfall variable. Variables: {names}")


def read_dates(ds: Dataset) -> list[datetime]:
    t = ds.variables["time"]
    dates = num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    out = []
    for d in dates:
        out.append(datetime(int(d.year), int(d.month), int(d.day)))
    return out


def season_key(date: datetime) -> tuple[int, int, str]:
    month = date.month
    year = date.year
    if month in (12, 1, 2):
        return (year if month == 12 else year - 1, 1, "DJF")
    if month in (3, 4, 5):
        return (year, 2, "MAM")
    if month in (6, 7, 8):
        return (year, 3, "JJA")
    return (year, 4, "SON")


def build_time_records() -> list[tuple[int, int, str, datetime]]:
    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        records.append((year, 2, "MAM", datetime(year, 3, 1)))
        records.append((year, 3, "JJA", datetime(year, 6, 1)))
        records.append((year, 4, "SON", datetime(year, 9, 1)))
        if year <= END_YEAR - 1:
            records.append((year, 1, "DJF", datetime(year, 12, 1)))
    return sorted(records, key=lambda x: x[3])


def copy_attrs(src, dst, skip_fill_value: bool = True) -> None:
    for attr in src.ncattrs():
        if skip_fill_value and attr == "_FillValue":
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            setattr(dst, attr, str(getattr(src, attr)))


def copy_variable(src_ds: Dataset, dst_ds: Dataset, name: str, compress: bool = True) -> None:
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


def create_output(template_path: Path, records: list[tuple[int, int, str, datetime]]) -> tuple[str, str]:
    if OUT_NC.exists() and START_FRESH:
        OUT_NC.unlink()

    src = Dataset(template_path, "r")
    rain_name = find_rainfall_variable(src)
    dims = src.variables[rain_name].dimensions
    spatial_dims = [d for d in dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected rainfall dimensions time,y,x; got {dims}")
    ydim, xdim = spatial_dims

    dst = Dataset(OUT_NC, "w", format="NETCDF4")
    dst.createDimension("time", len(records))
    dst.createDimension("bnds", 2)
    for d in [ydim, xdim]:
        dst.createDimension(d, len(src.dimensions[d]))

    # Copy common coordinates when present.
    for name in [ydim, xdim, "lat", "lon", "latitude", "longitude", "x", "y", "projection_y_coordinate", "projection_x_coordinate"]:
        if name in src.variables and name not in dst.variables:
            # Create any missing dims used by this coordinate.
            for dim in src.variables[name].dimensions:
                if dim not in dst.dimensions:
                    dst.createDimension(dim, len(src.dimensions[dim]))
            copy_variable(src, dst, name, compress=True)

    for name in ["crs", "crsOSGB", "transverse_mercator"]:
        if name in src.variables and name not in dst.variables:
            copy_variable(src, dst, name, compress=False)

    t = dst.createVariable("time", "f8", ("time",))
    t.units = "days since 1961-01-01 00:00:00"
    t.calendar = "gregorian"
    t.long_name = "season start time"
    t[:] = date2num([r[3] for r in records], units=t.units, calendar=t.calendar)

    tb = dst.createVariable("time_bnds", "f8", ("time", "bnds"))
    tb.units = t.units
    tb.calendar = t.calendar
    for i, (_, _, season, start) in enumerate(records):
        if season == "DJF":
            end = datetime(start.year + 1, 3, 1)
        elif season == "MAM":
            end = datetime(start.year, 6, 1)
        elif season == "JJA":
            end = datetime(start.year, 9, 1)
        else:
            end = datetime(start.year, 12, 1)
        tb[i, :] = date2num([start, end], units=t.units, calendar=t.calendar)

    sy = dst.createVariable("season_year", "i2", ("time",))
    sy.long_name = "season year; DJF is labelled by December start year"
    sy[:] = [r[0] for r in records]

    sc = dst.createVariable("season_code", "i1", ("time",))
    sc.long_name = "season code: 1=DJF, 2=MAM, 3=JJA, 4=SON"
    sc[:] = [r[1] for r in records]

    chunk = (1, min(256, len(src.dimensions[ydim])), min(256, len(src.dimensions[xdim])))
    outputs = {
        "precip_total": ("f4", "mm season-1", "Seasonal total daily rainfall from CEH-GEAR"),
        "precip_max_1day": ("f4", "mm day-1", "Maximum daily rainfall within the season from CEH-GEAR"),
        "wet_day_count": ("i2", "days season-1", f"Number of days with rainfall >= {WET_DAY_THRESHOLD_MM} mm day-1"),
        "heavy_day_count": ("i2", "days season-1", f"Number of days with rainfall >= {HEAVY_DAY_THRESHOLD_MM} mm day-1"),
        "antecedent_precip_total": ("f4", "mm season-1", "Previous-season seasonal total rainfall"),
        "antecedent_precip_max_1day": ("f4", "mm day-1", "Previous-season maximum daily rainfall"),
    }
    for name, (dtype, units, long_name) in outputs.items():
        fill = np.float32(np.nan) if dtype == "f4" else np.int16(-32768)
        v = dst.createVariable(
            name,
            dtype,
            ("time", ydim, xdim),
            zlib=True,
            complevel=4,
            chunksizes=chunk,
            fill_value=fill,
        )
        v.units = units
        v.long_name = long_name

    dst.title = "CEH-GEAR seasonal rainfall intensity and antecedent wetness features, Great Britain, 1961-2019"
    dst.source = "CEH-GEAR daily rainfall estimates, UKCEH EIDC, 1 km Great Britain"
    dst.source_url = BASE_URL
    dst.processing = "Annual daily NetCDF files were downloaded one at a time, aggregated to seasonal features, and deleted after use."
    dst.close()
    src.close()
    return rain_name, ydim


def init_accumulators(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "total": np.zeros(shape, dtype="float32"),
        "max": np.full(shape, np.nan, dtype="float32"),
        "wet": np.zeros(shape, dtype="int16"),
        "heavy": np.zeros(shape, dtype="int16"),
        "count": np.zeros(shape, dtype="int16"),
    }


def update_acc(acc: dict[str, np.ndarray], rain: np.ndarray) -> None:
    rain = np.asarray(rain, dtype="float32")
    if np.ma.isMaskedArray(rain):
        rain = rain.filled(np.nan)
    valid = np.isfinite(rain)
    acc["total"] += np.where(valid, rain, 0.0).astype("float32")
    acc["max"] = np.fmax(acc["max"], rain)
    acc["wet"] += ((rain >= WET_DAY_THRESHOLD_MM) & valid).astype("int16")
    acc["heavy"] += ((rain >= HEAVY_DAY_THRESHOLD_MM) & valid).astype("int16")
    acc["count"] += valid.astype("int16")


def write_season(idx: int, acc: dict[str, np.ndarray]) -> None:
    dst = Dataset(OUT_NC, "a")
    total = np.where(acc["count"] > 0, acc["total"], np.nan).astype("float32")
    max1 = np.where(acc["count"] > 0, acc["max"], np.nan).astype("float32")
    dst.variables["precip_total"][idx, :, :] = total
    dst.variables["precip_max_1day"][idx, :, :] = max1
    dst.variables["wet_day_count"][idx, :, :] = acc["wet"]
    dst.variables["heavy_day_count"][idx, :, :] = acc["heavy"]
    dst.close()


def fill_antecedent(records: list[tuple[int, int, str, datetime]]) -> None:
    dst = Dataset(OUT_NC, "a")
    precip_total = dst.variables["precip_total"]
    precip_max = dst.variables["precip_max_1day"]
    ante_total = dst.variables["antecedent_precip_total"]
    ante_max = dst.variables["antecedent_precip_max_1day"]

    # First complete season has no antecedent in this file.
    ante_total[0, :, :] = np.nan
    ante_max[0, :, :] = np.nan
    for i in tqdm(range(1, len(records)), desc="Writing antecedent features"):
        ante_total[i, :, :] = precip_total[i - 1, :, :]
        ante_max[i, :, :] = precip_max[i - 1, :, :]
    dst.close()


def process_year(session: requests.Session, year: int, record_index: dict[tuple[int, int], int], pending: dict) -> None:
    url = find_year_url(session, year)
    local = TMP_DIR / filename_from_url(url)
    print()
    print(f"Processing CEH-GEAR daily rainfall year {year}")
    try:
        download_file(session, url, local)
        src = Dataset(local, "r")
        rain_name = find_rainfall_variable(src)
        rain_var = src.variables[rain_name]
        dates = read_dates(src)

        shape = rain_var.shape[1:]
        for t, date in enumerate(tqdm(dates, desc=f"Aggregating {year}", leave=False)):
            key_year, code, _ = season_key(date)
            key = (key_year, code)
            if key not in record_index:
                continue
            if key not in pending:
                pending[key] = init_accumulators(shape)
            update_acc(pending[key], rain_var[t, :, :])

        src.close()

        # Write seasons that are complete after this year.
        completed = []
        for key in sorted(pending.keys()):
            season_year, code = key
            if code == 1:
                # DJF labelled by Dec start year completes in Feb of next calendar year.
                is_complete = year >= season_year + 1
            else:
                is_complete = year >= season_year
            if is_complete:
                idx = record_index[key]
                write_season(idx, pending[key])
                completed.append(key)
        for key in completed:
            del pending[key]

    finally:
        local.unlink(missing_ok=True)


def write_report() -> None:
    REPORT_TXT.write_text(
        f"""CEH-GEAR seasonal precipitation intensity processing report

Source:
CEH-GEAR daily rainfall estimates, UKCEH EIDC.
Download directory:
{BASE_URL}

Purpose:
This dataset provides precipitation intensity and antecedent wetness features for the seasonal 1 km soil moisture and runoff emulator.

Processing:
- Annual daily NetCDF files CEH_GEAR_daily_GB_YYYY.nc were downloaded one at a time for {START_YEAR}-{END_YEAR}.
- Each annual file was aggregated to seasonal 1 km indicators and then deleted immediately.
- Complete seasons retained: MAM/JJA/SON {START_YEAR}-{END_YEAR}, and DJF {START_YEAR}-{END_YEAR - 1}.
- DJF is labelled by the December start year.

Output:
{OUT_NC}

Output variables:
- precip_total: seasonal total rainfall, mm season-1.
- precip_max_1day: maximum daily rainfall within the season, mm day-1.
- wet_day_count: number of days with rainfall >= {WET_DAY_THRESHOLD_MM} mm day-1.
- heavy_day_count: number of days with rainfall >= {HEAVY_DAY_THRESHOLD_MM} mm day-1.
- antecedent_precip_total: previous-season precip_total.
- antecedent_precip_max_1day: previous-season precip_max_1day.

Reason for model use:
Runoff is sensitive to rainfall intensity and antecedent wetness, not only seasonal mean precipitation. These features help represent event-driven runoff generation and pre-season soil wetness memory.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "seasonal-emulator-ceh-gear-processor/1.0"})

    records = build_time_records()
    record_index = {(year, code): i for i, (year, code, _, _) in enumerate(records)}

    print("Output folder:", OUT_DIR)
    print("Final NetCDF:", OUT_NC)
    print("Temporary raw files:", TMP_DIR)
    print("This script downloads one annual CEH-GEAR daily file at a time and deletes it after processing.")

    # Template from first year.
    template_url = find_year_url(session, START_YEAR)
    template = TMP_DIR / filename_from_url(template_url)
    print()
    print("Downloading template file to create output grid...")
    download_file(session, template_url, template)
    create_output(template, records)
    template.unlink(missing_ok=True)

    pending = {}
    for year in range(START_YEAR, END_YEAR + 1):
        process_year(session, year, record_index, pending)

    if pending:
        print("Warning: incomplete seasons left in memory:", sorted(pending.keys()))

    fill_antecedent(records)
    write_report()

    try:
        TMP_DIR.rmdir()
    except OSError:
        pass

    print()
    print("All done.")
    print("Saved NetCDF:", OUT_NC)
    print("Saved report:", REPORT_TXT)


if __name__ == "__main__":
    main()
