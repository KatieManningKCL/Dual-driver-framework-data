"""
Download CEH-GEAR monthly rainfall and aggregate it to seasonal 1 km features.

This light version is designed for the seasonal emulator and avoids downloading
large daily files. It downloads one annual monthly NetCDF at a time, aggregates
monthly rainfall into DJF/MAM/JJA/SON seasonal indicators, writes a compact
NetCDF, and deletes the raw annual file immediately.

Output variables:
- precip_total: seasonal total rainfall, mm season-1
- precip_mean_monthly: mean monthly rainfall in the season, mm month-1
- precip_max_monthly: wettest-month rainfall in the season, mm month-1
- antecedent_precip_total: previous-season precip_total, mm season-1
- antecedent_precip_max_monthly: previous-season precip_max_monthly, mm month-1
"""

from __future__ import annotations

from datetime import datetime
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


BASE_URL = "https://catalogue.ceh.ac.uk/datastore/eidchub/dbf13dd5-90cd-457a-a986-f2f9dd97e93c/GB/monthly/"
OUT_ROOT = Path(r"E:\Emulator seasonal")
OUT_DIR = OUT_ROOT / "CEH-GEAR_monthly_precip_seasonal_1961_2019"
TMP_DIR = OUT_DIR / "raw_tmp"
OUT_NC = OUT_DIR / "CEH-GEAR_GB_1km_seasonal_precip_from_monthly_1961_2019.nc"
REPORT_TXT = OUT_DIR / "CEH-GEAR_monthly_to_seasonal_processing_notes.txt"

START_YEAR = 1961
END_YEAR = 2019
START_FRESH = True


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
    target = f"CEH_GEAR_monthly_GB_{year}.nc"
    links = sorted(set(re.findall(r'href="([^"]+\.nc)"', html)))
    if target not in links:
        raise FileNotFoundError(f"Could not find {target} at {BASE_URL}")
    return urljoin(BASE_URL, target)


def find_rainfall_variable(ds: Dataset) -> str:
    names = list(ds.variables.keys())
    for key in ["rainfall_amount", "precip", "rainfall", "rain", "pr"]:
        for name in names:
            if key in name.lower() and name not in ds.dimensions:
                return name
    for name in names:
        var = ds.variables[name]
        if "time" in var.dimensions and len(var.dimensions) >= 3:
            return name
    raise ValueError(f"Cannot identify rainfall variable. Variables: {names}")


def read_dates(ds: Dataset) -> list[datetime]:
    t = ds.variables["time"]
    dates = num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    return [datetime(int(d.year), int(d.month), 1) for d in dates]


def season_key(date: datetime) -> tuple[int, int, str]:
    y, m = date.year, date.month
    if m in (12, 1, 2):
        return (y if m == 12 else y - 1, 1, "DJF")
    if m in (3, 4, 5):
        return (y, 2, "MAM")
    if m in (6, 7, 8):
        return (y, 3, "JJA")
    return (y, 4, "SON")


def build_time_records() -> list[tuple[int, int, str, datetime]]:
    records = []
    for y in range(START_YEAR, END_YEAR + 1):
        records.append((y, 2, "MAM", datetime(y, 3, 1)))
        records.append((y, 3, "JJA", datetime(y, 6, 1)))
        records.append((y, 4, "SON", datetime(y, 9, 1)))
        if y <= END_YEAR - 1:
            records.append((y, 1, "DJF", datetime(y, 12, 1)))
    return sorted(records, key=lambda r: r[3])


def copy_attrs(src, dst, skip_fill_value=True) -> None:
    for attr in src.ncattrs():
        if skip_fill_value and attr == "_FillValue":
            continue
        try:
            setattr(dst, attr, getattr(src, attr))
        except Exception:
            setattr(dst, attr, str(getattr(src, attr)))


def copy_variable(src_ds: Dataset, dst_ds: Dataset, name: str, compress=True) -> None:
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


def create_output(template_path: Path, records: list[tuple[int, int, str, datetime]]) -> None:
    if OUT_NC.exists() and START_FRESH:
        OUT_NC.unlink()

    src = Dataset(template_path, "r")
    rain_name = find_rainfall_variable(src)
    rain_var = src.variables[rain_name]
    dims = rain_var.dimensions
    spatial_dims = [d for d in dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected rainfall dimensions time,y,x; got {dims}")
    ydim, xdim = spatial_dims

    dst = Dataset(OUT_NC, "w", format="NETCDF4")
    dst.createDimension("time", len(records))
    dst.createDimension("bnds", 2)
    for d in [ydim, xdim]:
        dst.createDimension(d, len(src.dimensions[d]))

    for name in [ydim, xdim, "lat", "lon", "latitude", "longitude", "x", "y"]:
        if name in src.variables and name not in dst.variables:
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

    sy = dst.createVariable("season_year", "i2", ("time",))
    sy.long_name = "season year; DJF is labelled by December start year"
    sy[:] = [r[0] for r in records]

    sc = dst.createVariable("season_code", "i1", ("time",))
    sc.long_name = "season code: 1=DJF, 2=MAM, 3=JJA, 4=SON"
    sc[:] = [r[1] for r in records]

    chunk = (1, min(256, len(src.dimensions[ydim])), min(256, len(src.dimensions[xdim])))
    meta = {
        "precip_total": ("mm season-1", "Seasonal total rainfall from CEH-GEAR monthly rainfall"),
        "precip_mean_monthly": ("mm month-1", "Mean monthly rainfall within the season"),
        "precip_max_monthly": ("mm month-1", "Wettest-month rainfall within the season"),
        "antecedent_precip_total": ("mm season-1", "Previous-season precip_total"),
        "antecedent_precip_max_monthly": ("mm month-1", "Previous-season precip_max_monthly"),
    }
    for name, (units, long_name) in meta.items():
        v = dst.createVariable(
            name,
            "f4",
            ("time", ydim, xdim),
            zlib=True,
            complevel=4,
            chunksizes=chunk,
            fill_value=np.float32(np.nan),
        )
        v.units = units
        v.long_name = long_name

    dst.title = "CEH-GEAR monthly rainfall aggregated to seasonal 1 km features, Great Britain, 1961-2019"
    dst.source = "CEH-GEAR monthly rainfall estimates, UKCEH EIDC, 1 km Great Britain"
    dst.source_url = BASE_URL
    dst.processing = "Annual monthly NetCDF files were downloaded one at a time, aggregated to seasonal features, and deleted after use."
    dst.close()
    src.close()


def init_acc(shape):
    return {
        "sum": np.zeros(shape, dtype="float32"),
        "max": np.full(shape, np.nan, dtype="float32"),
        "count": np.zeros(shape, dtype="int16"),
    }


def update_acc(acc, arr):
    arr = np.asarray(arr, dtype="float32")
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    valid = np.isfinite(arr)
    acc["sum"] += np.where(valid, arr, 0.0).astype("float32")
    acc["max"] = np.fmax(acc["max"], arr)
    acc["count"] += valid.astype("int16")


def write_season(idx, acc):
    dst = Dataset(OUT_NC, "a")
    total = np.where(acc["count"] > 0, acc["sum"], np.nan).astype("float32")
    mean = np.where(acc["count"] > 0, acc["sum"] / acc["count"], np.nan).astype("float32")
    mx = np.where(acc["count"] > 0, acc["max"], np.nan).astype("float32")
    dst.variables["precip_total"][idx, :, :] = total
    dst.variables["precip_mean_monthly"][idx, :, :] = mean
    dst.variables["precip_max_monthly"][idx, :, :] = mx
    dst.close()


def process_year(session, year, record_index, pending):
    url = find_year_url(session, year)
    local = TMP_DIR / filename_from_url(url)
    print()
    print(f"Processing CEH-GEAR monthly rainfall year {year}")
    try:
        download_file(session, url, local)
        src = Dataset(local, "r")
        rain_name = find_rainfall_variable(src)
        rain_var = src.variables[rain_name]
        dates = read_dates(src)
        shape = rain_var.shape[1:]

        for t, date in enumerate(dates):
            key = season_key(date)[:2]
            if key not in record_index:
                continue
            if key not in pending:
                pending[key] = init_acc(shape)
            update_acc(pending[key], rain_var[t, :, :])

        src.close()

        completed = []
        for key in sorted(pending.keys()):
            season_year, code = key
            complete = year >= season_year + 1 if code == 1 else year >= season_year
            if complete:
                write_season(record_index[key], pending[key])
                completed.append(key)
        for key in completed:
            del pending[key]
    finally:
        local.unlink(missing_ok=True)


def fill_antecedent(n_records):
    dst = Dataset(OUT_NC, "a")
    dst.variables["antecedent_precip_total"][0, :, :] = np.nan
    dst.variables["antecedent_precip_max_monthly"][0, :, :] = np.nan
    for i in tqdm(range(1, n_records), desc="Writing antecedent features"):
        dst.variables["antecedent_precip_total"][i, :, :] = dst.variables["precip_total"][i - 1, :, :]
        dst.variables["antecedent_precip_max_monthly"][i, :, :] = dst.variables["precip_max_monthly"][i - 1, :, :]
    dst.close()


def write_report():
    REPORT_TXT.write_text(
        f"""CEH-GEAR monthly-to-seasonal precipitation processing report

Source:
CEH-GEAR monthly rainfall estimates, UKCEH EIDC.
Download directory:
{BASE_URL}

Purpose:
This dataset provides seasonal precipitation and antecedent wetness features for the seasonal 1 km soil moisture and runoff emulator, without downloading large daily files.

Processing:
- Annual monthly NetCDF files CEH_GEAR_monthly_GB_YYYY.nc were downloaded one at a time for {START_YEAR}-{END_YEAR}.
- Each annual file was aggregated to seasonal 1 km indicators and deleted immediately.
- Complete seasons retained: MAM/JJA/SON {START_YEAR}-{END_YEAR}, and DJF {START_YEAR}-{END_YEAR - 1}.
- DJF is labelled by the December start year.

Output:
{OUT_NC}

Output variables:
- precip_total: seasonal total rainfall, mm season-1.
- precip_mean_monthly: seasonal mean monthly rainfall, mm month-1.
- precip_max_monthly: wettest-month rainfall within the season, mm month-1.
- antecedent_precip_total: previous-season seasonal total rainfall.
- antecedent_precip_max_monthly: previous-season wettest-month rainfall.

Note:
This monthly version is intentionally lighter than a daily-intensity product. It supports seasonal modelling and avoids storing large daily rainfall files.
""",
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "seasonal-emulator-ceh-gear-monthly/1.0"})

    records = build_time_records()
    record_index = {(year, code): i for i, (year, code, _, _) in enumerate(records)}

    print("Output folder:", OUT_DIR)
    print("Final NetCDF:", OUT_NC)
    print("Temporary raw files:", TMP_DIR)
    print("This script uses CEH-GEAR monthly files, not daily files.")

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
        print("Warning: incomplete seasons left:", sorted(pending.keys()))

    fill_antecedent(len(records))
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
