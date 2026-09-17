"""
Download OS Terrain 50 and aggregate it to the CHESS 1 km grid.

This script downloads the public OS Terrain 50 ASCII Grid package, processes the
50 m elevation tiles into CHESS-aligned 1 km terrain features, writes a compact
NetCDF, and deletes raw downloaded/extracted files afterwards.

Output variables:
- elevation_mean: mean elevation within each 1 km CHESS cell, m
- elevation_min: minimum elevation within each 1 km CHESS cell, m
- elevation_max: maximum elevation within each 1 km CHESS cell, m
- elevation_sd: standard deviation of 50 m elevations within each 1 km cell, m
- elevation_range: elevation_max - elevation_min, m
- terrain_sample_count: number of 50 m samples contributing to the cell

Run:
    python "E:\\Emulator seasonal\\download_process_os_terrain50_to_chess_1km.py"
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import math
import shutil
import zipfile

import numpy as np
import requests

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    print("Missing required Python package netCDF4.")
    print("Please run: pip install netCDF4 requests tqdm")
    print(f"Original import error: {exc}")
    raise SystemExit(1)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


DOWNLOADS_API = "https://api.os.uk/downloads/v1/products/Terrain50/downloads"
OUT_ROOT = Path(r"E:\Emulator seasonal")
OUT_DIR = OUT_ROOT / "OS_Terrain50_CHESS_1km_terrain"
TMP_DIR = OUT_DIR / "raw_tmp"
EXTRACT_DIR = TMP_DIR / "extracted"
OUT_NC = OUT_DIR / "OS_Terrain50_CHESS_1km_terrain_features.nc"
REPORT_TXT = OUT_DIR / "OS_Terrain50_CHESS_1km_processing_notes.txt"

CHESS_GRID_NC = OUT_ROOT / "HPC_input_data" / "01_historical_climate" / "CHESS-met_gb_1km_seasonal_1961-2019.nc"

START_FRESH = True


def get_os_terrain50_ascii_download() -> dict:
    items = requests.get(DOWNLOADS_API, timeout=60).json()
    for item in items:
        if item.get("format") == "ASCII Grid and GML (Grid)" and item.get("area") == "GB":
            return item
    raise RuntimeError("Could not find OS Terrain 50 ASCII Grid download in OS Downloads API.")


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")
    if part.exists():
        part.unlink()
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(part, "wb") as f:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=out_path.name,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    part.rename(out_path)


def load_chess_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with Dataset(CHESS_GRID_NC) as ds:
        x = np.asarray(ds.variables["x"][:], dtype="float64")
        y = np.asarray(ds.variables["y"][:], dtype="float64")
        lat = np.asarray(ds.variables["lat"][:], dtype="float32") if "lat" in ds.variables else None
        lon = np.asarray(ds.variables["lon"][:], dtype="float32") if "lon" in ds.variables else None
    return x, y, lat, lon


def find_ascii_files(root: Path) -> list[Path]:
    patterns = ["*.asc", "*.ASC", "*.txt", "*.TXT"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    out = []
    for f in files:
        name = f.name.lower()
        if "readme" in name or "metadata" in name:
            continue
        out.append(f)
    return sorted(out)


def parse_ascii_header(path: Path) -> tuple[dict, int]:
    header = {}
    n_header = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                break
            key = parts[0].lower()
            if key not in {"ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "nodata_value"}:
                break
            val = float(parts[1])
            header[key] = val
            n_header += 1
            if n_header >= 6 and "cellsize" in header:
                # ESRI ASCII usually has six header lines.
                if key == "nodata_value":
                    break
        return header, n_header


def read_ascii_grid(path: Path) -> tuple[np.ndarray, dict]:
    header, n_header = parse_ascii_header(path)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    data = np.loadtxt(path, skiprows=n_header, dtype="float32")
    if data.shape != (nrows, ncols):
        data = data.reshape((nrows, ncols))
    nodata = header.get("nodata_value")
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data).astype("float32")
    return data, header


def cell_centres(header: dict) -> tuple[np.ndarray, np.ndarray]:
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell = float(header["cellsize"])
    if "xllcenter" in header:
        x0 = float(header["xllcenter"])
    else:
        x0 = float(header["xllcorner"]) + cell / 2.0
    if "yllcenter" in header:
        y0 = float(header["yllcenter"])
    else:
        y0 = float(header["yllcorner"]) + cell / 2.0
    xs = x0 + np.arange(ncols, dtype="float64") * cell
    # ASCII rows are north-to-south.
    y_top = y0 + (nrows - 1) * cell
    ys = y_top - np.arange(nrows, dtype="float64") * cell
    return xs, ys


def update_aggregates(
    data: np.ndarray,
    header: dict,
    x_chess: np.ndarray,
    y_chess: np.ndarray,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    min_arr: np.ndarray,
    max_arr: np.ndarray,
    count_arr: np.ndarray,
) -> None:
    xs, ys = cell_centres(header)
    x_idx = np.rint((xs - 500.0) / 1000.0).astype("int64")
    y_idx = np.rint((ys - 500.0) / 1000.0).astype("int64")

    valid_x = (x_idx >= 0) & (x_idx < len(x_chess))
    valid_y = (y_idx >= 0) & (y_idx < len(y_chess))
    if not valid_x.any() or not valid_y.any():
        return

    row_indices = np.where(valid_y)[0]
    col_indices = np.where(valid_x)[0]
    yy = y_idx[row_indices]
    xx = x_idx[col_indices]
    sub = data[np.ix_(row_indices, col_indices)]

    # Flatten only the current tile. OS Terrain 50 tiles are small enough.
    yy2 = np.repeat(yy, len(xx))
    xx2 = np.tile(xx, len(yy))
    vals = sub.ravel()
    valid = np.isfinite(vals)
    if not valid.any():
        return

    yy2 = yy2[valid]
    xx2 = xx2[valid]
    vals = vals[valid].astype("float64")

    np.add.at(sum_arr, (yy2, xx2), vals)
    np.add.at(sumsq_arr, (yy2, xx2), vals * vals)
    np.add.at(count_arr, (yy2, xx2), 1)
    np.minimum.at(min_arr, (yy2, xx2), vals.astype("float32"))
    np.maximum.at(max_arr, (yy2, xx2), vals.astype("float32"))


def write_output(
    x: np.ndarray,
    y: np.ndarray,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    sum_arr: np.ndarray,
    sumsq_arr: np.ndarray,
    min_arr: np.ndarray,
    max_arr: np.ndarray,
    count_arr: np.ndarray,
    source_item: dict,
) -> None:
    valid = count_arr > 0
    mean = np.full(count_arr.shape, np.nan, dtype="float32")
    sd = np.full(count_arr.shape, np.nan, dtype="float32")
    elev_min = np.where(valid, min_arr, np.nan).astype("float32")
    elev_max = np.where(valid, max_arr, np.nan).astype("float32")
    elev_range = (elev_max - elev_min).astype("float32")

    mean[valid] = (sum_arr[valid] / count_arr[valid]).astype("float32")
    var = np.zeros(count_arr.shape, dtype="float64")
    var[valid] = sumsq_arr[valid] / count_arr[valid] - (sum_arr[valid] / count_arr[valid]) ** 2
    var = np.maximum(var, 0.0)
    sd[valid] = np.sqrt(var[valid]).astype("float32")

    if OUT_NC.exists() and START_FRESH:
        OUT_NC.unlink()
    nc = Dataset(OUT_NC, "w", format="NETCDF4")
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))
    nc.createDimension("bnds", 2)

    xv = nc.createVariable("x", "f4", ("x",))
    yv = nc.createVariable("y", "f4", ("y",))
    xv[:] = x.astype("float32")
    yv[:] = y.astype("float32")
    xv.units = "m"
    yv.units = "m"
    xv.long_name = "easting of British National Grid (BNG) coordinate system"
    yv.long_name = "northing of British National Grid (BNG) coordinate system"
    xv.standard_name = "projection_x_coordinate"
    yv.standard_name = "projection_y_coordinate"

    if lat is not None:
        v = nc.createVariable("lat", "f4", ("y", "x"), zlib=True, complevel=4)
        v[:] = lat
        v.units = "degrees_north"
        v.standard_name = "latitude"
    if lon is not None:
        v = nc.createVariable("lon", "f4", ("y", "x"), zlib=True, complevel=4)
        v[:] = lon
        v.units = "degrees_east"
        v.standard_name = "longitude"

    crs = nc.createVariable("crsOSGB", "i4")
    crs.grid_mapping_name = "transverse_mercator"
    crs.epsg_code = "EPSG:27700"
    crs.long_name = "OSGB 1936 / British National Grid"

    outputs = {
        "elevation_mean": (mean, "Mean elevation within each CHESS 1 km grid cell"),
        "elevation_min": (elev_min, "Minimum 50 m elevation within each CHESS 1 km grid cell"),
        "elevation_max": (elev_max, "Maximum 50 m elevation within each CHESS 1 km grid cell"),
        "elevation_sd": (sd, "Standard deviation of 50 m elevations within each CHESS 1 km grid cell"),
        "elevation_range": (elev_range, "Elevation range within each CHESS 1 km grid cell"),
    }
    for name, (arr, long_name) in outputs.items():
        v = nc.createVariable(
            name,
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            chunksizes=(min(256, len(y)), min(256, len(x))),
            fill_value=np.float32(np.nan),
        )
        v[:] = arr
        v.units = "m"
        v.long_name = long_name
        v.grid_mapping = "crsOSGB"

    c = nc.createVariable(
        "terrain_sample_count",
        "i2",
        ("y", "x"),
        zlib=True,
        complevel=4,
        chunksizes=(min(256, len(y)), min(256, len(x))),
        fill_value=np.int16(-32768),
    )
    c[:] = np.where(valid, count_arr, -32768).astype("int16")
    c.units = "1"
    c.long_name = "Number of 50 m OS Terrain samples contributing to each 1 km cell"

    nc.title = "OS Terrain 50 terrain features aggregated to the CHESS 1 km grid"
    nc.source = "OS Terrain 50, Ordnance Survey OpenData"
    nc.source_url = source_item.get("url", "")
    nc.source_file = source_item.get("fileName", "")
    nc.processing = "OS Terrain 50 ASCII grid tiles were aggregated to CHESS 1 km cells; raw zip and extracted files were deleted after processing."
    nc.close()


def write_report(source_item: dict, tile_count: int) -> None:
    REPORT_TXT.write_text(
        f"""OS Terrain 50 to CHESS 1 km processing report

Source:
OS Terrain 50, Ordnance Survey OpenData.
Download API:
{DOWNLOADS_API}

Downloaded file:
{source_item.get('fileName')}
Size reported by API:
{source_item.get('size')} bytes

Processing:
- Downloaded the GB ASCII Grid and GML package.
- Extracted OS Terrain 50 ASCII grid tiles temporarily.
- Aggregated 50 m elevation samples to the CHESS 1 km grid.
- Raw zip and extracted files were deleted after processing.

Output:
{OUT_NC}

Output variables:
- elevation_mean: mean elevation within each 1 km CHESS cell, m.
- elevation_min: minimum elevation within each 1 km CHESS cell, m.
- elevation_max: maximum elevation within each 1 km CHESS cell, m.
- elevation_sd: standard deviation of 50 m elevations within each 1 km cell, m.
- elevation_range: elevation_max - elevation_min, m.
- terrain_sample_count: number of 50 m samples per 1 km cell.

Reason for emulator use:
Terrain controls runoff generation, drainage and local water redistribution. Slope and relief-related features help the emulator distinguish flatter, storage-dominated areas from steeper, faster-draining areas.

Number of ASCII grid files processed:
{tile_count}
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Output folder:", OUT_DIR)
    print("Final NetCDF:", OUT_NC)
    print("Temporary folder:", TMP_DIR)

    source_item = get_os_terrain50_ascii_download()
    zip_path = TMP_DIR / source_item.get("fileName", "terr50_gagg_gb.zip")

    print()
    print("Downloading OS Terrain 50 ASCII Grid package...")
    print("URL:", source_item["url"])
    print("Reported size:", round(source_item.get("size", 0) / 1024 / 1024, 1), "MB")
    download_file(source_item["url"], zip_path)

    print()
    print("Extracting zip temporarily...")
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)

    x, y, lat, lon = load_chess_grid()
    shape = (len(y), len(x))
    sum_arr = np.zeros(shape, dtype="float64")
    sumsq_arr = np.zeros(shape, dtype="float64")
    min_arr = np.full(shape, np.inf, dtype="float32")
    max_arr = np.full(shape, -np.inf, dtype="float32")
    count_arr = np.zeros(shape, dtype="int16")

    ascii_files = find_ascii_files(EXTRACT_DIR)
    if not ascii_files:
        raise RuntimeError("No ASCII grid files found after extraction.")

    print()
    print("Processing ASCII grid tiles:", len(ascii_files))
    for path in tqdm(ascii_files, desc="Aggregating Terrain50"):
        try:
            data, header = read_ascii_grid(path)
            update_aggregates(data, header, x, y, sum_arr, sumsq_arr, min_arr, max_arr, count_arr)
        except Exception as exc:
            print(f"Warning: skipped {path} due to: {exc}")

    print()
    print("Writing compact NetCDF...")
    write_output(x, y, lat, lon, sum_arr, sumsq_arr, min_arr, max_arr, count_arr, source_item)
    write_report(source_item, len(ascii_files))

    print("Deleting raw zip and extracted files...")
    if zip_path.exists():
        zip_path.unlink()
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
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
