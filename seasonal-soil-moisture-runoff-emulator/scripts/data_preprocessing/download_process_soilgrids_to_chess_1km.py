"""
Download/process SoilGrids 250 m soil properties to the CHESS 1 km grid.

This script does not download global SoilGrids rasters. It uses GDAL remote VRT
access to clip/reproject/resample SoilGrids layers directly to the CHESS 1 km
British National Grid extent, writes one compact NetCDF, and deletes temporary
GeoTIFFs afterwards.

Variables retained as 0-30 cm weighted means:
- clay_0_30cm
- silt_0_30cm
- sand_0_30cm
- bulk_density_0_30cm
- coarse_fragments_0_30cm
- soil_organic_carbon_0_30cm
- soil_water_holding_proxy

Run:
    python "E:\\Emulator seasonal\\download_process_soilgrids_to_chess_1km.py"
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    print("Missing required Python package netCDF4.")
    print("Please run: pip install netCDF4 tqdm")
    print(f"Original import error: {exc}")
    raise SystemExit(1)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


OUT_ROOT = Path(r"E:\Emulator seasonal")
OUT_DIR = OUT_ROOT / "SoilGrids_CHESS_1km_soil_properties"
TMP_DIR = OUT_DIR / "raw_tmp"
OUT_NC = OUT_DIR / "SoilGrids_CHESS_1km_soil_properties_0_30cm.nc"
REPORT_TXT = OUT_DIR / "SoilGrids_CHESS_1km_processing_notes.txt"
CHESS_GRID_NC = OUT_ROOT / "HPC_input_data" / "01_historical_climate" / "CHESS-met_gb_1km_seasonal_1961-2019.nc"

SOILGRIDS_BASE = "https://files.isric.org/soilgrids/latest/data"

# SoilGrids standard depth intervals and weights for 0-30 cm.
DEPTHS = [
    ("0-5cm", 5.0),
    ("5-15cm", 10.0),
    ("15-30cm", 15.0),
]

# SoilGrids map units:
# clay/silt/sand: g kg-1
# bdod: cg cm-3
# cfvo: cm3 dm-3
# soc: dg kg-1
PROPERTIES = {
    "clay": {
        "out_name": "clay_0_30cm",
        "units": "g kg-1",
        "long_name": "0-30 cm weighted mean clay content",
    },
    "silt": {
        "out_name": "silt_0_30cm",
        "units": "g kg-1",
        "long_name": "0-30 cm weighted mean silt content",
    },
    "sand": {
        "out_name": "sand_0_30cm",
        "units": "g kg-1",
        "long_name": "0-30 cm weighted mean sand content",
    },
    "bdod": {
        "out_name": "bulk_density_0_30cm",
        "units": "cg cm-3",
        "long_name": "0-30 cm weighted mean bulk density of the fine earth fraction",
    },
    "cfvo": {
        "out_name": "coarse_fragments_0_30cm",
        "units": "cm3 dm-3",
        "long_name": "0-30 cm weighted mean coarse fragments",
    },
    "soc": {
        "out_name": "soil_organic_carbon_0_30cm",
        "units": "dg kg-1",
        "long_name": "0-30 cm weighted mean soil organic carbon",
    },
}


def load_chess_grid():
    with Dataset(CHESS_GRID_NC) as ds:
        x = np.asarray(ds.variables["x"][:], dtype="float64")
        y = np.asarray(ds.variables["y"][:], dtype="float64")
        lat = np.asarray(ds.variables["lat"][:], dtype="float32") if "lat" in ds.variables else None
        lon = np.asarray(ds.variables["lon"][:], dtype="float32") if "lon" in ds.variables else None
    return x, y, lat, lon


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def soilgrids_vrt(prop: str, depth: str) -> str:
    return f"/vsicurl/{SOILGRIDS_BASE}/{prop}/{prop}_{depth}_mean.vrt"


def warp_layer_to_chess(prop: str, depth: str, out_tif: Path, x: np.ndarray, y: np.ndarray) -> None:
    # CHESS x/y are cell centres. Use bounds at half-cell outside centre range.
    xmin = float(x.min() - 500.0)
    xmax = float(x.max() + 500.0)
    ymin = float(y.min() - 500.0)
    ymax = float(y.max() + 500.0)
    cmd = [
        "gdalwarp",
        "-overwrite",
        "-q",
        "-t_srs", "EPSG:27700",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-tr", "1000", "1000",
        "-r", "average",
        "-srcnodata", "-32768",
        "-dstnodata", "-9999",
        "-of", "GTiff",
        "-ot", "Float32",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        soilgrids_vrt(prop, depth),
        str(out_tif),
    ]
    run(cmd)


def read_tif_as_array(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    txt = subprocess.check_output(["gdalinfo", "-json", str(path)], text=True)
    info = json.loads(txt)
    nodata = info["bands"][0].get("noDataValue", -9999)
    tmp_xyz = path.with_suffix(".xyz")
    try:
        run(["gdal_translate", "-q", "-of", "XYZ", str(path), str(tmp_xyz)])
        data = np.loadtxt(tmp_xyz, dtype="float32")
        # XYZ columns: x, y, value. Output rows are usually top-to-bottom.
        vals = data[:, 2].reshape(expected_shape)
        vals = np.where(vals == nodata, np.nan, vals).astype("float32")
        return vals
    finally:
        tmp_xyz.unlink(missing_ok=True)


def weighted_0_30cm(prop: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    shape = (len(y), len(x))
    weighted_sum = np.zeros(shape, dtype="float64")
    weight_sum = np.zeros(shape, dtype="float64")

    for depth, weight in DEPTHS:
        tif = TMP_DIR / f"{prop}_{depth}_mean_chess1km.tif"
        print()
        print(f"Processing SoilGrids {prop} {depth}")
        warp_layer_to_chess(prop, depth, tif, x, y)
        arr = read_tif_as_array(tif, shape)
        valid = np.isfinite(arr)
        weighted_sum[valid] += arr[valid].astype("float64") * weight
        weight_sum[valid] += weight
        tif.unlink(missing_ok=True)

    out = np.full(shape, np.nan, dtype="float32")
    valid = weight_sum > 0
    out[valid] = (weighted_sum[valid] / weight_sum[valid]).astype("float32")
    return out


def soil_water_holding_proxy(sand, clay, soc, bdod, cfvo) -> np.ndarray:
    """
    A simple transparent proxy, not a calibrated pedotransfer function.

    Higher clay and organic carbon usually increase water holding capacity.
    Higher sand, bulk density and coarse fragments usually reduce it.
    The output is standardized over valid GB land/grid cells for modelling use.
    """
    raw = (
        0.45 * clay
        - 0.25 * sand
        + 0.20 * soc
        - 0.20 * bdod
        - 0.20 * cfvo
    ).astype("float32")
    valid = np.isfinite(raw)
    out = np.full(raw.shape, np.nan, dtype="float32")
    if valid.any():
        mu = np.nanmean(raw)
        sd = np.nanstd(raw)
        if sd > 0:
            out[valid] = ((raw[valid] - mu) / sd).astype("float32")
        else:
            out[valid] = 0.0
    return out


def write_output(x, y, lat, lon, arrays: dict[str, np.ndarray]) -> None:
    if OUT_NC.exists():
        OUT_NC.unlink()

    nc = Dataset(OUT_NC, "w", format="NETCDF4")
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))

    xv = nc.createVariable("x", "f4", ("x",))
    yv = nc.createVariable("y", "f4", ("y",))
    xv[:] = x.astype("float32")
    yv[:] = y.astype("float32")
    xv.units = "m"
    yv.units = "m"
    xv.standard_name = "projection_x_coordinate"
    yv.standard_name = "projection_y_coordinate"
    xv.long_name = "easting of British National Grid coordinate system"
    yv.long_name = "northing of British National Grid coordinate system"

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

    for prop, meta in PROPERTIES.items():
        name = meta["out_name"]
        v = nc.createVariable(
            name,
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            chunksizes=(min(256, len(y)), min(256, len(x))),
            fill_value=np.float32(np.nan),
        )
        v[:] = arrays[name]
        v.units = meta["units"]
        v.long_name = meta["long_name"]
        v.grid_mapping = "crsOSGB"

    v = nc.createVariable(
        "soil_water_holding_proxy",
        "f4",
        ("y", "x"),
        zlib=True,
        complevel=4,
        chunksizes=(min(256, len(y)), min(256, len(x))),
        fill_value=np.float32(np.nan),
    )
    v[:] = arrays["soil_water_holding_proxy"]
    v.units = "standardized index"
    v.long_name = "Transparent soil water holding proxy derived from SoilGrids texture, SOC, bulk density and coarse fragments"
    v.comment = "Higher values indicate greater expected water holding capacity; this is a modelling proxy, not a calibrated hydrological parameter."
    v.grid_mapping = "crsOSGB"

    nc.title = "SoilGrids 250 m soil properties aggregated to CHESS 1 km grid, 0-30 cm"
    nc.source = "SoilGrids, ISRIC World Soil Information"
    nc.source_url = SOILGRIDS_BASE
    nc.processing = "SoilGrids mean maps were accessed through remote VRT files, reprojected/resampled to CHESS 1 km using GDAL average resampling, and combined as 0-30 cm weighted means."
    nc.close()


def write_report() -> None:
    REPORT_TXT.write_text(
        f"""SoilGrids to CHESS 1 km processing report

Source:
SoilGrids 250 m, ISRIC World Soil Information.
Remote VRT base:
{SOILGRIDS_BASE}

Processing:
- No global SoilGrids rasters were downloaded.
- GDAL accessed SoilGrids remote VRT files directly.
- Each property/depth layer was reprojected and resampled to the CHESS 1 km British National Grid.
- Depth intervals 0-5, 5-15 and 15-30 cm were combined as a 0-30 cm thickness-weighted mean.
- Temporary GeoTIFF files were deleted after writing the NetCDF.

Output:
{OUT_NC}

Output variables:
- clay_0_30cm: 0-30 cm weighted mean clay content, g kg-1.
- silt_0_30cm: 0-30 cm weighted mean silt content, g kg-1.
- sand_0_30cm: 0-30 cm weighted mean sand content, g kg-1.
- bulk_density_0_30cm: 0-30 cm weighted mean bulk density, cg cm-3.
- coarse_fragments_0_30cm: 0-30 cm weighted mean coarse fragments, cm3 dm-3.
- soil_organic_carbon_0_30cm: 0-30 cm weighted mean soil organic carbon, dg kg-1.
- soil_water_holding_proxy: standardized modelling proxy for water holding capacity.

Reason for emulator use:
Soil texture, bulk density, coarse fragments and organic carbon influence infiltration, storage capacity, drainage and runoff partitioning. This open SoilGrids product is used as a practical substitute for restricted HOST/BFIHOST data.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Output folder:", OUT_DIR)
    print("Final NetCDF:", OUT_NC)
    print("Temporary folder:", TMP_DIR)

    x, y, lat, lon = load_chess_grid()
    arrays = {}
    for prop, meta in tqdm(PROPERTIES.items(), desc="SoilGrids properties"):
        arrays[meta["out_name"]] = weighted_0_30cm(prop, x, y)

    arrays["soil_water_holding_proxy"] = soil_water_holding_proxy(
        arrays["sand_0_30cm"],
        arrays["clay_0_30cm"],
        arrays["soil_organic_carbon_0_30cm"],
        arrays["bulk_density_0_30cm"],
        arrays["coarse_fragments_0_30cm"],
    )

    print()
    print("Writing compact NetCDF...")
    write_output(x, y, lat, lon, arrays)
    write_report()

    print("Cleaning temporary files...")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print()
    print("All done.")
    print("Saved NetCDF:", OUT_NC)
    print("Saved report:", REPORT_TXT)


if __name__ == "__main__":
    main()
