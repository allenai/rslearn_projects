"""Build stacked ESRI land cover GeoTIFFs from AWS S3 COGs.

For each Sentinel-2 grid tile, reads annual ESRI land cover COGs (2017-2024)
from the public S3 bucket ``s3://io-10m-annual-lulc`` and writes a single
8-band uint8 GeoTIFF where band index corresponds to year.

Band layout:
  0 = 2017, 1 = 2018, ..., 7 = 2024

ESRI class values (uint8):
  0 = No Data, 1 = Water, 2 = Trees, 4 = Flooded Vegetation,
  5 = Crops, 7 = Built Area, 8 = Bare Ground, 9 = Snow/Ice,
  10 = Clouds, 11 = Rangeland

Usage (single tile, for interactive testing)::

    python -m rslp.main change_finder_v2 esri_build_stack \
        --tile_id 10T \
        --out_path /tmp/esri_test/

Usage (batch, called by Beaker workers)::

    python -m rslp.main change_finder_v2 esri_build_stacks \
        --out_path /weka/.../esri_lc_stacks/ \
        --tile_ids '["10T","11T","12T"]'
"""

from __future__ import annotations

import json
import logging
import signal
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.env
from upath import UPath

logger = logging.getLogger(__name__)

S3_BASE_URL = "https://s3.us-west-2.amazonaws.com/io-10m-annual-lulc"
YEARS = list(range(2017, 2025))
NUM_YEARS = len(YEARS)

COG_READ_TIMEOUT = 600  # seconds per COG read before giving up

GDAL_HTTP_OPTS = {
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def _get_cog_url(tile_id: str, year: int) -> str:
    """Get the S3 HTTP URL for an ESRI land cover COG."""
    return f"{S3_BASE_URL}/{tile_id}_{year}.tif"


class _ReadTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _ReadTimeout(f"COG read timed out after {COG_READ_TIMEOUT}s")


def _read_cog(url: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Read a single-band COG from an HTTP URL.

    Returns (2D array, rasterio profile), or None if the file doesn't exist
    (404) or the read times out.
    """
    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COG_READ_TIMEOUT)
    try:
        with rasterio.env.Env(**GDAL_HTTP_OPTS):
            with rasterio.open(url) as src:
                arr = src.read(1)
                profile = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "height": src.height,
                    "width": src.width,
                }
        return arr, profile
    except rasterio.errors.RasterioIOError as e:
        if "404" in str(e):
            return None
        raise
    except _ReadTimeout:
        logger.error("timed out reading %s", url)
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _get_output_path(out_path: str, tile_id: str) -> UPath:
    """Get the output file path for a tile."""
    return UPath(out_path) / f"{tile_id}_lc_stack.tif"


def build_stack(
    tile_id: str,
    out_path: str,
) -> None:
    """Build a stacked land cover GeoTIFF for a single S2 tile.

    Reads 8 annual ESRI land cover COGs (2017-2024) from S3 and writes a
    single 8-band uint8 GeoTIFF.

    Args:
        tile_id: Sentinel-2 grid tile ID (e.g. "10T").
        out_path: directory to write the output GeoTIFF.
    """
    out_file = _get_output_path(out_path, tile_id)
    if out_file.exists():
        logger.info("output %s already exists, skipping", out_file)
        return

    logger.info("processing tile %s", tile_id)

    # Read all available years, skipping any that return 404.
    profile: dict[str, Any] | None = None
    year_arrays: dict[int, np.ndarray] = {}

    for year in YEARS:
        url = _get_cog_url(tile_id, year)
        result = _read_cog(url)
        if result is None:
            logger.warning("tile %s year %d: not found (404), skipping", tile_id, year)
            continue
        arr, prof = result
        year_arrays[year] = arr
        if profile is None:
            profile = prof
        logger.info(
            "tile %s: read %d (%d/%d)", tile_id, year, len(year_arrays), NUM_YEARS
        )

    if not year_arrays or profile is None:
        logger.warning("tile %s: no data for any year, skipping", tile_id)
        return

    height, width = next(iter(year_arrays.values())).shape
    logger.info(
        "tile %s: %dx%d, CRS=%s, %d/%d years",
        tile_id,
        width,
        height,
        profile["crs"],
        len(year_arrays),
        NUM_YEARS,
    )

    # Stack all years (missing years remain 0 = nodata)
    stack = np.zeros((NUM_YEARS, height, width), dtype=np.uint8)
    for i, year in enumerate(YEARS):
        if year in year_arrays:
            stack[i] = year_arrays[year]

    # Write output
    out_dir = UPath(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": NUM_YEARS,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Write to a local temp file first, then copy to final destination.
    # This avoids partial writes if the process is interrupted, and handles
    # remote filesystems (UPath/Weka) that rasterio can't write to directly.
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **write_profile) as dst:
            for i in range(NUM_YEARS):
                dst.write(stack[i], i + 1)
                dst.set_band_description(i + 1, str(YEARS[i]))

        # Copy to final destination
        with open(tmp_path, "rb") as f:
            with out_file.open("wb") as dst:
                dst.write(f.read())
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("wrote %s (%d bands, %dx%d)", out_file, NUM_YEARS, width, height)


def build_stacks(
    out_path: str,
    tile_ids: str,
) -> None:
    """Build stacked GeoTIFFs for a batch of tiles (Beaker worker entry point).

    Args:
        out_path: directory to write output GeoTIFFs.
        tile_ids: JSON-encoded list of tile ID strings.
    """
    ids = json.loads(tile_ids)
    logger.info("processing %d tiles", len(ids))
    for tile_id in ids:
        try:
            build_stack(tile_id=tile_id, out_path=out_path)
        except Exception:
            logger.exception("failed to process tile %s", tile_id)
