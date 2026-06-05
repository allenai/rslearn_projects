"""Build stacked ESRI land cover GeoTIFFs from Planetary Computer COGs.

For each Sentinel-2 grid tile, reads 9 annual ESRI land cover COGs (2017-2025)
from the Planetary Computer ``io-lulc-annual-v02`` collection and writes a single
9-band uint8 GeoTIFF where band index corresponds to year.

Band layout:
  0 = 2017, 1 = 2018, ..., 8 = 2025

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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import rasterio
from rslearn.data_sources.planetary_computer import PlanetaryComputerStacClient
from rslearn.utils.stac import StacItem
from upath import UPath

logger = logging.getLogger(__name__)

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "io-lulc-annual-v02"
YEARS = list(range(2017, 2026))
NUM_YEARS = len(YEARS)


def _get_stac_client() -> PlanetaryComputerStacClient:
    """Create a Planetary Computer STAC client."""
    return PlanetaryComputerStacClient(STAC_ENDPOINT)


def _get_items_for_tile(
    client: PlanetaryComputerStacClient,
    tile_id: str,
) -> dict[int, StacItem]:
    """Query STAC for all years of a given S2 tile.

    Uses direct item ID lookups (``{tile_id}-{year}``) which is reliable
    across all STAC implementations without needing query/CQL2 extensions.

    Returns a dict mapping year -> StacItem.
    """
    expected_ids = [f"{tile_id}-{year}" for year in YEARS]
    items = client.search(
        collections=[COLLECTION],
        ids=expected_ids,
    )

    year_to_item: dict[int, StacItem] = {}
    for item in items:
        parts = item.id.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            year = int(parts[1])
            if year in YEARS:
                year_to_item[year] = item
    return year_to_item


def _read_cog(item: StacItem) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single-band COG from a STAC item, returning (2D array, rasterio profile)."""
    asset = item.assets
    if asset is None:
        raise ValueError(f"Item {item.id} has no assets")

    # The main data asset key is "data" in io-lulc-annual-v02
    asset_key = "data"
    if asset_key not in asset:
        available = list(asset.keys())
        raise ValueError(
            f"Item {item.id} missing '{asset_key}' asset (available: {available})"
        )

    signed_url = planetary_computer.sign(asset[asset_key].href)
    with rasterio.open(signed_url) as src:
        arr = src.read(1)
        profile = dict(src.profile)
        profile["transform"] = src.transform
        profile["crs"] = src.crs
        profile["height"] = src.height
        profile["width"] = src.width
    return arr, profile


def _get_output_path(out_path: str, tile_id: str) -> UPath:
    """Get the output file path for a tile."""
    return UPath(out_path) / f"{tile_id}_lc_stack.tif"


def build_stack(
    tile_id: str,
    out_path: str,
) -> None:
    """Build a stacked land cover GeoTIFF for a single S2 tile.

    Reads 9 annual ESRI land cover COGs (2017-2025) and writes a single
    9-band uint8 GeoTIFF.

    Args:
        tile_id: Sentinel-2 grid tile ID (e.g. "10T").
        out_path: directory to write the output GeoTIFF.
    """
    out_file = _get_output_path(out_path, tile_id)
    if out_file.exists():
        logger.info("output %s already exists, skipping", out_file)
        return

    client = _get_stac_client()
    year_to_item = _get_items_for_tile(client, tile_id)

    if not year_to_item:
        logger.warning("no STAC items found for tile %s, skipping", tile_id)
        return

    missing_years = [y for y in YEARS if y not in year_to_item]
    if missing_years:
        logger.warning(
            "tile %s missing years %s — will fill with nodata", tile_id, missing_years
        )

    # Read the first available year to get spatial metadata
    first_year = min(year_to_item.keys())
    first_arr, profile = _read_cog(year_to_item[first_year])
    height, width = first_arr.shape

    # Stack all years
    stack = np.zeros((NUM_YEARS, height, width), dtype=np.uint8)
    for i, year in enumerate(YEARS):
        if year in year_to_item:
            if year == first_year:
                stack[i] = first_arr
            else:
                arr, _ = _read_cog(year_to_item[year])
                stack[i] = arr
            logger.debug("read year %d for tile %s", year, tile_id)
        # else: remains 0 (nodata)

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
