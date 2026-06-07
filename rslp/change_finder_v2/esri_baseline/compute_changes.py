"""Compute per-pixel land cover transitions from ESRI stacked GeoTIFFs.

Reads 8-band stacked GeoTIFFs (one band per year, 2017-2024) and produces:

1. A 7-band **transition raster** (one band per consecutive year pair).
   Pixel value = ``src_class * 16 + dst_class`` when a change occurred,
   0 when no change or either year is nodata.  Decode with
   ``src = value // 16; dst = value % 16``.

2. A 1-band **change count raster** where each pixel = the number of
   consecutive year pairs in which that pixel changed (0-7).

ESRI class values (uint8):
  0 = No Data, 1 = Water, 2 = Trees, 4 = Flooded Vegetation,
  5 = Crops, 7 = Built Area, 8 = Bare Ground, 9 = Snow/Ice,
  10 = Clouds, 11 = Rangeland

Usage (single tile)::

    python -m rslp.main change_finder_v2 esri_compute_changes \
        --tile_id 10T \
        --stack_path /weka/.../esri_lc_stacks/ \
        --transitions_path /weka/.../esri_lc_transitions/ \
        --change_count_path /weka/.../esri_lc_change_count/

Usage (all complete tiles)::

    python -m rslp.main change_finder_v2 esri_compute_all_changes \
        --stack_path /weka/.../esri_lc_stacks/ \
        --transitions_path /weka/.../esri_lc_transitions/ \
        --change_count_path /weka/.../esri_lc_change_count/
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from upath import UPath

logger = logging.getLogger(__name__)

YEARS = list(range(2017, 2025))
NUM_YEARS = len(YEARS)
NUM_PAIRS = NUM_YEARS - 1

YEAR_PAIR_NAMES = [f"{YEARS[i]}_{YEARS[i + 1]}" for i in range(NUM_PAIRS)]

_TILE_RE = re.compile(r"^(.+)_lc_stack\.tif$")


def _write_raster(
    arr: np.ndarray,
    profile: dict,
    out_file: UPath,
    band_descriptions: list[str] | None = None,
) -> None:
    """Write a numpy array to a GeoTIFF via a temp file, then copy to destination."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    count = arr.shape[0] if arr.ndim == 3 else 1
    write_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": profile["width"],
        "height": profile["height"],
        "count": count,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **write_profile) as dst:
            if arr.ndim == 3:
                for i in range(count):
                    dst.write(arr[i], i + 1)
                    if band_descriptions:
                        dst.set_band_description(i + 1, band_descriptions[i])
            else:
                dst.write(arr, 1)
                if band_descriptions:
                    dst.set_band_description(1, band_descriptions[0])

        with open(tmp_path, "rb") as f:
            with out_file.open("wb") as dst:
                dst.write(f.read())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_changes(
    tile_id: str,
    stack_path: str,
    transitions_path: str,
    change_count_path: str,
) -> None:
    """Compute transition and change count rasters for a single tile.

    Args:
        tile_id: tile identifier (e.g. "10T").
        stack_path: directory containing ``{tile_id}_lc_stack.tif`` files.
        transitions_path: directory to write the 7-band transition raster.
        change_count_path: directory to write the 1-band change count raster.
    """
    stack_file = UPath(stack_path) / f"{tile_id}_lc_stack.tif"
    trans_file = UPath(transitions_path) / f"{tile_id}_transitions.tif"
    count_file = UPath(change_count_path) / f"{tile_id}_change_count.tif"

    if trans_file.exists() and count_file.exists():
        logger.info("tile %s: outputs already exist, skipping", tile_id)
        return

    if not stack_file.exists():
        logger.warning("tile %s: stack file not found, skipping", tile_id)
        return

    logger.info("tile %s: reading stack", tile_id)
    with rasterio.open(str(stack_file)) as src:
        stack = src.read()  # (8, H, W)
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }

    if stack.shape[0] != NUM_YEARS:
        logger.warning(
            "tile %s: expected %d bands, got %d, skipping",
            tile_id,
            NUM_YEARS,
            stack.shape[0],
        )
        return

    logger.info(
        "tile %s: computing transitions (%dx%d)",
        tile_id,
        profile["width"],
        profile["height"],
    )

    transitions = np.zeros((NUM_PAIRS, stack.shape[1], stack.shape[2]), dtype=np.uint8)

    for i in range(NUM_PAIRS):
        src_band = stack[i]
        dst_band = stack[i + 1]

        # A valid change: both years have data (non-zero) and classes differ
        valid = (src_band > 0) & (dst_band > 0)
        changed = valid & (src_band != dst_band)

        transitions[i][changed] = (
            src_band[changed].astype(np.uint16) * 16 + dst_band[changed]
        ).astype(np.uint8)

    change_count = (transitions > 0).sum(axis=0).astype(np.uint8)

    logger.info(
        "tile %s: %d pixels changed at least once (%.2f%%)",
        tile_id,
        int((change_count > 0).sum()),
        (change_count > 0).sum() / change_count.size * 100,
    )

    if not trans_file.exists():
        _write_raster(
            transitions, profile, trans_file, band_descriptions=YEAR_PAIR_NAMES
        )
        logger.info("tile %s: wrote %s", tile_id, trans_file)

    if not count_file.exists():
        _write_raster(
            change_count, profile, count_file, band_descriptions=["change_count"]
        )
        logger.info("tile %s: wrote %s", tile_id, count_file)


def compute_all_changes(
    stack_path: str,
    transitions_path: str,
    change_count_path: str,
    only_complete: bool = True,
) -> None:
    """Compute transitions for all tiles in the stack directory.

    Args:
        stack_path: directory containing stacked GeoTIFFs.
        transitions_path: directory to write transition rasters.
        change_count_path: directory to write change count rasters.
        only_complete: if True, skip tiles with any nodata bands (incomplete years).
    """
    stack_dir = UPath(stack_path)
    tile_ids = sorted(
        m.group(1) for f in os.listdir(str(stack_dir)) if (m := _TILE_RE.match(f))
    )
    logger.info("found %d tiles in %s", len(tile_ids), stack_path)

    if only_complete:
        complete_ids = []
        for tile_id in tile_ids:
            with rasterio.open(str(stack_dir / f"{tile_id}_lc_stack.tif")) as src:
                has_all = True
                for b in range(1, NUM_YEARS + 1):
                    arr = src.read(b)
                    if arr.max() == 0:
                        has_all = False
                        break
            if has_all:
                complete_ids.append(tile_id)
            else:
                logger.info("tile %s: incomplete, skipping", tile_id)
        logger.info("%d complete tiles (of %d total)", len(complete_ids), len(tile_ids))
        tile_ids = complete_ids

    for idx, tile_id in enumerate(tile_ids):
        logger.info("processing tile %d/%d: %s", idx + 1, len(tile_ids), tile_id)
        try:
            compute_changes(
                tile_id=tile_id,
                stack_path=stack_path,
                transitions_path=transitions_path,
                change_count_path=change_count_path,
            )
        except Exception:
            logger.exception("failed to process tile %s", tile_id)


def compute_changes_batch(
    stack_path: str,
    transitions_path: str,
    change_count_path: str,
    tile_ids: str,
) -> None:
    """Compute transitions for a batch of tiles (Beaker worker entry point).

    Args:
        stack_path: directory containing stacked GeoTIFFs.
        transitions_path: directory to write transition rasters.
        change_count_path: directory to write change count rasters.
        tile_ids: JSON-encoded list of tile ID strings.
    """
    import json

    ids = json.loads(tile_ids)
    logger.info("processing %d tiles", len(ids))
    for tile_id in ids:
        try:
            compute_changes(
                tile_id=tile_id,
                stack_path=stack_path,
                transitions_path=transitions_path,
                change_count_path=change_count_path,
            )
        except Exception:
            logger.exception("failed to process tile %s", tile_id)
