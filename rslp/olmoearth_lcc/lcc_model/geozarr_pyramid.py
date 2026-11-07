"""Multiscale pyramid store for the OlmoEarth LCC GeoZarr predictions.

The overviews of a main store (see geozarr.py, which holds the 10 m level 0) are kept
in a separate GeoZarr store, so the 10 m store remains simple. A client reads level 0
from the main store and the overviews from this store, using the same zone group names.
The multiscales convention only allows levels in child groups of the group that lists
them, so the overviews can't sit next to the 10 m arrays as sibling groups.

Each zone group ``utm{NN}`` has child groups ``1``..``5`` (20, 40, 80, 160 and 320
m/pixel), each holding ``change`` and ``change_dates`` arrays with the same bands as
the main store plus band-name and x/y coordinate arrays. The zone group carries a
``multiscales`` attribute (zarr-conventions/multiscales) listing the levels. Level 1
is derived from level 0 in the main store, which the convention cannot reference, so
the root ``source_store`` and zone group ``source_group`` attributes record it.

Each level is a 2x reduction of the previous one (see reduce_2x): every output pixel
copies all bands from one input pixel of its 2x2 block, preferring changes with a pre
or post category, then higher binary change probabilities, then valid pixels over nodata.
The bands stay mutually consistent and small changes stay visible at coarse levels.

Zone grids are snapped to multiples of TILE_SIZE (tiling.get_zone_grid), so every
tile maps to an aligned, disjoint block at every level, and the shard size at level k
is ``min(2048, TILE_SIZE >> k)``. Each tile's job therefore writes its own pyramid
blocks (write_tile_pyramid) without touching shards of other tiles, so there is no
global pass and resuming still works. Staying in UTM also keeps the reduction exact:
a WebMercator pyramid would straddle tile and zone boundaries and resample
categorical data, so derive one offline from level 1 if needed.
"""

from typing import Any

import einops
import numpy as np
import zarr
from rslearn.utils.geometry import PixelBounds, Projection

from rslp.log_utils import get_logger

from . import geozarr
from .output_bands import (
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from .tiling import RESOLUTION, TILE_SIZE, check_bounds_aligned, get_zone_grid

logger = get_logger(__name__)

PYRAMID_LEVELS = [1, 2, 3, 4, 5]
RESAMPLING_METHOD = "max_change_pixel"
SOURCE_STORE_ATTR = "source_store"
SOURCE_GROUP_ATTR = "source_group"

MULTISCALES_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}
PYRAMID_ZARR_CONVENTIONS = [*geozarr.ZARR_CONVENTIONS, MULTISCALES_CONVENTION]

_PRE_CATEGORY_BAND = geozarr.CHANGE_BANDS.index("pre_change_category")
_POST_CATEGORY_BAND = geozarr.CHANGE_BANDS.index("post_change_category")
_BINARY_CHANGE_BAND = geozarr.CHANGE_BANDS.index("binary_change")


def level_shard_size(level: int) -> int:
    """Get the shard spatial size of a pyramid level."""
    return min(geozarr.SHARD_SIZE, TILE_SIZE >> level)


def level_projection(projection: Projection, level: int) -> Projection:
    """Get the projection of a pyramid level from the level-0 projection."""
    scale = 2**level
    return Projection(
        projection.crs, projection.x_resolution * scale, projection.y_resolution * scale
    )


def level_group_path(zone_number: int, level: int) -> str:
    """Get the path of a pyramid level group (e.g. 'utm10/1')."""
    return f"{geozarr.zone_group_name(zone_number)}/{level}"


def init_pyramid_store(
    store_path: str,
    source_store: str,
    zone_numbers: list[int] | None = None,
    overwrite: bool = False,
) -> None:
    """Create the pyramid store skeleton (root, zone and level groups, arrays).

    Args:
        store_path: the pyramid Zarr store path or URL.
        source_store: the main (level 0) store path or URL, recorded in the
            attributes.
        zone_numbers: the UTM zone numbers to create groups for (default all 60).
        overwrite: whether to overwrite an existing store.
    """
    if zone_numbers is None:
        zone_numbers = list(range(1, 61))

    root = zarr.open_group(store=store_path, mode="w" if overwrite else "w-")
    root.attrs.update(
        {
            "zarr_conventions": PYRAMID_ZARR_CONVENTIONS,
            SOURCE_STORE_ATTR: source_store,
            "spatial_layout": "utm_zones",
            "lc_class_names": LC_CLASS_NAMES,
            "pre_change_category_names": PRE_CHANGE_CATEGORY_NAMES,
            "post_change_category_names": POST_CHANGE_CATEGORY_NAMES,
            "resampling_method": RESAMPLING_METHOD,
            "description": (
                "Overviews (levels 1-5, 20-320 m/pixel) of the OlmoEarth Land Change GeoZarr "
                "store in source_store, which holds level 0 (10 m/pixel). Arrays and band "
                "encodings match the source store. Each pixel at a level one higher copies "
                "all bands from one pixel of its 2x2 block in the previous level, preferring "
                "changes with a pre or post category, then higher binary_change, then valid "
                "pixels over nodata."
            ),
        }
    )

    for zone_number in zone_numbers:
        projection, origin_px, shape_px = get_zone_grid(
            zone_number, RESOLUTION, TILE_SIZE
        )
        zone_group = root.create_group(geozarr.zone_group_name(zone_number))
        layout: list[dict[str, Any]] = []
        for level in PYRAMID_LEVELS:
            cur_projection = level_projection(projection, level)
            cur_origin = (origin_px[0] >> level, origin_px[1] >> level)
            cur_shape = (shape_px[0] >> level, shape_px[1] >> level)
            shard_size = level_shard_size(level)
            level_group = geozarr.create_zone_group(
                zone_group,
                str(level),
                cur_projection,
                cur_origin,
                cur_shape,
                geozarr.CHANGE_ARRAY_SPECS,
                chunk_size=min(geozarr.CHUNK_SIZE, shard_size),
                shard_size=shard_size,
            )
            entry: dict[str, Any] = {
                "asset": str(level),
                "spatial:transform": level_group.attrs["spatial:transform"],
                "spatial:shape": level_group.attrs["spatial:shape"],
            }
            if level > PYRAMID_LEVELS[0]:
                entry["derived_from"] = str(level - 1)
                entry["transform"] = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
            layout.append(entry)

        zone_group.attrs.update(
            {
                "zarr_conventions": PYRAMID_ZARR_CONVENTIONS,
                "proj:code": f"EPSG:{projection.crs.to_epsg()}",
                "spatial:dimensions": ["y", "x"],
                SOURCE_GROUP_ATTR: geozarr.zone_group_name(zone_number),
                "multiscales": {
                    "layout": layout,
                    "resampling_method": RESAMPLING_METHOD,
                },
            }
        )

    zarr.consolidate_metadata(root.store)
    logger.info(
        "initialized pyramid store %s with %d zones", store_path, len(zone_numbers)
    )


def read_pyramid_source_store(store_path: str) -> str | None:
    """Read the main store recorded in an existing pyramid store.

    Args:
        store_path: the pyramid Zarr store path or URL.

    Returns:
        the main (level 0) store path or URL, or None if the pyramid store does not
        exist.
    """
    try:
        root = zarr.open_group(store=store_path, mode="r")
    except FileNotFoundError:
        return None
    return str(root.attrs[SOURCE_STORE_ATTR])


def ensure_pyramid_store(store_path: str, source_store: str) -> None:
    """Create the pyramid store for a main store unless it already exists.

    Args:
        store_path: the pyramid Zarr store path or URL.
        source_store: the main store, which must exist.

    Raises:
        ValueError: if the main store is missing, or the pyramid store exists for a
            different main store.
    """
    if geozarr.read_store_detail(source_store) is None:
        raise ValueError(f"main store {source_store} does not exist")
    existing_source = read_pyramid_source_store(store_path)
    if existing_source is None:
        init_pyramid_store(store_path, source_store)
        return
    if existing_source != source_store:
        raise ValueError(
            f"pyramid store {store_path} is for {existing_source}, not {source_store}"
        )
    logger.info("using existing pyramid store %s", store_path)


def reduce_2x(
    change: np.ndarray, change_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce change/change_dates arrays by 2x, picking one pixel per 2x2 block.

    The chosen pixel maximizes ``valid + 2 * binary_change + 512 * pre_or_post_change``,
    where valid means the pixel has a prediction (a nonzero change category) and
    pre_or_post_change means a category other than "none". All bands are copied from it.

    Args:
        change: the uint8 (len(CHANGE_BANDS), H, W) array; H and W must be even.
        change_dates: the uint16 (len(CHANGE_DATES_BANDS), H, W) array.

    Returns:
        the reduced (change, change_dates) arrays of size (H/2, W/2).
    """
    _, height, width = change.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"array size {height}x{width} must be even")
    pre = change[_PRE_CATEGORY_BAND]
    post = change[_POST_CATEGORY_BAND]
    valid = (pre > 0) | (post > 0)
    pre_or_post_change = (pre > NONE_CATEGORY_IDX) | (post > NONE_CATEGORY_IDX)
    score = (
        valid.astype(np.int32)
        + 2 * change[_BINARY_CHANGE_BAND].astype(np.int32)
        + 512 * pre_or_post_change.astype(np.int32)
    )
    blocks = einops.rearrange(score, "(h dy) (w dx) -> h w (dy dx)", dy=2, dx=2)
    choice = blocks.argmax(axis=2)
    rows = 2 * np.arange(height // 2)[:, None] + choice // 2
    cols = 2 * np.arange(width // 2)[None, :] + choice % 2
    return change[:, rows, cols], change_dates[:, rows, cols]


def write_tile_pyramid(
    store_path: str,
    projection: Projection,
    bounds: PixelBounds,
    level1_blocks: list[tuple[PixelBounds, np.ndarray, np.ndarray]],
) -> None:
    """Write a tile's pyramid levels from the level-1 blocks of its windows.

    Args:
        store_path: the pyramid Zarr store path or URL.
        projection: the level-0 tile projection, a northern UTM projection
            (EPSG:326NN) at RESOLUTION.
        bounds: the level-0 tile bounds, which must be multiples of TILE_SIZE.
        level1_blocks: for each predicted window, its level-0 bounds and its
            level-1 (2x reduced, see reduce_2x) uint8 change and uint16
            change_dates arrays. The rest of the tile is nodata.
    """
    zone_number = geozarr.check_zone_projection(projection)
    check_bounds_aligned(bounds, TILE_SIZE)

    # Mosaic the windows' blocks into the tile's level-1 arrays.
    height = (bounds[3] - bounds[1]) // 2
    width = (bounds[2] - bounds[0]) // 2
    change = np.zeros((len(geozarr.CHANGE_BANDS), height, width), dtype=np.uint8)
    dates = np.zeros((len(geozarr.CHANGE_DATES_BANDS), height, width), dtype=np.uint16)
    for window_bounds, block_change, block_dates in level1_blocks:
        col = (window_bounds[0] - bounds[0]) // 2
        row = (window_bounds[1] - bounds[1]) // 2
        block_height, block_width = block_change.shape[1:]
        change[:, row : row + block_height, col : col + block_width] = block_change
        dates[:, row : row + block_height, col : col + block_width] = block_dates

    for level in PYRAMID_LEVELS:
        if level > PYRAMID_LEVELS[0]:
            change, dates = reduce_2x(change, dates)
        group = zarr.open_group(
            store=store_path, path=level_group_path(zone_number, level), mode="r+"
        )
        level_bounds = (
            bounds[0] >> level,
            bounds[1] >> level,
            bounds[2] >> level,
            bounds[3] >> level,
        )
        geozarr.write_arrays(
            group,
            level_bounds,
            {geozarr.CHANGE_ARRAY: change, geozarr.CHANGE_DATES_ARRAY: dates},
        )
    logger.info("wrote pyramid levels for tile %s to %s", bounds, store_path)
