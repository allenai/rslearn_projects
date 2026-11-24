"""Full pyramid store: the overview levels of a pyramid store plus coarse levels.

A pyramid store (geozarr_pyramid.py) has levels 1-5 (20-320 m/pixel), which each tile's
job writes for its own tile. Coarser levels can't be written per tile, since a tile
shrinks below a shard there, so a full pyramid store is built after the run, one UTM
zone at a time (build_pyramid_full_zone, e.g. one Beaker job per zone):

1. Levels 1-5 of the zone are copied object by object from the pyramid store.
2. Levels 6-12 (640 m to 40.96 km/pixel, down to roughly web zoom 2) are derived from
   level 5, each a 2x reduction of the previous one with reduce_2x_majority, and
   written with COARSE_SHARD_SIZE shards, or one shard per level once a zone's level
   fits in one.
3. The zone group zarr.json, whose multiscales layout lists all twelve levels, is
   written last.

finalize_pyramid_full then writes the root zarr.json with consolidated metadata, once
every zone is done, so a store whose root exists is complete (as geozarr.py's stores).
Each object is written once; GCS caches publicly readable objects, so rewriting one
could leave readers with a stale copy.

Levels 1-5 keep max_change_pixel (any change in a block survives). reduce_2x_majority
instead keeps the most common category of each 2x2 block, so sparse changes fade out
at the coarse levels rather than filling every cell.
"""

import json
import math
import posixpath
from typing import Any

import einops
import fsspec
import numpy as np
import zarr
from fsspec.implementations.local import LocalFileSystem
from rslearn.utils.geometry import Projection
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.storage import MemoryStore

from rslp.log_utils import get_logger
from rslp.utils.tiling import get_zone_grid

from . import geozarr, geozarr_pyramid
from .output_bands import NONE_CATEGORY_IDX
from .tiling import RESOLUTION, TILE_SIZE

logger = get_logger(__name__)

FULL_PYRAMID_LEVELS = list(range(1, 13))
COPIED_LEVELS = geozarr_pyramid.PYRAMID_LEVELS
COARSE_LEVELS = [level for level in FULL_PYRAMID_LEVELS if level not in COPIED_LEVELS]
COARSE_RESAMPLING_METHOD = "majority_category_pixel"
COARSE_SHARD_SIZE = 2048
# A coarse level whose zone grid fits in a shard this size is stored as a single shard.
SINGLE_SHARD_MAX_SIZE = 4096
FULL_PYRAMID_DESCRIPTION = (
    "Overviews (levels 1-12, 20 m-40.96 km/pixel) of the OlmoEarth Land Change GeoZarr "
    "store in source_store, which holds level 0 (10 m/pixel). Arrays and band encodings "
    "match the source store. Each pixel at a level one higher copies all bands from one "
    "pixel of its 2x2 block in the previous level. Levels 1-5 (resampling_method "
    "max_change_pixel) prefer changes with a pre or post category, then higher "
    "binary_change, then valid pixels over nodata. Levels 6-12 (coarse_resampling_method "
    "majority_category_pixel) prefer the most common category among the valid pixels "
    "(the post-change category, else the pre-change category, else none), a category over "
    "none on ties, then higher binary_change."
)

_PRE_BAND = geozarr.CHANGE_BANDS.index("pre_change_category")
_POST_BAND = geozarr.CHANGE_BANDS.index("post_change_category")
_BINARY_BAND = geozarr.CHANGE_BANDS.index("binary_change")
# Offset of pre-change categories in category keys, so they can't collide with post.
_PRE_KEY_OFFSET = 256


def category_keys(change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the majority-vote key and validity of each pixel.

    Args:
        change: the uint8 (len(CHANGE_BANDS), H, W) array.

    Returns:
        (keys, valid): int32 keys are the post-change category if it is not none, else
        _PRE_KEY_OFFSET + the pre-change category if that is not none, else 0 (none);
        valid means the pixel has a prediction (a nonzero category).
    """
    pre = change[_PRE_BAND].astype(np.int32)
    post = change[_POST_BAND].astype(np.int32)
    keys = np.where(
        post > NONE_CATEGORY_IDX,
        post,
        np.where(pre > NONE_CATEGORY_IDX, _PRE_KEY_OFFSET + pre, 0),
    )
    return keys, (pre > 0) | (post > 0)


def reduce_2x_majority(
    change: np.ndarray, change_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce change/change_dates arrays by 2x, keeping each block's majority category.

    Each valid pixel of a 2x2 block votes for its key (see category_keys: its post-change
    category, else its pre-change category, else none). The chosen pixel has the key with
    the most votes; ties go to a category over none, then to the higher binary_change,
    then to the first pixel (row-major). Nodata pixels don't vote and are only chosen if
    the whole block is nodata. All bands are copied from the chosen pixel.

    Args:
        change: the uint8 (len(CHANGE_BANDS), H, W) array; H and W must be even.
        change_dates: the uint16 (len(CHANGE_DATES_BANDS), H, W) array.

    Returns:
        the reduced (change, change_dates) arrays of size (H/2, W/2).
    """
    _, height, width = change.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"array size {height}x{width} must be even")
    keys, valid = category_keys(change)
    pattern = "(h dy) (w dx) -> h w (dy dx)"
    key_blocks = einops.rearrange(keys, pattern, dy=2, dx=2)
    valid_blocks = einops.rearrange(valid, pattern, dy=2, dx=2)
    binary_blocks = einops.rearrange(
        change[_BINARY_BAND].astype(np.int32), pattern, dy=2, dx=2
    )
    # votes[..., i]: how many valid pixels of the block share pixel i's key.
    same = key_blocks[..., :, None] == key_blocks[..., None, :]
    votes = (same & valid_blocks[..., None, :]).sum(axis=-1)
    # votes (<= 4) * 1024 outweighs the category bonus (512), which outweighs
    # 2 * binary_change + 1 (<= 511); invalid pixels score 0.
    score = valid_blocks * (
        1024 * votes + 512 * (key_blocks > 0) + 2 * binary_blocks + 1
    )
    choice = score.argmax(axis=2)
    rows = 2 * np.arange(height // 2)[:, None] + choice // 2
    cols = 2 * np.arange(width // 2)[None, :] + choice % 2
    return change[:, rows, cols], change_dates[:, rows, cols]


def coarse_shard_size(shape: tuple[int, int]) -> int:
    """Get the shard size of a coarse level from its zone grid shape (height, width)."""
    largest = max(shape)
    if largest > SINGLE_SHARD_MAX_SIZE:
        return COARSE_SHARD_SIZE
    return max(
        geozarr.CHUNK_SIZE, math.ceil(largest / geozarr.CHUNK_SIZE) * geozarr.CHUNK_SIZE
    )


def level_stats(change: np.ndarray) -> dict[str, int]:
    """Count valid pixels and pixels with a category in a level's change array."""
    keys, valid = category_keys(change)
    return {
        "pixels": int(keys.size),
        "valid": int(valid.sum()),
        "with_category": int((valid & (keys > 0)).sum()),
        "with_post_category": int((change[_POST_BAND] > NONE_CATEGORY_IDX).sum()),
    }


def _zone_layout(
    projection: Projection, origin_px: tuple[int, int], shape_px: tuple[int, int]
) -> list[dict[str, Any]]:
    """Get the multiscales layout entries of all full pyramid levels of a zone."""
    layout: list[dict[str, Any]] = []
    for level in FULL_PYRAMID_LEVELS:
        attrs = geozarr.build_zone_spatial_attrs(
            geozarr_pyramid.level_projection(projection, level),
            (origin_px[0] >> level, origin_px[1] >> level),
            (shape_px[0] >> level, shape_px[1] >> level),
        )
        entry: dict[str, Any] = {
            "asset": str(level),
            "spatial:transform": attrs["spatial:transform"],
            "spatial:shape": attrs["spatial:shape"],
        }
        if level > FULL_PYRAMID_LEVELS[0]:
            entry["derived_from"] = str(level - 1)
            entry["transform"] = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
        if level in COARSE_LEVELS:
            entry["resampling_method"] = COARSE_RESAMPLING_METHOD
        layout.append(entry)
    return layout


def _put_objects(fs: fsspec.AbstractFileSystem, objects: dict[str, bytes]) -> None:
    """Write objects to fs, creating local parent directories if needed."""
    if isinstance(fs, LocalFileSystem):
        for parent in sorted({posixpath.dirname(p) for p in objects}):
            fs.makedirs(parent, exist_ok=True)
    fs.pipe(objects)


def _copy_level_objects(
    source_store: str, dest_store: str, zone_name: str, level: int
) -> list[tuple[str, int]]:
    """Copy one pyramid level of a zone, object by object (server side on GCS).

    Returns:
        the copied (relative path, size) pairs.
    """
    fs, src = fsspec.core.url_to_fs(source_store)
    _, dst = fsspec.core.url_to_fs(dest_store)
    src, dst = src.rstrip("/"), dst.rstrip("/")
    prefix = f"{src}/{zone_name}/{level}"
    found = fs.find(prefix, detail=True)
    rel = sorted(posixpath.relpath(p, src) for p in found)
    sources = [f"{src}/{r}" for r in rel]
    dests = [f"{dst}/{r}" for r in rel]
    if isinstance(fs, LocalFileSystem):
        for parent in sorted({posixpath.dirname(p) for p in dests}):
            fs.makedirs(parent, exist_ok=True)
    fs.copy(sources, dests)
    return [(r, int(found[f"{src}/{r}"]["size"])) for r in rel]


def _row_band(
    copied: list[tuple[str, int]], zone_name: str, level: int, height: int
) -> tuple[int, int] | None:
    """Get the rows of a level spanned by its change shards, aligned for the reduction.

    Args:
        copied: the level's copied (relative path, size) pairs.
        zone_name: the zone group name.
        level: the level (its shards are level_shard_size(level) pixels).
        height: the level's zone grid height.

    Returns:
        (start, end) rows, multiples of 2 ** (max coarse level - level), or None if the
        level has no change shards.
    """
    prefix = f"{zone_name}/{level}/{geozarr.CHANGE_ARRAY}/c/"
    shard_rows = [
        int(r[len(prefix) :].split("/")[1]) for r, _ in copied if r.startswith(prefix)
    ]
    if not shard_rows:
        return None
    shard = geozarr_pyramid.level_shard_size(level)
    align = 2 ** (COARSE_LEVELS[-1] - level)
    start = min(shard_rows) * shard // align * align
    end = min(height, -(-(max(shard_rows) + 1) * shard // align) * align)
    return start, end


def build_pyramid_full_zone(
    source_store: str, dest_store: str, zone_number: int, stats_path: str | None = None
) -> dict[str, Any]:
    """Build one UTM zone of a full pyramid store (see the module docstring).

    Args:
        source_store: the pyramid store (levels 1-5) path or URL.
        dest_store: the full pyramid store path or URL; its root must not exist yet.
        zone_number: the UTM zone number.
        stats_path: optional path or URL to write the zone's stats JSON to.

    Returns:
        per-level stats (pixels, valid, with_category, with_post_category for levels
        5-12; copied object count and bytes for levels 1-5).
    """
    zone_name = geozarr.zone_group_name(zone_number)
    fs, dest_path = fsspec.core.url_to_fs(dest_store)
    dest_path = dest_path.rstrip("/")
    if fs.exists(f"{dest_path}/{geozarr.ZARR_JSON}"):
        raise FileExistsError(f"{dest_store} is already finalized")
    if fs.exists(f"{dest_path}/{zone_name}/{geozarr.ZARR_JSON}"):
        raise FileExistsError(f"{dest_store} zone {zone_name} is already built")
    stats: dict[str, Any] = {"zone": zone_number, "levels": {}}

    # 1. Copy levels 1-5.
    copied: list[tuple[str, int]] = []
    for level in COPIED_LEVELS:
        level_copied = _copy_level_objects(source_store, dest_store, zone_name, level)
        copied += level_copied
        stats["levels"][str(level)] = {
            "objects": len(level_copied),
            "bytes": sum(size for _, size in level_copied),
        }
        logger.info(
            "%s level %d: copied %d objects", zone_name, level, len(level_copied)
        )

    # 2. Create levels 6-12, then fill them from level 5. Only the band of level-5 rows
    # that has shards is read; the rest of every level is nodata (no shards written).
    projection, origin_px, shape_px = get_zone_grid(zone_number, RESOLUTION, TILE_SIZE)
    source_zone = zarr.open_group(store=source_store, path=zone_name, mode="r")
    store_dict: dict[str, Buffer] = {}
    mem_root = zarr.open_group(store=MemoryStore(store_dict), mode="w")
    mem_zone = mem_root.create_group(zone_name)
    for level in COARSE_LEVELS:
        shape = (shape_px[0] >> level, shape_px[1] >> level)
        shard_size = coarse_shard_size(shape)
        geozarr.create_zone_group(
            mem_zone,
            str(level),
            geozarr_pyramid.level_projection(projection, level),
            (origin_px[0] >> level, origin_px[1] >> level),
            shape,
            geozarr.CHANGE_ARRAY_SPECS,
            chunk_size=min(geozarr.CHUNK_SIZE, shard_size),
            shard_size=shard_size,
        )
    mem_zone.attrs.update(
        {
            **dict(source_zone.attrs),
            "multiscales": {
                "layout": _zone_layout(projection, origin_px, shape_px),
                "resampling_method": geozarr_pyramid.RESAMPLING_METHOD,
            },
        }
    )
    objects = {
        f"{dest_path}/{key}": value.to_bytes()
        for key, value in store_dict.items()
        if key != geozarr.ZARR_JSON
    }
    zone_json_key = f"{dest_path}/{zone_name}/{geozarr.ZARR_JSON}"
    zone_json = objects.pop(zone_json_key)
    _put_objects(fs, objects)

    top = COPIED_LEVELS[-1]
    band = _row_band(copied, zone_name, top, shape_px[0] >> top)
    stats["band"] = band
    if band is not None:
        change = np.asarray(
            source_zone[f"{top}/{geozarr.CHANGE_ARRAY}"][:, band[0] : band[1]]
        )
        dates = np.asarray(
            source_zone[f"{top}/{geozarr.CHANGE_DATES_ARRAY}"][:, band[0] : band[1]]
        )
        stats["levels"][str(top)].update(level_stats(change))
        for level in COARSE_LEVELS:
            change, dates = reduce_2x_majority(change, dates)
            scale = 2 ** (level - top)
            group = zarr.open_group(
                store=dest_store, path=f"{zone_name}/{level}", mode="r+"
            )
            rows = slice(band[0] // scale, band[1] // scale)
            group[geozarr.CHANGE_ARRAY][:, rows] = change
            group[geozarr.CHANGE_DATES_ARRAY][:, rows] = dates
            stats["levels"][str(level)] = level_stats(change)
            logger.info(
                "%s level %d: %s", zone_name, level, stats["levels"][str(level)]
            )

    # 3. The zone group last.
    _put_objects(fs, {zone_json_key: zone_json})
    if stats_path is not None:
        with fsspec.open(stats_path, "w") as f:
            json.dump(stats, f, indent=1)
    logger.info("built %s of %s", zone_name, dest_store)
    return stats


def verify_pyramid_full_zone(
    source_store: str, dest_store: str, zone_number: int
) -> None:
    """Check a built zone of a full pyramid store against its source.

    Levels 1-5 must hold the same objects (paths and sizes), level 5 the same data, and
    each coarse level must equal reduce_2x_majority of the level before it as stored.

    Raises:
        ValueError: if anything differs.
    """
    zone_name = geozarr.zone_group_name(zone_number)
    fs, src = fsspec.core.url_to_fs(source_store)
    _, dst = fsspec.core.url_to_fs(dest_store)
    src, dst = src.rstrip("/"), dst.rstrip("/")
    for level in COPIED_LEVELS:
        a = {
            posixpath.relpath(p, src): d["size"]
            for p, d in fs.find(f"{src}/{zone_name}/{level}", detail=True).items()
        }
        b = {
            posixpath.relpath(p, dst): d["size"]
            for p, d in fs.find(f"{dst}/{zone_name}/{level}", detail=True).items()
        }
        if a != b:
            raise ValueError(f"{zone_name} level {level}: copied objects differ")
    source_zone = zarr.open_group(store=source_store, path=zone_name, mode="r")
    dest_zone = zarr.open_group(store=dest_store, path=zone_name, mode="r")
    layout = dest_zone.attrs["multiscales"]["layout"]  # type: ignore[index]
    if [entry["asset"] for entry in layout] != [str(x) for x in FULL_PYRAMID_LEVELS]:  # type: ignore[index, union-attr]
        raise ValueError(f"{zone_name}: unexpected multiscales layout")
    top = COPIED_LEVELS[-1]
    height = dest_zone[f"{top}/{geozarr.CHANGE_ARRAY}"].shape[1]  # type: ignore[union-attr]
    copied = [
        (posixpath.relpath(p, dst), 0) for p in fs.find(f"{dst}/{zone_name}/{top}")
    ]
    band = _row_band(copied, zone_name, top, height)
    names = [geozarr.CHANGE_ARRAY, geozarr.CHANGE_DATES_ARRAY]
    if band is None:
        for level in COARSE_LEVELS:
            if fs.find(f"{dst}/{zone_name}/{level}/{geozarr.CHANGE_ARRAY}/c"):
                raise ValueError(
                    f"{zone_name} level {level} has data but level {top} has none"
                )
        logger.info("verified %s of %s (no data)", zone_name, dest_store)
        return
    prev = None
    for level in [top, *COARSE_LEVELS]:
        scale = 2 ** (level - top)
        rows = slice(band[0] // scale, band[1] // scale)
        full = (
            [np.asarray(dest_zone[f"{level}/{name}"][:]) for name in names]
            if level != top
            else None
        )
        cur = (
            [np.asarray(dest_zone[f"{level}/{name}"][:, rows]) for name in names]
            if full is None
            else [arr[:, rows] for arr in full]
        )
        if full is not None:
            for arr in full:
                outside = np.ones(arr.shape[1], dtype=bool)
                outside[rows] = False
                if arr[:, outside].any():
                    raise ValueError(
                        f"{zone_name} level {level} has data outside the band"
                    )
        if prev is None:
            for name, arr in zip(names, cur):
                if not np.array_equal(arr, source_zone[f"{level}/{name}"][:, rows]):
                    raise ValueError(f"{zone_name} level {level} {name} differs")
        else:
            expected = reduce_2x_majority(*prev)
            if not all(np.array_equal(x, y) for x, y in zip(expected, cur)):
                raise ValueError(
                    f"{zone_name} level {level} is not reduced from {level - 1}"
                )
        prev = cur
    logger.info("verified %s of %s", zone_name, dest_store)


def pyramid_full_zone(
    source_store: str, dest_store: str, zone_number: int, stats_path: str | None = None
) -> None:
    """Build and verify one zone of a full pyramid store (one Beaker job per zone).

    Args:
        source_store: the pyramid store (levels 1-5) path or URL.
        dest_store: the full pyramid store path or URL.
        zone_number: the UTM zone number.
        stats_path: optional path or URL for the zone's stats JSON.
    """
    build_pyramid_full_zone(source_store, dest_store, zone_number, stats_path)
    verify_pyramid_full_zone(source_store, dest_store, zone_number)


def finalize_pyramid_full(dest_store: str, source_store: str) -> None:
    """Write the root zarr.json (with consolidated metadata) of a full pyramid store.

    Run once after every zone is built; the root attributes come from the source
    pyramid store's root, with the full pyramid description.

    Args:
        dest_store: the full pyramid store path or URL.
        source_store: the pyramid store (levels 1-5) it was built from.
    """
    fs, dst = fsspec.core.url_to_fs(dest_store)
    dst = dst.rstrip("/")
    root_path = f"{dst}/{geozarr.ZARR_JSON}"
    if fs.exists(root_path):
        raise FileExistsError(f"{dest_store} is already finalized")
    source_root = zarr.open_group(store=source_store, mode="r")
    zone_names = sorted(
        posixpath.basename(posixpath.dirname(p))
        for p in fs.glob(f"{dst}/utm*/{geozarr.ZARR_JSON}")
    )
    if not zone_names:
        raise ValueError(f"{dest_store} has no built zones")
    paths = [
        p
        for p in fs.find(dst)
        if p.endswith(f"/{geozarr.ZARR_JSON}") and p != root_path
    ]
    proto = default_buffer_prototype()
    store_dict: dict[str, Buffer] = {
        posixpath.relpath(p, dst): proto.buffer.from_bytes(data)
        for p, data in fs.cat(paths).items()
    }
    root = zarr.open_group(store=MemoryStore(store_dict), mode="a")
    root.attrs.update(
        {
            **dict(source_root.attrs),
            "description": FULL_PYRAMID_DESCRIPTION,
            "coarse_resampling_method": COARSE_RESAMPLING_METHOD,
            "pyramid_levels": FULL_PYRAMID_LEVELS,
            "zones": zone_names,
        }
    )
    zarr.consolidate_metadata(root.store)
    fs.pipe_file(root_path, store_dict[geozarr.ZARR_JSON].to_bytes())
    logger.info("finalized %s with %d zones", dest_store, len(zone_names))
