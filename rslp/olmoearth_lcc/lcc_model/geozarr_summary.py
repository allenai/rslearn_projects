"""Coarse change summary ("heatmap") store for an OlmoEarth LCC GeoZarr store.

A map zoomed out over a continent can't read 10 m pixels, and the pyramid store's
max-pick overviews (geozarr_pyramid.py) can't be filtered: hiding the class of a
block's representative pixel hides the whole block. This store holds, per coarse
cell, how much of the cell changed for every change category and month, so a client
can filter by category and month and still show how much matches.

Layout: a Zarr v3 store in Web Mercator (EPSG:3857) with one group per level ``L`` in
0..MAX_LEVEL, each holding one array ``values`` with dimensions
``(category, month, y, x)``. Level L has TILE_CELLS cells per side of every XYZ map
tile at zoom L, i.e. 2**L * TILE_CELLS cells across the world, and one chunk per map
tile, so a client reads one object per map tile and draws a cell as a block of
256 / TILE_CELLS screen pixels. Chunks without any change aren't stored (fill 0).

- category: CATEGORIES, the post change categories ("none" and 18 others) followed by
  the pre change categories ("none" and 6 others). A change pixel (a pre or post change
  category other than none) counts once in its post category and once in its pre
  category. Pixels whose categories are both none, including those with a high
  binary change score, are left out.
- month: NUM_MONTHS buckets of the change's end month, the last one being the month of
  the run's reference time (root attribute ``first_month`` names the first). Changes
  that ended earlier are left out.
- value: the fraction of the cell's ground area that changed, on a log scale: 0 = none,
  otherwise fraction = 10 ** ((value - 1) * LOG_DECADES / 254 - LOG_DECADES), so values
  1..255 span 1e-10..1 in steps of ~9.5% (rounding error <= ~4.7%). A linear or
  square-root uint8 can't hold one 10 m pixel in a zoom-0 cell (~2.6e-10 of it) without
  rounding it up by orders of magnitude, which inflates sums of many small values.
  Decode before summing; how to draw it (e.g. sqrt of the fraction) is up to the client.

Building it takes three steps, each of which parallelizes:

1. ``partials``: per 16K tile of the source store (from its ``meta/*_meta.json``),
   read every window at 10 m and count change pixels per (MAX_LEVEL cell, category,
   month), assigning each pixel to the cell containing its center and keeping only
   pixels inside its UTM zone's 6 degree band (zone grids overlap their neighbours).
   Writes one ``<tile>.npz`` of sparse counts per tile.
2. ``merge``: per zoom-MERGE_LEVEL map tile, sum the partials into levels MAX_LEVEL
   down to MERGE_LEVEL, write their chunks, and save the MERGE_LEVEL counts.
3. ``top``: sum the MERGE_LEVEL counts into levels MERGE_LEVEL - 1 down to 0.

Counts stay integers until a chunk is written, so every level is an exact sum of the
10 m pixels (each written value is quantized once). ``init`` writes the store's
metadata once, before any chunk, since a cached read of overwritten metadata on GCS
can be stale.

Self-contained (numpy, zarr, pyproj, fsspec/gcsfs) so it runs in a plain Python
image: ``python geozarr_summary.py <step> ...``.
"""

import argparse
import json
import math
import multiprocessing
import os
import posixpath
import time
from collections.abc import Iterable

import fsspec
import numpy as np
import zarr
from pyproj import Transformer
from zarr.codecs import BytesCodec, ZstdCodec

PRE_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "forest_loss",
    "removed_building",
    "wetland_loss",
    "water_to_land",
    "removed_crop_structure",
    "mangrove_loss",
]
POST_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_agriculture",
    "new_aquafarm",
    "site_clearing",
    "land_to_water",
    "new_mining_or_quarry",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
    "agricultural_activity",
    "burned",
    "flooding",
    "storm_damage",
    "vegetation_dieback",
]
# Category planes: post classes 1.. then pre classes 1.. (nodata dropped).
NUM_POST = len(POST_CHANGE_CATEGORY_NAMES) - 1
NUM_PRE = len(PRE_CHANGE_CATEGORY_NAMES) - 1
CATEGORIES = [f"post:{n}" for n in POST_CHANGE_CATEGORY_NAMES[1:]] + [
    f"pre:{n}" for n in PRE_CHANGE_CATEGORY_NAMES[1:]
]
NUM_CATEGORIES = len(CATEGORIES)
NUM_MONTHS = 24
TILE_CELLS = 64
MAX_LEVEL = 10
MERGE_LEVEL = 5
# Bands of the source store's arrays (geozarr.CHANGE_BANDS / CHANGE_DATES_BANDS).
PRE_CATEGORY_BAND = 3
POST_CATEGORY_BAND = 4
END_DAYS_BAND = 1
PIXEL_AREA_M2 = 100.0
EARTH_CIRCUMFERENCE_M = 40075016.686
MONTH_EPOCH_YEAR = 2015  # month values: 1 + whole months since 2015-01 (viewer's)
ENCODING = "log10_fraction_u8"
LOG_DECADES = 10.0


def month_value(year: int, month: int) -> int:
    """The viewer's month value (1 = 2015-01) of a year and month."""
    return (year - MONTH_EPOCH_YEAR) * 12 + month


def month_iso(value: int) -> str:
    """'YYYY-MM' of a month value."""
    months = value - 1
    return f"{MONTH_EPOCH_YEAR + months // 12}-{months % 12 + 1:02d}"


def days_to_month_value(days: np.ndarray) -> np.ndarray:
    """Month values of day counts since 1970-01-01 (0 stays 0)."""
    out = np.zeros(days.shape, np.int32)
    nz = days > 0
    d = days[nz].astype("datetime64[D]")
    years = d.astype("datetime64[Y]").astype(np.int64) + 1970
    months = d.astype("datetime64[M]").astype(np.int64) % 12 + 1
    out[nz] = (years - MONTH_EPOCH_YEAR) * 12 + months
    return out


def cells_per_side(level: int) -> int:
    """Summary cells along each side of the world at a zoom level."""
    return (1 << level) * TILE_CELLS


def lonlat_to_cells(
    lon: np.ndarray, lat: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Web Mercator cell (row, col) at a level of lon/lat points."""
    n = cells_per_side(level)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    u = (lon + 180.0) / 360.0
    s = np.sin(np.radians(lat))
    v = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    col = np.clip(np.floor(u * n), 0, n - 1).astype(np.int64)
    row = np.clip(np.floor(v * n), 0, n - 1).astype(np.int64)
    return row, col


def cell_row_lat(rows: np.ndarray, level: int) -> np.ndarray:
    """Latitude of the centers of cell rows at a level."""
    n = cells_per_side(level)
    y = math.pi * (1 - 2 * (rows + 0.5) / n)
    return np.degrees(np.arctan(np.sinh(y)))


def cell_area_m2(rows: np.ndarray, level: int) -> np.ndarray:
    """Ground area of cells in the given rows (Mercator scale at the cell center)."""
    side = EARTH_CIRCUMFERENCE_M / cells_per_side(level)
    return (side * np.cos(np.radians(cell_row_lat(rows, level)))) ** 2


def encode_key(
    row: np.ndarray, col: np.ndarray, cat: np.ndarray, month: np.ndarray, level: int
) -> np.ndarray:
    """Pack (row, col, category, month) into one int64 key per entry."""
    n = cells_per_side(level)
    return ((row * n + col) * NUM_CATEGORIES + cat) * NUM_MONTHS + month


def decode_key(
    key: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of encode_key: (row, col, category, month)."""
    n = cells_per_side(level)
    month = key % NUM_MONTHS
    rest = key // NUM_MONTHS
    cat = rest % NUM_CATEGORIES
    cell = rest // NUM_CATEGORIES
    return cell // n, cell % n, cat, month


def sum_by_key(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum counts of equal keys; returns sorted unique keys and their sums."""
    if keys.size == 0:
        return keys.astype(np.int64), counts.astype(np.int64)
    uniq, inverse = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inverse, weights=counts, minlength=uniq.size).astype(
        np.int64
    )


def coarsen(
    keys: np.ndarray, counts: np.ndarray, level: int, to_level: int
) -> tuple[np.ndarray, np.ndarray]:
    """Counts at `level` summed into the cells of the coarser `to_level`."""
    row, col, cat, month = decode_key(keys, level)
    shift = level - to_level
    return sum_by_key(
        encode_key(row >> shift, col >> shift, cat, month, to_level), counts
    )


def encode_fraction(frac: np.ndarray) -> np.ndarray:
    """Positive fractions as log-scale uint8 values 1..255 (see the module docstring)."""
    v = 1 + np.round(
        (np.log10(np.minimum(frac, 1.0)) + LOG_DECADES) * 254 / LOG_DECADES
    )
    return np.clip(v, 1, 255).astype(np.uint8)


def decode_fraction(values: np.ndarray) -> np.ndarray:
    """Fractions of log-scale uint8 values (0 -> 0)."""
    frac = 10.0 ** ((values.astype(np.float64) - 1) * LOG_DECADES / 254 - LOG_DECADES)
    return np.where(values > 0, frac, 0.0)


def quantize(counts: np.ndarray, rows: np.ndarray, level: int) -> np.ndarray:
    """Pixel counts of cells in `rows` as log-scale fraction values."""
    return encode_fraction(counts * PIXEL_AREA_M2 / cell_area_m2(rows, level))


# ---------------------------------------------------------------------------
# Step 1: per-tile partial counts.


def _strip(path: str) -> str:
    return path.split("://", 1)[1] if "://" in path else path


def _fs(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.core.url_to_fs(path)[0]


def tile_window_counts(
    change: np.ndarray,
    end_days: np.ndarray,
    x0: float,
    y0: float,
    transformer: Transformer,
    zone: int,
    first_month: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse MAX_LEVEL counts of one window.

    `change` holds the pre and post category bands, (x0, y0) is the window's top-left corner in EPSG:326NN.
    """
    pre, post = change[0], change[1]
    is_change = (pre > 0) & (post > 0) & ((pre > 1) | (post > 1))
    month = days_to_month_value(end_days) - first_month
    is_change &= (month >= 0) & (month < NUM_MONTHS)
    rows, cols = np.nonzero(is_change)
    if rows.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    lon, lat = transformer.transform(x0 + (cols + 0.5) * 10.0, y0 - (rows + 0.5) * 10.0)
    west = (zone - 1) * 6 - 180
    in_band = (lon >= west) & (lon < west + 6)
    rows, cols, lon, lat = rows[in_band], cols[in_band], lon[in_band], lat[in_band]
    crow, ccol = lonlat_to_cells(lon, lat, MAX_LEVEL)
    m = month[rows, cols].astype(np.int64)
    post_plane = post[rows, cols].astype(np.int64) - 1
    pre_plane = NUM_POST + pre[rows, cols].astype(np.int64) - 1
    keys = np.concatenate(
        [
            encode_key(crow, ccol, post_plane, m, MAX_LEVEL),
            encode_key(crow, ccol, pre_plane, m, MAX_LEVEL),
        ]
    )
    return sum_by_key(keys, np.ones(keys.size, np.int64))


def tile_partial(store: str, meta_path: str, out_dir: str, first_month: int) -> dict:
    """Count one tile's change pixels into `<out_dir>/<tile>.npz`."""
    name = posixpath.basename(meta_path).removesuffix("_meta.json")
    out_path = posixpath.join(out_dir, f"{name}.npz")
    out_fs = _fs(out_dir)
    if out_fs.exists(out_path):
        return {"tile": name, "skipped": True}
    t0 = time.time()
    with fsspec.open(meta_path) as f:
        meta = json.load(f)
    epsg = int(meta["projection"]["crs"].split(":")[1])
    zone = epsg % 100
    south = epsg // 100 == 327
    root = zarr.open_group(store, mode="r")
    group = root[f"utm{zone:02d}"]
    transform = group.attrs["spatial:transform"]
    change_arr = group["change"]
    dates_arr = group["change_dates"]
    transformer = Transformer.from_crs(
        f"EPSG:{32600 + zone}", "EPSG:4326", always_xy=True
    )
    all_keys, all_counts = [], []
    lon_lat_bounds = [180.0, 90.0, -180.0, -90.0]
    for w in meta["windows"]:
        wx0, wy0, wx1, wy1 = w["bounds"]
        gx, gy = wx0 * 10.0, -wy0 * 10.0
        if south:
            gy -= 10_000_000.0
        col = int(round((gx - transform[2]) / transform[0]))
        row = int(round((gy - transform[5]) / transform[4]))
        h, wd = wy1 - wy0, wx1 - wx0
        change = change_arr[
            PRE_CATEGORY_BAND : POST_CATEGORY_BAND + 1, row : row + h, col : col + wd
        ]
        end_days = dates_arr[END_DAYS_BAND, row : row + h, col : col + wd]
        keys, counts = tile_window_counts(
            change, end_days, gx, gy, transformer, zone, first_month
        )
        all_keys.append(keys)
        all_counts.append(counts)
        lons, lats = transformer.transform(
            [gx, gx + wd * 10, gx, gx + wd * 10], [gy, gy, gy - h * 10, gy - h * 10]
        )
        lon_lat_bounds = [
            min(lon_lat_bounds[0], *lons),
            min(lon_lat_bounds[1], *lats),
            max(lon_lat_bounds[2], *lons),
            max(lon_lat_bounds[3], *lats),
        ]
    keys, counts = sum_by_key(np.concatenate(all_keys), np.concatenate(all_counts))
    tmp = out_path + ".tmp"
    with out_fs.open(tmp, "wb") as f:
        np.savez(
            f,
            keys=keys,
            counts=counts.astype(np.uint32),
            bounds=np.array(lon_lat_bounds),
        )
    out_fs.mv(tmp, out_path)
    return {
        "tile": name,
        "windows": len(meta["windows"]),
        "entries": int(keys.size),
        "pixels": int(counts.sum()) // 2,
        "seconds": round(time.time() - t0, 1),
    }


def _tile_partial_star(args: tuple) -> dict:
    try:
        return tile_partial(*args)
    except Exception as e:  # noqa: BLE001 (logged and retried on the next run)
        return {"tile": args[1], "error": repr(e)}


def run_partials(
    store: str,
    out_dir: str,
    end_month: str,
    shard: int,
    num_shards: int,
    workers: int,
) -> None:
    """Step 1 for every num_shards-th tile starting at `shard`."""
    meta_dir = posixpath.join(posixpath.dirname(store.rstrip("/")), "meta")
    fs = _fs(meta_dir)
    protocol = meta_dir.split("://", 1)[0] + "://" if "://" in meta_dir else ""
    metas = sorted(
        protocol + p for p in fs.ls(_strip(meta_dir)) if p.endswith("_meta.json")
    )
    mine = metas[shard::num_shards]
    year, month = map(int, end_month.split("-"))
    first_month = month_value(year, month) - NUM_MONTHS + 1
    _fs(out_dir).makedirs(out_dir, exist_ok=True)
    print(
        f"{len(mine)} of {len(metas)} tiles, first month {month_iso(first_month)}",
        flush=True,
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        jobs = [(store, m, out_dir, first_month) for m in mine]
        for i, r in enumerate(pool.imap_unordered(_tile_partial_star, jobs), 1):
            print(f"[{i}/{len(mine)}] {json.dumps(r)}", flush=True)


# ---------------------------------------------------------------------------
# Store metadata.


def init_store(out_store: str, source_store: str, end_month: str) -> None:
    """Create the summary store's groups and arrays (no chunks)."""
    year, month = map(int, end_month.split("-"))
    first_month = month_value(year, month) - NUM_MONTHS + 1
    root = zarr.open_group(out_store, mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "description": (
                "OlmoEarth Land Change summary: per Web Mercator cell, the fraction of "
                "the cell's ground area that changed, per change category and end "
                "month, as 0 (none) or fraction = 10 ** ((value - 1) * log_decades / 254 "
                "- log_decades). A change pixel counts in "
                "its post category and in its pre category; pixels whose categories "
                "are both none are left out."
            ),
            "crs": "EPSG:3857",
            "source_store": source_store,
            "encoding": ENCODING,
            "log_decades": LOG_DECADES,
            "categories": CATEGORIES,
            "num_post_categories": NUM_POST,
            "first_month": month_iso(first_month),
            "num_months": NUM_MONTHS,
            "tile_cells": TILE_CELLS,
            "min_level": 0,
            "max_level": MAX_LEVEL,
            "pixel_area_m2": PIXEL_AREA_M2,
        }
    )
    for level in range(MAX_LEVEL + 1):
        n = cells_per_side(level)
        group = root.create_group(str(level))
        group.create_array(
            "values",
            shape=(NUM_CATEGORIES, NUM_MONTHS, n, n),
            chunks=(NUM_CATEGORIES, NUM_MONTHS, TILE_CELLS, TILE_CELLS),
            dtype="uint8",
            fill_value=0,
            serializer=BytesCodec(),
            compressors=[ZstdCodec(level=9)],
            dimension_names=["category", "month", "y", "x"],
        )
    zarr.consolidate_metadata(out_store)


# ---------------------------------------------------------------------------
# Steps 2 and 3: merge and top.


def write_level(
    root: zarr.Group, keys: np.ndarray, counts: np.ndarray, level: int
) -> int:
    """Write the chunks of one level from its counts; returns chunks written."""
    if keys.size == 0:
        return 0
    row, col, cat, month = decode_key(keys, level)
    values = quantize(counts, row, level)
    tile_id = (row // TILE_CELLS) * (1 << level) + col // TILE_CELLS
    order = np.argsort(tile_id, kind="stable")
    tile_id, row, col, cat, month, values = (
        a[order] for a in (tile_id, row, col, cat, month, values)
    )
    starts = np.flatnonzero(np.r_[True, tile_id[1:] != tile_id[:-1]])
    ends = np.r_[starts[1:], tile_id.size]
    arr = root[f"{level}/values"]
    for s, e in zip(starts, ends):
        ty, tx = divmod(int(tile_id[s]), 1 << level)
        chunk = np.zeros((NUM_CATEGORIES, NUM_MONTHS, TILE_CELLS, TILE_CELLS), np.uint8)
        chunk[cat[s:e], month[s:e], row[s:e] % TILE_CELLS, col[s:e] % TILE_CELLS] = (
            values[s:e]
        )
        y0, x0 = ty * TILE_CELLS, tx * TILE_CELLS
        arr[:, :, y0 : y0 + TILE_CELLS, x0 : x0 + TILE_CELLS] = chunk
    return len(starts)


def tile_lonlat_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a Web Mercator tile."""
    n = 1 << z

    def lat(yy: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def merge_tiles(partials_dir: str) -> list[tuple[int, int]]:
    """Zoom-MERGE_LEVEL map tiles that any partial's bounds touch."""
    out: set[tuple[int, int]] = set()
    for _, bounds in _partial_bounds(partials_dir):
        w, s, e, n = bounds
        r0, c0 = lonlat_to_cells(np.array([w]), np.array([n]), MERGE_LEVEL)
        r1, c1 = lonlat_to_cells(np.array([e]), np.array([s]), MERGE_LEVEL)
        for ty in range(int(r0[0]) // TILE_CELLS, int(r1[0]) // TILE_CELLS + 1):
            for tx in range(int(c0[0]) // TILE_CELLS, int(c1[0]) // TILE_CELLS + 1):
                out.add((ty, tx))
    return sorted(out)


_BOUNDS_CACHE: dict[str, list[tuple[str, np.ndarray]]] = {}


def _partial_bounds(partials_dir: str) -> list[tuple[str, np.ndarray]]:
    """(path, lon/lat bounds) of every partial, from an index built on first use."""
    if partials_dir in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[partials_dir]
    fs = _fs(partials_dir)
    index_path = posixpath.join(partials_dir, "_bounds.json")
    if fs.exists(index_path):
        with fs.open(index_path) as f:
            index = json.load(f)
    else:
        index = {}
        for p in sorted(fs.ls(_strip(partials_dir))):
            if p.endswith(".npz"):
                with fs.open(p, "rb") as f:
                    index[posixpath.basename(p)] = np.load(f)["bounds"].tolist()
        with fs.open(index_path, "w") as f:
            json.dump(index, f)
    out = [(posixpath.join(partials_dir, k), np.array(v)) for k, v in index.items()]
    _BOUNDS_CACHE[partials_dir] = out
    return out


def merge_one(
    root: zarr.Group, partials_dir: str, counts_dir: str, ty: int, tx: int
) -> dict:
    """Step 2 for one zoom-MERGE_LEVEL map tile."""
    t0 = time.time()
    shift = MAX_LEVEL - MERGE_LEVEL
    span = TILE_CELLS << shift  # MAX_LEVEL cells per side of the map tile
    w, s, e, n = tile_lonlat_bounds(MERGE_LEVEL, tx, ty)
    fs = _fs(partials_dir)
    keys_list, counts_list = [], []
    for path, b in _partial_bounds(partials_dir):
        if b[0] > e or b[2] < w or b[1] > n or b[3] < s:
            continue
        with fs.open(path, "rb") as f:
            d = np.load(f)
            keys, counts = d["keys"], d["counts"].astype(np.int64)
        row, col, _, _ = decode_key(keys, MAX_LEVEL)
        inside = (row // span == ty) & (col // span == tx)
        keys_list.append(keys[inside])
        counts_list.append(counts[inside])
    if not keys_list:
        return {"tile": [MERGE_LEVEL, tx, ty], "partials": 0}
    keys, counts = sum_by_key(np.concatenate(keys_list), np.concatenate(counts_list))
    chunks = {}
    level = MAX_LEVEL
    while True:
        chunks[level] = write_level(root, keys, counts, level)
        if level == MERGE_LEVEL:
            break
        keys, counts = coarsen(keys, counts, level, level - 1)
        level -= 1
    out_path = posixpath.join(counts_dir, f"{MERGE_LEVEL}_{tx}_{ty}.npz")
    with _fs(counts_dir).open(out_path, "wb") as f:
        np.savez(f, keys=keys, counts=counts)
    return {
        "tile": [MERGE_LEVEL, tx, ty],
        "partials": len(keys_list),
        "chunks": chunks,
        "seconds": round(time.time() - t0, 1),
    }


def run_merge(
    partials_dir: str, out_store: str, counts_dir: str, shard: int, num_shards: int
) -> None:
    """Write levels MERGE_LEVEL..MAX_LEVEL for this shard's tiles.

    Also saves the MERGE_LEVEL counts that run_top builds the coarser levels from.
    """
    tiles = merge_tiles(partials_dir)[shard::num_shards]
    _fs(counts_dir).makedirs(counts_dir, exist_ok=True)
    root = zarr.open_group(out_store, mode="r+")
    print(f"{len(tiles)} zoom-{MERGE_LEVEL} tiles", flush=True)
    for i, (ty, tx) in enumerate(tiles, 1):
        print(
            f"[{i}/{len(tiles)}] {json.dumps(merge_one(root, partials_dir, counts_dir, ty, tx))}",
            flush=True,
        )


def run_top(counts_dir: str, out_store: str) -> None:
    """Step 3: levels MERGE_LEVEL - 1 .. 0 from the saved MERGE_LEVEL counts."""
    fs = _fs(counts_dir)
    keys_list, counts_list = [], []
    for p in sorted(fs.ls(_strip(counts_dir))):
        if p.endswith(".npz"):
            with fs.open(p, "rb") as f:
                d = np.load(f)
                keys_list.append(d["keys"])
                counts_list.append(d["counts"])
    keys, counts = sum_by_key(np.concatenate(keys_list), np.concatenate(counts_list))
    root = zarr.open_group(out_store, mode="r+")
    level = MERGE_LEVEL
    while level > 0:
        keys, counts = coarsen(keys, counts, level, level - 1)
        level -= 1
        print(
            f"level {level}: {write_level(root, keys, counts, level)} chunks",
            flush=True,
        )


def main(argv: Iterable[str] | None = None) -> None:
    """Command-line entry point: partials, init, merge or top."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("partials")
    a.add_argument("--store", required=True, help="source lcc.zarr (gs://...)")
    a.add_argument("--out_dir", required=True)
    a.add_argument(
        "--end_month", required=True, help="YYYY-MM of the run's reference time"
    )
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--num_shards", type=int, default=1)
    a.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    i = sub.add_parser("init")
    i.add_argument("--out_store", required=True)
    i.add_argument("--source_store", required=True)
    i.add_argument("--end_month", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--partials_dir", required=True)
    m.add_argument("--out_store", required=True)
    m.add_argument("--counts_dir", required=True)
    m.add_argument("--shard", type=int, default=0)
    m.add_argument("--num_shards", type=int, default=1)
    t = sub.add_parser("top")
    t.add_argument("--counts_dir", required=True)
    t.add_argument("--out_store", required=True)
    args = p.parse_args(argv)
    if args.step == "partials":
        run_partials(
            args.store,
            args.out_dir,
            args.end_month,
            args.shard,
            args.num_shards,
            args.workers,
        )
    elif args.step == "init":
        init_store(args.out_store, args.source_store, args.end_month)
    elif args.step == "merge":
        run_merge(
            args.partials_dir,
            args.out_store,
            args.counts_dir,
            args.shard,
            args.num_shards,
        )
    elif args.step == "top":
        run_top(args.counts_dir, args.out_store)


if __name__ == "__main__":
    main()
