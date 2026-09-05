"""Reproject the per-zone PCA pyramid into a single web-mercator pyramid.

Why this exists
---------------
The PCA store is a visualization artifact and its only consumer is a web map, so the
embeddings' UTM layout costs that consumer dearly:

* a view spanning two UTM zones cannot be drawn without client-side reprojection;
* every UTM pyramid level keeps ``shard == one prediction window``, so downsampling
  happens inside a window and never across windows, and the object count for a given
  extent is identical at every level.

Here the grid is EPSG:3857 on the standard XYZ scheme, so a level *is* a web-map zoom
and a shard is a fixed 2048 px regardless of level. Ground coverage doubles per level
and object count falls 4x, which is what a pyramid is supposed to do.

The cost is that mercator samples uniformly in the projected plane, so holding ground
resolution costs sec^2(latitude) in pixels. Against a PCA layer that is a few percent
of the archive, that is cheap for removing an entire class of problem.

Ordering matters
----------------
PCA is applied per-pixel to unresampled embeddings, then the RGB is reprojected here.
The reverse would interpolate embeddings, which is not a meaningful operation. So this
stage reads the PCA store, never ``embeddings.zarr``, and the embeddings stay in UTM at
10 m as the analysis-grade path. One consequence to state plainly: after this stage the
visualization pixels no longer align 1:1 with embedding pixels.
"""

import json
import math
import re
import urllib.parse
import urllib.request
import xml.dom.minidom
from typing import Any

import fsspec
import numpy as np
import rasterio.warp
import zarr
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from upath import UPath
from zarr.codecs import ZstdCodec

from rslp.log_utils import get_logger

logger = get_logger(__name__)

# EPSG:3857 half-extent in metres. The projected world is square and spans this either
# side of the origin, which is what makes the XYZ scheme powers of two.
WEB_HALF_EXTENT = 20037508.342789244
WEB_EPSG = 3857

# One XYZ tile is 256 px, so level z is a 256 * 2**z square. Level 14 is 9.55 m/px at
# the equator, the natural home for 10 m source data.
XYZ_TILE = 256
DEFAULT_MAX_ZOOM = 14

# Deliberately constant across levels, unlike the UTM pyramid. This is the whole point:
# a fixed pixel shard means ground coverage doubles per level.
WEB_SHARD = 2048
WEB_CHUNK = 256

# The PCA arrays reserve 0 for nodata, so valid pixels occupy 1-255 and compositing
# several source zones into one output shard is just "keep what is non-zero".
WEB_NODATA = 0
WEB_BANDS = 3


def web_array_name(zoom: int) -> str:
    """Name of the RGB array at a zoom level.

    Named by zoom rather than by a downsample factor: levels here run in the web-map
    direction (0 is the whole world, higher is finer), the opposite of the UTM pyramid's
    overview numbering. Encoding the zoom in the name stops the two conventions being
    confused at a glance.

    Args:
        zoom: the XYZ zoom level.

    Returns:
        the array name.
    """
    return f"pca_rgb_z{zoom}"


def web_resolution(zoom: int) -> float:
    """Ground resolution in metres per pixel at the equator for a zoom level.

    Args:
        zoom: the XYZ zoom level.

    Returns:
        metres per pixel at the equator.
    """
    return 2.0 * WEB_HALF_EXTENT / (XYZ_TILE * 2**zoom)


def web_size(zoom: int) -> int:
    """Side length in pixels of the whole-world array at a zoom level.

    Args:
        zoom: the XYZ zoom level.

    Returns:
        the array's width and height in pixels.
    """
    return XYZ_TILE * 2**zoom


def web_transform(zoom: int) -> tuple[float, float, float, float, float, float]:
    """Affine transform mapping (col, row) to (x, y) in EPSG:3857 metres.

    Args:
        zoom: the XYZ zoom level.

    Returns:
        (x_res, 0, origin_x, 0, y_res, origin_y), y_res negative as the row axis runs
        south.
    """
    res = web_resolution(zoom)
    return (res, 0.0, -WEB_HALF_EXTENT, 0.0, -res, WEB_HALF_EXTENT)


def shard_bounds(
    zoom: int, shard_row: int, shard_col: int
) -> tuple[float, float, float, float]:
    """Projected bounds of one output shard.

    Args:
        zoom: the XYZ zoom level.
        shard_row: shard index down the array.
        shard_col: shard index across the array.

    Returns:
        (min_x, min_y, max_x, max_y) in EPSG:3857 metres.
    """
    res = web_resolution(zoom)
    min_x = -WEB_HALF_EXTENT + shard_col * WEB_SHARD * res
    max_x = min_x + WEB_SHARD * res
    max_y = WEB_HALF_EXTENT - shard_row * WEB_SHARD * res
    min_y = max_y - WEB_SHARD * res
    return (min_x, min_y, max_x, max_y)


def shards_per_side(zoom: int) -> int:
    """How many shards span the world at a zoom level.

    Low zooms are smaller than one shard, so the array is a single partial shard there.

    Args:
        zoom: the XYZ zoom level.

    Returns:
        the number of shards across the array, at least 1.
    """
    return max(1, web_size(zoom) // WEB_SHARD)


def build_multiscales(min_zoom: int, max_zoom: int) -> list[dict[str, Any]]:
    """Describe the pyramid for readers.

    Carries the resolution explicitly rather than only a level index. The UTM pyramid
    numbers levels coarse-upward and this one numbers them fine-upward, so a client that
    keys off metres never has to know which convention a store follows.

    Args:
        min_zoom: shallowest zoom written.
        max_zoom: deepest zoom written.

    Returns:
        a list of level descriptors, coarsest first.
    """
    return [
        {
            "array": web_array_name(z),
            "zoom": z,
            "resolution": web_resolution(z),
            "scheme": "xyz",
        }
        for z in range(min_zoom, max_zoom + 1)
    ]


def init_web_store(
    store_path: str,
    years: list[int],
    min_zoom: int = 0,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    zstd_level: int = 1,
    source_store_path: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Create the web-mercator pyramid skeleton.

    Every level is a whole-world array. They are enormous and almost entirely empty --
    level 14 is 4.2 million pixels square -- but a zarr array only materialises the
    shards that are written, exactly as the UTM stores already rely on. That keeps the
    grid globally addressable without paying for the parts nobody has computed, and a
    later run covering new ground writes into the same coordinates.

    Args:
        store_path: where to create the store.
        years: reference years, defining the time axis in order.
        min_zoom: shallowest zoom to create.
        max_zoom: deepest zoom to create.
        zstd_level: compression level for the RGB arrays.
        source_store_path: the UTM PCA store this is derived from, recorded in the
            provenance so the lineage is not lost.
        storage_options: fsspec options for remote stores.

    Raises:
        ValueError: if the zoom range is empty or inverted.
    """
    if max_zoom < min_zoom:
        raise ValueError(f"max_zoom {max_zoom} is below min_zoom {min_zoom}")

    root = zarr.open_group(
        _zarr_store(store_path, storage_options), mode="w", zarr_format=3
    )
    root.attrs.update(
        {
            "geoemb:spatial_layout": "web_mercator",
            "geoemb:multiscales": build_multiscales(min_zoom, max_zoom),
            "proj:code": f"EPSG:{WEB_EPSG}",
            "spatial:registration": "pixel",
            "geoemb:derived_from": source_store_path,
            "geoemb:note": (
                "False-color visualization derived from embeddings by PCA, then "
                "reprojected to web mercator for display. Three components capture "
                "only a minority of embedding variance; do not use these bands as "
                "features. Reprojection means these pixels no longer align with the "
                "embedding pixels -- use the UTM store for anything positional."
            ),
        }
    )

    for zoom in range(min_zoom, max_zoom + 1):
        size = web_size(zoom)
        # Low zooms are smaller than one shard, so clamp rather than emit a shard
        # larger than the array it lives in.
        shard = min(WEB_SHARD, size)
        chunk = min(WEB_CHUNK, shard)
        root.create_array(
            web_array_name(zoom),
            shape=(len(years), WEB_BANDS, size, size),
            chunks=(1, WEB_BANDS, chunk, chunk),
            shards=(1, WEB_BANDS, shard, shard),
            dtype="uint8",
            fill_value=WEB_NODATA,
            compressors=[ZstdCodec(level=zstd_level)],
            dimension_names=("time", "band", "y", "x"),
        )
        root[web_array_name(zoom)].attrs.update(
            {
                "spatial:transform": list(web_transform(zoom)),
                "spatial:shape": [size, size],
                "geoemb:zoom": zoom,
                "geoemb:gsd": web_resolution(zoom),
            }
        )

    time_arr = root.create_array(
        "time", shape=(len(years),), dtype="int32", dimension_names=("time",)
    )
    time_arr[:] = np.array(years, dtype=np.int32)
    zarr.consolidate_metadata(root.store)
    logger.info(
        "created web store %s with zooms %d..%d for years %s",
        store_path,
        min_zoom,
        max_zoom,
        years,
    )


def _zarr_store(path: str, storage_options: dict[str, Any] | None) -> Any:
    """Open a zarr store for a local path or a remote URL.

    Args:
        path: the store path or URL.
        storage_options: fsspec options for remote stores.

    Returns:
        something zarr.open_group accepts.
    """
    if "://" not in path:
        return path
    return fsspec.get_mapper(path, **(storage_options or {}))


# The UTM grid every zone shares. get_zone_grid snaps the same canonical wedge in each
# zone's own northern CRS, and transverse mercator has identical parameters per zone, so
# origin and shape come out identical for all 60. Verified against nine zones and two
# live stores.
UTM_ORIGIN_X = 0.0
UTM_ORIGIN_Y = 9502720.0
UTM_RES = 10.0
UTM_NORTH_EPSG_BASE = 32600


def zones_for_bounds(min_lon: float, max_lon: float) -> list[int]:
    """UTM zone numbers a longitude span touches.

    Args:
        min_lon: western edge in degrees.
        max_lon: eastern edge in degrees.

    Returns:
        the zone numbers, ascending.
    """
    lo = max(1, min(60, int((min_lon + 180.0) // 6.0) + 1))
    hi = max(1, min(60, int((max_lon + 180.0) // 6.0) + 1))
    return list(range(min(lo, hi), max(lo, hi) + 1))


def reproject_shard(
    source_group: Any,
    dest_array: Any,
    zoom: int,
    time_index: int,
    shard_row: int,
    shard_col: int,
    zone_numbers: list[int],
) -> int:
    """Warp every contributing UTM zone into one web-mercator shard.

    A shard can straddle a zone boundary, which is the case the UTM layout cannot serve
    at all. Each zone is warped in turn and composited where it has data; the wedge
    filter means a ground point is computed in exactly one zone, so contributions never
    disagree and compositing is just "keep what is non-zero".

    Args:
        source_group: opened zarr group of the UTM PCA store.
        dest_array: the destination array for this zoom.
        zoom: the XYZ zoom level.
        time_index: index on the time axis.
        shard_row: shard index down the destination array.
        shard_col: shard index across the destination array.
        zone_numbers: candidate zone numbers present in the source store.

    Returns:
        the number of valid (non-nodata) pixels written.
    """
    min_x, min_y, max_x, max_y = shard_bounds(zoom, shard_row, shard_col)
    res = web_resolution(zoom)
    dst = np.zeros((WEB_BANDS, WEB_SHARD, WEB_SHARD), dtype=np.uint8)
    dst_transform = Affine(res, 0.0, min_x, 0.0, -res, max_y)

    to_wgs = Transformer.from_crs(f"EPSG:{WEB_EPSG}", "EPSG:4326", always_xy=True)
    lons, _ = to_wgs.transform(
        [min_x, max_x, min_x, max_x], [min_y, min_y, max_y, max_y]
    )
    candidates = zones_for_bounds(min(lons), max(lons))

    for zone in candidates:
        name = f"utm{zone:02d}"
        if zone not in zone_numbers or name not in source_group:
            continue
        src_epsg = UTM_NORTH_EPSG_BASE + zone
        to_utm = Transformer.from_crs(
            f"EPSG:{WEB_EPSG}", f"EPSG:{src_epsg}", always_xy=True
        )
        xs, ys = to_utm.transform(
            [min_x, max_x, min_x, max_x], [min_y, min_y, max_y, max_y]
        )
        # Pad by a pixel so the warp has neighbours at the edges to interpolate from.
        col0 = math.floor((min(xs) - UTM_ORIGIN_X) / UTM_RES) - 1
        col1 = math.ceil((max(xs) - UTM_ORIGIN_X) / UTM_RES) + 1
        row0 = math.floor((UTM_ORIGIN_Y - max(ys)) / UTM_RES) - 1
        row1 = math.ceil((UTM_ORIGIN_Y - min(ys)) / UTM_RES) + 1

        src = source_group[name]["pca_rgb"]
        _, _, height, width = src.shape
        col0, row0 = max(0, col0), max(0, row0)
        col1, row1 = min(width, col1), min(height, row1)
        if col1 <= col0 or row1 <= row0:
            continue

        block = np.asarray(src[time_index, :, row0:row1, col0:col1])
        if not block.any():
            continue

        src_transform = Affine(
            UTM_RES,
            0.0,
            UTM_ORIGIN_X + col0 * UTM_RES,
            0.0,
            -UTM_RES,
            UTM_ORIGIN_Y - row0 * UTM_RES,
        )
        warped = np.zeros_like(dst)
        rasterio.warp.reproject(
            source=block,
            destination=warped,
            src_transform=src_transform,
            src_crs=f"EPSG:{src_epsg}",
            src_nodata=WEB_NODATA,
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{WEB_EPSG}",
            dst_nodata=WEB_NODATA,
            # Nearest, not bilinear. The destination grid never lines up with the UTM
            # source, so bilinear blends neighbours for *every* pixel, not just where it
            # is resampling to a different scale: measured against the source's own
            # edge energy it keeps 92% at the equator and 93% at Seattle, and the loss
            # is all in the high frequencies that make an edge look like an edge.
            #
            # Latitude makes it worse rather than causing it. A z14 pixel is 9.55 m at
            # the equator but 6.44 m at 47.6N, so Seattle stores a grid 1.55x finer than
            # the 10 m data behind it, and every feature boundary becomes a ramp three
            # pixels wide instead of a step. That is the blurring people notice.
            #
            # Nearest measures 100% of source edge energy at the equator and 103% at
            # Seattle (above 100% because a duplicated pixel can sharpen a step). It also
            # invents no values, so the visible pixel blocks state the true 10 m
            # resolution rather than implying a finer one. Coarser levels are unaffected:
            # they come from downsample_shard's 2x2 mean, which is the right operator for
            # reducing and is not touched by this.
            resampling=Resampling.nearest,
        )
        fill = (dst == WEB_NODATA).all(axis=0) & (warped != WEB_NODATA).any(axis=0)
        dst[:, fill] = warped[:, fill]

    valid = int((dst != WEB_NODATA).any(axis=0).sum())
    if valid:
        r0, c0 = shard_row * WEB_SHARD, shard_col * WEB_SHARD
        dest_array[time_index, :, r0 : r0 + WEB_SHARD, c0 : c0 + WEB_SHARD] = dst
    return valid


def downsample_shard(
    dest_group: Any,
    zoom: int,
    time_index: int,
    shard_row: int,
    shard_col: int,
) -> int:
    """Build one shard at `zoom` from the four shards above it at `zoom + 1`.

    Cascading rather than re-warping from UTM at every level: each level is a quarter of
    the pixels of the one below, so the whole pyramid above the base costs about a third
    of the base again, and the expensive projection maths is done exactly once.

    Averaging ignores nodata so a coastline does not bleed toward black as it coarsens;
    a 2x2 block with any valid pixel yields the mean of the valid ones.

    Args:
        dest_group: the opened web store.
        zoom: the zoom level being written.
        time_index: index on the time axis.
        shard_row: shard index down the destination array.
        shard_col: shard index across the destination array.

    Returns:
        the number of valid pixels written.
    """
    fine = dest_group[web_array_name(zoom + 1)]
    size = web_size(zoom)
    shard = min(WEB_SHARD, size)
    r0, c0 = shard_row * shard, shard_col * shard
    # The corresponding region one level down is twice the size in each direction.
    src = np.asarray(
        fine[time_index, :, r0 * 2 : (r0 + shard) * 2, c0 * 2 : (c0 + shard) * 2]
    )
    if not src.any():
        return 0

    bands, h, w = src.shape
    blocks = src.reshape(bands, h // 2, 2, w // 2, 2).astype(np.uint16)
    valid = (src != WEB_NODATA).any(axis=0).reshape(h // 2, 2, w // 2, 2)
    count = valid.sum(axis=(1, 3))
    total = (blocks * valid[None]).sum(axis=(2, 4))
    out = np.zeros((bands, h // 2, w // 2), dtype=np.uint8)
    ok = count > 0
    for b in range(bands):
        out[b][ok] = (total[b][ok] // count[ok]).astype(np.uint8)

    n = int(ok.sum())
    if n:
        dest_group[web_array_name(zoom)][
            time_index, :, r0 : r0 + shard, c0 : c0 + shard
        ] = out
    return n


def source_shard_positions(store_url: str, zone: str) -> set[tuple[int, int]]:
    """Written shard positions of a zone's UTM pca_rgb array, from the object keys.

    The keys under the chunk root *are* the footprint -- one object per written shard --
    so listing them is exact and needs no read of the data. The alternative, probing the
    destination grid, is hopeless: level 14 has 2048x2048 shard slots, and almost all of
    them are ocean.

    Args:
        store_url: the UTM PCA store, as either a gs:// or an
            https://storage.googleapis.com/ URL.
        zone: zone group name, e.g. "utm10".

    Returns:
        (y_shard, x_shard) pairs, in units of the UTM array's 2048 px shards.
    """
    url = store_url.rstrip("/")
    m = re.match(r"https://storage\.googleapis\.com/([^/]+)/(.+)", url) or re.match(
        r"gs://([^/]+)/(.+)", url
    )
    if not m:
        raise ValueError(
            f"cannot derive a bucket listing from {store_url}; expected a gs:// or "
            "https://storage.googleapis.com/ URL"
        )
    bucket, root = m.group(1), m.group(2)
    prefix = f"{root}/{zone}/pca_rgb/c/"

    out: set[tuple[int, int]] = set()
    token = None
    for _ in range(200):
        url = (
            f"https://storage.googleapis.com/{bucket}?list-type=2"
            f"&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
            + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        )
        with urllib.request.urlopen(url, timeout=90) as resp:
            doc = xml.dom.minidom.parseString(resp.read())
        for node in doc.getElementsByTagName("Key"):
            if not node.firstChild:
                continue
            parts = node.firstChild.data[len(prefix) :].split("/")
            # <time>/<band>/<y>/<x>; anything else is not a chunk key.
            if len(parts) == 4 and all(p.isdigit() for p in parts):
                out.add((int(parts[2]), int(parts[3])))
        trunc = doc.getElementsByTagName("IsTruncated")
        if not (
            trunc and trunc[0].firstChild and trunc[0].firstChild.data.strip() == "true"
        ):
            return out
        nxt = doc.getElementsByTagName("NextContinuationToken")
        token = nxt[0].firstChild.data if nxt and nxt[0].firstChild else None
        if not token:
            return out
    return out


def web_shards_for_source(
    positions: set[tuple[int, int]], zone_number: int, zoom: int
) -> set[tuple[int, int]]:
    """Output shards at `zoom` that the given UTM source shards fall into.

    Args:
        positions: (y_shard, x_shard) pairs in the UTM array's 2048 px shard grid.
        zone_number: the UTM zone number the positions belong to.
        zoom: the destination XYZ zoom level.

    Returns:
        (shard_row, shard_col) pairs in the destination grid.
    """
    to_web = Transformer.from_crs(
        f"EPSG:{UTM_NORTH_EPSG_BASE + zone_number}", f"EPSG:{WEB_EPSG}", always_xy=True
    )
    res = web_resolution(zoom)
    size = web_size(zoom)
    shard = min(WEB_SHARD, size)
    span = 2048 * UTM_RES  # one UTM shard on the ground, metres
    out: set[tuple[int, int]] = set()
    for sy, sx in positions:
        x0 = UTM_ORIGIN_X + sx * span
        y1 = UTM_ORIGIN_Y - sy * span
        # All four corners: the shard is a rectangle in UTM but a curved quad in 3857.
        xs, ys = to_web.transform(
            [x0, x0 + span, x0, x0 + span], [y1 - span, y1 - span, y1, y1]
        )
        c0 = int((min(xs) + WEB_HALF_EXTENT) / res) // shard
        c1 = int((max(xs) + WEB_HALF_EXTENT) / res) // shard
        r0 = int((WEB_HALF_EXTENT - max(ys)) / res) // shard
        r1 = int((WEB_HALF_EXTENT - min(ys)) / res) // shard
        n = max(1, size // shard)
        for r in range(max(0, r0), min(n - 1, r1) + 1):
            for c in range(max(0, c0), min(n - 1, c1) + 1):
                out.add((r, c))
    return out


def parent_shards(shards: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Shards one zoom coarser that cover the given ones.

    Args:
        shards: (row, col) pairs at some zoom.

    Returns:
        (row, col) pairs at zoom - 1.
    """
    return {(r // 2, c // 2) for r, c in shards}


def render_web_pca_pipeline_all(
    source_store_path: str,
    web_store_path: str,
    years: list[int],
    zone_numbers: list[int],
    min_zoom: int = 8,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    source_url: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Build the whole web-mercator pyramid for one run.

    The base zoom is warped from the UTM store; every coarser zoom is cascaded from the
    one below. Work is enumerated from the source's own object keys rather than probed,
    so the cost tracks the data that exists rather than the size of the global grid.

    Args:
        source_store_path: the UTM PCA store to read, as a zarr-openable path.
        web_store_path: where to write the web store.
        years: reference years in store order.
        zone_numbers: UTM zone numbers present in the source.
        min_zoom: shallowest zoom to build.
        max_zoom: deepest zoom to build, warped directly from UTM.
        source_url: https base of the source, for listing its keys. Defaults to
            source_store_path when that is already an https URL.
        storage_options: fsspec options for the destination.
    """
    listing_url = source_url or source_store_path
    src = zarr.open_group(_zarr_store(source_store_path, None), mode="r")

    init_web_store(
        web_store_path,
        years=years,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        source_store_path=source_store_path,
        storage_options=storage_options,
    )
    dest = zarr.open_group(_zarr_store(web_store_path, storage_options), mode="a")

    base: set[tuple[int, int]] = set()
    for zone in zone_numbers:
        positions = source_shard_positions(listing_url, f"utm{zone:02d}")
        base |= web_shards_for_source(positions, zone, max_zoom)
        logger.info("zone %d: %d source shard(s)", zone, len(positions))
    logger.info("z%d: %d output shard(s)", max_zoom, len(base))

    for t, year in enumerate(years):
        written = 0
        for row, col in sorted(base):
            written += (
                1
                if reproject_shard(
                    src,
                    dest[web_array_name(max_zoom)],
                    max_zoom,
                    t,
                    row,
                    col,
                    zone_numbers,
                )
                else 0
            )
        logger.info(
            "year %d z%d: %d/%d shard(s) with data", year, max_zoom, written, len(base)
        )

        shards = base
        for zoom in range(max_zoom - 1, min_zoom - 1, -1):
            shards = parent_shards(shards)
            got = 0
            for row, col in sorted(shards):
                got += 1 if downsample_shard(dest, zoom, t, row, col) else 0
            logger.info(
                "year %d z%d: %d/%d shard(s) with data", year, zoom, got, len(shards)
            )


def web_marker_name(time_index: int, row: int, col: int) -> str:
    """Filename of one shard's marker within its zoom directory.

    The single source of truth for that name. The writer and the enumerator must agree
    exactly: if they drift, enumeration never sees its own completed work and the stage
    rebuilds everything on every cycle, forever, with no error to notice.

    Args:
        time_index: index on the time axis.
        row: shard row.
        col: shard column.

    Returns:
        the marker filename.
    """
    return f"{time_index}_{row}_{col}.json"


def web_marker_fname(
    completed_path: str, zoom: int, time_index: int, row: int, col: int
) -> UPath:
    """Marker path for one finished output shard.

    Keyed by every coordinate that identifies the unit of work, so a resume skips
    exactly what is done and nothing else.

    Args:
        completed_path: the marker directory.
        zoom: the XYZ zoom level.
        time_index: index on the time axis.
        row: shard row.
        col: shard column.

    Returns:
        the marker path.
    """
    return UPath(completed_path) / f"z{zoom}" / web_marker_name(time_index, row, col)


def render_web_pca_pipeline(
    source_store_path: str,
    web_store_path: str,
    completed_path: str,
    zoom: int,
    time_index: int,
    shard_row: int,
    shard_col: int,
    zone_numbers: list[int],
    base_zoom: int = DEFAULT_MAX_ZOOM,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Build one output shard: the unit of work a queue worker claims.

    At the base zoom this warps from the UTM store; above it, it downsamples the level
    below, which must already be complete. That dependency is why the stage runs one
    zoom at a time rather than as a single flat pass.

    Args:
        source_store_path: the UTM PCA store.
        web_store_path: the web store to write.
        completed_path: marker directory.
        zoom: the XYZ zoom level to write.
        time_index: index on the time axis.
        shard_row: shard row in the destination.
        shard_col: shard column in the destination.
        zone_numbers: UTM zones present in the source.
        base_zoom: the deepest zoom, warped directly from UTM.
        storage_options: fsspec options for the destination.
    """
    marker = web_marker_fname(completed_path, zoom, time_index, shard_row, shard_col)
    if marker.exists():
        logger.info("marker %s already exists", marker)
        return

    dest = zarr.open_group(_zarr_store(web_store_path, storage_options), mode="a")
    if zoom == base_zoom:
        src = zarr.open_group(_zarr_store(source_store_path, None), mode="r")
        written = reproject_shard(
            src,
            dest[web_array_name(zoom)],
            zoom,
            time_index,
            shard_row,
            shard_col,
            zone_numbers,
        )
    else:
        written = downsample_shard(dest, zoom, time_index, shard_row, shard_col)

    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w") as f:
        json.dump(
            {
                "zoom": zoom,
                "time_index": time_index,
                "shard": [shard_row, shard_col],
                "valid_pixels": written,
            },
            f,
        )
    logger.info(
        "z%d t%d shard (%d,%d): %d valid px",
        zoom,
        time_index,
        shard_row,
        shard_col,
        written,
    )


def get_web_jobs(
    source_store_path: str,
    web_store_path: str,
    completed_path: str,
    zoom: int,
    years: list[int],
    zone_numbers: list[int],
    base_zoom: int = DEFAULT_MAX_ZOOM,
    source_url: str | None = None,
) -> list[list[str]]:
    """Enumerate the outstanding shards for one zoom level.

    Work is derived from the source's object keys rather than probed: the destination
    grid has millions of shard slots and almost all are ocean, so probing it is
    hopeless while listing what exists is exact.

    Args:
        source_store_path: the UTM PCA store.
        web_store_path: the web store.
        completed_path: marker directory.
        zoom: the zoom level to enumerate.
        years: reference years in store order.
        zone_numbers: UTM zones present in the source.
        base_zoom: the deepest zoom.
        source_url: https base of the source, for listing.

    Returns:
        argument lists for the queue.
    """
    listing = source_url or source_store_path
    shards: set[tuple[int, int]] = set()
    for zone in zone_numbers:
        positions = source_shard_positions(listing, f"utm{zone:02d}")
        shards |= web_shards_for_source(positions, zone, base_zoom)
    for _ in range(base_zoom - zoom):
        shards = parent_shards(shards)

    done: set[str] = set()
    level_dir = UPath(completed_path) / f"z{zoom}"
    if level_dir.exists():
        done = {fname.name for fname in level_dir.iterdir()}

    jobs: list[list[str]] = []
    total = 0
    for time_index in range(len(years)):
        for row, col in sorted(shards):
            total += 1
            if web_marker_name(time_index, row, col) in done:
                continue
            jobs.append(
                [
                    "--source_store_path",
                    source_store_path,
                    "--web_store_path",
                    web_store_path,
                    "--completed_path",
                    completed_path,
                    "--zoom",
                    str(zoom),
                    "--time_index",
                    str(time_index),
                    "--shard_row",
                    str(row),
                    "--shard_col",
                    str(col),
                    "--zone_numbers",
                    json.dumps(zone_numbers),
                    "--base_zoom",
                    str(base_zoom),
                ]
            )
    logger.info("z%d: %d shard-year(s), %d still to build", zoom, total, len(jobs))
    return jobs
