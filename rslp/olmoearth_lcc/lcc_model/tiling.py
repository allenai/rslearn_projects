"""UTM zone tiling for scaled LCC prediction.

The world is processed one UTM zone number (1-60) at a time. Each zone is handled in
its northern CRS (EPSG:326NN) with a continuous northing axis that goes negative
south of the equator, so a single zone covers both hemispheres (matching the geoemb
``utm_zones`` layout used by the GeoZarr store). Enumerating tiles over a zone's
projected bounding box covers a lot of area that belongs to neighboring zones, so
tiles and windows are only kept if they intersect their own zone's canonical
6-degree longitude wedge (80S to 84N).
"""

import functools
import math
from collections.abc import Generator

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds

# Number of vertices to use when densifying the wedge meridian (north-south) and
# parallel (east-west) edges. The meridian edges are curved in projected coordinates
# so they need dense sampling.
NUM_MERIDIAN_VERTICES = 4096
NUM_PARALLEL_VERTICES = 256

# EPSG code bases for WGS84 UTM zones (326NN north, 327NN south).
NORTH_EPSG_BASE = 32600
SOUTH_EPSG_BASE = 32700
# Latitude range covered by UTM (80S to 84N). Each zone wedge spans this full range
# so one northern-CRS zone covers both hemispheres.
UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0

# Projection resolution in m/pixel.
RESOLUTION = 10
# Per-window size. The tile size must be a multiple of this.
PATCH_SIZE = 2048
# Tile size used by the job writer and to snap the per-zone grid (and so the GeoZarr
# array extents). The prediction pipeline supports any multiple of PATCH_SIZE.
TILE_SIZE = 32768


def zone_number_from_epsg(epsg_code: int) -> int:
    """Get the UTM zone number (1-60) from a 326NN or 327NN EPSG code."""
    if NORTH_EPSG_BASE < epsg_code <= NORTH_EPSG_BASE + 60:
        return epsg_code - NORTH_EPSG_BASE
    if SOUTH_EPSG_BASE < epsg_code <= SOUTH_EPSG_BASE + 60:
        return epsg_code - SOUTH_EPSG_BASE
    raise ValueError(f"EPSG:{epsg_code} is not a WGS84 UTM zone")


def get_zone_projection(zone_number: int, resolution: float) -> Projection:
    """Get the northern UTM projection (EPSG:326NN) for a zone number."""
    return Projection(
        CRS.from_epsg(NORTH_EPSG_BASE + zone_number), resolution, -resolution
    )


@functools.cache
def _get_zone_wedge(epsg_code: int, resolution: float) -> shapely.Polygon:
    """Cached implementation of get_zone_wedge keyed on the EPSG code."""
    utm_zone = CRS.from_epsg(epsg_code)
    # Use the zone's canonical 6-degree longitude band but span the full UTM latitude
    # range so a single wedge covers both hemispheres.
    min_lon, _, max_lon, _ = get_wgs84_bounds(utm_zone)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    lons = np.concatenate(
        [
            np.linspace(min_lon, max_lon, NUM_PARALLEL_VERTICES),
            np.full(NUM_MERIDIAN_VERTICES, max_lon),
            np.linspace(max_lon, min_lon, NUM_PARALLEL_VERTICES),
            np.full(NUM_MERIDIAN_VERTICES, min_lon),
        ]
    )
    lats = np.concatenate(
        [
            np.full(NUM_PARALLEL_VERTICES, UTM_MIN_LAT),
            np.linspace(UTM_MIN_LAT, UTM_MAX_LAT, NUM_MERIDIAN_VERTICES),
            np.full(NUM_PARALLEL_VERTICES, UTM_MAX_LAT),
            np.linspace(UTM_MAX_LAT, UTM_MIN_LAT, NUM_MERIDIAN_VERTICES),
        ]
    )
    xs, ys = transformer.transform(lons, lats)
    # Convert projected meters to pixel coordinates. The pixel y axis points south
    # (y_resolution is -resolution).
    wedge = shapely.Polygon(
        np.stack([np.array(xs) / resolution, -np.array(ys) / resolution], axis=1)
    )
    shapely.prepare(wedge)
    return wedge


def get_zone_wedge(utm_zone: CRS, resolution: float) -> shapely.Polygon:
    """Get the canonical wedge of a UTM zone as a polygon in pixel coordinates.

    Args:
        utm_zone: the CRS which must correspond to a UTM EPSG.
        resolution: the projection resolution in m/pixel (the y resolution is assumed
            to be its negation).

    Returns:
        the wedge polygon in pixel coordinates, prepared for fast intersection tests.
    """
    return _get_zone_wedge(utm_zone.to_epsg(), resolution)


def get_zone_grid(
    zone_number: int,
    resolution: float,
    tile_size: int,
) -> tuple[Projection, tuple[int, int], tuple[int, int]]:
    """Get the northern-CRS pixel grid for a UTM zone number spanning both hemispheres.

    The array origin and shape are snapped outward to a multiple of tile_size so
    tiles, windows, and zarr shards stay aligned. This is the single source of truth
    for the per-zone grid, used by both the job enumeration and the zarr store init.

    Args:
        zone_number: the UTM zone number (1-60).
        resolution: the projection resolution in m/pixel.
        tile_size: the tile size in pixels; the origin and shape are snapped outward
            to a multiple of this.

    Returns:
        a tuple (projection, origin_px, shape_px) where projection is the northern
        UTM projection, origin_px is the (x, y) pixel coordinate of the array's
        top-left corner, and shape_px is the (height, width) of the array in pixels.
    """
    projection = get_zone_projection(zone_number, resolution)
    wedge = get_zone_wedge(projection.crs, resolution)
    min_x, min_y, max_x, max_y = wedge.bounds
    origin_x = math.floor(min_x / tile_size) * tile_size
    origin_y = math.floor(min_y / tile_size) * tile_size
    end_x = math.ceil(max_x / tile_size) * tile_size
    end_y = math.ceil(max_y / tile_size) * tile_size
    return projection, (origin_x, origin_y), (end_y - origin_y, end_x - origin_x)


def enumerate_zone_tiles(
    zone_number: int, resolution: float, tile_size: int
) -> Generator[tuple[int, int], None, None]:
    """List the (column, row) of tiles in a zone's grid that intersect its wedge.

    Args:
        zone_number: the UTM zone number (1-60).
        resolution: the projection resolution in m/pixel.
        tile_size: the tile size in pixels.

    Returns:
        generator of (column, row) of the tiles, in the zone's northern projection.
    """
    projection, origin_px, shape_px = get_zone_grid(zone_number, resolution, tile_size)
    wedge = get_zone_wedge(projection.crs, resolution)
    origin_x, origin_y = origin_px
    height, width = shape_px
    for col in range(origin_x // tile_size, (origin_x + width) // tile_size):
        for row in range(origin_y // tile_size, (origin_y + height) // tile_size):
            bounds = (
                col * tile_size,
                row * tile_size,
                (col + 1) * tile_size,
                (row + 1) * tile_size,
            )
            if wedge.intersects(shapely.box(*bounds)):
                yield (col, row)


def window_in_zone_wedge(projection: Projection, bounds: PixelBounds) -> bool:
    """Check whether pixel bounds intersect the wedge of the projection's UTM zone.

    Works for both 326NN and 327NN projections. Projections that are not UTM, or
    whose pixels are not square with a south-pointing y axis, are always kept.

    Args:
        projection: the projection of the bounds.
        bounds: the pixel bounds to check.

    Returns:
        whether the bounds intersect the zone wedge.
    """
    epsg_code = projection.crs.to_epsg()
    if epsg_code is None:
        return True
    try:
        zone_number_from_epsg(epsg_code)
    except ValueError:
        return True
    if projection.y_resolution != -projection.x_resolution:
        return True
    wedge = get_zone_wedge(projection.crs, projection.x_resolution)
    return wedge.intersects(shapely.box(*bounds))
