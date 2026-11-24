"""UTM zone wedges and grids shared by the global tiled prediction pipelines.

Each UTM zone number (1-60) is processed once in its northern CRS (EPSG:326NN), with a
continuous northing axis that goes negative south of the equator, so one grid covers
both hemispheres. Zones' projected extents overlap, so pipelines keep only tiles that
intersect the zone's own canonical 6-degree longitude wedge (80S to 84N).
"""

import functools
import math

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds

# Number of vertices to use when densifying the wedge meridian (north-south) and
# parallel (east-west) edges. The meridian edges are curved in projected coordinates
# so they need dense sampling.
NUM_MERIDIAN_VERTICES = 4096
NUM_PARALLEL_VERTICES = 256

# EPSG code base for the northern WGS84 UTM zones (326NN).
NORTH_EPSG_BASE = 32600
# Latitude range covered by UTM (80S to 84N). Each zone wedge spans this full range
# so one northern-CRS zone covers both hemispheres.
UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0


@functools.cache
def _get_zone_wedge(epsg_code: int, resolution: float) -> shapely.Polygon:
    """Cached implementation of get_zone_wedge keyed on the EPSG code."""
    utm_zone = CRS.from_epsg(epsg_code)
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

    The origin and shape are snapped outward to multiples of tile_size so tiles,
    windows and zarr shards stay aligned. Used by both job enumeration and store
    creation.

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
    utm_zone = CRS.from_epsg(NORTH_EPSG_BASE + zone_number)
    projection = Projection(utm_zone, resolution, -resolution)
    wedge = get_zone_wedge(utm_zone, resolution)
    min_x, min_y, max_x, max_y = wedge.bounds
    origin_x = math.floor(min_x / tile_size) * tile_size
    origin_y = math.floor(min_y / tile_size) * tile_size
    end_x = math.ceil(max_x / tile_size) * tile_size
    end_y = math.ceil(max_y / tile_size) * tile_size
    return projection, (origin_x, origin_y), (end_y - origin_y, end_x - origin_x)
