"""Tiling helpers for global embedding inference.

The world is processed one UTM zone number (1-60) at a time. Each zone is handled in
its northern CRS (EPSG:326NN) with a continuous northing axis that goes negative
south of the equator, so a single zone covers both hemispheres (matching the geoemb
utm_zones layout). Each zone is divided into TILE_SIZE tiles (see write_jobs.py), and
each tile into PATCH_SIZE crops (see predict_pipeline.py). Tiles are enumerated from
the zone's projected extent, so without filtering there is substantial duplication
across zones (each point on land would be computed in ~2.2 zones on average). This
module provides the filters that reduce the duplication to ~1.04x:

- Zone wedge: each UTM zone has a canonical 6-degree longitude wedge spanning the
  full UTM latitude range (80S to 84N). We only keep tiles and crops that intersect
  their own zone's wedge, so areas covered by multiple zones' projected extents are
  only processed in the zone that owns them.
- Ocean: we skip crops whose four corners are all ocean according to the
  global_land_mask package.
"""

import functools
import math

import numpy as np
import shapely
from global_land_mask import globe
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds

# Number of vertices to use when densifying the wedge meridian (north-south) and
# parallel (east-west) edges. The meridian edges are curved in projected coordinates
# so they need dense sampling.
NUM_MERIDIAN_VERTICES = 4096
NUM_PARALLEL_VERTICES = 256

# EPSG code base for northern-hemisphere WGS84 UTM zones (326NN).
NORTH_EPSG_BASE = 32600
# Latitude range covered by UTM (80S to 84N). Each zone wedge spans this full range
# in its northern CRS so one zone group covers both hemispheres.
UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0


@functools.cache
def _get_zone_wedge(epsg_code: int, resolution: float) -> shapely.Polygon:
    """Cached implementation of get_zone_wedge keyed on the EPSG code."""
    utm_zone = CRS.from_epsg(epsg_code)
    # Use the zone's canonical 6-degree longitude band but span the full UTM latitude
    # range so a single northern-CRS wedge covers both hemispheres (see module docs).
    min_lon, _, max_lon, _ = get_wgs84_bounds(utm_zone)
    min_lat, max_lat = UTM_MIN_LAT, UTM_MAX_LAT
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
            np.full(NUM_PARALLEL_VERTICES, min_lat),
            np.linspace(min_lat, max_lat, NUM_MERIDIAN_VERTICES),
            np.full(NUM_PARALLEL_VERTICES, max_lat),
            np.linspace(max_lat, min_lat, NUM_MERIDIAN_VERTICES),
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

    The zone is processed once, in its northern CRS (EPSG:326NN), with a continuous
    northing axis that goes negative south of the equator so a single grid covers
    both hemispheres. The array origin and shape are snapped outward to a multiple of
    tile_size (itself a multiple of PATCH_SIZE) so tiles, crops, and zarr shards stay
    aligned. This is the single source of truth for the per-zone grid, used by both
    the job enumeration (write_jobs.py) and the zarr store init (zarr_store.py).

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
    origin_px = (origin_x, origin_y)
    shape_px = (end_y - origin_y, end_x - origin_x)
    return projection, origin_px, shape_px


@functools.cache
def _get_to_wgs84_transformer(epsg_code: int) -> Transformer:
    """Get a cached transformer from the given EPSG code to WGS84."""
    return Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)


def bounds_intersect_wedge(wedge: shapely.Polygon, bounds: PixelBounds) -> bool:
    """Check whether the given pixel bounds intersect the zone wedge.

    Args:
        wedge: the zone wedge from get_zone_wedge (in matching pixel coordinates).
        bounds: the pixel bounds to check.

    Returns:
        whether the bounds intersect the wedge.
    """
    return wedge.intersects(shapely.box(*bounds))


def list_kept_crops(
    projection: Projection,
    bounds: PixelBounds,
    crop_size: int,
    wedge: shapely.Polygon | None = None,
) -> list[PixelBounds]:
    """List the crops within the given bounds that should be processed.

    The bounds are divided into a grid of crop_size x crop_size crops. A crop is kept
    if it intersects the zone's canonical wedge, and at least one of its four corners
    is land according to global_land_mask.

    Args:
        projection: the UTM projection (with negative y resolution).
        bounds: the pixel bounds to divide into crops. Each value must be a multiple
            of crop_size.
        crop_size: the size of each crop in pixels.
        wedge: the zone wedge from get_zone_wedge; computed if not provided.

    Returns:
        list of pixel bounds of the crops to process.
    """
    for value in bounds:
        assert value % crop_size == 0
    if wedge is None:
        wedge = get_zone_wedge(projection.crs, projection.x_resolution)

    cols = list(range(bounds[0] // crop_size, bounds[2] // crop_size))
    rows = list(range(bounds[1] // crop_size, bounds[3] // crop_size))

    # Compute the land mask at the lattice of crop corners, in one vectorized pass.
    corner_xs_px = np.array([col * crop_size for col in cols] + [bounds[2]])
    corner_ys_px = np.array([row * crop_size for row in rows] + [bounds[3]])
    xs_m, ys_m = np.meshgrid(
        corner_xs_px * projection.x_resolution,
        corner_ys_px * projection.y_resolution,
    )
    transformer = _get_to_wgs84_transformer(projection.crs.to_epsg())
    lons, lats = transformer.transform(xs_m, ys_m)
    # Wrap/clip into the domain expected by global_land_mask.
    lons = ((np.array(lons) + 180) % 360) - 180
    lats = np.clip(np.array(lats), -89.99, 89.99)
    # is_land is indexed [row, col] following the meshgrid above.
    is_land = globe.is_land(lats, lons)

    kept: list[PixelBounds] = []
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            # Skip if all four corners of the crop are ocean.
            if not is_land[row_idx : row_idx + 2, col_idx : col_idx + 2].any():
                continue
            crop_bounds = (
                col * crop_size,
                row * crop_size,
                (col + 1) * crop_size,
                (row + 1) * crop_size,
            )
            # Skip if the crop does not intersect the canonical zone wedge (it will be
            # processed by the neighboring UTM zone instead).
            if not bounds_intersect_wedge(wedge, crop_bounds):
                continue
            kept.append(crop_bounds)
    return kept
