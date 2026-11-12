"""Tiling helpers for global embedding inference.

Each UTM zone (1-60) is processed once in its northern CRS (EPSG:326NN), with northing
going negative south of the equator so one grid covers both hemispheres. Zones'
projected extents overlap, so two filters keep each point in about one zone:

- Zone wedge: keep only tiles and crops that intersect the zone's own 6-degree
  longitude band (80S to 84N).
- Coverage: skip crops where no point on a LAND_STEP_SIZE lattice falls inside the
  coverage mask (see coverage.py).
"""

import functools
import math

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds

from rslp.large_scale_embeddings.coverage import is_covered

# Vertices along the wedge's meridian and parallel edges. Meridians curve in UTM, so
# they are sampled densely.
NUM_MERIDIAN_VERTICES = 4096
NUM_PARALLEL_VERTICES = 256

# Spacing in pixels of the points sampled to decide whether a crop is covered. Must stay
# below the mask's cell size (about 926 m) so no covered cell falls between samples.
LAND_STEP_SIZE = 64

NORTH_EPSG_BASE = 32600
UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0


@functools.cache
def _get_zone_wedge(epsg_code: int, resolution: float) -> shapely.Polygon:
    """Cached implementation of get_zone_wedge keyed on the EPSG code."""
    utm_zone = CRS.from_epsg(epsg_code)
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
    # Pixel y points south (y_resolution is -resolution).
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

    The origin and shape are snapped outward to multiples of tile_size so tiles, crops
    and zarr shards stay aligned. Used by both job enumeration and store creation.

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
    if it intersects the zone's canonical wedge, and at least one point sampled in a
    LAND_STEP_SIZE grid within the crop falls inside the coverage mask.

    Args:
        projection: the UTM projection (with negative y resolution).
        bounds: the pixel bounds to divide into crops. Each value must be a multiple
            of crop_size.
        crop_size: the size of each crop in pixels. Must be a multiple of
            LAND_STEP_SIZE.
        wedge: the zone wedge from get_zone_wedge; computed if not provided.

    Returns:
        list of pixel bounds of the crops to process.
    """
    if any(value % crop_size != 0 for value in bounds):
        raise ValueError(
            f"bounds coords must be multiples of crop size {crop_size} but got {bounds}"
        )
    if crop_size % LAND_STEP_SIZE != 0:
        raise ValueError(
            f"crop size {crop_size} must be a multiple of "
            f"LAND_STEP_SIZE {LAND_STEP_SIZE}"
        )
    if wedge is None:
        wedge = get_zone_wedge(projection.crs, projection.x_resolution)

    cols = list(range(bounds[0] // crop_size, bounds[2] // crop_size))
    rows = list(range(bounds[1] // crop_size, bounds[3] // crop_size))

    # Sample the mask on a LAND_STEP_SIZE lattice, including crop edges.
    samples_per_crop = crop_size // LAND_STEP_SIZE
    sample_xs_px = np.arange(bounds[0], bounds[2] + 1, LAND_STEP_SIZE)
    sample_ys_px = np.arange(bounds[1], bounds[3] + 1, LAND_STEP_SIZE)
    xs_m, ys_m = np.meshgrid(
        sample_xs_px * projection.x_resolution,
        sample_ys_px * projection.y_resolution,
    )
    transformer = _get_to_wgs84_transformer(projection.crs.to_epsg())
    lons, lats = transformer.transform(xs_m, ys_m)
    lons = ((np.array(lons) + 180) % 360) - 180
    lats = np.clip(np.array(lats), -89.99, 89.99)
    is_land = is_covered(lats, lons)

    kept: list[PixelBounds] = []
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            sample_row = row_idx * samples_per_crop
            sample_col = col_idx * samples_per_crop
            if not is_land[
                sample_row : sample_row + samples_per_crop + 1,
                sample_col : sample_col + samples_per_crop + 1,
            ].any():
                continue
            crop_bounds = (
                col * crop_size,
                row * crop_size,
                (col + 1) * crop_size,
                (row + 1) * crop_size,
            )
            # Outside the wedge, the neighboring zone covers it.
            if not bounds_intersect_wedge(wedge, crop_bounds):
                continue
            kept.append(crop_bounds)
    return kept
