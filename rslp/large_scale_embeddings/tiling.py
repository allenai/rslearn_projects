"""Tiling helpers for global embedding inference.

The world is processed as TILE_SIZE tiles in each UTM zone (see write_jobs.py), and
each tile is processed as PATCH_SIZE crops (see predict_pipeline.py). Tiles are
enumerated from the bounding box of each zone's projected extent, so without
filtering there is substantial duplication across zones (each point on land would be
computed in ~2.2 zones on average). This module provides the filters that reduce the
duplication to ~1.04x:

- Zone wedge: each UTM zone has a canonical 6-degree longitude wedge (0 to 84N for
  northern zones, 80S to 0 for southern zones). We only keep tiles and crops that
  intersect their own zone's wedge, so areas covered by multiple zones' projected
  extents are only processed in the zone that owns them.
- Ocean: we skip crops where every point sampled in a LAND_STEP_SIZE grid is ocean
  according to the global_land_mask package.
"""

import functools

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

# Step size in pixels of the grid of points sampled to decide whether each crop
# contains land. We process a crop if at least one point sampled in a grid along this
# step size intersects land, meaning we will capture islands that are at least this
# large in both height and width (2.56 km at 10 m/pixel). The crop size must be a
# multiple of this step size.
LAND_STEP_SIZE = 256


@functools.cache
def _get_zone_wedge(epsg_code: int, resolution: float) -> shapely.Polygon:
    """Cached implementation of get_zone_wedge keyed on the EPSG code."""
    utm_zone = CRS.from_epsg(epsg_code)
    min_lon, min_lat, max_lon, max_lat = get_wgs84_bounds(utm_zone)
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
    LAND_STEP_SIZE grid within the crop is land according to global_land_mask.

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

    # Compute the land mask at a lattice of points spaced LAND_STEP_SIZE apart
    # (including the crop corners and edges), in one vectorized pass.
    samples_per_crop = crop_size // LAND_STEP_SIZE
    sample_xs_px = np.arange(bounds[0], bounds[2] + 1, LAND_STEP_SIZE)
    sample_ys_px = np.arange(bounds[1], bounds[3] + 1, LAND_STEP_SIZE)
    xs_m, ys_m = np.meshgrid(
        sample_xs_px * projection.x_resolution,
        sample_ys_px * projection.y_resolution,
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
            # Skip if all points sampled within the crop (including its far edges)
            # are ocean.
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
            # Skip if the crop does not intersect the canonical zone wedge (it will be
            # processed by the neighboring UTM zone instead).
            if not bounds_intersect_wedge(wedge, crop_bounds):
                continue
            kept.append(crop_bounds)
    return kept
