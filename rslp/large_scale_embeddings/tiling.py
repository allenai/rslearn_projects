"""Tiling helpers for global embedding inference.

Each UTM zone (1-60) is processed once in its northern CRS (EPSG:326NN), with northing
going negative south of the equator so one grid covers both hemispheres. Zones'
projected extents overlap, so two filters keep each point in about one zone:

- Zone wedge: keep only tiles and crops that intersect the zone's own 6-degree
  longitude band (80S to 84N).
- Coverage: skip crops where no point on a LAND_STEP_SIZE lattice falls inside the
  coverage mask (see coverage.py).

The zone wedge and grid come from rslp.utils.tiling.
"""

import functools

import numpy as np
import shapely
from pyproj import Transformer
from rslearn.utils.geometry import PixelBounds, Projection

from rslp.large_scale_embeddings.coverage import is_covered
from rslp.utils.tiling import get_zone_wedge

# Spacing in pixels of the points sampled to decide whether a crop is covered. Must stay
# below the mask's cell size (about 926 m) so no covered cell falls between samples.
LAND_STEP_SIZE = 64


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
