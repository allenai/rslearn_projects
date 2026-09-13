"""Unit tests for rslp.large_scale_embeddings.tiling."""

import numpy as np
import pytest
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from rslp.large_scale_embeddings import tiling
from rslp.large_scale_embeddings.tiling import (
    LAND_STEP_SIZE,
    bounds_intersect_wedge,
    get_zone_grid,
    get_zone_wedge,
    list_kept_crops,
)

RESOLUTION = 10
TILE_SIZE = 32768
SHARD_SIZE = 2048

# EPSG:32610 is UTM zone 10N, covering -126 to -120 longitude, 0 to 84 latitude.
UTM_ZONE_10N = CRS.from_epsg(32610)
PROJECTION_10N = Projection(UTM_ZONE_10N, RESOLUTION, -RESOLUTION)


def _pixel_bounds_around(
    x_m: float, y_m: float, size: int
) -> tuple[int, int, int, int]:
    """Get size x size pixel bounds aligned to size containing the given point."""
    col = int(x_m / RESOLUTION) // size
    row = int(-y_m / RESOLUTION) // size
    return (col * size, row * size, (col + 1) * size, (row + 1) * size)


def test_wedge_contains_zone_interior() -> None:
    """A tile at the zone's central meridian is inside the wedge."""
    wedge = get_zone_wedge(UTM_ZONE_10N, RESOLUTION)
    # The central meridian is at x=500000m; Seattle-ish latitude is y~5270000m.
    bounds = _pixel_bounds_around(500000, 5270000, 2048)
    assert bounds_intersect_wedge(wedge, bounds)


def test_wedge_excludes_far_outside_zone() -> None:
    """A tile far east of the zone's wedge (in zone 11's territory) is excluded."""
    wedge = get_zone_wedge(UTM_ZONE_10N, RESOLUTION)
    # x=1100000m in zone 10N is around -114 longitude, well inside zone 11/12.
    bounds = _pixel_bounds_around(1100000, 5270000, 2048)
    assert not bounds_intersect_wedge(wedge, bounds)


def test_list_kept_crops_land() -> None:
    """Crops on land at the zone center are all kept."""
    # Near Portland, OR (~-122.6, 45.5): x~530000m, y~5040000m in zone 10N.
    bounds = _pixel_bounds_around(530000, 5040000, 4096)
    kept = list_kept_crops(PROJECTION_10N, bounds, 2048)
    assert len(kept) == 4


def test_list_kept_crops_ocean() -> None:
    """Crops in the open ocean are all skipped."""
    # Pacific ocean far off the coast (~-126 to -125, ~40N): x~150000m, y~4440000m.
    bounds = _pixel_bounds_around(150000, 4440000, 4096)
    kept = list_kept_crops(PROJECTION_10N, bounds, 2048)
    assert len(kept) == 0


def test_list_kept_crops_outside_wedge() -> None:
    """Crops on land but outside the zone's wedge are skipped."""
    # x=1100000m in zone 10N is around -114 longitude (Idaho, land), well inside
    # zone 11/12's territory.
    bounds = _pixel_bounds_around(1100000, 5040000, 4096)
    kept = list_kept_crops(PROJECTION_10N, bounds, 2048)
    assert len(kept) == 0


def test_list_kept_crops_small_island(monkeypatch: pytest.MonkeyPatch) -> None:
    """An island smaller than a crop but >= LAND_STEP_SIZE is captured.

    The land mask is mocked so that only one grid sample point in the interior of one
    crop (away from all crop corners) is land; the crop must still be kept.
    """
    crop_size = 2048
    bounds = _pixel_bounds_around(530000, 5040000, 2 * crop_size)
    # A sample point in the interior of the first crop: crop_size / 2 is a multiple
    # of LAND_STEP_SIZE but not on any crop corner or edge.
    assert (crop_size // 2) % LAND_STEP_SIZE == 0
    island_x_px = bounds[0] + crop_size // 2
    island_y_px = bounds[1] + crop_size // 2
    island_geom = STGeometry(
        PROJECTION_10N, shapely.Point(island_x_px, island_y_px), None
    ).to_projection(WGS84_PROJECTION)
    island_lon, island_lat = island_geom.shp.x, island_geom.shp.y

    # Land only within ~1 km of the island point, which is smaller than the
    # LAND_STEP_SIZE spacing (2.56 km) so no other sample point registers as land.
    def fake_is_land(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (np.abs(lat - island_lat) < 0.01) & (np.abs(lon - island_lon) < 0.01)

    monkeypatch.setattr(tiling.globe, "is_land", fake_is_land)

    kept = list_kept_crops(PROJECTION_10N, bounds, crop_size)
    assert kept == [
        (bounds[0], bounds[1], bounds[0] + crop_size, bounds[1] + crop_size)
    ]


def test_wedge_spans_both_hemispheres() -> None:
    """The northern-CRS wedge covers southern latitudes (negative northing)."""
    wedge = get_zone_wedge(UTM_ZONE_10N, RESOLUTION)
    # A point at ~10S on zone 10's central meridian has negative northing in the
    # northern CRS (EPSG:32610); it must still fall inside the full-latitude wedge.
    bounds = _pixel_bounds_around(500000, -1105000, 2048)
    assert bounds_intersect_wedge(wedge, bounds)


def test_get_zone_grid_alignment_and_hemispheres() -> None:
    """get_zone_grid is tile/shard aligned and spans both hemispheres."""
    projection, origin, shape = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    origin_x, origin_y = origin
    height, width = shape
    assert projection.crs.to_epsg() == 32610
    # Origin snapped to the tile grid, shape snapped to the shard grid.
    assert origin_x % TILE_SIZE == 0 and origin_y % TILE_SIZE == 0
    assert height % SHARD_SIZE == 0 and width % SHARD_SIZE == 0
    # Row 0 is north of the equator (negative northing -> negative pixel y) and the
    # array extends south past the equator (positive pixel y).
    assert origin_y < 0 < origin_y + height
