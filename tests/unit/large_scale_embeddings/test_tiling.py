"""Unit tests for rslp.large_scale_embeddings.tiling."""

from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.large_scale_embeddings.tiling import (
    bounds_intersect_wedge,
    get_zone_wedge,
    list_kept_crops,
)

RESOLUTION = 10

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
