import numpy as np
import shapely
from global_land_mask import globe
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    enumerate_zone_tiles,
    get_zone_grid,
    get_zone_projection,
    get_zone_wedge,
    window_has_land,
    window_in_zone_wedge,
    zone_number_from_epsg,
)
from rslp.olmoearth_lcc.lcc_model.write_jobs import get_tasks


def _lonlat_to_pixel(zone_number: int, lon: float, lat: float) -> tuple[int, int]:
    """Get the (col, row) pixel of a point in the zone's northern projection."""
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{32600 + zone_number}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return int(x // RESOLUTION), int(-y // RESOLUTION)


def test_zone_number_from_epsg() -> None:
    assert zone_number_from_epsg(32610) == 10
    assert zone_number_from_epsg(32710) == 10
    assert zone_number_from_epsg(32660) == 60


def test_zone_grid_is_tile_aligned_and_covers_tiles() -> None:
    projection, origin_px, shape_px = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    assert projection.crs.to_epsg() == 32610
    assert all(value % TILE_SIZE == 0 for value in origin_px + shape_px)
    wedge = get_zone_wedge(projection.crs, RESOLUTION)
    tiles = list(enumerate_zone_tiles(10, RESOLUTION, TILE_SIZE))
    assert len(tiles) > 0
    for col, row in tiles:
        assert origin_px[0] <= col * TILE_SIZE < origin_px[0] + shape_px[1]
        assert origin_px[1] <= row * TILE_SIZE < origin_px[1] + shape_px[0]
        tile = shapely.box(
            col * TILE_SIZE,
            row * TILE_SIZE,
            (col + 1) * TILE_SIZE,
            (row + 1) * TILE_SIZE,
        )
        assert wedge.intersects(tile)


def test_southern_point_is_in_northern_crs_tile() -> None:
    # Zone 10 is 126W to 120W. South of the equator the northing is negative, so
    # pixel rows are positive.
    col, row = _lonlat_to_pixel(10, -123, -30)
    assert row > 0
    tiles = set(enumerate_zone_tiles(10, RESOLUTION, TILE_SIZE))
    assert (col // TILE_SIZE, row // TILE_SIZE) in tiles


def test_window_in_zone_wedge() -> None:
    projection = get_zone_projection(10, RESOLUTION)
    for lon, lat, expected in [(-123, 45, True), (-123, -30, True), (-110, 45, False)]:
        col, row = _lonlat_to_pixel(10, lon, lat)
        col -= col % PATCH_SIZE
        row -= row % PATCH_SIZE
        bounds = (col, row, col + PATCH_SIZE, row + PATCH_SIZE)
        assert window_in_zone_wedge(projection, bounds) == expected

    # Non-UTM projections are always kept.
    assert window_in_zone_wedge(
        Projection(CRS.from_epsg(3857), 10, -10), (0, 0, PATCH_SIZE, PATCH_SIZE)
    )


def test_window_has_land() -> None:
    projection = get_zone_projection(10, RESOLUTION)
    # Inland Washington, open Pacific, and Pacific south of the equator.
    for lon, lat, expected in [
        (-121, 47.5, True),
        (-125, 40, False),
        (-123, -30, False),
    ]:
        col, row = _lonlat_to_pixel(10, lon, lat)
        col -= col % PATCH_SIZE
        row -= row % PATCH_SIZE
        bounds = (col, row, col + PATCH_SIZE, row + PATCH_SIZE)
        assert window_has_land(projection, bounds) == expected


def test_window_has_land_between_corners() -> None:
    # A window centered on San Nicolas Island (about 14 km long) has water at all
    # four corners but is kept.
    projection = get_zone_projection(11, RESOLUTION)
    center_col, center_row = _lonlat_to_pixel(11, -119.5, 33.25)
    bounds = (
        center_col - PATCH_SIZE // 2,
        center_row - PATCH_SIZE // 2,
        center_col + PATCH_SIZE // 2,
        center_row + PATCH_SIZE // 2,
    )
    transformer = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    for col, row in [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
    ]:
        lon, lat = transformer.transform(col * RESOLUTION, -row * RESOLUTION)
        assert not globe.is_land(lat, lon)
    assert window_has_land(projection, bounds)


def test_get_tasks_epsg_hemispheres_match() -> None:
    assert get_tasks(epsg_code=32610) == get_tasks(epsg_code=32710)


def test_little_duplication_across_zones() -> None:
    # Windows of all tasks intersecting a region spanning several zones. Count the
    # windows (by center) inside the region that would be processed, against the
    # number of distinct locations (windows in their own zone's longitude band).
    region = (0.0, 40.0, 12.0, 50.0)
    num_kept = 0
    num_own = 0
    for projection, bounds in get_tasks(wgs84_bounds=region):
        zone_number = projection.crs.to_epsg() - 32600
        min_lon = -180 + 6 * (zone_number - 1)
        transformer = Transformer.from_crs(projection.crs, "EPSG:4326", always_xy=True)
        offsets = np.arange(bounds[0], bounds[2], PATCH_SIZE)
        rows = np.arange(bounds[1], bounds[3], PATCH_SIZE)
        cols_grid, rows_grid = np.meshgrid(offsets, rows)
        lons, lats = transformer.transform(
            (cols_grid + PATCH_SIZE / 2) * RESOLUTION,
            -(rows_grid + PATCH_SIZE / 2) * RESOLUTION,
        )
        for col, row, lon, lat in zip(
            cols_grid.flat, rows_grid.flat, np.ravel(lons), np.ravel(lats)
        ):
            if not (region[0] <= lon < region[2] and region[1] <= lat < region[3]):
                continue
            window_bounds = (col, row, col + PATCH_SIZE, row + PATCH_SIZE)
            if not window_in_zone_wedge(projection, window_bounds):
                continue
            num_kept += 1
            if min_lon <= lon < min_lon + 6:
                num_own += 1
    assert num_own > 0
    assert num_kept / num_own < 1.1
