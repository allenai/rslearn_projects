"""Write random 2048x2048 land tiles for LCC prediction to a job directory.

Samples random (lat, lon) points in a region (the whole world between 60S and 70N, or
a rough outline of mainland China or of Africa), snaps each to the 2048-pixel
grid (matching PATCH_SIZE) of its UTM zone's northern projection (EPSG:326NN, like the
GeoZarr store), and keeps tiles with land (tiling.window_has_land, as the prediction
pipeline checks its windows). Each tile is exactly one prediction window. This is
useful for getting diverse outputs to inspect model predictions and identify
annotation gaps.

Each tile gets its own random reference timestamp, recorded in its ``_meta.json``,
so one GeoZarr store (see geozarr.py) holds tiles with different timestamps.
"""

import math
import multiprocessing
import random
from datetime import datetime, timedelta

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from rslp.log_utils import get_logger

from . import geozarr
from .job_dir import PredictTaskArgs
from .tiling import (
    PATCH_SIZE,
    RESOLUTION,
    get_lonlat_zone_projection,
    window_has_land,
)
from .write_jobs import write_tasks

logger = get_logger(__name__)

# Rough polygon (lon, lat) of mainland China. Intentionally approximate; it just
# needs to keep samples roughly inside the country.
CHINA_POLYGON = shapely.Polygon(
    [
        (73.5, 39.5),
        (79.0, 42.0),
        (82.5, 45.0),
        (87.0, 49.0),
        (96.0, 53.5),
        (120.0, 53.5),
        (126.0, 53.0),
        (134.5, 48.0),
        (131.0, 43.0),
        (124.0, 40.0),
        (122.0, 30.0),
        (121.0, 24.0),
        (110.0, 21.0),
        (108.5, 18.5),
        (98.0, 24.0),
        (85.0, 28.0),
        (78.0, 32.0),
        (74.0, 36.0),
    ]
)

# Rough outline (lon, lat) of the whole African continent: mainland Africa without the
# Sinai, plus Madagascar and the Comoros. Natural Earth 50m countries with CONTINENT ==
# "Africa", unioned, buffered by 0.25 degrees and simplified with tolerance 0.15, minus
# Europe and the Sinai / Arabian side of the Red Sea; parts under 1 square degree (e.g.
# Cape Verde, the Canary Islands) are dropped. The margin includes some coastal water;
# sampled tiles still need land (window_has_land).
# fmt: off
AFRICA_POLYGON = shapely.MultiPolygon(
    [
        shapely.Polygon(  # mainland Africa (Sinai excluded)
            [
                (-17.62, 14.47), (-17.78, 14.82), (-16.8, 15.84), (-16.28, 17.9),
                (-16.45, 18.91), (-16.76, 19.35),
                (-16.49, 20.19), (-17.34, 20.8), (-17.15, 22.02), (-16.59, 22.69),
                (-16.1, 24.0), (-15.13, 24.84),
                (-14.61, 26.41), (-13.77, 26.9), (-13.09, 28.12), (-11.66, 28.53),
                (-10.39, 29.54), (-9.91, 30.19),
                (-10.12, 30.7), (-10.06, 31.46), (-9.46, 32.7), (-8.64, 33.47),
                (-7.06, 34.17), (-6.1, 35.95),
                (-5.12, 35.95), (-4.57, 35.45), (-2.97, 35.66), (-2.69, 35.38),
                (-1.97, 35.34), (1.22, 36.77),
                (3.78, 37.15), (4.8, 37.14), (5.3, 36.91), (6.47, 37.34),
                (7.9, 37.11), (9.7, 37.59),
                (10.34, 37.41), (10.46, 37.1), (11.14, 37.31), (11.37, 36.92),
                (10.73, 36.17), (11.22, 35.76),
                (11.52, 34.69), (11.18, 34.44), (10.91, 34.55), (10.3, 34.11),
                (10.44, 33.94), (10.99, 34.14),
                (11.45, 33.47), (12.36, 33.1), (13.32, 33.16), (15.26, 32.63),
                (15.85, 31.63), (17.91, 31.17),
                (19.01, 30.53), (19.56, 30.69), (19.88, 31.07), (19.68, 31.83),
                (19.94, 32.39), (21.62, 33.19),
                (23.18, 32.85), (23.38, 32.46), (24.98, 32.21), (25.38, 31.77),
                (26.82, 31.72), (29.1, 31.08),
                (31.07, 31.85), (31.96, 31.78), (32.17, 31.64), (32.3, 31.5),
                (34.52, 27.74), (34.51, 27.73),
                (34.25, 27.52), (33.81, 27.67), (35.4, 24.62), (36.01, 24.04),
                (35.76, 23.69), (35.91, 23.09),
                (37.09, 22.13), (37.5, 21.17), (37.69, 18.96), (38.78, 18.21),
                (39.69, 15.77), (39.88, 16.29),
                (40.26, 16.25), (40.66, 15.63), (40.3, 15.28), (41.34, 14.81),
                (43.13, 13.13), (43.64, 12.26),
                (43.66, 12.22), (43.36, 11.73), (44.39, 10.7), (45.78, 11.08),
                (46.5, 10.99), (47.35, 11.42),
                (49.31, 11.58), (50.27, 12.0), (51.33, 12.0), (51.48, 11.93),
                (51.39, 10.73), (51.64, 10.41),
                (51.49, 10.16), (51.14, 10.15), (51.05, 9.32), (48.16, 4.33),
                (46.23, 2.3), (43.88, 0.67),
                (41.56, -2.05), (40.44, -2.82), (39.65, -4.68), (40.0, -4.69),
                (40.12, -4.95), (39.98, -5.57),
                (39.54, -5.62), (39.82, -6.4), (39.54, -7.59), (39.94, -7.4),
                (40.16, -7.65), (39.56, -8.42),
                (40.03, -9.83), (40.86, -10.61), (40.69, -11.64), (41.08, -14.85),
                (40.0, -16.63), (37.39, -17.94),
                (36.56, -18.96), (35.0, -20.06), (34.95, -20.39), (35.79, -22.26),
                (35.64, -24.32), (35.11, -24.87),
                (33.0, -25.79), (33.2, -26.04), (33.14, -26.87), (32.58, -28.65),
                (31.52, -29.54), (30.16, -31.49),
                (28.0, -33.26), (26.68, -33.95), (25.9, -33.98), (25.64, -34.28),
                (24.6, -34.42), (22.59, -34.26),
                (20.02, -35.04), (18.37, -34.58), (18.13, -34.31), (18.17, -33.77),
                (17.6, -32.85), (18.08, -32.44),
                (17.98, -31.83), (16.53, -29.14), (15.13, -27.52), (14.73, -26.39),
                (14.25, -24.23), (14.23, -22.54),
                (11.53, -18.07), (11.51, -15.76), (12.33, -13.31), (13.45, -12.03),
                (13.6, -11.05), (12.75, -9.1),
                (13.12, -8.41), (12.05, -6.22), (11.82, -5.16), (9.42, -2.51),
                (8.47, -0.67), (8.57, -0.38),
                (9.06, -0.25), (9.09, 0.68), (9.35, 0.88), (9.14, 1.19),
                (9.54, 1.98), (9.69, 3.1),
                (9.34, 3.68), (8.61, 2.97), (8.18, 3.33), (8.77, 4.01),
                (8.68, 4.27), (5.87, 4.11),
                (5.38, 4.51), (4.68, 5.85), (4.1, 6.16), (1.39, 5.91),
                (1.06, 5.58), (0.32, 5.51),
                (-1.96, 4.52), (-3.97, 4.99), (-5.84, 4.77), (-7.52, 4.1),
                (-8.36, 4.36), (-11.87, 6.87),
                (-13.13, 7.39), (-13.55, 9.12), (-14.9, 10.57), (-15.54, 11.01),
                (-16.38, 10.88), (-16.26, 11.58),
                (-17.02, 12.38), (-17.07, 13.37), (-16.89, 13.63), (-17.33, 14.43),
            ]
        ),
        shapely.Polygon(  # Madagascar
            [
                (44.16, -19.88), (43.69, -17.59), (44.23, -16.12), (46.16, -15.49),
                (47.54, -14.47), (47.99, -13.11),
                (48.32, -12.95), (48.56, -13.15), (48.6, -12.3), (49.39, -11.86),
                (50.17, -12.99), (50.73, -15.41),
                (50.45, -16.07), (50.07, -16.18), (50.26, -16.77), (49.99, -17.27),
                (49.69, -17.31), (49.6, -18.41),
                (47.22, -25.15), (45.21, -25.82), (43.88, -25.19), (43.45, -24.43),
                (43.45, -23.49), (43.02, -22.43),
                (43.2, -21.55),
            ]
        ),
        shapely.Polygon(  # Comoros
            [
                (43.57, -11.23), (43.7, -12.0), (43.97, -12.14), (44.64, -11.89),
                (44.7, -12.51), (43.5, -12.54),
                (42.99, -11.83), (43.07, -11.26),
            ]
        ),
    ]
)
# fmt: on

# Sampling region by name: a polygon, or None for the whole world (between
# GLOBAL_MIN_LAT and GLOBAL_MAX_LAT).
REGIONS: dict[str, shapely.Polygon | shapely.MultiPolygon | None] = {
    "global": None,
    "china": CHINA_POLYGON,
    "africa": AFRICA_POLYGON,
}
GLOBAL_MIN_LAT = -60.0
GLOBAL_MAX_LAT = 70.0

# Each sampling round draws this many points per requested tile, to account for
# water, points outside the region polygon, and duplicates.
OVERSAMPLE_FACTOR = 5
# Give up after this many sampling rounds, e.g. when the region has fewer unique
# land tiles than requested.
MAX_SAMPLING_ROUNDS = 10


def snap_random_tile(lon: float, lat: float) -> tuple[Projection, PixelBounds] | None:
    """Snap a point to the PATCH_SIZE grid of its UTM zone's northern projection.

    Args:
        lon: the longitude.
        lat: the latitude.

    Returns:
        the (projection, bounds) of the tile, or None if the point is in a UPS zone.
    """
    projection = get_lonlat_zone_projection(lon, lat, RESOLUTION)
    if projection is None:
        return None
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = math.floor(dst_geom.shp.x / PATCH_SIZE)
    row = math.floor(dst_geom.shp.y / PATCH_SIZE)
    bounds = (
        col * PATCH_SIZE,
        row * PATCH_SIZE,
        (col + 1) * PATCH_SIZE,
        (row + 1) * PATCH_SIZE,
    )
    return projection, bounds


class RandomTileCollector:
    """Deduplicate random tiles and assign each a random reference timestamp."""

    def __init__(self, start_time: datetime, end_time: datetime, rng: random.Random):
        """Create a new RandomTileCollector.

        Args:
            start_time: earliest reference timestamp (inclusive).
            end_time: latest reference timestamp (inclusive).
            rng: the random number generator for the timestamps.
        """
        self.start_time = start_time
        self.time_range_seconds = int((end_time - start_time).total_seconds())
        self.rng = rng
        self.tasks: list[PredictTaskArgs] = []
        self.seen: set[tuple[str, PixelBounds]] = set()

    def add(self, projection: Projection, bounds: PixelBounds) -> bool:
        """Add a tile unless it is a duplicate.

        Args:
            projection: the tile projection.
            bounds: the tile bounds.

        Returns:
            whether the tile was added.
        """
        key = (str(projection.crs), bounds)
        if key in self.seen:
            return False
        self.seen.add(key)
        offset = timedelta(seconds=self.rng.randint(0, self.time_range_seconds))
        timestamp = self.start_time + offset
        self.tasks.append(
            PredictTaskArgs(
                projection=projection, bounds=bounds, time_range=(timestamp, timestamp)
            )
        )
        return True


def sample_region_point(rng: random.Random, region: str) -> tuple[float, float]:
    """Sample a uniformly random (lon, lat) point within a region of REGIONS."""
    polygon = REGIONS[region]
    if polygon is None:
        return rng.uniform(-180, 180), rng.uniform(GLOBAL_MIN_LAT, GLOBAL_MAX_LAT)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    while True:
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        if polygon.contains(shapely.Point(lon, lat)):
            return lon, lat


def _land_tile(lon: float, lat: float) -> tuple[Projection, PixelBounds] | None:
    """Snap a point to a tile and check that it has land. Returns None if rejected."""
    tile = snap_random_tile(lon, lat)
    if tile is None or not window_has_land(*tile):
        return None
    return tile


def write_jobs_random_2048(
    start_time: datetime,
    end_time: datetime,
    out_path: str,
    job_dir: str,
    zarr_store_path: str,
    zarr_detail: str = geozarr.DETAIL_COMPACT,
    region: str = "global",
    count: int = 100,
    fast_inference: bool = True,
    workers: int = 32,
) -> None:
    """Sample random 2048x2048 land tiles and write prediction tasks to a job dir.

    Samples random lat/lon points in the region, snaps each to a PATCH_SIZE-aligned
    grid, checks that the tile has land, deduplicates, and writes the tiles to the
    job directory (which skips tiles that are already done or queued). Each tile
    gets a randomly chosen reference timestamp between start_time and end_time.

    Args:
        start_time: earliest reference timestamp (inclusive). Must have timezone.
        end_time: latest reference timestamp (inclusive). Must have timezone.
        out_path: the output directory for the per-tile _meta.json.
        job_dir: the job directory (see job_dir.py).
        zarr_store_path: the GeoZarr store, created if missing.
        zarr_detail: the detail mode of the store, one of geozarr.DETAILS.
        region: where to sample tiles, one of REGIONS.
        count: number of land tiles to collect.
        fast_inference: whether workers should use fast inference.
        workers: number of worker processes for land checks.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region}, expected one of {list(REGIONS)}")
    rng = random.Random()
    collector = RandomTileCollector(start_time, end_time, rng)

    with multiprocessing.Pool(workers) as pool:
        for _ in range(MAX_SAMPLING_ROUNDS):
            if len(collector.tasks) >= count:
                break
            sample_args = []
            for _ in range(OVERSAMPLE_FACTOR * count):
                lon, lat = sample_region_point(rng, region)
                sample_args.append(dict(lon=lon, lat=lat))
            results = star_imap_unordered(pool, _land_tile, sample_args)
            for tile in tqdm.tqdm(
                results, total=len(sample_args), desc=f"Sampling {region} land tiles"
            ):
                if tile is not None and len(collector.tasks) < count:
                    collector.add(*tile)

    logger.info(
        "Collected %d %s land tiles (requested %d)",
        len(collector.tasks),
        region,
        count,
    )
    write_tasks(
        job_dir,
        collector.tasks,
        out_path=out_path,
        zarr_store_path=zarr_store_path,
        zarr_detail=zarr_detail,
        fast_inference=fast_inference,
    )
