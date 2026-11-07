"""Write random 2048x2048 land tiles for LCC prediction to a job directory.

Samples random (lat, lon) points in a region (the whole world between 60S and 70N, or
a rough polygon of mainland China or sub-Saharan Africa), snaps each to the 2048-pixel
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

# Rough exterior polygon (lon, lat) covering a large swath of sub-Saharan Africa
# (among others Guinea, Mali, Burkina Faso, Niger, Sierra Leone, Liberia, Cote
# d'Ivoire, Ghana, Togo, Benin, Nigeria, Cameroon, Central African Republic, South
# Sudan, Ethiopia, Kenya, Uganda, DRC, Republic of the Congo, Gabon, Angola, Zambia,
# Rwanda, Burundi, Tanzania, Malawi, Mozambique and Zimbabwe). Being a single coarse
# exterior, it also spills over into some neighboring countries. Traced roughly
# clockwise starting from the West African coast.
AFRICA_POLYGON = shapely.Polygon(
    [
        (-13.5, 12.5),  # West coast (Guinea / Sierra Leone)
        (-12.0, 25.0),  # Mali NW (Sahara edge)
        (3.0, 25.0),  # Mali / Niger north
        (16.0, 23.0),  # Niger NE
        (24.0, 22.0),  # toward Sudan
        (37.0, 18.0),  # Sudan / Ethiopia north
        (40.0, 15.0),  # Ethiopia north
        (43.0, 11.0),  # Ethiopia east
        (48.0, 8.0),  # Ethiopia SE (Ogaden)
        (42.0, 4.0),  # Kenya east / Somalia border
        (41.0, -2.0),  # Kenya coast
        (40.0, -10.0),  # Tanzania coast
        (41.0, -16.0),  # Mozambique coast
        (37.0, -26.0),  # Mozambique south
        (31.0, -26.0),  # Zimbabwe / Mozambique south
        (25.0, -22.0),  # Zimbabwe / Botswana
        (19.0, -18.0),  # Angola / Namibia border
        (11.5, -17.0),  # Angola SW coast
        (13.5, -6.0),  # Angola coast
        (9.0, -1.0),  # Gabon coast
        (8.0, 4.0),  # Cameroon / Nigeria coast (Gulf of Guinea)
        (-0.5, 4.0),  # Ghana / Togo coast
        (-8.0, 4.0),  # Cote d'Ivoire / Liberia coast
        (-13.5, 6.5),  # Sierra Leone coast
    ]
)

# Sampling region by name: a polygon, or None for the whole world (between
# GLOBAL_MIN_LAT and GLOBAL_MAX_LAT).
REGIONS: dict[str, shapely.Polygon | None] = {
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
