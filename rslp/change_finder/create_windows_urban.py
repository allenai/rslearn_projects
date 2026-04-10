"""Create rslearn dataset windows for change finder around major world cities."""

import hashlib
import multiprocessing
import random
from datetime import datetime, timedelta, timezone

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.planetary_computer import PlanetaryComputerStacClient
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import retry
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from ._cities import CITIES

PIXEL_SIZE = 10
WINDOW_SIZE = 128
GRID_RADIUS = 15  # cells in each direction from center → 31x31 grid
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
DURATION_DAYS = 120

_stac_client: PlanetaryComputerStacClient | None = None


def _get_stac_client() -> PlanetaryComputerStacClient:
    global _stac_client
    if _stac_client is None:
        _stac_client = PlanetaryComputerStacClient(STAC_ENDPOINT)
    return _stac_client


def _has_period_coverage(lat: float, lon: float, base_time: datetime) -> bool:
    """Check that there is at least one Sentinel-2 item per 30-day period."""
    client = _get_stac_client()
    end_time = base_time + timedelta(days=DURATION_DAYS)

    eps = 0.01
    bbox = (lon - eps, lat - eps, lon + eps, lat + eps)
    items = retry(
        lambda: client.search(
            collections=[S2_COLLECTION],
            bbox=bbox,
            date_time=(base_time, end_time),
        ),
        retry_max_attempts=5,
        retry_backoff=timedelta(seconds=5),
    )

    period_length = 30
    num_periods = DURATION_DAYS // period_length

    periods_found: set[int] = set()
    for item in items:
        if item.time_range is None:
            continue
        ts = item.time_range[0]
        offset_days = (ts - base_time).total_seconds() / 86400
        if 0 <= offset_days < DURATION_DAYS:
            periods_found.add(int(offset_days) // period_length)

    return len(periods_found) >= num_periods


BASE_MONTHS: list[datetime] = []
for _y in range(2016, 2020):
    for _m in range(1, 13):
        BASE_MONTHS.append(datetime(_y, _m, 1, tzinfo=timezone.utc))


def _save_window_urban(
    dataset: Dataset,
    group: str,
    name: str,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    base_time: datetime,
) -> None:
    """Check STAC coverage and save a single pre-computed grid window."""
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    src_geometry = STGeometry(projection, shapely.Point(cx, cy), None)
    dst_geometry = src_geometry.to_projection(WGS84_PROJECTION)
    lon, lat = dst_geometry.shp.x, dst_geometry.shp.y

    from global_land_mask import globe

    if not globe.is_land(lat, lon):
        return

    if not _has_period_coverage(lat, lon, base_time):
        return

    time_range = (base_time, base_time + timedelta(days=DURATION_DAYS))

    is_val = hashlib.sha256(name.encode()).hexdigest()[0] in ["0", "1"]
    split = "val" if is_val else "train"

    window = Window(
        storage=dataset.storage,
        group=group,
        name=name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(split=split),
    )
    window.save()


def create_windows_urban(
    ds_path: str,
    group: str = "default",
    samples_per_city: int = 100,
    workers: int = 32,
) -> None:
    """Create windows by randomly sampling grid cells around ~1000 major world cities.

    For each city the center is projected to UTM, snapped to the 128px grid,
    and ``samples_per_city`` cells are randomly drawn from the 31x31 neighborhood.
    Duplicates (across and within cities) are discarded.

    Args:
        ds_path: output dataset path.
        group: window group name.
        samples_per_city: number of grid cells to sample per city (before dedup).
        workers: number of worker processes.
    """
    ds_upath = UPath(ds_path)
    dataset = Dataset(ds_upath)
    rng = random.Random()

    seen: set[tuple[int, int, int]] = set()  # (epsg, grid_x, grid_y)
    jobs: list[dict] = []

    for lat, lon, city_name in tqdm.tqdm(CITIES, desc="Building grid"):
        projection = get_utm_ups_projection(lon, lat, PIXEL_SIZE, -PIXEL_SIZE)
        epsg = projection.crs.to_epsg()

        src_point = shapely.Point(lon, lat)
        src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
        dst_geometry = src_geometry.to_projection(projection)

        center_x = int(dst_geometry.shp.x) // WINDOW_SIZE * WINDOW_SIZE
        center_y = int(dst_geometry.shp.y) // WINDOW_SIZE * WINDOW_SIZE

        for _ in range(samples_per_city):
            base_time = rng.choice(BASE_MONTHS)
            sample_x = center_x + rng.randint(-GRID_RADIUS, GRID_RADIUS) * WINDOW_SIZE
            sample_y = center_y + rng.randint(-GRID_RADIUS, GRID_RADIUS) * WINDOW_SIZE

            key = (epsg, sample_x, sample_y)
            if key in seen:
                continue
            seen.add(key)

            name = f"{city_name}_EPSG:{epsg}_{sample_x}_{sample_y}"
            bounds = (
                sample_x,
                sample_y,
                sample_x + WINDOW_SIZE,
                sample_y + WINDOW_SIZE,
            )

            jobs.append(
                dict(
                    dataset=dataset,
                    group=group,
                    name=name,
                    projection=projection,
                    bounds=bounds,
                    base_time=base_time,
                )
            )

    print(
        f"Processing {len(jobs)} unique grid cells from {len(CITIES)} cities with {workers} workers"
    )
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, _save_window_urban, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Creating windows"):
        pass
    p.close()
