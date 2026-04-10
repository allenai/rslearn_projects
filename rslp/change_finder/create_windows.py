"""Create rslearn dataset windows for change finder on a global land grid."""

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
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

PIXEL_SIZE = 10
WINDOW_SIZE = 128
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
DURATION_DAYS = 120

_stac_client: PlanetaryComputerStacClient | None = None


def _get_stac_client() -> PlanetaryComputerStacClient:
    """Lazily create a per-process STAC client."""
    global _stac_client
    if _stac_client is None:
        _stac_client = PlanetaryComputerStacClient(STAC_ENDPOINT)
    return _stac_client


def _has_period_coverage(lat: float, lon: float, base_time: datetime) -> bool:
    """Check that there is at least one Sentinel-2 item per 30-day period in the window."""
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


# All first-of-month dates in [Jan 2016, Dec 2019].
BASE_MONTHS: list[datetime] = []
for y in range(2016, 2020):
    for m in range(1, 13):
        BASE_MONTHS.append(datetime(y, m, 1, tzinfo=timezone.utc))


def _save_window(
    dataset: Dataset,
    group: str,
    lat: float,
    lon: float,
    base_time: datetime,
) -> None:
    """Create and save a single window from a lat/lon sample."""
    from global_land_mask import globe

    if not globe.is_land(lat, lon):
        return

    if not _has_period_coverage(lat, lon, base_time):
        return

    dst_projection = get_utm_ups_projection(lon, lat, PIXEL_SIZE, -PIXEL_SIZE)

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_geometry = src_geometry.to_projection(dst_projection)

    grid_x = int(dst_geometry.shp.x) // WINDOW_SIZE * WINDOW_SIZE
    grid_y = int(dst_geometry.shp.y) // WINDOW_SIZE * WINDOW_SIZE
    bounds = (grid_x, grid_y, grid_x + WINDOW_SIZE, grid_y + WINDOW_SIZE)

    time_range = (base_time, base_time + timedelta(days=DURATION_DAYS))

    name = f"{lat:.4f}_{lon:.4f}"
    is_val = hashlib.sha256(name.encode()).hexdigest()[0] in ["0", "1"]
    split = "val" if is_val else "train"

    window = Window(
        storage=dataset.storage,
        group=group,
        name=name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(split=split),
    )
    window.save()


def create_windows(
    ds_path: str,
    group: str = "default",
    num_samples: int = 50000,
    workers: int = 32,
) -> None:
    """Create windows by sampling random land locations and snapping to a 128px grid.

    Each window is a 128x128 pixel tile (10 m/pixel) in UTM. Locations are sampled
    randomly, ocean points are skipped, and each point is snapped to the nearest
    grid-aligned tile so that bounds are always multiples of WINDOW_SIZE.

    Args:
        ds_path: output dataset path, e.g.
            gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/
        group: window group name.
        num_samples: number of random points to sample (ocean points are skipped).
        workers: number of worker processes for saving windows.
    """
    ds_upath = UPath(ds_path)
    dataset = Dataset(ds_upath)
    rng = random.Random()

    jobs: list[dict] = []
    for _ in range(num_samples):
        jobs.append(
            dict(
                dataset=dataset,
                group=group,
                lat=rng.uniform(-60, 70),
                lon=rng.uniform(-180, 180),
                base_time=rng.choice(BASE_MONTHS),
            )
        )

    print(f"Processing {len(jobs)} samples with {workers} workers")
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, _save_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Creating windows"):
        pass
    p.close()
