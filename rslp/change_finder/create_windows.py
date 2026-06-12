"""Create rslearn dataset windows for change finder on a global land grid."""

import hashlib
import multiprocessing
import random
import traceback
from datetime import datetime, timedelta, timezone

import shapely
import tqdm
from global_land_mask import globe
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.planetary_computer import PlanetaryComputerStacClient
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import retry
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

PIXEL_SIZE = 10
WINDOW_SIZE = 128
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
DURATION_DAYS = 180
MIN_CENTER_DIST = 256  # minimum distance in pixels between window centers
_CELL_RADIUS = MIN_CENTER_DIST // WINDOW_SIZE  # neighborhood radius in cell indices

_stac_client: PlanetaryComputerStacClient | None = None


def _get_stac_client() -> PlanetaryComputerStacClient:
    """Lazily create a per-process STAC client."""
    global _stac_client
    if _stac_client is None:
        _stac_client = PlanetaryComputerStacClient(STAC_ENDPOINT)
    return _stac_client


def _compute_cell(
    lat: float,
    lon: float,
    base_time: datetime,
) -> tuple[int, int, int, datetime] | None:
    """Project lat/lon to UTM, check land/coverage, return (epsg, cell_x, cell_y, base_time) or None."""
    if not globe.is_land(lat, lon):
        return None

    if not _has_period_coverage(lat, lon, base_time):
        return None

    dst_projection = get_utm_ups_projection(lon, lat, PIXEL_SIZE, -PIXEL_SIZE)
    epsg = dst_projection.crs.to_epsg()

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_geometry = src_geometry.to_projection(dst_projection)
    cell_x = int(dst_geometry.shp.x) // WINDOW_SIZE
    cell_y = int(dst_geometry.shp.y) // WINDOW_SIZE
    return (epsg, cell_x, cell_y, base_time)


def _has_period_coverage(lat: float, lon: float, base_time: datetime) -> bool:
    """Check that there is at least one Sentinel-2 item per 30-day period in the window."""
    client = _get_stac_client()
    end_time = base_time + timedelta(days=DURATION_DAYS)

    eps = 0.01
    bbox = (lon - eps, lat - eps, lon + eps, lat + eps)
    try:
        items = retry(
            lambda: client.search(
                collections=[S2_COLLECTION],
                bbox=bbox,
                date_time=(base_time, end_time),
            ),
            retry_max_attempts=5,
            retry_backoff=timedelta(seconds=5),
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Skipping lat={lat}, lon={lon}, base_time={base_time}: {e}")
        return False

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


# All first-of-month dates in [Jan 2016, Oct 2016].
# Only months in 2016 so that end will be before 1 April 2026
# (with six months per year, ten years).
BASE_MONTHS: list[datetime] = []
for m in range(1, 11):
    BASE_MONTHS.append(datetime(2016, m, 1, tzinfo=timezone.utc))


def _save_window(
    dataset: Dataset,
    group: str,
    epsg: int,
    cell_x: int,
    cell_y: int,
    base_time: datetime,
) -> None:
    """Save a single grid-cell window."""
    projection = Projection(CRS.from_epsg(epsg), PIXEL_SIZE, -PIXEL_SIZE)

    grid_x = cell_x * WINDOW_SIZE
    grid_y = cell_y * WINDOW_SIZE
    bounds = (grid_x, grid_y, grid_x + WINDOW_SIZE, grid_y + WINDOW_SIZE)

    time_range = (base_time, base_time + timedelta(days=DURATION_DAYS))

    name = f"EPSG:{epsg}_{cell_x}_{cell_y}"
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

    # Build index of occupied grid cells from existing windows.
    # Two cells are "too close" if centers are < 256 px apart (i.e. < 128 px
    # between borders).  With WINDOW_SIZE=128 that equals a 3×3 neighborhood
    # in cell-index space.
    occupied: set[tuple[int, int, int]] = set()  # (epsg, cell_x, cell_y)
    print("Loading existing windows...")
    for w in dataset.load_windows(workers=workers, show_progress=True):
        epsg = w.projection.crs.to_epsg()
        if epsg is None:
            continue
        occupied.add((epsg, w.bounds[0] // WINDOW_SIZE, w.bounds[1] // WINDOW_SIZE))
    print(f"Found {len(occupied)} existing cells")

    # 1) Generate random lat/lon samples.
    samples = [
        dict(
            lat=rng.uniform(-60, 70),
            lon=rng.uniform(-180, 180),
            base_time=rng.choice(BASE_MONTHS),
        )
        for _ in range(num_samples)
    ]

    # 2) Compute grid cells in parallel (reprojection + land/coverage checks).
    print(f"Computing grid cells for {len(samples)} samples with {workers} workers...")
    p = multiprocessing.Pool(workers)
    cell_results = list(
        tqdm.tqdm(
            star_imap_unordered(p, _compute_cell, samples),
            total=len(samples),
            desc="Computing cells",
        )
    )
    p.close()

    # 3) Sequentially filter by proximity to existing/planned windows.
    jobs: list[dict] = []
    skipped = 0
    for cell in cell_results:
        if cell is None:
            continue
        epsg, cell_x, cell_y, base_time = cell
        if any(
            (epsg, cell_x + dx, cell_y + dy) in occupied
            for dx in range(-_CELL_RADIUS + 1, _CELL_RADIUS)
            for dy in range(-_CELL_RADIUS + 1, _CELL_RADIUS)
        ):
            skipped += 1
            continue

        occupied.add((epsg, cell_x, cell_y))
        jobs.append(
            dict(
                dataset=dataset,
                group=group,
                epsg=epsg,
                cell_x=cell_x,
                cell_y=cell_y,
                base_time=base_time,
            )
        )

    # 4) Save windows in parallel (STAC checks + write).
    print(f"Skipped {skipped} samples too close to existing/planned windows")
    print(f"Processing {len(jobs)} samples with {workers} workers")
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, _save_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Creating windows"):
        pass
    p.close()
