"""Enqueue random 2048x2048 land tiles for LCC prediction on a Beaker queue.

Samples random (lat, lon) points, projects to UTM, snaps to the 2048-pixel grid
(matching PATCH_SIZE), and filters to tiles where at least one corner is on land.
Each tile is exactly one prediction window. This is useful for getting diverse
outputs across the globe to inspect model predictions and identify annotation gaps.
"""

import json
import multiprocessing
import random
from datetime import datetime, timedelta

import shapely
import tqdm
from global_land_mask import globe
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .predict_pipeline import (
    DEFAULT_THRESHOLD,
    PATCH_SIZE,
    RESOLUTION,
    PredictTaskArgs,
    get_output_fname,
)

logger = get_logger(__name__)


def _any_corner_on_land(
    projection: Projection, bounds: tuple[int, int, int, int]
) -> bool:
    """Check whether at least one corner of the tile is on land."""
    corners = [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
    ]
    for px, py in corners:
        geom = STGeometry(projection, shapely.Point(px, py), None)
        wgs84 = geom.to_projection(WGS84_PROJECTION)
        if globe.is_land(wgs84.shp.y, wgs84.shp.x):
            return True
    return False


def _process_sample(
    lat: float,
    lon: float,
) -> tuple[int, int, int, Projection, tuple[int, int, int, int]] | None:
    """Project a lat/lon sample to UTM, snap to grid, and check land. Returns None if rejected."""
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    epsg = projection.crs.to_epsg()
    if epsg is None:
        return None

    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    col = int(dst_geom.shp.x) // PATCH_SIZE
    row = int(dst_geom.shp.y) // PATCH_SIZE

    bounds = (
        col * PATCH_SIZE,
        row * PATCH_SIZE,
        (col + 1) * PATCH_SIZE,
        (row + 1) * PATCH_SIZE,
    )

    if not _any_corner_on_land(projection, bounds):
        return None

    return (epsg, col, row, projection, bounds)


def write_jobs_random_2048(
    start_time: datetime,
    end_time: datetime,
    out_path: str,
    queue_name: str,
    count: int = 100,
    batch_size: int = 1,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
    workers: int = 32,
) -> None:
    """Sample random 2048x2048 land tiles and write prediction jobs to a Beaker queue.

    Samples random lat/lon points, projects each to UTM, snaps to a PATCH_SIZE-aligned
    grid, checks that at least one corner is on land, deduplicates, skips tiles whose
    output already exists, and writes the remainder to the queue. Each tile gets a
    randomly chosen reference timestamp between start_time and end_time.

    Args:
        start_time: earliest reference timestamp (inclusive). Must have timezone.
        end_time: latest reference timestamp (inclusive). Must have timezone.
        out_path: the output directory (also used to skip already-computed tiles).
        queue_name: the Beaker queue to write the job entries to.
        count: number of land tiles to enqueue.
        batch_size: how many tasks to run per worker job.
        write_raster: whether workers should also write the merged raster.
        threshold: binary change probability threshold (0-255) for polygonization.
        workers: number of worker processes for land checks.
    """
    time_range_seconds = int((end_time - start_time).total_seconds())

    # Collect existing outputs for dedup.
    out_upath = UPath(out_path)
    existing: set[str] = set()
    if out_upath.exists():
        existing = {fname.name for fname in out_upath.iterdir()}

    # Over-sample to account for ocean points, duplicates, and already-computed tiles.
    num_samples = count * 5

    rng = random.Random()
    sample_args = [
        dict(lat=rng.uniform(-60, 70), lon=rng.uniform(-180, 180))
        for _ in range(num_samples)
    ]

    # Project and land-check all samples in parallel.
    p = multiprocessing.Pool(workers)
    results = list(
        tqdm.tqdm(
            star_imap_unordered(p, _process_sample, sample_args),
            total=len(sample_args),
            desc="Sampling random land tiles",
        )
    )
    p.close()

    # Sequentially dedup, skip existing, and cap at count.
    seen: set[tuple[int, int, int]] = set()  # (epsg, col, row)
    tasks: list[
        tuple[Projection, tuple[int, int, int, int], tuple[datetime, datetime]]
    ] = []

    for result in results:
        if result is None:
            continue

        epsg, col, row, projection, bounds = result

        key = (epsg, col, row)
        if key in seen:
            continue
        seen.add(key)

        if get_output_fname(out_path, projection, bounds).name in existing:
            continue

        offset = timedelta(seconds=rng.randint(0, time_range_seconds))
        timestamp = start_time + offset
        time_range = (timestamp, timestamp)

        tasks.append((projection, bounds, time_range))
        if len(tasks) >= count:
            break

    logger.info("Collected %d land tiles (requested %d)", len(tasks), count)

    # Convert to batched worker jobs.
    jobs = []
    for i in range(0, len(tasks), batch_size):
        cur_tasks = tasks[i : i + batch_size]
        predict_tasks = [
            PredictTaskArgs(
                projection_json=projection.serialize(),
                bounds=bounds,
                time_range=time_range,
            )
            for projection, bounds, time_range in cur_tasks
        ]
        cur_args = [
            "--out_path",
            out_path,
            "--scratch_path",
            "/tmp/scratch/",
            "--tasks",
            json.dumps([predict_task.serialize() for predict_task in predict_tasks]),
        ]
        if write_raster:
            cur_args += ["--write_raster", "true"]
        cur_args += ["--threshold", str(threshold)]
        jobs.append(cur_args)

    random.shuffle(jobs)
    rslp.common.worker.write_jobs(queue_name, "change_finder_v2", "predict_multi", jobs)
    logger.info("Wrote %d jobs to queue %s", len(jobs), queue_name)
