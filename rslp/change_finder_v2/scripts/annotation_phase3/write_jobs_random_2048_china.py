"""Enqueue random 2048x2048 tiles within China for LCC prediction on a Beaker queue.

Same idea as ``lcc_model.write_jobs_random_2048`` but instead of a global land check,
samples random (lat, lon) points inside a rough polygon of mainland China, projects to
UTM, and snaps to the 2048-pixel grid (matching PATCH_SIZE). Each tile is exactly one
prediction window. Useful for inspecting LCC model predictions over China and finding
annotation gaps.
"""

import json
import random
from datetime import datetime, timedelta

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

import rslp.common.worker
from rslp.change_finder_v2.lcc_model.predict_pipeline import (
    DEFAULT_THRESHOLD,
    PATCH_SIZE,
    RESOLUTION,
    PredictTaskArgs,
    get_output_fname,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Rough polygon (lon, lat) of mainland China. Intentionally approximate (<20 points);
# it just needs to keep samples roughly inside the country.
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
CHINA_MIN_LON, CHINA_MIN_LAT, CHINA_MAX_LON, CHINA_MAX_LAT = CHINA_POLYGON.bounds


def _process_sample(
    lat: float,
    lon: float,
) -> tuple[int, int, int, Projection, tuple[int, int, int, int]]:
    """Project a lat/lon sample to UTM and snap to the PATCH_SIZE grid."""
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    epsg = projection.crs.to_epsg()

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

    return (epsg, col, row, projection, bounds)


def write_jobs_random_2048_china(
    start_time: datetime,
    end_time: datetime,
    out_path: str,
    queue_name: str,
    count: int = 200,
    batch_size: int = 1,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> None:
    """Sample random 2048x2048 tiles in China and write prediction jobs to a queue.

    Samples random lat/lon points within a rough China polygon, projects each to UTM,
    snaps to a PATCH_SIZE-aligned grid, deduplicates, skips tiles whose output already
    exists, and writes the remainder to the queue. Each tile gets a randomly chosen
    reference timestamp between start_time and end_time.

    Args:
        start_time: earliest reference timestamp (inclusive). Must have timezone.
        end_time: latest reference timestamp (inclusive). Must have timezone.
        out_path: the output directory (also used to skip already-computed tiles).
        queue_name: the Beaker queue to write the job entries to.
        count: number of tiles to enqueue.
        batch_size: how many tasks to run per worker job.
        write_raster: whether workers should also write the merged raster.
        threshold: binary change probability threshold (0-255) for polygonization.
    """
    time_range_seconds = int((end_time - start_time).total_seconds())

    # Collect existing outputs for dedup.
    out_upath = UPath(out_path)
    existing: set[str] = set()
    if out_upath.exists():
        existing = {fname.name for fname in out_upath.iterdir()}

    rng = random.Random()

    # Sequentially sample (rejecting points outside the polygon), dedup, skip existing,
    # and cap at count. Rejection sampling is cheap enough that we don't parallelize.
    seen: set[tuple[int, int, int]] = set()  # (epsg, col, row)
    tasks: list[
        tuple[Projection, tuple[int, int, int, int], tuple[datetime, datetime]]
    ] = []

    pbar = tqdm.tqdm(total=count, desc="Sampling random China tiles")
    while len(tasks) < count:
        lon = rng.uniform(CHINA_MIN_LON, CHINA_MAX_LON)
        lat = rng.uniform(CHINA_MIN_LAT, CHINA_MAX_LAT)
        if not CHINA_POLYGON.contains(shapely.Point(lon, lat)):
            continue

        epsg, col, row, projection, bounds = _process_sample(lat, lon)

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
        pbar.update(1)
    pbar.close()

    logger.info("Collected %d China tiles (requested %d)", len(tasks), count)

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
