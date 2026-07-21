"""Enqueue random 2048x2048 tiles within sub-Saharan Africa for LCC prediction.

Same idea as ``lcc_model.write_jobs_random_2048`` but instead of a global land check,
samples random (lat, lon) points inside a rough polygon covering a large swath of
sub-Saharan Africa, projects to UTM, and snaps to the 2048-pixel grid (matching
PATCH_SIZE). Each tile is exactly one prediction window. Useful for inspecting LCC
model predictions over Africa and finding annotation gaps.

The polygon is an approximate exterior outline covering (among others): Guinea, Mali,
Burkina Faso, Niger, Sierra Leone, Liberia, Cote d'Ivoire, Ghana, Togo, Benin,
Nigeria, Cameroon, Central African Republic, South Sudan, Ethiopia, Kenya, Uganda,
DRC, Republic of the Congo, Gabon, Angola, Zambia, Rwanda, Burundi, Tanzania, Malawi,
Mozambique, and Zimbabwe. Being a single coarse exterior, it also spills over into
some neighboring countries; that is acceptable for prediction-inspection sampling.
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

# Rough exterior polygon (lon, lat) covering a large swath of sub-Saharan Africa.
# Intentionally approximate; it just needs to keep samples roughly inside the region
# spanning West, Central, East, and Southern Africa. Traced roughly clockwise starting
# from the West African coast.
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
AFRICA_MIN_LON, AFRICA_MIN_LAT, AFRICA_MAX_LON, AFRICA_MAX_LAT = AFRICA_POLYGON.bounds


def _process_sample(
    lat: float,
    lon: float,
) -> tuple[int, int, int, Projection, tuple[int, int, int, int]] | None:
    """Project a lat/lon sample to UTM and snap to the PATCH_SIZE grid.

    Returns None if the UTM projection has no EPSG code.
    """
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

    return (epsg, col, row, projection, bounds)


def write_jobs_random_2048_africa(
    start_time: datetime,
    end_time: datetime,
    out_path: str,
    queue_name: str,
    count: int = 200,
    batch_size: int = 1,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> None:
    """Sample random 2048x2048 tiles in Africa and write prediction jobs to a queue.

    Samples random lat/lon points within a rough sub-Saharan Africa polygon, projects
    each to UTM, snaps to a PATCH_SIZE-aligned grid, deduplicates, skips tiles whose
    output already exists, and writes the remainder to the queue. Each tile gets a
    randomly chosen reference timestamp between start_time and end_time.

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

    pbar = tqdm.tqdm(total=count, desc="Sampling random Africa tiles")
    while len(tasks) < count:
        lon = rng.uniform(AFRICA_MIN_LON, AFRICA_MAX_LON)
        lat = rng.uniform(AFRICA_MIN_LAT, AFRICA_MAX_LAT)
        if not AFRICA_POLYGON.contains(shapely.Point(lon, lat)):
            continue

        result = _process_sample(lat, lon)
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
        pbar.update(1)
    pbar.close()

    logger.info("Collected %d Africa tiles (requested %d)", len(tasks), count)

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
