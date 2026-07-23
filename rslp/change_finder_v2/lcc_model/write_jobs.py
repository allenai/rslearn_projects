"""Enqueue change_finder_v2 LCC prediction jobs on a Beaker queue.

The world is divided into TILE_SIZE x TILE_SIZE UTM tiles. Each tile becomes one
prediction task for a single user-provided reference timestamp. Tasks are batched and
written to a Beaker queue, where they are processed by rslp.common workers running the
``predict_multi`` workflow.

The tile size is fixed to 32768x32768 here; the prediction pipeline itself accepts any
tile size that is a multiple of PATCH_SIZE.
"""

import json
import random
from collections.abc import Generator
from datetime import datetime

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_proj_bounds
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .predict_pipeline import (
    DEFAULT_THRESHOLD,
    RESOLUTION,
    PredictTaskArgs,
    get_output_fname,
)

logger = get_logger(__name__)

# Fixed tile size for this job-writer (the prediction pipeline supports any multiple of
# PATCH_SIZE).
TILE_SIZE = 32768


def enumerate_tiles_in_zone(utm_zone: CRS) -> Generator[tuple[int, int], None, None]:
    """List the (column, row) of all TILE_SIZE tiles within a UTM zone.

    Args:
        utm_zone: the CRS which must correspond to a UTM EPSG.

    Returns:
        generator of (column, row) of the tiles that are needed.
    """
    crs_bbox = STGeometry(
        Projection(utm_zone, 1, 1),
        shapely.box(*get_proj_bounds(utm_zone)),
        None,
    )
    projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
    pixel_bbox = crs_bbox.to_projection(projection)
    zone_bounds = tuple(int(value) for value in pixel_bbox.shp.bounds)

    for col in range(zone_bounds[0] // TILE_SIZE, zone_bounds[2] // TILE_SIZE + 1):
        for row in range(zone_bounds[1] // TILE_SIZE, zone_bounds[3] // TILE_SIZE + 1):
            yield (col, row)


def get_jobs(
    timestamp: datetime,
    out_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[list[str]]:
    """Get batches of prediction tasks.

    Tasks whose outputs already exist are excluded.

    Args:
        timestamp: the reference timestamp. Must have timezone.
        out_path: the output directory.
        epsg_code: limit tasks to this UTM zone (EPSG code); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        batch_size: how many tasks to run per worker job.
        count: limit to this many tasks (randomly sampled).
        write_raster: whether workers should also write the merged raster.
        threshold: binary change probability threshold (0-255) for polygonization.

    Returns:
        a list of worker argument lists, one per batch of TILE_SIZE tiles.
    """
    if epsg_code:
        utm_zones = [CRS.from_epsg(epsg_code)]
    else:
        utm_zones = [CRS.from_epsg(code) for code in range(32601, 32661)]
        utm_zones += [CRS.from_epsg(code) for code in range(32701, 32761)]

    time_range = (timestamp, timestamp)

    tasks: list[tuple[Projection, PixelBounds]] = []
    for utm_zone in tqdm.tqdm(utm_zones, desc="Enumerating tasks across UTM zones"):
        projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)

        user_bounds_in_proj: PixelBounds | None = None
        if wgs84_bounds is not None:
            dst_geom = STGeometry(
                WGS84_PROJECTION, shapely.box(*wgs84_bounds), None
            ).to_projection(projection)
            user_bounds_in_proj = (
                int(dst_geom.shp.bounds[0]),
                int(dst_geom.shp.bounds[1]),
                int(dst_geom.shp.bounds[2]),
                int(dst_geom.shp.bounds[3]),
            )

        for col, row in enumerate_tiles_in_zone(utm_zone):
            if user_bounds_in_proj is not None:
                if (col + 1) * TILE_SIZE < user_bounds_in_proj[0]:
                    continue
                if col * TILE_SIZE >= user_bounds_in_proj[2]:
                    continue
                if (row + 1) * TILE_SIZE < user_bounds_in_proj[1]:
                    continue
                if row * TILE_SIZE >= user_bounds_in_proj[3]:
                    continue

            bounds = (
                col * TILE_SIZE,
                row * TILE_SIZE,
                (col + 1) * TILE_SIZE,
                (row + 1) * TILE_SIZE,
            )
            tasks.append((projection, bounds))

    logger.info("Got %d total tasks", len(tasks))

    # Remove tasks where outputs are already computed.
    out_upath = UPath(out_path)
    if out_upath.exists():
        existing_output_fnames = {fname.name for fname in out_upath.iterdir()}
        tasks = [
            (projection, bounds)
            for projection, bounds in tasks
            if get_output_fname(out_path, projection, bounds).name
            not in existing_output_fnames
        ]
    logger.info("Got %d tasks that are uncompleted", len(tasks))

    # Sample down to count if requested.
    if count is not None and len(tasks) > count:
        tasks = random.sample(tasks, count)
        logger.info("Randomly sampled %d tasks", len(tasks))

    # Convert tasks to batched worker jobs.
    jobs = []
    for i in range(0, len(tasks), batch_size):
        cur_tasks = tasks[i : i + batch_size]
        predict_tasks = [
            PredictTaskArgs(
                projection_json=projection.serialize(),
                bounds=bounds,
                time_range=time_range,
            )
            for projection, bounds in cur_tasks
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

    return jobs


def write_jobs(
    timestamp: datetime,
    out_path: str,
    queue_name: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> None:
    """Enumerate tiles for one reference timestamp and write jobs to a Beaker queue.

    Args:
        timestamp: the reference timestamp. Must have timezone.
        out_path: the output directory.
        queue_name: the Beaker queue to write the job entries to.
        epsg_code: limit tasks to this UTM zone (EPSG code); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        batch_size: how many tasks to run per worker job.
        count: limit to this many tasks (randomly sampled).
        write_raster: whether workers should also write the merged raster.
        threshold: binary change probability threshold (0-255) for polygonization.
    """
    jobs = get_jobs(
        timestamp=timestamp,
        out_path=out_path,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        batch_size=batch_size,
        count=count,
        write_raster=write_raster,
        threshold=threshold,
    )
    # Shuffle so outputs start appearing from random parts of the world (aids debugging).
    random.shuffle(jobs)
    rslp.common.worker.write_jobs(queue_name, "change_finder_v2", "predict_multi", jobs)
