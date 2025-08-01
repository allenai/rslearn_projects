"""Launch Satlas prediction jobs on Beaker."""

import json
import random
from collections.abc import Generator
from datetime import datetime, timedelta, timezone

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_proj_bounds
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .predict_pipeline import Application, PredictTaskArgs, get_output_fname

logger = get_logger(__name__)

TILE_SIZE = 32768
RESOLUTION = 10

# Days to add before a provided date.
DEFAULT_DAYS_BEFORE = 120

# Days to add after a provided date.
DEFAULT_DAYS_AFTER = 90


class Task:
    """Represents a task that processes one tile at one point in time."""

    def __init__(
        self,
        application: Application,
        projection: Projection,
        bounds: PixelBounds,
        time_range: tuple[datetime, datetime],
        out_path: str,
    ) -> None:
        """Create a new Task.

        Args:
            application: the application to run
            projection: the projection of the tile
            bounds: the bounds of the tile
            time_range: the time range to process
            out_path: where to write outputs
        """
        self.application = application
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.out_path = out_path

    def get_output_fname(self) -> UPath:
        """Get the output filename that will be used for this task."""
        # The filename format is defined by get_output_fname in predict_pipeline.py.
        return get_output_fname(
            self.application, self.out_path, self.projection, self.bounds
        )


def enumerate_tiles_in_zone(utm_zone: CRS) -> Generator[tuple[int, int], None, None]:
    """List all of the tiles in the zone where outputs should be computed.

    The tiles are all TILE_SIZE x TILE_SIZE so only the column/row of the tile along
    that grid are returned.

    Args:
        utm_zone: the CRS which must correspond to a UTM EPSG.

    Returns:
        generator of (column, row) of the tiles that are needed.
    """
    # We use get_proj_bounds to get the bounds of the UTM zone in CRS units.
    # We then convert to pixel units in order to determine the tiles that are needed.
    crs_bbox = STGeometry(
        Projection(utm_zone, 1, 1),
        shapely.box(*get_proj_bounds(utm_zone)),
        None,
    )
    projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
    pixel_bbox = crs_bbox.to_projection(projection)

    # Convert the resulting shape to integer bbox.
    zone_bounds = tuple(int(value) for value in pixel_bbox.shp.bounds)

    for col in range(zone_bounds[0] // TILE_SIZE, zone_bounds[2] // TILE_SIZE + 1):
        for row in range(zone_bounds[1] // TILE_SIZE, zone_bounds[3] // TILE_SIZE + 1):
            yield (col, row)


def get_jobs(
    application: Application,
    time_range: tuple[datetime, datetime],
    out_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
) -> list[list[str]]:
    """Get batches of tasks for Satlas prediction.

    Tasks where outputs have already been computed are excluded.

    Args:
        application: which application to run.
        time_range: the time range to run within. Must have timezone.
        out_path: the output path. It should be specific to the time range.
        epsg_code: limit tasks to this UTM zone (specified by its EPSG code), default
            run in all UTM zones.
        wgs84_bounds: limit tasks to ones that intersect these WGS84 bounds.
        batch_size: how many tasks to run in each batch.
        count: limit to this many tasks.

    Returns:
        the list of worker tasks where each worker task
    """
    # Generate tasks.
    if epsg_code:
        utm_zones = [CRS.from_epsg(epsg_code)]
    else:
        utm_zones = []
        for epsg_code in range(32601, 32661):
            utm_zones.append(CRS.from_epsg(epsg_code))
        for epsg_code in range(32701, 32761):
            utm_zones.append(CRS.from_epsg(epsg_code))

    tasks: list[Task] = []
    for utm_zone in tqdm.tqdm(utm_zones, desc="Enumerating tasks across UTM zones"):
        projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)

        # If the user provided WGS84 bounds, then we convert it to pixel coordinates so
        # we can check each tile easily.
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
                # Check if this task intersects the bounds specified by the user.
                if (col + 1) * TILE_SIZE < user_bounds_in_proj[0]:
                    continue
                if col * TILE_SIZE >= user_bounds_in_proj[2]:
                    continue
                if (row + 1) * TILE_SIZE < user_bounds_in_proj[1]:
                    continue
                if row * TILE_SIZE >= user_bounds_in_proj[3]:
                    continue

            tasks.append(
                Task(
                    application=application,
                    projection=projection,
                    bounds=(
                        col * TILE_SIZE,
                        row * TILE_SIZE,
                        (col + 1) * TILE_SIZE,
                        (row + 1) * TILE_SIZE,
                    ),
                    time_range=time_range,
                    out_path=out_path,
                )
            )

    logger.info("Got %d total tasks", len(tasks))

    # Remove tasks where outputs are already computed.
    out_upath = UPath(out_path)
    if out_upath.exists():
        existing_output_fnames = {
            out_fname.name for out_fname in UPath(out_path).iterdir()
        }
        tasks = [
            task
            for task in tasks
            if task.get_output_fname().name not in existing_output_fnames
        ]
    logger.info("Got %d tasks that are uncompleted", len(tasks))

    # Sample tasks down to user-provided count (max # tasks to run), if provided.
    if count is not None and len(tasks) > count:
        tasks = random.sample(tasks, count)
        logger.info("Randomly sampled %d tasks", len(tasks))

    # Convert tasks to jobs for use with rslp.common.worker.
    # This is what will be written to the Pub/Sub topic.
    jobs = []
    for i in range(0, len(tasks), batch_size):
        cur_tasks = tasks[i : i + batch_size]

        # Get list of PredictTaskArgs that we can serialize.
        # These just specify the projection, time range, and bounds.
        predict_tasks = []
        for task in cur_tasks:
            predict_tasks.append(
                PredictTaskArgs(
                    projection_json=task.projection.serialize(),
                    bounds=task.bounds,
                    time_range=task.time_range,
                )
            )

        cur_args = [
            "--application",
            application.value.upper(),
            "--out_path",
            out_path,
            "--scratch_path",
            "/tmp/scratch/",
            "--tasks",
            json.dumps([predict_task.serialize() for predict_task in predict_tasks]),
        ]
        jobs.append(cur_args)

    return jobs


def write_jobs(
    application: Application,
    time_range: tuple[datetime, datetime],
    out_path: str,
    queue_name: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
) -> None:
    """Write jobs for the specified application and time range.

    Args:
        application: which application to run.
        time_range: the time range to run within. Must have timezone.
        out_path: the output path. It should be specific to the time range.
        queue_name: the Beaker queue to write the job entries to.
        epsg_code: limit tasks to this UTM zone (specified by its EPSG code), default
            run in all UTM zones.
        wgs84_bounds: limit tasks to ones that intersect these WGS84 bounds.
        batch_size: how many tasks to run in each batch.
        count: limit to this many tasks.
    """
    jobs = get_jobs(
        application,
        time_range,
        out_path,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        batch_size=batch_size,
        count=count,
    )
    rslp.common.worker.write_jobs(queue_name, "satlas", "predict_multi", jobs)


def write_jobs_for_year_months(
    year_months: list[tuple[int, int]],
    application: Application,
    out_path: str,
    queue_name: str,
    batch_size: int = 1,
    days_before: int = DEFAULT_DAYS_BEFORE,
    days_after: int = DEFAULT_DAYS_AFTER,
    count: int | None = None,
) -> None:
    """Write Satlas prediction jobs for the given year and month.

    Args:
        year_months: list of year-month pairs.
        application: the application to run.
        out_path: the output path with year and month placeholders.
        queue_name: the Beaker queue to write the job entries to.
        worker_params: the worker parameters.
        batch_size: the batch size.
        days_before: how much to pad windows before the year/month.
        days_after: how much to pad windows after the year/month.
        count: limit each year-month to this many tasks.
    """
    jobs = []
    for year, month in year_months:
        ts = datetime(year, month, 1, tzinfo=timezone.utc)
        time_range = (
            ts - timedelta(days=days_before),
            ts + timedelta(days=days_after),
        )
        cur_out_path = out_path.format(year=year, month=month)
        logger.info(
            f"collecting jobs for year={year}, month={month}, time_range={time_range}, out_path={cur_out_path}"
        )
        cur_jobs = get_jobs(
            application=application,
            time_range=time_range,
            out_path=cur_out_path,
            batch_size=batch_size,
            count=count,
        )
        logger.info("got %d jobs for %04d-%02d", len(cur_jobs), year, month)
        jobs.extend(cur_jobs)

    logger.info("got a total of %d jobs across year-months", len(jobs))
    rslp.common.worker.write_jobs(queue_name, "satlas", "predict_multi", jobs)
