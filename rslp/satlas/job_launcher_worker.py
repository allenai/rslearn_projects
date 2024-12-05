"""Launch Satlas prediction jobs on Beaker."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import shapely
import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
    TaskResources,
)
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_proj_bounds
from upath import UPath

from rslp.launch_beaker import BUDGET, DEFAULT_WORKSPACE, IMAGE_NAME, get_base_env_vars
from rslp.log_utils import get_logger

from .predict_pipeline import Application, PredictTaskArgs

logger = get_logger(__name__)

TILE_SIZE = 32768
RESOLUTION = 10

# Days to add before a provided date.
DAYS_BEFORE = 120

# Days to add after a provided date.
DAYS_AFTER = 90


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


class WorkerParams:
    """Parameters that worker pipeline needs to know."""

    def __init__(self, job_fname: str, claim_bucket_name: str, claim_dir: str) -> None:
        """Create a new WorkerParams.

        Args:
            job_fname: the filename containing list of jobs.
            claim_bucket_name: the bucket where workers will claim jobs.
            claim_dir: the path in the bucket to write claim files.
        """
        self.job_fname = job_fname
        self.claim_bucket_name = claim_bucket_name
        self.claim_dir = claim_dir


def launch_worker(worker_params: WorkerParams) -> None:
    """Launch a worker job.

    Args:
        worker_params: the parameters to pass to the worker.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    with beaker.session():
        env_vars = get_base_env_vars(use_weka_prefix=False)
        env_vars.append(
            EnvVar(
                name="RSLEARN_LOGLEVEL",
                value="DEBUG",
            )
        )

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description="worker",
            beaker_image=IMAGE_NAME,
            priority=Priority.low,
            command=["python", "-m", "rslp.main"],
            arguments=[
                "common",
                "worker",
                "satlas",
                "predict_multi",
                worker_params.job_fname,
                worker_params.claim_bucket_name,
                worker_params.claim_dir,
            ],
            constraints=Constraints(
                cluster=[
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                    "ai2/saturn-cirrascale",
                    "ai2/augusta-google-1",
                ]
            ),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),  # nosec
                    mount_path="/etc/credentials/gcp_credentials.json",  # nosec
                ),
            ],
            env_vars=env_vars,
            resources=TaskResources(gpu_count=1, shared_memory="256GiB"),
        )
        unique_id = str(uuid.uuid4())[0:8]
        beaker.experiment.create(f"worker_{unique_id}", spec)


def get_jobs(
    application: Application,
    time_range: tuple[datetime, datetime],
    out_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
) -> list[list[str]]:
    """Get batches of tasks for Satlas prediction.

    Args:
        application: which application to run.
        time_range: the time range to run within. Must have timezone.
        out_path: the output path. It should be specific to the time range.
        epsg_code: limit tasks to this UTM zone (specified by its EPSG code), default
            run in all UTM zones.
        wgs84_bounds: limit tasks to ones that intersect these WGS84 bounds.
        batch_size: how many tasks to run in each batch.

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
        # get_proj_bounds returns bounds in CRS units so we need to convert to pixel
        # units.
        crs_bbox = STGeometry(
            Projection(utm_zone, 1, 1),
            shapely.box(*get_proj_bounds(utm_zone)),
            None,
        )
        projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
        pixel_bbox = crs_bbox.to_projection(projection)
        zone_bounds = tuple(int(value) for value in pixel_bbox.shp.bounds)

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

        for col in range(zone_bounds[0] // TILE_SIZE, zone_bounds[2] // TILE_SIZE + 1):
            for row in range(
                zone_bounds[1] // TILE_SIZE, zone_bounds[3] // TILE_SIZE + 1
            ):
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

    print(f"Got {len(tasks)} total tasks")

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
            application.value.upper(),
            out_path,
            "/tmp/scratch/",
            json.dumps([predict_task.serialize() for predict_task in predict_tasks]),
        ]
        jobs.append(cur_args)

    return jobs


def write_jobs(
    application: Application,
    time_range: tuple[datetime, datetime],
    out_path: str,
    job_fname: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
) -> None:
    """Write jobs for the specified application and time range.

    Args:
        application: which application to run.
        time_range: the time range to run within. Must have timezone.
        out_path: the output path. It should be specific to the time range.
        job_fname: where to write the list of jobs for workers to read.
        epsg_code: limit tasks to this UTM zone (specified by its EPSG code), default
            run in all UTM zones.
        wgs84_bounds: limit tasks to ones that intersect these WGS84 bounds.
        batch_size: how many tasks to run in each batch.
    """
    jobs = get_jobs(
        application,
        time_range,
        out_path,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        batch_size=batch_size,
    )
    with UPath(job_fname).open("w") as f:
        json.dump(jobs, f)


def write_jobs_for_year_months(
    year_months: list[tuple[int, int]],
    application: Application,
    out_path: str,
    job_fname: str,
    batch_size: int = 1,
) -> None:
    """Write Satlas prediction jobs for the given year and month.

    Args:
        year_months: list of year-month pairs.
        application: the application to run.
        out_path: the output path with year and month placeholders.
        job_fname: where to write the list of jobs for workers to read.
        worker_params: the worker parameters.
        batch_size: the batch size.
    """
    jobs = []
    for year, month in year_months:
        ts = datetime(year, month, 1, tzinfo=timezone.utc)
        time_range = (
            ts - timedelta(days=DAYS_BEFORE),
            ts + timedelta(days=DAYS_AFTER),
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
        )
        logger.info("got %d jobs for %04d-%02d", len(cur_jobs), year, month)
        jobs.extend(cur_jobs)

    logger.info("got a total of %d jobs across year-months", len(jobs))
    with UPath(job_fname).open("w") as f:
        json.dump(jobs, f)


def launch_workers(worker_params: WorkerParams, num_workers: int) -> None:
    """Start workers for the prediction jobs.

    Args:
        worker_params: the parameters for the workers, including job file where the
            list of jobs has been written.
        num_workers: number of workers to launch
    """
    for _ in tqdm.tqdm(range(num_workers)):
        launch_worker(worker_params)
