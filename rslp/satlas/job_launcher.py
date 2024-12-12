"""Launch Satlas prediction jobs on Beaker."""

import json
import multiprocessing
import random
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

from rslp.launch_beaker import BUDGET, DEFAULT_WORKSPACE, IMAGE_NAME, get_base_env_vars

from .predict_pipeline import Application, PredictTaskArgs, get_output_fname

TILE_SIZE = 32768
RESOLUTION = 10

# Days to add before a provided date.
DEFAULT_DAYS_BEFORE = 120

# Days to add after a provided date.
DEFAULT_DAYS_AFTER = 90


class Task:
    """Represents a task that will correspond to one Beaker job."""

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


def launch_job(batch: list[Task]) -> None:
    """Launch job for this task.

    Args:
        batch: list of Task objects for which to create a job.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    # Convert tasks to PredictTask.
    # These just set projection/bounds/time range, so the application and output path
    # come from the first task.
    predict_tasks = []
    for task in batch:
        predict_tasks.append(
            PredictTaskArgs(
                projection_json=task.projection.serialize(),
                bounds=task.bounds,
                time_range=task.time_range,
            )
        )

    with beaker.session():
        env_vars = get_base_env_vars(use_weka_prefix=False)
        env_vars.append(
            EnvVar(
                name="RSLEARN_LOGLEVEL",
                value="DEBUG",
            )
        )

        # Name the job based on the first task.
        task = batch[0]
        experiment_name = (
            f"satlas_{task.application.value}_{task.projection.crs.to_epsg()}_"
            + f"{task.bounds[0]}_{task.bounds[1]}"
        )

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=experiment_name,
            beaker_image=IMAGE_NAME,
            priority=Priority.low,
            command=["python", "-m", "rslp.main"],
            arguments=[
                "satlas",
                "predict_multi",
                task.application.value.upper(),
                task.out_path,
                "/tmp/scratch/",
                json.dumps(
                    [predict_task.serialize() for predict_task in predict_tasks]
                ),
            ],
            constraints=Constraints(
                cluster=[
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                    "ai2/saturn-cirrascale",
                    "ai2/augusta-google-1",
                    # "ai2/prior-cirrascale",
                    # "ai2/prior-elanding",
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
        beaker.experiment.create(experiment_name + "_" + unique_id, spec)


def check_task_done(task: Task) -> tuple[Task, bool]:
    """Checks whether this task is done processing already.

    It is determined based on existence of output file for the task.

    Args:
        task: the task.

    Returns:
        whether the task was completed
    """
    out_fname = get_output_fname(
        task.application, task.out_path, task.projection, task.bounds
    )
    return task, out_fname.exists()


def launch_jobs(
    application: Application,
    time_range: tuple[datetime, datetime],
    out_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    count: int | None = None,
    batch_size: int = 1,
) -> None:
    """Launch Beaker jobs for Satlas prediction.

    Args:
        application: which application to run.
        time_range: the time range to run within. Must have timezone.
        out_path: the output path. It should be specific to the time range.
        epsg_code: limit tasks to this UTM zone (specified by its EPSG code), default
            run in all UTM zones.
        wgs84_bounds: limit tasks to ones that intersect these WGS84 bounds.
        count: only run up to this many tasks.
        batch_size: how many tasks to run in each Beaker job.
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

    # See which tasks are not done yet.
    p = multiprocessing.Pool(32)
    outputs = p.imap_unordered(check_task_done, tasks)

    pending_tasks: list[Task] = []
    for task, is_done in tqdm.tqdm(
        outputs, desc="Check which tasks are completed", total=len(tasks)
    ):
        if is_done:
            continue
        pending_tasks.append(task)

    p.close()

    # Run up to count of them.
    if count is not None and len(pending_tasks) > count:
        run_tasks = random.sample(pending_tasks, count)
    else:
        run_tasks = pending_tasks

    print(
        f"Got {len(tasks)} total tasks, {len(pending_tasks)} pending, running {len(run_tasks)} of them"
    )

    batches = []
    for i in range(0, len(run_tasks), batch_size):
        batches.append(run_tasks[i : i + batch_size])

    for batch in tqdm.tqdm(batches, desc="Starting Beaker jobs"):
        launch_job(batch)


def launch_jobs_for_year_month(
    year: int,
    month: int,
    application: Application,
    out_path: str,
    batch_size: int = 1,
    count: int | None = None,
    days_before: int = DEFAULT_DAYS_BEFORE,
    days_after: int = DEFAULT_DAYS_AFTER,
) -> None:
    """Launch Satlas prediction jobs on Beaker for the given year and month.

    Args:
        year: the year.
        month: the month.
        application: the application to run.
        out_path: the output path with year and month placeholders.
        batch_size: the batch size.
        count: only run up to this many tasks.
        days_before: how much to pad windows before the year/month.
        days_after: how much to pad windows after the year/month.
    """
    ts = datetime(year, month, 1, tzinfo=timezone.utc)
    time_range = (
        ts - timedelta(days=days_before),
        ts + timedelta(days=days_after),
    )
    cur_out_path = out_path.format(year=year, month=month)
    print(f"launching jobs with time_range={time_range} and out_path={cur_out_path}")
    launch_jobs(
        application=application,
        time_range=time_range,
        out_path=cur_out_path,
        batch_size=batch_size,
        count=count,
    )
