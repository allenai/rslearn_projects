"""Launch Satlas prediction jobs on Beaker."""

import json
import multiprocessing
import os
import random
import uuid
from datetime import datetime

import rslearn.utils.get_utm_ups_crs
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

from .predict_pipeline import Application, get_output_fname

WORKSPACE = "ai2/earth-systems"
BUDGET = "ai2/d5"
IMAGE_NAME = "favyen/rslearn"
TILE_SIZE = 16384
RESOLUTION = 10


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


def launch_job(task: Task) -> None:
    """Launch job for this task.

    Args:
        task: the Task object for which to create a job.
    """
    beaker = Beaker.from_env(default_workspace=WORKSPACE)

    with beaker.session():
        env_vars = [
            EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
                value="/etc/credentials/gcp_credentials.json",  # nosec
            ),
            EnvVar(
                name="GCLOUD_PROJECT",  # nosec
                value="skylight-proto-1",  # nosec
            ),
            EnvVar(
                name="RSLP_BUCKET",
                value=os.environ["RSLP_BUCKET"],
            ),
            EnvVar(
                name="MKL_THREADING_LAYER",
                value="GNU",
            ),
        ]

        experiment_name = (
            f"satlas_{task.application.value}_{str(task.projection.crs)}_"
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
                "predict",
                task.application,
                json.dumps(task.projection.serialize()),
                json.dumps(task.bounds),
                json.dumps(
                    [task.time_range[0].isoformat(), task.time_range[1].isoformat()]
                ),
                task.out_path,
                "/tmp/scratch/",
            ],
            constraints=Constraints(
                cluster=[
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                    "ai2/saturn-cirrascale",
                    "ai2/pluto-cirrascale",
                    "ai2/general-cirrascale",
                    "ai2/prior-cirrascale",
                    "ai2/prior-elanding",
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
            resources=TaskResources(gpu_count=1),
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
    wgs84_bounds: PixelBounds | None = None,
    count: int | None = None,
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
    """
    # Generate tasks.
    if epsg_code:
        utm_zones = [CRS.from_epsg(epsg_code)]
    else:
        for epsg_code in range(32601, 32661):
            utm_zones.append(CRS.from_epsg(epsg_code))
        for epsg_code in range(32701, 32761):
            utm_zones.append(CRS.from_epsg(epsg_code))

    tasks: list[Task] = []
    for utm_zone in utm_zones:
        zone_bounds = rslearn.utils.get_utm_ups_crs.get_proj_bounds(utm_zone)
        projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
        for col in range(zone_bounds[0], zone_bounds[2], TILE_SIZE):
            for row in range(zone_bounds[1], zone_bounds[3], TILE_SIZE):
                if wgs84_bounds is not None:
                    # Check if the longitude/latitude of this task is in wgs84_bounds.
                    src_geom = STGeometry(projection, shapely.Point(col, row), None)
                    wgs84_point = src_geom.to_projection(WGS84_PROJECTION).shp
                    if wgs84_point.x < wgs84_bounds[0]:
                        continue
                    if wgs84_point.x >= wgs84_bounds[2]:
                        continue
                    if wgs84_point.y < wgs84_bounds[1]:
                        continue
                    if wgs84_point.y >= wgs84_bounds[3]:
                        continue

                tasks.append(
                    Task(
                        application=application,
                        projection=projection,
                        bounds=(col, row, col + TILE_SIZE, row + TILE_SIZE),
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
    for task in tqdm.tqdm(run_tasks, desc="Starting Beaker jobs"):
        launch_job(task)
