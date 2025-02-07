"""Launch a Beaker job that executes one or more rslp workflows."""

import uuid
from dataclasses import dataclass
from datetime import datetime

from beaker import Beaker, DataMount, DataSource, EnvVar, ExperimentSpec, ImageSource
from beaker.exceptions import ImageNotFound

from rslp.log_utils import get_logger
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    create_gcp_credentials_mount,
    get_base_env_vars,
    upload_image,
)

logger = get_logger(__name__)

# The default priority for jobs launched in this way is high since there is expected to
# be only a small number of jobs. For launching many jobs in parallel,
# `rslp.common.worker` would likely be used instead.
DEFAULT_PRIORITY = "high"


def get_command(project: str, workflow: str, extra_args: list[str]) -> list[str]:
    """Get the command that would run the specified workflow(s).

    Args:
        project: The project to execute a workflow for.
        workflow: the workflow to run.
        extra_args: list of arguments to pass to the workflow.
    """
    return ["python", "-m", "rslp.main", project, workflow] + extra_args


@dataclass
class WekaMount:
    """Specification of a Weka mount within a Beaker job."""

    bucket_name: str
    mount_path: str
    sub_path: str | None = None

    def to_data_mount(self) -> DataMount:
        """Convert this WekaMount to a Beaker DataMount object."""
        return DataMount(
            source=DataSource(weka=self.bucket_name),
            mount_path=self.mount_path,
            sub_path=self.sub_path,
        )


def launch_job(
    project: str,
    workflow: str,
    extra_args: list[str],
    image: str,
    clusters: list[str],
    task_name: str | None,
    gpu_count: int = 0,
    shared_memory: str | None = None,
    priority: str = DEFAULT_PRIORITY,
    task_specific_env_vars: list[EnvVar] = [],
    budget: str = DEFAULT_BUDGET,
    workspace: str = DEFAULT_WORKSPACE,
    preemptible: bool = True,
    weka_mounts: list[WekaMount] = [],
) -> None:
    """Launch a Beaker job to run an rslp workflow.

    The BEAKER_ADDR, BEAKER_CONFIG, and BEAKER_TOKEN environment variables must be
    configured.

    Args:
        project: the rslp project containing the workflow(s) to run.
        workflow: the workflow to run.
        extra_args: a list of arguments for the workflow.
        image: the name of the Beaker image to use. If it doesn't exist, we look for a
            local Docker image matching this name and attempt to register it into
            Beaker.
        clusters: list of Beaker clusters to target.
        task_name: name for the Beaker job.
        gpu_count: number of GPUs to assign.
        shared_memory: amount of shared memory.
        priority: priority of the Beaker job.
        task_specific_env_vars: additional task-specific environment variables to pass
            to the Beaker job.
        budget: the Beaker budget.
        workspace: the Beaker workspace.
        preemptible: whether to make the Beaker job preemptible.
        weka_mounts: list of weka mounts for Beaker job.
    """
    if task_name is None:
        task_name = f"{project}_{workflow}"

    logger.info("Starting Beaker client...")
    logger.info(f"Workspace: {workspace}")
    beaker = Beaker.from_env(default_workspace=workspace)
    with beaker.session():
        logger.info("Getting base env vars...")
        base_env_vars = get_base_env_vars()
        logger.info("Generating task name...")
        task_uuid = str(uuid.uuid4())[0:8]
        unique_task_name = f"{task_name}_{task_uuid}"

        # Check for existing image and create image if it doesn't exist.
        try:
            beaker.image.get(image)
            logger.info(f"Image already exists: {image}")
            image_source = ImageSource(beaker=image)
        except ImageNotFound:
            logger.info(f"Uploading image: {image}")
            # Handle image upload
            image_source = upload_image(image, workspace, beaker)
            logger.info(f"Image uploaded: {image_source.beaker}")

        logger.info("Creating experiment spec...")
        datasets = [create_gcp_credentials_mount()]
        datasets += [weka_mount.to_data_mount() for weka_mount in weka_mounts]
        experiment_spec = ExperimentSpec.new(
            budget=budget,
            task_name=unique_task_name,
            beaker_image=image_source.beaker,
            result_path="/models",
            priority=priority,
            cluster=clusters,
            command=get_command(project, workflow, extra_args),
            env_vars=base_env_vars + task_specific_env_vars,
            datasets=datasets,
            resources={"gpuCount": gpu_count, "sharedMemory": shared_memory},
            preemptible=preemptible,
        )
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{task_name}_{task_uuid}_{current_time}"
        logger.info(f"Creating experiment: {experiment_name}")
        beaker.experiment.create(experiment_name, experiment_spec)
