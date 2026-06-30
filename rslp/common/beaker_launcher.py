"""Launch a Beaker job that executes one or more rslp workflows."""

import uuid
from datetime import datetime

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerImageSource,
)
from beaker.exceptions import BeakerImageNotFound

from rslp.log_utils import get_logger
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WekaMount,
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


def launch_job(
    image: str,
    clusters: list[str] | None = None,
    hostname: str | None = None,
    project: str | None = None,
    workflow: str | None = None,
    extra_args: list[str] = [],
    command: list[str] | None = None,
    task_name: str | None = None,
    gpu_count: int = 0,
    shared_memory: str | None = None,
    priority: str = DEFAULT_PRIORITY,
    task_specific_env_vars: list[BeakerEnvVar] = [],
    budget: str = DEFAULT_BUDGET,
    workspace: str = DEFAULT_WORKSPACE,
    preemptible: bool = True,
    weka_mounts: list[WekaMount] = [],
) -> None:
    """Launch a Beaker job to run an rslp workflow or an arbitrary command.

    The BEAKER_ADDR, BEAKER_CONFIG, and BEAKER_TOKEN environment variables must be
    configured.

    The job's command is either ``command`` (run verbatim) or, if ``command`` is not
    set, the rslp workflow indicated by ``project``/``workflow``/``extra_args``. Exactly
    one of these two modes must be specified.

    Args:
        image: the name of the Beaker image to use. If it doesn't exist, we look for a
            local Docker image matching this name and attempt to register it into
            Beaker.
        clusters: list of Beaker clusters to target. Exactly one of clusters or
            hostname must be set.
        hostname: a specific Beaker host to constrain the job to. Exactly one of
            clusters or hostname must be set.
        project: the rslp project containing the workflow to run (workflow mode).
        workflow: the workflow to run (workflow mode).
        extra_args: a list of arguments for the workflow (workflow mode).
        command: an arbitrary command to run instead of an rslp workflow. If set,
            project/workflow/extra_args are ignored.
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
    if command is None:
        if project is None or workflow is None:
            raise ValueError("must set either command or both project and workflow")
        command = get_command(project, workflow, extra_args)
    if (clusters is None) == (hostname is None):
        raise ValueError("exactly one of clusters or hostname must be set")

    if task_name is None:
        if project is not None and workflow is not None:
            task_name = f"{project}_{workflow}"
        else:
            task_name = "command"

    logger.info("Starting Beaker client...")
    logger.info(f"Workspace: {workspace}")
    with Beaker.from_env(default_workspace=workspace) as beaker:
        logger.info("Getting base env vars...")
        base_env_vars = get_base_env_vars()
        logger.info("Generating task name...")
        task_uuid = str(uuid.uuid4())[0:8]
        unique_task_name = f"{task_name}_{task_uuid}"

        # Check for existing image and create image if it doesn't exist.
        try:
            beaker.image.get(image)
            logger.info(f"Image already exists: {image}")
            image_source = BeakerImageSource(beaker=image)
        except BeakerImageNotFound:
            logger.info(f"Uploading image: {image}")
            # Handle image upload
            image_source = upload_image(image, workspace, beaker)
            logger.info(f"Image uploaded: {image_source.beaker}")

        logger.info("Creating experiment spec...")
        datasets = [create_gcp_credentials_mount()]
        datasets += [weka_mount.to_data_mount() for weka_mount in weka_mounts]
        if hostname is not None:
            constraints = BeakerConstraints(hostname=[hostname])
        else:
            constraints = BeakerConstraints(cluster=clusters)
        experiment_spec = BeakerExperimentSpec.new(
            budget=budget,
            task_name=unique_task_name,
            beaker_image=image_source.beaker,
            result_path="/models",
            priority=priority,
            constraints=constraints,
            command=command,
            env_vars=base_env_vars + task_specific_env_vars,
            datasets=datasets,
            resources={"gpuCount": gpu_count, "sharedMemory": shared_memory},
            preemptible=preemptible,
        )
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{task_name}_{task_uuid}_{current_time}"
        logger.info(f"Creating experiment: {experiment_name}")
        beaker.experiment.create(name=experiment_name, spec=experiment_spec)
