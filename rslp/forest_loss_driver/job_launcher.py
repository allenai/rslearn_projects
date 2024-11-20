"""Launch a Beaker Job for forest loss driver prediction.

First get something working then think about how to make integrated shared code

"""

import argparse
import os
import uuid
from datetime import datetime

import dotenv
from beaker import Beaker, EnvVar, ExperimentSpec, ImageSource
from beaker.exceptions import ImageNotFound
from rslp import launcher_lib
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Should be in a shared defaults folder
DEFAULT_WORKSPACE = "ai2/earth-systems"


def get_inference_job_command(project: str, workflow: str) -> list[str]:
    """Get the command for the inference job.

    Args:
        project: The project to execute a workflow for.
        workflow: The name of the workflow.
    """
    return ["python", "-m", "rslp.main", project, workflow]


# Must Auth with     ADDRESS_KEY: ClassVar[str] = "BEAKER_ADDR" CONFIG_PATH_KEY: ClassVar[str] = "BEAKER_CONFIG" TOKEN_KEY: ClassVar[str] = "BEAKER_TOKEN"
# Maybe allow setting a group for this to keep these tasks  orgnaized on beake
def launch_job(
    project: str,
    workflow: str,
    image: str,
    gpu_count: int,
    shared_memory: str,
    cluster: list,
    priority: str,
    task_name: str,
    task_specific_env_vars: list[EnvVar],
    budget: str,
    workspace: str,
) -> None:
    beaker = Beaker.from_env(default_workspace=workspace)
    with beaker.session():
        logger.info("Starting Beaker client...")
        logger.info(f"Workspace: {workspace}")
        logger.info("Getting base env vars...")
        base_env_vars = launcher_lib.get_base_env_vars()
        logger.info("Generating task name...")
        task_uuid = str(uuid.uuid4())[0:8]
        task_name = f"{task_name}_{task_uuid}"
        try:
            beaker.image.get(image)
            logger.info(f"Image already exists: {image}")
            image_source = ImageSource(beaker=image)
        except ImageNotFound:
            logger.info(f"Uploading image: {image}")
            # Handle image upload
            image_source = launcher_lib.upload_image(image, workspace, beaker)
            logger.info(f"Image uploaded: {image_source.beaker}")
        # Potentially we might want to have many different tasks as part of a job but this is very simple for now
        logger.info("Creating experiment spec...")
        experiment_spec = ExperimentSpec.new(
            budget=budget,
            task_name=task_name,
            beaker_image=image_source.beaker,
            result_path="/models",
            priority=priority,
            cluster=cluster,
            command=get_inference_job_command(project, workflow),
            env_vars=base_env_vars + task_specific_env_vars,
            datasets=[launcher_lib.create_gcp_credentials_mount()],
            resources={"gpuCount": gpu_count, "sharedMemory": shared_memory},
            preemptible=True,
        )
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{task_name}_{task_uuid}_{current_time}"
        logger.info(f"Creating experiment: {experiment_name}")
        beaker.experiment.create(experiment_name, experiment_spec)


if __name__ == "__main__":
    dotenv.load_dotenv()

    # Ensure that RSLP_Prefix is set to get base nev vars
    if "RSLP_PREFIX" not in os.environ:
        raise ValueError("RSLP_PREFIX is not set")

    # i want to be able to specify project andd workflow here
    parser = argparse.ArgumentParser(
        description="Launch Inference Job on Beaker",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="The project to execute a workflow for",
        required=True,
    )
    parser.add_argument(
        "--workflow",
        type=str,
        help="The name of the workflow to execute",
        required=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Docker image to use for the job",
        required=True,
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        help="Number of GPUs to allocate",
        required=True,
    )
    parser.add_argument(
        "--shared_memory",
        type=str,
        help="Amount of shared memory to allocate",
        required=True,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        nargs="+",
        help="List of clusters to run on",
        required=True,
    )
    parser.add_argument(
        "--priority",
        type=str,
        help="Priority of the task",
        required=True,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the task",
        required=True,
    )
    parser.add_argument(
        "--task_specific_env_vars",
        type=lambda x: x.split(","),
        nargs="+",
        help="List of task-specific environment variables in the format: NAME=VALUE,NAME=VALUE",
        required=False,
        default=[],
    )
    parser.add_argument(
        "--budget",
        type=str,
        help="Budget for the experiment",
        required=True,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Beaker workspace to run the job in",
        required=False,
        default=DEFAULT_WORKSPACE,
    )
    args = parser.parse_args()

    task_specific_env_vars = []
    # Note make a single funciton that does this parsing on pass in directly
    if args.task_specific_env_vars:
        task_specific_env_vars = [
            EnvVar(name=var.split("=")[0], value=var.split("=")[1])
            for var in args.task_specific_env_vars
        ]
    logger.info(f"Launching job with task-specific env vars: {task_specific_env_vars}")
    logger.info(f"Arguments used for launching job: {vars(args)}")
    # handle case where image is already uploaded

    launch_job(
        project=args.project,
        workflow=args.workflow,
        image=args.image,
        gpu_count=args.gpu_count,
        shared_memory=args.shared_memory,
        cluster=args.cluster,
        priority=args.priority,
        task_name=args.task_name,
        task_specific_env_vars=task_specific_env_vars,
        budget=args.budget,
        workspace=args.workspace,
    )