"""Launch train jobs on Beaker."""

import argparse
import os
import uuid

import dotenv
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

from rslp import launcher_lib

DEFAULT_WORKSPACE = "ai2/earth-systems"
BUDGET = "ai2/prior"
IMAGE_NAME = "favyen/rslearn"


def launch_job(
    config_path: str, workspace: str = DEFAULT_WORKSPACE, username: str | None = None
):
    """Launch training for the specified config on Beaker.

    Args:
        config_path: the relative path from rslearn_projects/ to the YAML configuration
            file.
        workspace: the Beaker workspace to run the job in.
        username: optional W&B username to associate with the W&B run for this job.
    """
    project_id, experiment_id = launcher_lib.get_project_and_experiment(config_path)
    launcher_lib.upload_code(project_id, experiment_id)
    beaker = Beaker.from_env(default_workspace=workspace)

    with beaker.session():
        env_vars = [
            EnvVar(
                name="WANDB_API_KEY",
                secret="RSLEARN_WANDB_API_KEY",
            ),
            EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",
                value="/etc/credentials/gcp_credentials.json",
            ),
            EnvVar(
                name="GCLOUD_PROJECT",
                value="prior-satlas",
            ),
            EnvVar(
                name="S3_ACCESS_KEY_ID",
                secret="RSLEARN_WEKA_KEY",
            ),
            EnvVar(
                name="S3_SECRET_ACCESS_KEY",
                secret="RSLEARN_WEKA_SECRET",
            ),
            EnvVar(
                name="RSLP_PROJECT",
                value=project_id,
            ),
            EnvVar(
                name="RSLP_EXPERIMENT",
                value=experiment_id,
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
        if username:
            env_vars.append(
                EnvVar(
                    name="WANDB_USERNAME",
                    value=username,
                )
            )

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=f"{project_id}/{experiment_id}",
            beaker_image=IMAGE_NAME,
            priority=Priority.high,
            command=["python", "-m", "rslp.docker_entrypoint"],
            arguments=["model", "fit", "--config", config_path, "--autoresume=true"],
            constraints=Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),
                    mount_path="/etc/credentials/gcp_credentials.json",
                ),
            ],
            env_vars=env_vars,
            resources=TaskResources(gpu_count=1),
        )
        unique_id = str(uuid.uuid4())[0:8]
        beaker.experiment.create(f"{project_id}_{experiment_id}_{unique_id}", spec)


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Launch beaker experiment for rslearn_projects",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration file relative to rslearn_projects repository root",
        required=True,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Which workspace to run the experiment in",
        default=DEFAULT_WORKSPACE,
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Associate a W&B user with this run in W&B",
        default=None,
    )
    args = parser.parse_args()
    launch_job(args.config_path, workspace=args.workspace, username=args.username)
