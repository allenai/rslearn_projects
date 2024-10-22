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
IMAGE_NAME = "favyen/rslearn"  # Update image if needed


def launch_job(
    config_path: str,
    hparams_config_path: str | None = None,
    mode: str = "fit",
    workspace: str = DEFAULT_WORKSPACE,
    username: str | None = None,
) -> None:
    """Launch training for the specified config on Beaker.

    Args:
        config_path: the relative path from rslearn_projects/ to the YAML configuration
            file.
        hparams_config_path: the relative path from rslearn_projects/ to the YAML configuration
            file containing the hyperparameters to be combined with the base config.
        mode: Mode to run the model ('fit', 'validate', 'test', or 'predict').
        workspace: the Beaker workspace to run the job in.
        username: optional W&B username to associate with the W&B run for this job.
    """
    project_id, experiment_id = launcher_lib.get_project_and_experiment(config_path)
    launcher_lib.upload_code(project_id, experiment_id)
    beaker = Beaker.from_env(default_workspace=workspace)

    with beaker.session():
        env_vars = [
            EnvVar(
                name="WANDB_API_KEY",  # nosec
                secret="RSLEARN_WANDB_API_KEY",  # nosec
            ),
            EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
                value="/etc/credentials/gcp_credentials.json",  # nosec
            ),
            EnvVar(
                name="GCLOUD_PROJECT",  # nosec
                value="prior-satlas",  # nosec
            ),
            EnvVar(
                name="WEKA_ACCESS_KEY_ID",  # nosec
                secret="RSLEARN_WEKA_KEY",  # nosec
            ),
            EnvVar(
                name="WEKA_SECRET_ACCESS_KEY",  # nosec
                secret="RSLEARN_WEKA_SECRET",  # nosec
            ),
            EnvVar(
                name="WEKA_ENDPOINT_URL",  # nosec
                value="https://weka-aus.beaker.org:9000",  # nosec
            ),
            EnvVar(
                name="RSLP_PROJECT",  # nosec
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

        # if hparams_config_path is provided, create custom configs
        if hparams_config_path:
            custom_dir = os.path.dirname(config_path)
            custom_configs = launcher_lib.create_custom_configs(
                config_path, hparams_config_path, custom_dir
            )
        else:
            custom_configs = [config_path]
        for custom_config in custom_configs:
            spec = ExperimentSpec.new(
                budget=BUDGET,
                description=f"{project_id}/{experiment_id}",
                beaker_image=IMAGE_NAME,
                priority=Priority.high,
                command=["python", "-m", "rslp.docker_entrypoint"],
                arguments=[
                    "model",
                    mode,
                    "--config",
                    custom_config,
                    "--autoresume=true",
                ],
                constraints=Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
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
        "--hparams_config_path",
        type=str,
        help="Path to hyperparameters configuration file relative to rslearn_projects repository root",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fit", "validate", "test", "predict"],
        help="Mode to run the model ('fit', 'validate', 'test', or 'predict')",
        required=False,
        default="fit",
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
    launch_job(
        args.config_path,
        hparams_config_path=args.hparams_config_path,
        mode=args.mode,
        workspace=args.workspace,
        username=args.username,
    )
