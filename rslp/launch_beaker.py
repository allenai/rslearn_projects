"""Launch train jobs on Beaker."""

import argparse
import os
import shutil
import uuid

import dotenv
from beaker import (
    Beaker,
    Constraints,
    EnvVar,
    ExperimentSpec,
    Priority,
    TaskResources,
)

from rslp import launcher_lib

DEFAULT_WORKSPACE = "ai2/earth-systems"
BUDGET = "ai2/prior"
# Update when new image is needed.
IMAGE_NAME = "favyen/rslearn-20241025"

# I should make a docker image specifc to this project
# Need to add the following functionality
# upload a specified image


def launch_job(
    config_path: str,
    hparams_config_path: str | None = None,
    mode: str = "fit",
    run_id: str = "",
    workspace: str = DEFAULT_WORKSPACE,
    username: str | None = None,
    gpus: int = 1,
) -> None:
    """Launch training for the specified config on Beaker.

    Args:
        config_path: the relative path from rslearn_projects/ to the YAML configuration
            file.
        hparams_config_path: the relative path from rslearn_projects/ to the YAML configuration
            file containing the hyperparameters to be combined with the base config.
        mode: Mode to run the model ('fit', 'validate', 'test', or 'predict').
        run_id: The run ID to associate with this job.
        workspace: the Beaker workspace to run the job in.
        username: optional W&B username to associate with the W&B run for this job.
        gpus: number of GPUs to use.
    """
    hparams_configs_dir = None

    if hparams_config_path:
        config_dir = os.path.dirname(config_path)
        hparams_configs_dir = os.path.join(config_dir, "hparams_configs")
        os.makedirs(hparams_configs_dir, exist_ok=True)
        config_paths = launcher_lib.create_custom_configs(
            config_path, hparams_config_path, hparams_configs_dir
        )
    else:
        # run_id can be specified in predict jobs
        config_paths = {run_id: config_path}

    project_id, experiment_id = launcher_lib.get_project_and_experiment(config_path)
    launcher_lib.upload_code(project_id, experiment_id)

    if hparams_configs_dir is not None:
        shutil.rmtree(hparams_configs_dir)

    beaker = Beaker.from_env(default_workspace=workspace)

    for run_id, config_path in config_paths.items():
        with beaker.session():
            env_vars = launcher_lib.get_base_env_vars()
            env_vars.extend(
                [
                    EnvVar(
                        name="RSLP_PROJECT",  # nosec
                        value=project_id,
                    ),
                    EnvVar(
                        name="RSLP_EXPERIMENT",
                        value=experiment_id,
                    ),
                    EnvVar(
                        name="RSLP_RUN_ID",
                        value=run_id,
                    ),
                ]
            )
            if username:
                env_vars.append(
                    EnvVar(
                        name="WANDB_USERNAME",
                        value=username,
                    )
                )
            spec = ExperimentSpec.new(
                budget=BUDGET,
                description=f"{project_id}/{experiment_id}/{run_id}",
                beaker_image=IMAGE_NAME,
                priority=Priority.high,
                command=["python", "-m", "rslp.docker_entrypoint"],
                arguments=[
                    "model",
                    mode,
                    "--config",
                    config_path,
                    "--autoresume=true",
                ],
                constraints=Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
                preemptible=True,
                datasets=[launcher_lib.create_gcp_credentials_mount()],
                env_vars=env_vars,
                resources=TaskResources(gpu_count=gpus),
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
        "--run_id",
        type=str,
        help="The run ID to associate with this job, used to specify an existing run on GCS",
        required=False,
        default="",
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
    parser.add_argument(
        "--gpus",
        type=int,
        help="Number of GPUs",
        default=1,
    )
    args = parser.parse_args()
    launch_job(
        config_path=args.config_path,
        hparams_config_path=args.hparams_config_path,
        mode=args.mode,
        run_id=args.run_id,
        workspace=args.workspace,
        username=args.username,
        gpus=args.gpus,
    )
