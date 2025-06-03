"""Launch train jobs on Beaker."""

import os
import shutil
import uuid

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerRetrySpec,
    BeakerTaskResources,
    BeakerTaskSpec,
)

from rslp import launcher_lib
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)


def beaker_train(
    config_path: str,
    image_name: str,
    cluster: list[str],
    hparams_config_path: str | None = None,
    mode: str = "fit",
    run_id: str = "",
    workspace: str = DEFAULT_WORKSPACE,
    username: str | None = None,
    gpus: int = 1,
    shared_memory: str = "256GiB",
    weka_mounts: list[WekaMount] = [],
    project_id: str | None = None,
    experiment_id: str | None = None,
    extra_args: list[str] = [],
    priority: str = "high",
    retries: int = 0,
) -> None:
    """Launch training for the specified config on Beaker.

    Args:
        config_path: the relative path from rslearn_projects/ to the YAML configuration
            file.
        image_name: the name of the Beaker image to use for the job.
        cluster: list of Beaker clusters to target.
        hparams_config_path: the relative path from rslearn_projects/ to the YAML configuration
            file containing the hyperparameters to be combined with the base config.
        mode: Mode to run the model ('fit', 'validate', 'test', or 'predict').
        run_id: The run ID to associate with this job.
        workspace: the Beaker workspace to run the job in.
        username: optional W&B username to associate with the W&B run for this job.
        gpus: number of GPUs to use.
        shared_memory: shared memory resource string to use, e.g. "256GiB".
        weka_mounts: list of Weka mounts to include.
        project_id: override the project ID.
        experiment_id: override the experiment ID.
        extra_args: extra arguments to pass in the Beaker job.
        priority: the priority to assign to the Beaker job.
        retries: how many times to retry the Beaker job.
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

    # Get the project and experiment ID to use based on the config or user-provided
    # override. This is used for uploading code here and then downloading it back in
    # in the Beaker job.
    config_project_id, config_experiment_id = launcher_lib.get_project_and_experiment(
        config_path
    )
    if project_id is None:
        project_id = config_project_id
    if experiment_id is None:
        experiment_id = config_experiment_id
    launcher_lib.upload_code(project_id, experiment_id)

    if hparams_configs_dir is not None:
        shutil.rmtree(hparams_configs_dir)

    with Beaker.from_env(default_workspace=workspace) as beaker:
        for run_id, config_path in config_paths.items():
            env_vars = get_base_env_vars()
            env_vars.extend(
                [
                    BeakerEnvVar(
                        name="RSLP_PROJECT",  # nosec
                        value=project_id,
                    ),
                    BeakerEnvVar(
                        name="RSLP_EXPERIMENT",
                        value=experiment_id,
                    ),
                    BeakerEnvVar(
                        name="RSLP_RUN_ID",
                        value=run_id,
                    ),
                ]
            )
            if username:
                env_vars.append(
                    BeakerEnvVar(
                        name="WANDB_USERNAME",
                        value=username,
                    )
                )
            datasets = [create_gcp_credentials_mount()]
            datasets += [weka_mount.to_data_mount() for weka_mount in weka_mounts]
            spec = BeakerExperimentSpec(
                budget=DEFAULT_BUDGET,
                description=f"{project_id}/{experiment_id}/{run_id}",
                retry=BeakerRetrySpec(allowed_task_retries=retries),
                tasks=[
                    BeakerTaskSpec.new(
                        beaker_image=image_name,
                        priority=priority,
                        command=["python", "-m", "rslp.docker_entrypoint"],
                        arguments=[
                            "model",
                            mode,
                            "--config",
                            config_path,
                            "--autoresume=true",
                            # Ensure that the experiment/project are correctly set (in case the
                            # user overwrote the one in the configuration file).
                            "--rslp_experiment",
                            experiment_id,
                            "--rslp_project",
                            project_id,
                        ]
                        + extra_args,
                        constraints=BeakerConstraints(
                            cluster=cluster,
                        ),
                        preemptible=True,
                        datasets=datasets,
                        env_vars=env_vars,
                        resources=BeakerTaskResources(
                            gpu_count=gpus, shared_memory=shared_memory
                        ),
                        name="main",
                    )
                ],
            )
            unique_id = str(uuid.uuid4())[0:8]
            beaker.experiment.create(
                name=f"{project_id}_{experiment_id}_{unique_id}", spec=spec
            )
