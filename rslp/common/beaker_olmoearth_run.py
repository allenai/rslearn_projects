"""Launch OlmoEarth run jobs on Beaker."""

import os
import uuid
from pathlib import Path

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerDataMount,
    BeakerDataSource,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerRetrySpec,
    BeakerTaskResources,
    BeakerTaskSpec,
)

from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    create_gcp_credentials_mount,
    get_base_env_vars,
)


def beaker_olmoearth_run(
    image_name: str,
    cluster: list[str],
    project_path: str,
    scratch_path: str,
    workflow: str = "finetune",
    workspace: str = DEFAULT_WORKSPACE,
    project_name: str | None = None,
    username: str | None = None,
    gpus: int = 1,
    shared_memory: str = "256GiB",
    extra_args: list[str] = [],
    priority: str = "high",
    retries: int = 0,
    extra_env_vars: dict[str, str] = {},
    checkpoint_path: str | None = None,
) -> None:
    """Launch OlmoEarth run workflow on Beaker.

    Args:
        image_name: the name of the Beaker image to use for the job.
        cluster: list of Beaker clusters to target.
        project_path: the path to the project directory containing configuration files.
        scratch_path: the path to use for scratch space.
        workflow: the OlmoEarth workflow to run ('finetune', 'olmoearth_run', 'prepare_labeled_windows', 'one_stage').
        workspace: the Beaker workspace to run the job in.
        project_name: optional project name to use for the W&B run for this job.
        username: optional W&B username to associate with the W&B run for this job.
        gpus: number of GPUs to use.
        shared_memory: shared memory resource string to use, e.g. "256GiB".
        extra_args: extra arguments to pass in the Beaker job.
        priority: the priority to assign to the Beaker job.
        retries: how many times to retry the Beaker job.
        extra_env_vars: additional environment variables to set in the Beaker job.
        checkpoint_path: path to model checkpoint (required for 'olmoearth_run' and 'one_stage' workflows).
    """
    # Validate workflow-specific requirements
    if workflow in ["olmoearth_run", "one_stage"] and checkpoint_path is None:
        raise ValueError(f"checkpoint_path is required for workflow '{workflow}'")

    with Beaker.from_env(default_workspace=workspace) as beaker:
        env_vars = get_base_env_vars(use_weka_prefix=False)

        if username:
            env_vars.append(
                BeakerEnvVar(
                    name="WANDB_USERNAME",
                    value=username,
                )
            )

        # Add common OlmoEarth environment variables from local environment
        olmoearth_env_vars = [
            "DATASET_PATH",
            "EXTRA_FILES_PATH",
            "NUM_WORKERS",
            "WANDB_ENTITY",
            "WANDB_NAME",
            "WANDB_PROJECT",
        ]

        for env_name in olmoearth_env_vars:
            if env_name in os.environ:
                env_vars.append(
                    BeakerEnvVar(
                        name=env_name,
                        value=os.environ[env_name],
                    )
                )

        # Add any extra environment variables
        for env_name, env_value in extra_env_vars.items():
            env_vars.append(
                BeakerEnvVar(
                    name=env_name,
                    value=env_value,
                )
            )

        # Create command list for the OlmoEarth run command
        command = [
            "python",
            "-m",
            "rslp.main",
            "olmoearth_run",
            workflow,
            "--project_path",
            project_path,
            "--scratch_path",
            scratch_path,
        ]

        # Add checkpoint path if provided
        if checkpoint_path is not None:
            command.extend(["--checkpoint_path", checkpoint_path])

        # Add any extra arguments
        command.extend(extra_args)

        # Set up datasets (mounts)
        datasets = [
            BeakerDataMount(
                source=BeakerDataSource(weka="dfive-default"),
                mount_path="/weka/dfive-default",
            ),
            create_gcp_credentials_mount(),
        ]

        if project_name is None:
            project_name = Path(project_path).name

        # Create the experiment spec
        spec = BeakerExperimentSpec(
            budget=DEFAULT_BUDGET,
            description=f"olmoearth_run_{workflow}_{project_name}",
            retry=BeakerRetrySpec(allowed_task_retries=retries),
            tasks=[
                BeakerTaskSpec.new(
                    beaker_image=image_name,
                    priority=priority,
                    command=command,
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

        # Generate unique experiment name
        unique_id = str(uuid.uuid4())[0:8]
        experiment_name = f"olmoearth_run_{workflow}_{project_name}_{unique_id}"

        beaker.experiment.create(name=experiment_name, spec=spec)
        print(f"Created experiment: {experiment_name}")
