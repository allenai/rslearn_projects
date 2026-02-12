"""Launch Beaker jobs to parallelize data materialization."""

import os
import uuid

import tqdm
from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerDataMount,
    BeakerDataSource,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerJobPriority,
)

from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    create_gcp_credentials_mount,
    create_gee_credentials_mount,
    get_base_env_vars,
)

DEFAULT_COMMAND = [
    "rslearn",
    "dataset",
    "materialize",
    "--root",
    "{ds_path}",
    "--workers",
    "64",
    "--no-use-initial-job",
    "--retry-max-attempts",
    "8",
    "--retry-backoff-seconds",
    "60",
    "--ignore-errors",
]


def launch_job(
    image: str,
    command: list[str],
    clusters: list[str] | None = None,
    hostname: str | None = None,
    priority: BeakerJobPriority = BeakerJobPriority.high,
) -> None:
    """Launch a Beaker job that materializes the rslearn dataset.

    Args:
        image: the name of the Beaker image to use.
        command: the command to run in the Beaker job.
        clusters: optional list of Beaker clusters to target. One of hostname or
            clusters must be set.
        hostname: optional Beaker host to constrain to.
        priority: the priority to assign to the Beaker job.
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        experiment_name = str(uuid.uuid4())[0:16]
        weka_mount = BeakerDataMount(
            source=BeakerDataSource(weka="dfive-default"),
            mount_path="/weka/dfive-default",
        )
        env_vars = get_base_env_vars(use_weka_prefix=False)
        if "NASA_EARTHDATA_USERNAME" in os.environ:
            env_vars += [
                BeakerEnvVar(
                    name="NASA_EARTHDATA_USERNAME",
                    value=os.environ["NASA_EARTHDATA_USERNAME"],
                ),
                BeakerEnvVar(
                    name="NASA_EARTHDATA_PASSWORD",
                    value=os.environ["NASA_EARTHDATA_PASSWORD"],
                ),
            ]
        if "HTTP_PROXY" in os.environ:
            env_vars += [
                BeakerEnvVar(
                    name="HTTP_PROXY",
                    value=os.environ["HTTP_PROXY"],
                )
            ]
        if "HTTPS_PROXY" in os.environ:
            env_vars += [
                BeakerEnvVar(
                    name="HTTPS_PROXY",
                    value=os.environ["HTTPS_PROXY"],
                ),
            ]
        if "EARTHDATAHUB_TOKEN" in os.environ:
            env_vars += [
                BeakerEnvVar(
                    name="EARTHDATAHUB_TOKEN",
                    value=os.environ["EARTHDATAHUB_TOKEN"],
                ),
            ]
        # Set one GPU if not targeting a specific host, otherwise we might have
        # hundreds of jobs scheduled on the same host.
        # Also we can only set cluster constraint if we do not specify hostname.
        resources: dict | None
        constraints: BeakerConstraints
        if hostname is None:
            resources = {"gpuCount": 1}
            constraints = BeakerConstraints(
                cluster=clusters,
            )
        else:
            resources = None
            constraints = BeakerConstraints(
                hostname=[hostname],
            )

        experiment_spec = BeakerExperimentSpec.new(
            budget=DEFAULT_BUDGET,
            task_name=experiment_name,
            beaker_image=image,
            priority=priority,
            command=command,
            datasets=[
                weka_mount,
                create_gcp_credentials_mount(),
                create_gee_credentials_mount(),
            ],
            resources=resources,
            preemptible=True,
            constraints=constraints,
            env_vars=env_vars,
        )
        beaker.experiment.create(name=experiment_name, spec=experiment_spec)


def launch_jobs(
    image: str,
    ds_path: str,
    group: str | None = None,
    clusters: list[str] | None = None,
    num_jobs: int | None = None,
    hosts: list[str] | None = None,
    command: list[str] | None = None,
    priority: BeakerJobPriority = BeakerJobPriority.high,
) -> None:
    """Launch Beaker jobs to materialize an rslearn dataset.

    Args:
        image: the name of the Beaker image to use.
        ds_path: the dataset path.
        group: the group to use for the jobs.
        clusters: optional list of Beaker clusters to target. One of hostname or
            clusters+num_jobs must be set.
        num_jobs: when specifying clusters, the number of jobs to launch.
        hosts: optional list of Beaker hosts to launch jobs on (one job per host),
            as an alternative to specifying clusters+num_jobs.
        command: override the default materialization command.
        priority: the priority to assign to the Beaker jobs.
    """
    if (clusters is not None and hosts is not None) or (
        clusters is None and hosts is None
    ):
        raise ValueError("exactly one of clusters or hosts must be set")

    # Determine the command to run.
    if command is None:
        command = DEFAULT_COMMAND
    command = [arg.format(ds_path=ds_path) for arg in command]
    if group is not None:
        command += ["--group", group]

    if clusters is not None:
        if num_jobs is None:
            raise ValueError("when specifying clusters, num_jobs must also be set")
        for i in tqdm.tqdm(list(range(num_jobs)), desc="Launching jobs"):
            launch_job(
                image=image,
                command=command,
                clusters=clusters,
                priority=priority,
            )

    else:
        for host in tqdm.tqdm(hosts, desc="Launching jobs"):
            launch_job(
                image=image,
                command=command,
                hostname=host,
                priority=priority,
            )
