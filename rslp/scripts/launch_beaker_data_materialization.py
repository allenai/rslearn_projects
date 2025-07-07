"""Launch Beaker jobs to parallelize data materialization."""

import argparse
import os
import uuid

import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
)

from rslp.utils.beaker import DEFAULT_BUDGET, DEFAULT_WORKSPACE

PLANETARY_COMPUTER_COMMAND = [
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
    project: str,
    image: str,
    ds_path: str,
    group: str,
    clusters: list[str] | None = None,
    hostname: str | None = None,
) -> None:
    """Launch a Beaker job that materializes the rslearn dataset.

    Args:
        project: the project to use for the jobs.
        image: the name of the Beaker image to use.
        ds_path: the dataset path.
        group: the group to use for the jobs.
        clusters: optional list of Beaker clusters to target. One of hostname or
            clusters must be set.
        hostname: optional Beaker host to constrain to.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)
    with beaker.session():
        # Add random string since experiment names must be unique.
        task_uuid = str(uuid.uuid4())[0:16]
        experiment_name = f"{project}-{task_uuid}"

        command = [
            arg.format(ds_path=ds_path)
            for arg in PLANETARY_COMPUTER_COMMAND + ["--group", group]
        ]
        weka_mount = DataMount(
            source=DataSource(weka="dfive-default"),
            mount_path="/weka/dfive-default",
        )

        # Set one GPU if not targeting a specific host, otherwise we might have
        # hundreds of jobs scheduled on the same host.
        # Also we can only set cluster constraint if we do not specify hostname.
        resources: dict | None
        constraints: Constraints
        if hostname is None:
            resources = {"gpuCount": 1}
            constraints = Constraints(
                cluster=clusters,
            )
        else:
            resources = None
            constraints = Constraints(
                hostname=[hostname],
            )

        experiment_spec = ExperimentSpec.new(
            budget=DEFAULT_BUDGET,
            task_name=experiment_name,
            beaker_image=image,
            priority=Priority.high,
            command=command,
            datasets=[weka_mount],
            resources=resources,
            preemptible=True,
            constraints=constraints,
            env_vars=[
                EnvVar(
                    name="NASA_EARTHDATA_USERNAME",
                    value=os.environ.get("NASA_EARTHDATA_USERNAME"),
                ),
                EnvVar(
                    name="NASA_EARTHDATA_PASSWORD",
                    value=os.environ.get("NASA_EARTHDATA_PASSWORD"),
                ),
            ],
        )
        beaker.experiment.create(experiment_name, experiment_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs to materialize rslearn datasets",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="The project to use for the jobs",
        required=True,
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to the rslearn dataset for dataset creation assuming /weka/dfive-default/ is mounted",
        required=True,
    )
    parser.add_argument(
        "--group",
        type=str,
        help="The group to use for the jobs",
        required=True,
    )
    parser.add_argument(
        "--image_name",
        type=str,
        help="Name of the Beaker image to use for the job",
        required=True,
    )
    parser.add_argument(
        "--clusters",
        type=str,
        help="Comma-separated list of clusters to target",
        default=None,
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        help="Number of Beaker jobs to start (one of clusters+num_jobs or hosts must be set)",
        default=None,
    )
    parser.add_argument(
        "--hosts",
        type=str,
        help="Comma-separated list of hosts to start jobs on, one job per host (one of clusters+num_jobs or hosts must be set)",
        default=None,
    )
    args = parser.parse_args()

    if args.num_jobs is not None:
        for i in tqdm.tqdm(list(range(args.num_jobs)), desc="Launching jobs"):
            launch_job(
                image=args.image_name,
                project=args.project,
                ds_path=args.ds_path,
                group=args.group,
                clusters=args.clusters.split(","),
            )
    elif args.hosts is not None:
        for host in args.hosts.split(","):
            launch_job(
                image=args.image_name,
                project=args.project,
                ds_path=args.ds_path,
                group=args.group,
                hostname=host,
            )
    else:
        raise ValueError("one of num_jobs and hosts must be set")
