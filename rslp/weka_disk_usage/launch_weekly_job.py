"""Launch the Beaker job that produces the weekly weka disk usage report.

This is typically executed from the GitHub Action in
``.github/workflows/weka_disk_usage.yaml``, which runs it inside the published
``ghcr.io/allenai/rslearn_projects:latest`` image. The Beaker job itself also runs
that same public image (as a Docker image source, so nothing needs to be uploaded
to Beaker) with the weka bucket mounted, and executes
``rslp.weka_disk_usage.weekly_report``.

The BEAKER_TOKEN (and optionally BEAKER_ADDR) environment variables must be set.
"""

import uuid
from datetime import datetime, timezone

from beaker import (
    Beaker,
    BeakerExperimentSpec,
    BeakerJobPriority,
)

from rslp.log_utils import get_logger
from rslp.utils.beaker import DEFAULT_BUDGET, DEFAULT_WORKSPACE, WekaMount

logger = get_logger(__name__)

DEFAULT_DOCKER_IMAGE = "ghcr.io/allenai/rslearn_projects:latest"
WEKA_BUCKET_NAME = "dfive-default"
DEFAULT_ROOT = f"/weka/{WEKA_BUCKET_NAME}"
DEFAULT_OUT_DIR = f"/weka/{WEKA_BUCKET_NAME}/weka_disk_utilization"


def launch_weekly_job(
    date: str | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
    root: str = DEFAULT_ROOT,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    cluster: str = "ai2/jupiter",
    workspace: str = DEFAULT_WORKSPACE,
    budget: str = DEFAULT_BUDGET,
) -> None:
    """Create a Beaker experiment that runs ``weekly_report`` for the given date.

    The date is fixed at launch time (rather than computed inside the job) so that
    when the preemptible job is auto-resumed it re-runs the exact same command and
    finds its own checkpoint / done marker files.

    Args:
        date: YYYYMMDD prefix for the output files. Defaults to the current UTC
            date; it only needs to differ from week to week.
        out_dir: directory on the weka mount to write the outputs to.
        root: directory to scan.
        docker_image: public Docker image to run the job with. It must contain
            rslearn_projects.
        cluster: Beaker cluster to run on.
        workspace: Beaker workspace.
        budget: Beaker budget.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")

    task_name = f"weka_disk_usage_{date}_{str(uuid.uuid4())[0:8]}"
    command = [
        "python",
        "-m",
        "rslp.main",
        "weka_disk_usage",
        "weekly_report",
        "--out_dir",
        out_dir,
        "--date",
        date,
        "--root",
        root,
    ]
    datasets = [
        WekaMount(
            bucket_name=WEKA_BUCKET_NAME, mount_path=f"/weka/{WEKA_BUCKET_NAME}"
        ).to_data_mount()
    ]

    with Beaker.from_env(default_workspace=workspace) as beaker:
        experiment_spec = BeakerExperimentSpec.new(
            budget=budget,
            task_name=task_name,
            docker_image=docker_image,
            priority=BeakerJobPriority.urgent,
            cluster=cluster,
            command=command,
            datasets=datasets,
            # The job is preemptible, but protected from preemption for the first
            # 8 hours (the maximum allowed on most clusters). If it is preempted
            # after that, it is re-queued and weekly_report resumes from the scan
            # checkpoint (or skips the scan if it already finished).
            min_runtime="8h",
            auto_resume=True,
        )
        logger.info("Creating experiment %s: %s", task_name, " ".join(command))
        experiment = beaker.experiment.create(name=task_name, spec=experiment_spec)
        logger.info("Created experiment %s", experiment.id)
