"""Pipelines common across projects."""

from .beaker_launcher import launch_job
from .worker import launch_workers, worker_pipeline

workflows = {
    "worker": worker_pipeline,
    "launch": launch_workers,
    "beaker_launcher": launch_job,
}
