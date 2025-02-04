"""Pipelines common across projects."""

from .worker import launch_workers, worker_pipeline

workflows = {
    "worker": worker_pipeline,
    "launch": launch_workers,
}
