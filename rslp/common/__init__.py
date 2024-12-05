"""Pipelines common across projects."""

from .worker import worker_pipeline

workflows = {
    "worker": worker_pipeline,
}
