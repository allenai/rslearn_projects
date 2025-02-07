"""Pipelines common across projects."""

from .worker import launch_workers, worker_pipeline
from .write_file import write_file

workflows = {
    "worker": worker_pipeline,
    "launch": launch_workers,
    "write_file": write_file,
}
