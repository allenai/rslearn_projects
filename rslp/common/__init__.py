"""Pipelines common across projects."""

from .beaker_launcher import launch_job
from .beaker_train import beaker_train
from .worker import launch_workers, worker_pipeline
from .write_file import write_file

workflows = {
    "worker": worker_pipeline,
    "launch": launch_workers,
    "beaker_launcher": launch_job,
    "beaker_train": beaker_train,
    "write_file": write_file,
}
