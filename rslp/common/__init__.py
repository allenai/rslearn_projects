"""Pipelines common across projects."""

from .beaker_data_materialization import launch_jobs as launch_data_materialization_jobs
from .beaker_launcher import launch_job
from .beaker_olmoearth_run import beaker_olmoearth_run
from .beaker_train import beaker_train
from .worker import launch_workers, worker_pipeline
from .write_file import write_file

workflows = {
    "launch_data_materialization_jobs": launch_data_materialization_jobs,
    "beaker_launcher": launch_job,
    "beaker_olmoearth_run": beaker_olmoearth_run,
    "beaker_train": beaker_train,
    "launch": launch_workers,
    "worker": worker_pipeline,
    "write_file": write_file,
}
