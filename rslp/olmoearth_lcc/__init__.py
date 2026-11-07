"""OlmoEarth LCC: land cover change detection from Sentinel-2 time series."""

from .lcc_model.job_dir import job_status
from .lcc_model.predict_pipeline import predict_pipeline
from .lcc_model.worker import launch_workers, worker
from .lcc_model.write_jobs import write_jobs
from .lcc_model.write_jobs_random_2048 import write_jobs_random_2048
from .ten_year_dataset.create_windows import create_windows
from .ten_year_dataset.create_windows_africa import create_windows_africa
from .ten_year_dataset.create_windows_urban import create_windows_urban

workflows = {
    "create_windows": create_windows,
    "create_windows_africa": create_windows_africa,
    "create_windows_urban": create_windows_urban,
    "job_status": job_status,
    "launch_workers": launch_workers,
    "predict": predict_pipeline,
    "worker": worker,
    "write_jobs": write_jobs,
    "write_jobs_random_2048": write_jobs_random_2048,
}
