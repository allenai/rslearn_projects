"""OlmoEarth LCC: land cover change detection from Sentinel-2 time series."""

from .annotation_scripts.phase03_random_tiles_china_africa.write_jobs_random_2048_africa import (
    write_jobs_random_2048_africa,
)
from .annotation_scripts.phase03_random_tiles_china_africa.write_jobs_random_2048_china import (
    write_jobs_random_2048_china,
)
from .lcc_model.predict_pipeline import predict_multi, predict_pipeline
from .lcc_model.write_jobs import write_jobs
from .lcc_model.write_jobs_random_2048 import write_jobs_random_2048
from .ten_year_dataset.create_windows import create_windows
from .ten_year_dataset.create_windows_africa import create_windows_africa
from .ten_year_dataset.create_windows_urban import create_windows_urban

workflows = {
    "create_windows": create_windows,
    "create_windows_africa": create_windows_africa,
    "create_windows_urban": create_windows_urban,
    "predict": predict_pipeline,
    "predict_multi": predict_multi,
    "write_jobs": write_jobs,
    "write_jobs_random_2048": write_jobs_random_2048,
    "write_jobs_random_2048_africa": write_jobs_random_2048_africa,
    "write_jobs_random_2048_china": write_jobs_random_2048_china,
}
