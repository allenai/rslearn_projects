"""Change finder v2: point-based annotation format for land cover change."""

from .lcc_model.predict_pipeline import predict_multi, predict_pipeline
from .lcc_model.write_jobs import write_jobs
from .lcc_model.write_jobs_random_2048 import write_jobs_random_2048
from .scripts.annotation_phase3.write_jobs_random_2048_china import (
    write_jobs_random_2048_china,
)

workflows = {
    "predict": predict_pipeline,
    "predict_multi": predict_multi,
    "write_jobs": write_jobs,
    "write_jobs_random_2048": write_jobs_random_2048,
    "write_jobs_random_2048_china": write_jobs_random_2048_china,
}
