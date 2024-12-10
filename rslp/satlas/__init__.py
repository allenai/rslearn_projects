"""Satlas batch jobs.

Specifically, training and inference for these fine-tuned models on satlas.allen.ai:
- Marine infrastructure
- On-shore wind turbines
- Solar farms
- Tree cover
"""

from .job_launcher_worker import launch_workers, write_jobs, write_jobs_for_year_months
from .postprocess import merge_points, smooth_points
from .predict_pipeline import predict_multi, predict_pipeline
from .publish import publish_points

workflows = {
    "predict": predict_pipeline,
    "predict_multi": predict_multi,
    "write_jobs": write_jobs,
    "write_jobs_for_year_months": write_jobs_for_year_months,
    "launch_workers": launch_workers,
    "merge_points": merge_points,
    "smooth_points": smooth_points,
    "publish_points": publish_points,
}
