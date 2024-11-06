"""Satlas batch jobs.

Specifically, training and inference for these fine-tuned models on satlas.allen.ai:
- Marine infrastructure
- On-shore wind turbines
- Solar farms
- Tree cover
"""

from .job_launcher import launch_jobs
from .postprocess import postprocess_points
from .predict_pipeline import predict_multi, predict_pipeline

workflows = {
    "predict": predict_pipeline,
    "predict_multi": predict_multi,
    "launch": launch_jobs,
    "postprocess_points": postprocess_points,
}
