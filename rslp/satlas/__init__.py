"""Satlas batch jobs.

Specifically, training and inference for these fine-tuned models on satlas.allen.ai:
- Marine infrastructure
- On-shore wind turbines
- Solar farms
- Tree cover
"""

from .predict_pipeline import predict_pipeline

workflows = {
    "predict": predict_pipeline,
}
