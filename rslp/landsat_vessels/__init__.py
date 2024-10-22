"""Landsat vessel detection project."""

from .predict_pipeline import predict_pipeline

workflows = {
    "predict": (None, predict_pipeline),
}
