"""Sentinel-1 vessel detection project."""

from .predict_pipeline import predict_pipeline

workflows = {
    "predict": predict_pipeline,
}
