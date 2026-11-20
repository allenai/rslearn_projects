"""Sentinel-1 vessel detection project."""

from .predict_pipeline import predict_pipeline
from .prepare import prepare

workflows = {
    "predict": predict_pipeline,
    "prepare": prepare,
}
