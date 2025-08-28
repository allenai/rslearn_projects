"""Sentinel-2 vessel detection project."""

from .predict_pipeline import predict_pipeline
from .write_entries import write_entries

workflows = {
    "predict": predict_pipeline,
    "write_entries": write_entries,
}
