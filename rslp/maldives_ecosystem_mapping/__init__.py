"""Maldives ecosystem mapping project."""

from .data_pipeline import data_pipeline
from .predict_pipeline import maxar_predict_pipeline, sentinel2_predict_pipeline

workflows = {
    "data": data_pipeline,
    "predict_maxar": maxar_predict_pipeline,
    "predict_sentinel2": sentinel2_predict_pipeline,
}
