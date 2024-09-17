"""Maldives ecosystem mapping project."""

from rslp.config import BaseTrainPipelineConfig

from .data_pipeline import DataPipelineConfig, data_pipeline
from .predict_pipeline import maxar_predict_pipeline, sentinel2_predict_pipeline
from .train_pipeline import maxar_train_pipeline, sentinel2_train_pipeline

workflows = {
    "data": (DataPipelineConfig, data_pipeline),
    "train_maxar": (BaseTrainPipelineConfig, maxar_train_pipeline),
    "train_sentinel2": (BaseTrainPipelineConfig, sentinel2_train_pipeline),
    "predict_maxar": (None, maxar_predict_pipeline),
    "predict_sentinel2": (None, sentinel2_predict_pipeline),
}
