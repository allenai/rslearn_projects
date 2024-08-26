"""Maldives ecosystem mapping project."""

from rslp.config import BaseTrainPipelineConfig

from .data_pipeline import DataPipelineConfig, data_pipeline
from .train_pipeline import maxar_train_pipeline, sentinel2_train_pipeline

workflows = {
    "data": (DataPipelineConfig, data_pipeline),
    "train_maxar": (BaseTrainPipelineConfig, maxar_train_pipeline),
    "train_sentinel2": (BaseTrainPipelineConfig, sentinel2_train_pipeline),
}
