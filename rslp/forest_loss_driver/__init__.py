"""Forest loss driver classification project."""

from .predict_pipeline import (
    PredictPipelineConfig,
    predict_pipeline,
    select_best_images_pipeline,
)

workflows = {
    "predict": (PredictPipelineConfig, predict_pipeline),
    "select_best_images": (None, select_best_images_pipeline),
}
