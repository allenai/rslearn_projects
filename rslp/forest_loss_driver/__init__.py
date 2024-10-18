"""Forest loss driver classification project."""

from .predict_pipeline import (
    predict_pipeline,
    select_best_images_pipeline,
)

workflows = {
    "predict": predict_pipeline,
    "select_best_images": select_best_images_pipeline,
}
