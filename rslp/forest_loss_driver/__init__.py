"""Forest loss driver classification project."""

from .inference.best_image_selector import select_best_images_pipeline
from .predict_pipeline import predict_pipeline

workflows = {
    "predict": predict_pipeline,
    "select_best_images": select_best_images_pipeline,
}
