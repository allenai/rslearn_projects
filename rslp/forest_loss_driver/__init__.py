"""Forest loss driver classification project."""

from .predict_pipeline import predict_pipeline_main

workflows = {
    "predict": predict_pipeline_main,
}
