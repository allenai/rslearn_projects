"""Forest loss driver classification project."""

from .predict_pipeline import extract_dataset_main, run_model_predict_main

workflows = {
    "extract_dataset": extract_dataset_main,
    "predict": run_model_predict_main,
}
