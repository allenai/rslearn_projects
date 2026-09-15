"""NISAR vessel detection."""

# Aliased so it does not shadow the module of the same name on this package.
from .predict_pipeline import predict_pipeline as predict
from .scripts.create_dataset import create_dataset
from .scripts.create_predict_windows import create_predict_windows

workflows = {
    "create_dataset": create_dataset,
    "create_predict_windows": create_predict_windows,
    "predict": predict,
}
