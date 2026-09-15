"""NISAR vessel detection."""

# Imported under different names than the modules they come from: binding a function to
# the same name as its module shadows the module on this package, so that
# `rslp.nisar_vessels.predict_pipeline` would be the function rather than the module.
from .predict_pipeline import predict_pipeline as predict
from .scripts.create_dataset import create_dataset as create_dataset_workflow
from .scripts.create_predict_windows import (
    create_predict_windows as create_predict_windows_workflow,
)

workflows = {
    "create_dataset": create_dataset_workflow,
    "create_predict_windows": create_predict_windows_workflow,
    "predict": predict,
}
