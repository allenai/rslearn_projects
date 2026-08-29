"""NISAR vessel detection."""

from .scripts.create_dataset import create_dataset
from .scripts.create_predict_windows import create_predict_windows

workflows = {
    "create_dataset": create_dataset,
    "create_predict_windows": create_predict_windows,
}
