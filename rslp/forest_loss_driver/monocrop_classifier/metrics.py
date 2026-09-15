"""Per-window metrics for monocrop classification.

Each val/test sample is one full 128x128 window whose valid (non-nodata) pixels
belong to a single annotated class, so these metrics reduce every batch sample
to one prediction and one label. The window prediction is the majority vote of
the per-pixel argmax over valid pixels; ties break toward the lowest class ID.
The window label is the majority class among valid label pixels (a single class
by construction).
"""

from __future__ import annotations

from typing import Any

import torch
from rslearn.train.metrics import ConfusionMatrixOutput
from torchmetrics import Metric


def per_window_classes(
    preds: list[Any] | torch.Tensor,
    targets: list[dict[str, Any]],
    num_classes: int,
) -> list[tuple[int, int]]:
    """Reduce each batch sample to one (label, prediction) class pair.

    Args:
        preds: per-pixel class probabilities, as a BCHW tensor or list of CHW
            tensors.
        targets: per-sample target dicts with "classes" and "valid" RasterImages,
            as produced by SegmentationTask.process_inputs.
        num_classes: the number of classes, which must match the number of
            prediction channels.

    Returns:
        one (label_class, predicted_class) pair per sample with at least one
        valid pixel.
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.stack(list(preds))
    if preds.shape[1] != num_classes:
        raise ValueError(
            f"expected {num_classes} prediction channels, got {preds.shape[1]}"
        )

    pairs: list[tuple[int, int]] = []
    for pred, target in zip(preds, targets):
        valid = target["valid"].get_hw_tensor() > 0
        if not valid.any():
            continue
        labels = target["classes"].get_hw_tensor().long()
        label_counts = torch.bincount(labels[valid], minlength=num_classes)
        pred_counts = torch.bincount(pred.argmax(dim=0)[valid], minlength=num_classes)
        pairs.append((int(label_counts.argmax()), int(pred_counts.argmax())))
    return pairs


class PerWindowAccuracy(Metric):
    """Accuracy over majority-vote per-window predictions."""

    def __init__(self, num_classes: int) -> None:
        """Initialize a new PerWindowAccuracy.

        Args:
            num_classes: the number of classes, which must match the number of
                prediction channels.
        """
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self, preds: list[Any] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update metric with one batch of per-pixel predictions and targets."""
        for label, pred in per_window_classes(preds, targets, self.num_classes):
            self.correct += int(label == pred)
            self.total += 1

    def compute(self) -> torch.Tensor:
        """Return the fraction of windows predicted correctly."""
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total.float()


class PerWindowConfusionMatrix(Metric):
    """Confusion matrix over majority-vote per-window predictions.

    The computed value is a ConfusionMatrixOutput that rslearn logs to wandb.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize a new PerWindowConfusionMatrix.

        Args:
            num_classes: the number of classes, which must match the number of
                prediction channels.
            class_names: optional class names for the confusion matrix axes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.add_state(
            "confusion_matrix",
            default=torch.zeros(num_classes, num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(
        self, preds: list[Any] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update metric with one batch of per-pixel predictions and targets."""
        for label, pred in per_window_classes(preds, targets, self.num_classes):
            self.confusion_matrix[label, pred] += 1

    def compute(self) -> ConfusionMatrixOutput:
        """Return the confusion matrix wrapped for wandb logging."""
        return ConfusionMatrixOutput(
            confusion_matrix=self.confusion_matrix,
            class_names=self.class_names,
        )
