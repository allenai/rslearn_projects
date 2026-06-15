"""LCCMultiTaskV2: multi-task wrapper for the 20260610 temporal model.

Same annotation injection as LCCMultiTask, but the timestamp metrics measure
per-pixel start/end timestep classification accuracy instead of per-image
change-period membership.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rslearn.train.model_context import SampleMetadata
from rslearn.train.tasks.multi_task import MetricWrapper
from torchmetrics import Metric, MetricCollection
from typing_extensions import override

from .tasks import LCCMultiTask


class TimestampBoundaryAccuracy(Metric):
    """Accuracy of per-pixel start/end timestep predictions at change pixels.

    Compares the argmax over T of the predicted boundary distribution against
    the target timestep index, counting a prediction correct when the absolute
    index difference is within ``tolerance``. Only valid (change) pixels count.
    """

    def __init__(self, key: str, tolerance: int = 0) -> None:
        """Initialize counters.

        Args:
            key: which boundary to score, "start" or "end".
            tolerance: max absolute timestep-index difference counted as correct.
        """
        super().__init__()
        self.key = key
        self.tolerance = tolerance
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        for pred, target in zip(preds, targets):
            pred_idx = pred[self.key].argmax(dim=0)  # (H, W)
            target_idx = target[self.key].get_hw_tensor().to(pred_idx.device).long()
            valid = target["valid"].get_hw_tensor().to(pred_idx.device) > 0
            within = (pred_idx - target_idx).abs() <= self.tolerance
            self.correct += (within & valid).sum()
            self.total += valid.sum()

    @override
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total.float()


class LCCMultiTaskV2(LCCMultiTask):
    """MultiTask with start/end timestamp accuracy metrics."""

    @override
    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[np.uint8]:
        """Stack per-task probabilities into the base 49-band uint8 CHW array.

        The 20260611 model emits start/end timestamp distributions (two softmaxes
        over the T timesteps) rather than the base model's single 20-channel
        membership map. To stay compatible with the base ``output_change`` layer
        (49 bands) and downstream tooling, the start/end distributions are collapsed
        back into a per-timestep "in change window" membership:

            membership[t] = P(start <= t) * P(end >= t)
                          = cumsum(start)[t] * reverse_cumsum(end)[t]

        Band layout (identical to LCCMultiTask.process_output):
        0..2   = binary (softmax probs)
        3..15  = src (softmax probs)
        16..28 = dst (softmax probs)
        29..48 = timestamp membership (reconstructed, T=20)
        """
        parts: list[torch.Tensor] = []
        for task_name in ("binary", "src", "dst"):
            probs = raw_output[task_name].float()
            parts.append((probs * 255).clamp(0, 255).to(torch.uint8))

        timestamps = raw_output["timestamps"]
        start_cdf = timestamps["start"].float().cumsum(dim=0)  # P(start <= t)
        end_rev = timestamps["end"].float().flip(0).cumsum(dim=0).flip(0)  # P(end >= t)
        membership = (start_cdf * end_rev).clamp(0, 1)
        parts.append((membership * 255).clamp(0, 255).to(torch.uint8))

        stacked = torch.cat(parts, dim=0)
        return stacked.cpu().numpy()

    def get_metrics(self) -> MetricCollection:
        """Get binary/src/dst metrics plus start/end timestamp accuracy."""
        # Skip LCCMultiTask.get_metrics (per-image timestamp metrics) and use the
        # grandparent MultiTask metrics for binary/src/dst.
        metrics = super(LCCMultiTask, self).get_metrics()
        metrics.add_metrics(
            {
                "timestamps/start_accuracy": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("start", tolerance=0)
                ),
                "timestamps/end_accuracy": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("end", tolerance=0)
                ),
                "timestamps/start_within1": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("start", tolerance=1)
                ),
                "timestamps/end_within1": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("end", tolerance=1)
                ),
            }
        )
        return metrics
