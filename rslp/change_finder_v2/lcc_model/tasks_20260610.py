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
from .timestamp_output import start_end_day_bands


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
    ) -> npt.NDArray[np.uint16]:
        """Stack per-task outputs into a 56-band uint16 CHW array.

        The two timestamp bands hold the predicted pre-change and post-change
        dates as integer days since ``TIMESTAMP_EPOCH``, from the per-pixel argmax
        of the start/end distributions over the input timesteps (mapped to real
        dates via ``raw_output["timestep_days"]``). Probability bands are stored as
        0..255 within the uint16 raster.

        Band layout:
        0..2   = binary (softmax probs)
        3..15  = src (softmax probs)
        16..28 = dst (softmax probs)
        29     = ts_pre_days (days since epoch)
        30     = ts_post_days (days since epoch)
        31..37 = pre_change (softmax probs)
        38..49 = post_change (softmax probs)
        50..55 = same_change (softmax probs)
        """
        parts: list[npt.NDArray[np.uint16]] = []
        for task_name in ("binary", "src", "dst"):
            probs = raw_output[task_name].float()
            parts.append(
                (probs * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint16)
            )

        timestamps = raw_output["timestamps"]
        timestep_days = raw_output.get("timestep_days")
        if timestep_days is not None:
            day_bands = start_end_day_bands(
                timestamps["start"], timestamps["end"], timestep_days
            )
            parts.append(day_bands.cpu().numpy().astype(np.uint16))
        else:
            h, w = timestamps["start"].shape[-2:]
            parts.append(np.zeros((2, h, w), dtype=np.uint16))

        for task_name in ("pre_change", "post_change", "same_change"):
            probs = raw_output[task_name].float()
            parts.append(
                (probs * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint16)
            )

        return np.concatenate(parts, axis=0)

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
                "timestamps/start_within2": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("start", tolerance=2)
                ),
                "timestamps/end_within2": MetricWrapper(
                    "timestamps", TimestampBoundaryAccuracy("end", tolerance=2)
                ),
            }
        )
        return metrics
