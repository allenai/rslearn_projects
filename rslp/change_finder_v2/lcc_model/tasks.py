"""LCCMultiTask: injects per-window annotation metadata for the transform.

Similar to the v1 ChangeMultiTask, this loads a sidecar JSON with per-window
annotation data and injects it into input_dict so the FrequentOptionSampler
transform can compute temporal references and timestamp targets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.train.tasks.multi_task import MetricWrapper, MultiTask
from rslearn.train.tasks.task import Task
from rslearn.utils import Feature
from torchmetrics import Metric, MetricCollection
from typing_extensions import override
from upath import UPath

from .transforms import ANNOTATION_KEY

OUTPUT_TASK_ORDER = ("binary", "src", "dst", "timestamps")


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class TimestampAccuracyMetric(Metric):
    """Binary accuracy for per-timestamp change predictions.

    Receives per-sample sigmoid probabilities and targets with classes/valid
    masks, thresholds at 0.5, and accumulates correct/total across batches.
    """

    def __init__(self) -> None:
        """Initialize counters for correct and total predictions."""
        super().__init__()
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        for pred, target in zip(preds, targets):
            classes = target["classes"].image[:, 0, :, :]  # (num_ts, H, W)
            valid = target["valid"].get_hw_tensor() > 0  # (H, W)
            valid_exp = valid.unsqueeze(0).expand_as(pred)  # (num_ts, H, W)
            binary_pred = (pred > 0.5).float()
            self.correct += ((binary_pred == classes) & valid_exp).sum()
            self.total += valid_exp.sum()

    @override
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total.float()


class TimestampAUROCMetric(Metric):
    """AUROC for per-timestamp binary change predictions.

    Flattens all valid (timestamp, pixel) pairs into a single binary
    classification problem and computes AUROC via torchmetrics.
    """

    def __init__(self) -> None:
        """Initialize with an internal BinaryAUROC metric."""
        super().__init__()
        print("initialize timestamp auroc metric")
        from torchmetrics.classification import BinaryAUROC

        self.auroc = BinaryAUROC()

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        try:
            all_preds: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []
            for pred, target in zip(preds, targets):
                classes = target["classes"].image[:, 0, :, :]  # (num_ts, H, W)
                valid = target["valid"].get_hw_tensor() > 0  # (H, W)
                valid_exp = valid.unsqueeze(0).expand_as(pred)  # (num_ts, H, W)
                all_preds.append(pred[valid_exp])
                all_labels.append(classes[valid_exp].long())
            print("all_preds", all_preds)
            print("all_labels", all_labels)
            if all_preds:
                self.auroc.update(torch.cat(all_preds), torch.cat(all_labels))
        except Exception as e:
            print("got update error", e)

    @override
    def compute(self) -> torch.Tensor:
        return self.auroc.compute()


class LCCMultiTask(MultiTask):
    """MultiTask that injects per-window LCC annotations into input_dict.

    Annotations are loaded from a sidecar JSON written by the prepare script,
    keyed by "{group}/{name}".
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        annotations_path: str,
    ):
        """Create a new LCCMultiTask.

        Args:
            tasks: map from task name to task object (binary, src, dst).
            input_mapping: per-task raw-input remapping.
            annotations_path: path to lcc_annotations.json sidecar.
        """
        super().__init__(tasks=tasks, input_mapping=input_mapping)
        self.annotations_path = annotations_path
        self._annotations: dict[str, dict[str, Any]] | None = None

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        if self._annotations is None:
            with UPath(self.annotations_path).open() as f:
                self._annotations = json.load(f)
        return self._annotations

    def get_metrics(self) -> MetricCollection:
        """Get metrics including timestamp accuracy and AUROC."""
        metrics = super().get_metrics()
        metrics.add_metrics(
            {
                "timestamps/accuracy": MetricWrapper(
                    "timestamps", TimestampAccuracyMetric()
                ),
                "timestamps/auroc": MetricWrapper("timestamps", TimestampAUROCMetric()),
            }
        )
        return metrics

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process inputs and inject annotation metadata.

        When load_targets=False (predict mode), the annotation is not injected
        since the FrequentOptionSampler transform is not used during prediction.
        """
        input_dict, target_dict = super().process_inputs(
            raw_inputs, metadata=metadata, load_targets=load_targets
        )
        if load_targets:
            ann = self._load_annotations().get(
                f"{metadata.window_group}/{metadata.window_name}"
            )
            if ann is None:
                raise KeyError(
                    f"No annotation found for {metadata.window_group}/{metadata.window_name} "
                    f"in {self.annotations_path}"
                )
            input_dict[ANNOTATION_KEY] = {
                "pre_change": _parse_date(ann["pre_change"]),
                "post_change": _parse_date(ann["post_change"]),
                "first_noticeable": _parse_date(ann["first_noticeable"]),
            }
        return input_dict, target_dict

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[np.uint8]:
        """Stack per-task probabilities into a single 49-band uint8 CHW array.

        Band layout:
        0..2 = binary (softmax probs)
        3..15 = src (softmax probs)
        16..28 = dst (softmax probs)
        29..48 = timestamp (sigmoid probs)
        """
        parts: list[torch.Tensor] = []
        for task_name in ("binary", "src", "dst"):
            probs = raw_output[task_name].float()
            parts.append((probs * 255).clamp(0, 255).to(torch.uint8))
        ts_probs = raw_output["timestamps"].float()
        parts.append((ts_probs * 255).clamp(0, 255).to(torch.uint8))
        stacked = torch.cat(parts, dim=0)
        return stacked.cpu().numpy()
