"""Task for annotation timestamp classification."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
import shapely
import torch
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.train.tasks.task import Task
from rslearn.utils import Feature, STGeometry
from torchmetrics import Metric, MetricCollection

from .constants import NUM_CROP_MONTHS, SENTINEL2_LAYER, TIMESTAMP_HEADS


def _decode_monotonic(
    pre_probs: torch.Tensor, first_probs: torch.Tensor, post_probs: torch.Tensor
) -> tuple[int, int, int]:
    """Decode the best monotonic index triplet."""
    pre = torch.log(pre_probs + 1e-12)
    first = torch.log(first_probs + 1e-12)
    post = torch.log(post_probs + 1e-12)
    if (
        pre.numel() != NUM_CROP_MONTHS
        or first.numel() != NUM_CROP_MONTHS
        or post.numel() != NUM_CROP_MONTHS
    ):
        raise ValueError("timestamp probability vectors must each have 60 entries")

    best_score = float("-inf")
    best = (0, 0, 0)
    for pre_idx in range(NUM_CROP_MONTHS):
        for first_idx in range(pre_idx, NUM_CROP_MONTHS):
            post_idx = first_idx + int(torch.argmax(post[first_idx:]).item())
            score = float((pre[pre_idx] + first[first_idx] + post[post_idx]).item())
            if score > best_score:
                best_score = score
                best = (pre_idx, first_idx, post_idx)
    return best


class TimestampAccuracy(Metric):
    """Accuracy over valid timestamp heads, optionally with tolerance."""

    def __init__(self, head: str | None = None, tolerance: int = 0) -> None:
        """Initialize metric counters."""
        super().__init__()
        self.head = head
        self.tolerance = tolerance
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self, preds: list[dict[str, torch.Tensor]], targets: list[dict[str, Any]]
    ) -> None:
        """Accumulate correct predictions."""
        heads = (self.head,) if self.head is not None else TIMESTAMP_HEADS
        for pred, target in zip(preds, targets):
            for head in heads:
                if head not in target:
                    continue
                valid = target[head]["valid"] > 0
                if not bool(valid.item()):
                    continue
                cls = target[head]["class"]
                pred_cls = pred[head].argmax()
                self.correct += ((pred_cls - cls).abs() <= self.tolerance).long()
                self.total += 1

    def compute(self) -> torch.Tensor:
        """Return accuracy."""
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()


class TimestampHelperTask(Task):
    """Task that reads timestamp labels and writes prediction vectors."""

    def __init__(
        self,
        input_key: str = SENTINEL2_LAYER,
        target_key: str = "targets",
        allow_invalid: bool = False,
    ) -> None:
        """Initialize the task."""
        self.input_key = input_key
        self.target_key = target_key
        self.allow_invalid = allow_invalid

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pass through imagery and parse vector labels."""
        input_dict = {self.input_key: raw_inputs[self.input_key]}
        if not load_targets:
            return input_dict, {}

        features = raw_inputs.get(self.target_key, [])
        assert isinstance(features, list)
        for feat in features:
            props = feat.properties or {}
            if all(f"{head}_date" in props for head in TIMESTAMP_HEADS):
                target_dict: dict[str, Any] = {}
                for head in TIMESTAMP_HEADS:
                    target_dict[f"{head}_date"] = {
                        "date": str(props[f"{head}_date"]),
                        "valid": torch.tensor(1.0, dtype=torch.float32),
                    }
                return input_dict, target_dict

        if not self.allow_invalid:
            raise ValueError(
                f"no timestamp label feature found for "
                f"{metadata.window_group}/{metadata.window_name}"
            )

        return input_dict, {
            f"{head}_date": {
                "date": "1970-01-01T00:00:00+00:00",
                "valid": torch.tensor(0.0, dtype=torch.float32),
            }
            for head in TIMESTAMP_HEADS
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any] | list[Feature] | dict[str, Any]:
        """Convert three probability vectors to one GeoJSON feature."""
        if not isinstance(raw_output, dict):
            raise ValueError("TimestampHelperTask expects dict model outputs")

        props: dict[str, Any] = {
            "window_group": metadata.window_group,
            "window_name": metadata.window_name,
        }
        probs_by_head = {
            head: raw_output[head].detach().float().cpu() for head in TIMESTAMP_HEADS
        }
        frame_dates = raw_output.get("frame_dates")
        if not isinstance(frame_dates, list) or len(frame_dates) != NUM_CROP_MONTHS:
            raise ValueError("timestamp output is missing 60 frame_dates")

        decoded = _decode_monotonic(
            probs_by_head["pre"], probs_by_head["first"], probs_by_head["post"]
        )
        for head, idx in zip(TIMESTAMP_HEADS, decoded, strict=True):
            props[f"{head}_idx"] = idx
            props[f"{head}_date"] = frame_dates[idx]
            props[f"{head}_probs"] = probs_by_head[head].tolist()

        x = (metadata.crop_bounds[0] + metadata.crop_bounds[2]) / 2
        y = (metadata.crop_bounds[1] + metadata.crop_bounds[3]) / 2
        return [
            Feature(STGeometry(metadata.projection, shapely.Point(x, y), None), props)
        ]

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """No custom visualization."""
        return {}

    def get_metrics(self) -> MetricCollection:
        """Get timestamp metrics."""
        metrics: dict[str, Metric] = {
            "accuracy": TimestampAccuracy(),
            "accuracy_off_by_1": TimestampAccuracy(tolerance=1),
            "accuracy_off_by_2": TimestampAccuracy(tolerance=2),
        }
        for head in TIMESTAMP_HEADS:
            metrics[f"{head}_accuracy"] = TimestampAccuracy(head=head)
            metrics[f"{head}_accuracy_off_by_1"] = TimestampAccuracy(
                head=head, tolerance=1
            )
            metrics[f"{head}_accuracy_off_by_2"] = TimestampAccuracy(
                head=head, tolerance=2
            )
        return MetricCollection(metrics, compute_groups=False)
