"""Task and metrics for the LCC change model.

- ``LCCMultiTask``: injects per-window annotation metadata (consumed by
  ``transforms.StackSampler``), merges the same_change label raster into the
  pre_change raster so a single head predicts both, adds start/end timestamp
  accuracy metrics, and stacks the per-task outputs into the uint16 raster
  consumed by ``postprocess``.
- ``BalancedBinaryMetric``: sample-balanced accuracy / AUROC / PRAUC for the
  binary change task.
- ``TimestampBoundaryAccuracy``: accuracy of the per-pixel start/end timestep
  predictions at change pixels.
- ``NegativeWindowMaxScore``: mean over negative-only crops of the max change
  probability anywhere in the crop (a hallucination proxy).
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities import dim_zero_cat
from typing_extensions import override
from upath import UPath

from .prepare import PRE_CHANGE_CATEGORY_NAMES, SAME_CHANGE_CATEGORY_NAMES
from .timestamp_output import start_end_day_bands
from .transforms import ANNOTATION_KEY

# Class names for the merged pre+same change-category head: the pre_change
# classes followed by the same_change option classes (their nodata/none classes
# fold into pre's).
MERGED_PRE_SAME_CATEGORY_NAMES = (
    PRE_CHANGE_CATEGORY_NAMES + SAME_CHANGE_CATEGORY_NAMES[2:]
)

# Offset added to same_change option classes (>= 2) when merging them into the
# pre_change label raster (same class 2 becomes the first class after pre's).
SAME_TO_MERGED_OFFSET = len(PRE_CHANGE_CATEGORY_NAMES) - 2


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class BalancedBinaryMetric(Metric):
    """Sample-wise balanced accuracy, AUROC, or PRAUC for the binary change task.

    Each valid point is weighted by ``1 / (count of points of its own class
    within that sample)``, so that every sample contributes equally regardless
    of how many points it has. This prevents samples with many (auto-annotated)
    negatives from dominating samples with few (strong) negatives, mirroring the
    per-sample balancing in the model's balanced binary loss.

    Binary labels follow the segmentation convention: class 1 = no_change
    (negative), class 2 = change (positive); class 0 / invalid points are
    ignored. The AUROC/PRAUC score is the softmax probability of the change class.
    """

    def __init__(self, metric: str = "auroc") -> None:
        """Initialize accumulators for the requested balanced metric.

        Args:
            metric: which metric to compute, "accuracy", "auroc", or "prauc".
        """
        super().__init__()
        if metric not in ("accuracy", "auroc", "prauc"):
            raise ValueError(
                f"metric must be 'accuracy', 'auroc', or 'prauc', got {metric!r}"
            )
        self.metric_name = metric
        if metric == "accuracy":
            self.add_state(
                "weighted_correct",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "weighted_total",
                default=torch.tensor(0.0, dtype=torch.float64),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state("scores", default=[], dist_reduce_fx="cat")
            self.add_state("bin_labels", default=[], dist_reduce_fx="cat")
            self.add_state("weights", default=[], dist_reduce_fx="cat")

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        for pred, target in zip(preds, targets):
            label = target["classes"].get_hw_tensor().long()  # (H, W)
            valid = target["valid"].get_hw_tensor() > 0  # (H, W)

            neg = valid & (label == 1)
            pos = valid & (label == 2)
            neg_count = neg.sum()
            pos_count = pos.sum()

            weight = torch.zeros_like(label, dtype=torch.float64)
            if neg_count > 0:
                weight[neg] = 1.0 / neg_count.double()
            if pos_count > 0:
                weight[pos] = 1.0 / pos_count.double()

            # Only the two binary classes (1, 2) carry weight; everything else
            # stays at zero so it does not contribute.
            point_mask = neg | pos
            if not point_mask.any():
                continue

            if self.metric_name == "accuracy":
                pred_cls = pred.argmax(dim=0)  # (H, W)
                correct = (pred_cls == label) & point_mask
                self.weighted_correct += (weight * correct).sum().double()
                self.weighted_total += weight.sum().double()
            else:
                # Keep accumulated tensors on the metric's device so that DDP
                # all_gather works (NCCL cannot gather CPU tensors); move to CPU
                # only at compute() time for sklearn.
                change_prob = pred[2]  # (H, W) softmax prob of change class
                self.scores.append(change_prob[point_mask].detach())
                self.bin_labels.append(pos[point_mask].detach().long())
                self.weights.append(weight[point_mask].detach())

    @override
    def compute(self) -> torch.Tensor:
        if self.metric_name == "accuracy":
            if self.weighted_total > 0:
                return (self.weighted_correct / self.weighted_total).float()
            return torch.tensor(0.0)

        # After DDP sync the "cat" list states become a single tensor, so use
        # dim_zero_cat to handle both the list (pre-sync) and tensor cases.
        has_scores = (
            len(self.scores) > 0
            if isinstance(self.scores, list)
            else self.scores.numel() > 0
        )
        if not has_scores:
            return torch.tensor(float("nan"))

        scores = dim_zero_cat(self.scores).cpu().numpy()
        labels = dim_zero_cat(self.bin_labels).cpu().numpy()
        weights = dim_zero_cat(self.weights).cpu().numpy()
        # roc_auc_score/average_precision_score require both classes to be present.
        if labels.min() == labels.max():
            return torch.tensor(float("nan"))
        if self.metric_name == "prauc":
            return torch.tensor(
                float(average_precision_score(labels, scores, sample_weight=weights))
            )
        return torch.tensor(float(roc_auc_score(labels, scores, sample_weight=weights)))


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


class NegativeWindowMaxScore(Metric):
    """Mean over negative-only samples of the max change probability anywhere.

    For each sample whose labeled pixels contain no change point, this records
    the maximum predicted change probability over ALL pixels of the crop
    (labeled or not). Per-labeled-pixel metrics like the balanced AUROC cannot
    see hallucinations on the unlabeled 99.99% of each window; this metric is a
    direct proxy for the in-the-wild false-positive density. Lower is better.
    """

    def __init__(self) -> None:
        """Initialize accumulators."""
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        for pred, target in zip(preds, targets):
            label = target["classes"].get_hw_tensor().long()
            valid = target["valid"].get_hw_tensor() > 0
            has_pos = bool((valid & (label == 2)).any())
            has_neg = bool((valid & (label == 1)).any())
            if has_pos or not has_neg:
                continue
            self.total += pred[2].max().double()
            self.count += 1

    @override
    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return (self.total / self.count).float()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class LCCMultiTask(MultiTask):
    """MultiTask that injects per-window LCC annotations and start/end metrics.

    Annotations are loaded from a sidecar JSON written by the prepare script,
    keyed by "{group}/{name}". The injected metadata is consumed by
    ``transforms.StackSampler`` to compute start/end timestamp targets.

    The same_change label raster is merged into the pre_change raster so a
    single head predicts both (class layout ``MERGED_PRE_SAME_CATEGORY_NAMES``).
    On conflict (a point with both a pre and a same category) the pre category
    wins. The config must therefore have no same_change task and a pre_change
    task with the merged number of classes.
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        annotations_path: str,
    ):
        """Create a new LCCMultiTask.

        Args:
            tasks: map from task name to task object (binary, src, dst,
                pre_change, post_change).
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

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Merge same into pre, process inputs, and inject annotation metadata.

        When load_targets=False (predict mode), neither the merge nor the
        annotation injection happens since there are no label rasters and the
        StackSampler transform is not used during prediction.
        """
        if load_targets:
            raw_inputs = dict(raw_inputs)
            pre = raw_inputs["label_pre_change"]
            same = raw_inputs.pop("label_same_change")
            assert isinstance(pre, RasterImage) and isinstance(same, RasterImage)
            # Both label rasters share the same valid mask (class 0 = nodata,
            # class 1 = "none" when a sibling category field is set), so the
            # merge only fills same categories where pre has no category.
            merged = pre.image.clone()
            fill = (pre.image == 1) & (same.image >= 2)
            merged[fill] = same.image[fill] + SAME_TO_MERGED_OFFSET
            raw_inputs["label_pre_change"] = RasterImage(image=merged)

        input_dict, target_dict = super().process_inputs(
            raw_inputs, metadata=metadata, load_targets=load_targets
        )
        if load_targets:
            ann = self._load_annotations().get(
                f"{metadata.window_group}/{metadata.window_name}"
            )
            if ann is None:
                raise KeyError(
                    f"No annotation found for {metadata.window_group}/"
                    f"{metadata.window_name} in {self.annotations_path}"
                )
            input_dict[ANNOTATION_KEY] = {
                "pre_change": _parse_date(ann["pre_change"]),
                "post_change": _parse_date(ann["post_change"]),
                "first_noticeable": _parse_date(ann["first_noticeable"]),
            }
        return input_dict, target_dict

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[np.uint16]:
        """Stack per-task outputs into a 57-band uint16 CHW array.

        The two timestamp bands hold the predicted pre-change and post-change
        dates as integer days since ``TIMESTAMP_EPOCH``, derived from the per-pixel
        argmax of the start/end distributions over the input timesteps (mapped to
        real dates via ``raw_output["timestep_days"]``). Probability bands are
        stored as 0..255 within the uint16 raster.

        Band layout (see ``postprocess.OUTPUT_BANDS``):
        0..2   = binary (softmax probs)
        3..15  = src (softmax probs)
        16..28 = dst (softmax probs)
        29     = ts_pre_days (days since epoch)
        30     = ts_post_days (days since epoch)
        31..41 = pre_change, merged pre+same categories (softmax probs)
        42..56 = post_change (softmax probs)
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

        for task_name in ("pre_change", "post_change"):
            probs = raw_output[task_name].float()
            parts.append(
                (probs * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint16)
            )

        return np.concatenate(parts, axis=0)

    def get_metrics(self) -> MetricCollection:
        """Get binary/src/dst metrics plus start/end timestamp accuracy."""
        metrics = super().get_metrics()
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
