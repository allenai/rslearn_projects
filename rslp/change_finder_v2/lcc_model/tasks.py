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
from sklearn.metrics import roc_auc_score
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC
from torchmetrics.utilities import dim_zero_cat
from typing_extensions import override
from upath import UPath

from .timestamp_output import membership_day_bands
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
        self.auroc = BinaryAUROC()

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for pred, target in zip(preds, targets):
            classes = target["classes"].image[:, 0, :, :]  # (num_ts, H, W)
            valid = target["valid"].get_hw_tensor() > 0  # (H, W)
            valid_exp = valid.unsqueeze(0).expand_as(pred)  # (num_ts, H, W)
            all_preds.append(pred[valid_exp])
            all_labels.append(classes[valid_exp].long())
        if all_preds:
            self.auroc.update(torch.cat(all_preds), torch.cat(all_labels))

    @override
    def compute(self) -> torch.Tensor:
        return self.auroc.compute()


class BalancedBinaryMetric(Metric):
    """Sample-wise balanced accuracy or AUROC for the binary change task.

    Each valid point is weighted by ``1 / (count of points of its own class
    within that sample)``, so that every sample contributes equally regardless
    of how many points it has. This prevents samples with many (auto-annotated)
    negatives from dominating samples with few (strong) negatives, mirroring the
    per-sample balancing in DualPassChangeModel._balanced_binary_loss.

    Binary labels follow the segmentation convention: class 1 = no_change
    (negative), class 2 = change (positive); class 0 / invalid points are
    ignored. The AUROC score is the softmax probability of the change class.

    Returns a single scalar. Instantiate once per ``metric`` ("accuracy" or
    "auroc") so the metric name (and therefore the logged key, e.g.
    ``val_binary/auroc``) comes from the config key rather than a dict.
    """

    def __init__(self, metric: str = "auroc") -> None:
        """Initialize accumulators for the requested balanced metric.

        Args:
            metric: which metric to compute, "accuracy" or "auroc".
        """
        super().__init__()
        if metric not in ("accuracy", "auroc"):
            raise ValueError(f"metric must be 'accuracy' or 'auroc', got {metric!r}")
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
        # roc_auc_score requires both classes to be present.
        if labels.min() == labels.max():
            return torch.tensor(float("nan"))
        return torch.tensor(float(roc_auc_score(labels, scores, sample_weight=weights)))


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
    ) -> npt.NDArray[np.uint16]:
        """Stack per-task outputs into a single 56-band uint16 CHW array.

        The two timestamp bands hold the predicted pre-change and post-change
        dates as integer days since ``TIMESTAMP_EPOCH``: the earliest and latest
        timesteps whose per-timestep membership exceeds 0.5, mapped to real dates
        via ``raw_output["timestep_days"]``. Probability bands are stored as 0..255
        within the uint16 raster.

        Band layout:
        0..2 = binary (softmax probs)
        3..15 = src (softmax probs)
        16..28 = dst (softmax probs)
        29 = ts_pre_days (days since epoch)
        30 = ts_post_days (days since epoch)
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

        membership = raw_output["timestamps"].float()
        timestep_days = raw_output.get("timestep_days")
        if timestep_days is not None:
            day_bands = membership_day_bands(membership, timestep_days)
            parts.append(day_bands.cpu().numpy().astype(np.uint16))
        else:
            h, w = membership.shape[-2:]
            parts.append(np.zeros((2, h, w), dtype=np.uint16))

        for task_name in ("pre_change", "post_change", "same_change"):
            probs = raw_output[task_name].float()
            parts.append(
                (probs * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint16)
            )

        return np.concatenate(parts, axis=0)
