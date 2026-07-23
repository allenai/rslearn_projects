"""Self-contained single-pass LCC change model, task, and transform.

This module is intentionally self-contained: it does NOT import from the other
``model*``/``transform*``/``task*`` modules in this package. It bundles together
everything ``config_pass20_v1_2.yaml`` needs:

- ``SinglePassSampler``: builds a 20-image ``sentinel2_l2a`` stack (16 quarterly +
  4 frequent) and per-pixel start/end timestep-index targets.
- ``SinglePassMultiTask``: injects per-window annotation metadata and adds
  start/end timestamp accuracy metrics. ``BalancedBinaryMetric`` is provided here
  too (referenced by the config for the binary task's accuracy/auroc).
- ``SinglePassChangeModel``: a SINGLE OlmoEarth forward pass over all 20 timesteps
  (``token_pooling=False`` so per-timestep tokens are preserved), then:
    * binary/src/dst: mean-pool the per-timestep tokens over time and run per-task
      conv decoders that upsample back to full resolution.
    * start/end timestamps: a per-token linear logit over the T timesteps,
      upsampled to full resolution, trained with cross-entropy at change pixels.
  There is NO temporal transformer and NO temporal positional encoding.

This is the single-pass analogue of the dual-pass temporal models, relying on the
latest OlmoEarth being able to ingest all 20 timesteps in one forward pass.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.train.tasks.multi_task import MetricWrapper, MultiTask
from rslearn.train.tasks.task import Task
from rslearn.train.transforms.transform import Transform
from rslearn.utils import Feature
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities import dim_zero_cat
from typing_extensions import override
from upath import UPath

from .timestamp_encoding import timestamps_to_days
from .timestamp_output import start_end_day_bands

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUARTERLY_KEY = "sentinel2_quarterly"
FREQUENT_KEY_PREFIX = "sentinel2_frequent_"
ANNOTATION_KEY = "_lcc_annotation"
INPUT_KEY = "sentinel2_l2a"
OUTPUT_KEY = "sentinel2_l2a"

NUM_QUARTERLY = 16
NUM_FREQUENT = 4

# Change-category heads (predicted per positive point, "none"-aware). These mirror
# the src/dst segmentation heads and are trained/masked independently.
CHANGE_CATEGORY_TASKS = ("pre_change", "post_change", "same_change")

# Spacing used for the fake timestamps assigned to padding (duplicated) images.
QUARTERLY_PERIOD = timedelta(days=90)

# A stage is a list of (out_channels, kernel_size) conv specs.
StageSpec = list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _change_index(centers: list[datetime], target: datetime, is_start: bool) -> int:
    """Index of the timestep closest to a change boundary.

    For the start boundary, returns the latest center that is <= target (the
    last image before/at the change start). For the end boundary, returns the
    earliest center that is >= target. Defaults to the first/last index when no
    center satisfies the condition.
    """
    if is_start:
        candidates = [i for i, c in enumerate(centers) if c <= target]
        return candidates[-1] if candidates else 0
    candidates = [i for i, c in enumerate(centers) if c >= target]
    return candidates[0] if candidates else len(centers) - 1


def _build_quarterly_stack(
    quarterly: RasterImage, valid_indices: list[int]
) -> tuple[torch.Tensor, list[tuple[datetime, datetime]]]:
    """Build exactly NUM_QUARTERLY quarterly images and their timestamps.

    Uses the (already-trimmed) ``valid_indices`` in chronological order. When
    fewer than NUM_QUARTERLY are available, prepends copies of the earliest
    selected image with progressively-older fake timestamps so the encoder sees
    distinct timesteps (it rejects duplicate timestamps within a modality when
    ``token_pooling=False``).
    """
    assert quarterly.timestamps is not None
    q_ts = [quarterly.timestamps[i] for i in valid_indices]

    img_parts: list[torch.Tensor] = []
    if valid_indices:
        img_parts.append(quarterly.image[:, valid_indices, :, :])

    if len(valid_indices) < NUM_QUARTERLY:
        deficit = NUM_QUARTERLY - len(valid_indices)
        base_idx = valid_indices[0] if valid_indices else 0
        base_ts = q_ts[0][0] if q_ts else quarterly.timestamps[0][0]
        pad_img = quarterly.image[:, base_idx : base_idx + 1, :, :].repeat(
            1, deficit, 1, 1
        )
        # Oldest first; each padding step is one QUARTERLY_PERIOD older than the
        # next, all strictly older than base_ts.
        pad_ts = [
            (base_ts - QUARTERLY_PERIOD * (deficit - k),) * 2 for k in range(deficit)
        ]
        img_parts.insert(0, pad_img)
        q_ts = pad_ts + q_ts

    return torch.cat(img_parts, dim=1), q_ts


def _make_decoder(
    in_dim: int, stages: list[StageSpec], num_classes: int
) -> nn.Sequential:
    """Build a per-task convolutional decoder from explicit stage specs.

    Each stage is a list of (out_channels, kernel_size) convs (each followed by
    ReLU). A 2x bilinear upsample precedes every stage after the first, so the
    output is at 2^(len(stages)-1) times the input resolution. A final 1x1 conv
    produces num_classes channels.

    Takes (B, in_dim, H, W) and produces (B, num_classes, H*2^(n-1), W*2^(n-1))
    where n = len(stages).
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for i, stage in enumerate(stages):
        if i > 0:
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        for out_ch, k in stage:
            layers.append(nn.Conv2d(prev, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU(inplace=True))
            prev = out_ch
    layers.append(nn.Conv2d(prev, num_classes, kernel_size=1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class SinglePassSampler(Transform):
    """Pick a frequent option, take 16 quarterly, build sentinel2_l2a + idx targets.

    Builds the 20-image stack (16 quarterly + 4 frequent) and emits the per-pixel
    start/end timestep-index targets consumed by the start/end heads. Unlike the
    dual-pass samplers, the whole stack is fed to a single encoder pass, so the
    start/end indices index over all 20 chronological timesteps.
    """

    def __init__(self, deterministic: bool = False) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test). Option 2
                is a mid-range temporal context rather than the hardest
                immediate-detection case.
        """
        super().__init__()
        self.deterministic = deterministic

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample a frequent option, take 16 quarterly, produce sentinel2_l2a."""
        ann = input_dict.pop(ANNOTATION_KEY, None)
        if ann is None:
            raise KeyError(
                f"Expected {ANNOTATION_KEY!r} in input_dict; is SinglePassMultiTask "
                "in use?"
            )

        pre_change: datetime = ann["pre_change"]
        post_change: datetime = ann["post_change"]

        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        # Collect available frequent options.
        frequent_options: list[RasterImage] = []
        for i in range(8):
            key = f"{FREQUENT_KEY_PREFIX}{i}"
            if key not in input_dict:
                continue
            freq_img = input_dict.pop(key)
            if freq_img.image.shape[1] == NUM_FREQUENT:
                frequent_options.append(freq_img)

        if not frequent_options:
            raise ValueError("No valid frequent options available")

        # Pick an option.
        if self.deterministic:
            opt_idx = 2 if len(frequent_options) > 2 else 0
        else:
            opt_idx = random.randrange(len(frequent_options))
        chosen_frequent = frequent_options[opt_idx]

        if not chosen_frequent.timestamps:
            raise ValueError("Frequent option must have timestamps")

        # Quarterly images end where the frequent block begins.
        earliest_freq_ts = min(ts[0] for ts in chosen_frequent.timestamps)

        # Strict inequality so a quarterly scene captured exactly at the frequent
        # block start (the same Sentinel-2 scene) is not pulled in as a baseline
        # image, which would create a duplicate timestamp with the first frequent.
        valid_indices = [
            i for i, ts in enumerate(quarterly.timestamps) if ts[1] < earliest_freq_ts
        ]
        # Take the most recent NUM_QUARTERLY candidates (consecutive, no skipping).
        valid_indices = valid_indices[-NUM_QUARTERLY:]
        q_img, q_ts = _build_quarterly_stack(quarterly, valid_indices)

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + chosen_frequent.timestamps
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Compute start/end timestamp index targets over the chronological steps.
        centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in combined_ts]
        start_idx = _change_index(centers, pre_change, is_start=True)
        end_idx = _change_index(centers, post_change, is_start=False)

        H, W = quarterly.image.shape[2], quarterly.image.shape[3]
        start_map = torch.zeros(H, W, dtype=torch.long)
        end_map = torch.zeros(H, W, dtype=torch.long)

        if "binary" in target_dict:
            binary_classes = target_dict["binary"]["classes"].get_hw_tensor()
            change_mask = binary_classes == 2
            start_map[change_mask] = start_idx
            end_map[change_mask] = end_idx
            valid_mask = change_mask.float()
        else:
            valid_mask = torch.ones(H, W, dtype=torch.float32)

        target_dict["timestamps"] = {
            "start": RasterImage(image=start_map[None, None, :, :]),
            "end": RasterImage(image=end_map[None, None, :, :]),
            "valid": RasterImage(image=valid_mask[None, None, :, :]),
        }

        # Mask dst loss when the latest frequent image is before post_change, since
        # the model can't predict destination land cover without post-change imagery.
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        if latest_freq_ts < post_change and "dst" in target_dict:
            dst_valid = target_dict["dst"]["valid"]
            target_dict["dst"]["valid"] = RasterImage(
                image=torch.zeros_like(dst_valid.image)
            )

        return input_dict, target_dict


class SinglePassPredictBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is one ``sentinel2_frequent_0`` layer with four 15-day
    periods and no annotation sidecar. This takes the last 16 quarterly images and
    concatenates 16 quarterly + 4 frequent = 20 into ``sentinel2_l2a``, matching the
    20-timestep stack the model is trained on.
    """

    def __init__(self) -> None:
        """Initialize the prediction pass builder."""
        super().__init__()

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last 16 quarterly + 4 frequent into sentinel2_l2a."""
        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        frequent: RasterImage = input_dict.pop(f"{FREQUENT_KEY_PREFIX}0")
        if frequent.image.shape[1] != NUM_FREQUENT:
            raise ValueError(
                f"Expected prediction frequent layer to have {NUM_FREQUENT} "
                f"timesteps, got {frequent.image.shape[1]}"
            )

        T = quarterly.image.shape[1]
        start = max(0, T - NUM_QUARTERLY)
        indices = list(range(start, T))
        if len(indices) < NUM_QUARTERLY:
            deficit = NUM_QUARTERLY - len(indices)
            pad_idx = indices[0] if indices else 0
            indices = [pad_idx] * deficit + indices

        q_img = quarterly.image[:, indices, :, :]
        q_ts = [quarterly.timestamps[i] for i in indices]

        combined_img = torch.cat([q_img, frequent.image], dim=1)
        combined_ts = q_ts + (frequent.timestamps or [])
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        return input_dict, target_dict


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


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class SinglePassMultiTask(MultiTask):
    """MultiTask that injects per-window LCC annotations and start/end metrics.

    Annotations are loaded from a sidecar JSON written by the prepare script,
    keyed by "{group}/{name}". The injected metadata is consumed by
    ``SinglePassSampler`` to compute start/end timestamp targets.
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        annotations_path: str,
    ):
        """Create a new SinglePassMultiTask.

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

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process inputs and inject annotation metadata.

        When load_targets=False (predict mode), the annotation is not injected
        since the SinglePassSampler transform is not used during prediction.
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
        """Stack per-task outputs into a 56-band uint16 CHW array.

        The two timestamp bands hold the predicted pre-change and post-change
        dates as integer days since ``TIMESTAMP_EPOCH``, derived from the per-pixel
        argmax of the start/end distributions over the input timesteps (mapped to
        real dates via ``raw_output["timestep_days"]``). Probability bands are
        stored as 0..255 within the uint16 raster.

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

        for task_name in CHANGE_CATEGORY_TASKS:
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SinglePassChangeModel(nn.Module):
    """Single OlmoEarth pass over all timesteps with per-task decoders.

    The model expects ``sentinel2_l2a`` with ``num_timesteps`` timesteps (e.g. 16
    quarterly + 4 frequent = 20) built by ``SinglePassSampler``. It runs a SINGLE
    OlmoEarth forward pass with ``token_pooling=False`` so per-timestep tokens are
    preserved, giving ``(B, C, H, W, T)`` features. There is no temporal
    transformer and no temporal positional encoding:

    - binary/src/dst: the per-timestep tokens are aggregated over time
      (``temporal_aggregation``) and run through per-task conv decoders that
      upsample the 1/patch_size features back to full resolution. The default
      ``"mean"`` mean-pools over time; ``"diff"`` uses the last-minus-first
      timestep embedding (post minus pre) to emphasize change.
    - start/end: a per-token linear head produces one logit per timestep,
      upsampled to full resolution, trained with cross-entropy over the T
      timesteps at change pixels.
    """

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_classes_pre_change: int = 7,
        num_classes_post_change: int = 12,
        num_classes_same_change: int = 6,
        num_timesteps: int = 20,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 2.0,
        temporal_aggregation: str = "mean",
    ):
        """Initialize the single-pass LCC model.

        Args:
            encoder: the OlmoEarth encoder. Must be configured with
                ``token_pooling=False`` so per-token features are returned.
            num_classes_binary: number of classes for the binary change task.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_classes_pre_change: number of pre_change_category classes
                (including nodata and "none").
            num_classes_post_change: number of post_change_category classes
                (including nodata and "none").
            num_classes_same_change: number of same_change_category classes
                (including nodata and "none").
            num_timesteps: expected number of input timesteps (e.g. 20).
            embedding_dim: per-token encoder embedding size (768 for BASE). This
                is the decoder input dim (a single pass, aggregated over time).
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size)
                so outputs are full resolution. Required.
            binary_loss_weight: multiplier applied to the binary change loss.
            temporal_aggregation: how the per-timestep tokens are collapsed into
                the segmentation feature. ``"mean"`` mean-pools over time;
                ``"diff"`` uses the last-minus-first timestep embedding.
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.binary_loss_weight = binary_loss_weight

        if temporal_aggregation not in ("mean", "diff"):
            raise ValueError(
                "temporal_aggregation must be 'mean' or 'diff', got "
                f"{temporal_aggregation!r}"
            )
        self.temporal_aggregation = temporal_aggregation

        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        # Segmentation decoders consume the mean-over-time feature.
        self.decoder_binary = _make_decoder(
            embedding_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(embedding_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(embedding_dim, decoder_stages, num_classes_dst)

        # Change-category decoders (pre/post/same), consuming the same aggregated
        # feature as src/dst.
        self.decoder_pre_change = _make_decoder(
            embedding_dim, decoder_stages, num_classes_pre_change
        )
        self.decoder_post_change = _make_decoder(
            embedding_dim, decoder_stages, num_classes_post_change
        )
        self.decoder_same_change = _make_decoder(
            embedding_dim, decoder_stages, num_classes_same_change
        )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst
        self.num_classes_pre_change = num_classes_pre_change
        self.num_classes_post_change = num_classes_post_change
        self.num_classes_same_change = num_classes_same_change

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Single forward pass with aggregated-time decoders and start/end heads.

        Args:
            context: ModelContext with ``sentinel2_l2a`` RasterImage (num_timesteps).
            targets: optional target dicts with "binary", "src", "dst",
                "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses.
        """
        # Single encoder pass; token_pooling=False -> per-timestep tokens.
        token_feature_maps = self.encoder(context)
        feature = token_feature_maps.feature_maps[0]  # (B, C, H, W, T)

        # Segmentation: aggregate over time, then per-task decoders.
        if self.temporal_aggregation == "diff":
            seg_feat = feature[..., -1] - feature[..., 0]  # (B, C, H, W)
        else:
            seg_feat = feature.mean(dim=-1)  # (B, C, H, W)
        logits_binary = self.decoder_binary(seg_feat)
        logits_src = self.decoder_src(seg_feat)
        logits_dst = self.decoder_dst(seg_feat)
        change_logits = self._change_category_logits(seg_feat)

        # Per-token timestamp logits over T, upsampled to full resolution.
        xt = feature.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
        start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)  # (B,T,H,W)
        end_logits = self.end_head(xt).squeeze(-1).permute(0, 3, 1, 2)
        scale = self.encoder.patch_size
        start_logits = F.interpolate(
            start_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )
        end_logits = F.interpolate(
            end_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )

        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            self._add_change_category_losses(change_logits, targets, losses)
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            ts_image = context.inputs[i].get(INPUT_KEY)
            timestep_days = (
                timestamps_to_days(ts_image.timestamps)
                if isinstance(ts_image, RasterImage) and ts_image.timestamps is not None
                else None
            )
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    **{
                        name: F.softmax(change_logits[name][i], dim=0)
                        for name in change_logits
                    },
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                    "timestep_days": timestep_days,
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)

    def _change_category_logits(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the pre/post/same change-category decoders on ``feat``.

        Args:
            feat: aggregated segmentation feature (B, C, H, W), same as src/dst.

        Returns:
            Map from change-category task name to logits (B, num_classes, H', W').
        """
        return {
            "pre_change": self.decoder_pre_change(feat),
            "post_change": self.decoder_post_change(feat),
            "same_change": self.decoder_same_change(feat),
        }

    def _add_change_category_losses(
        self,
        change_logits: dict[str, torch.Tensor],
        targets: list[dict[str, Any]] | None,
        losses: dict[str, torch.Tensor],
    ) -> None:
        """Add masked cross-entropy losses for any configured change-category head.

        Each head is optional: a loss is only added when the corresponding target
        is present (so configs without these tasks are unaffected). Per-pixel
        masking (via the label rasters) already encodes the "train only when at
        least one field is set" rule.
        """
        if targets is None:
            return
        for name, logits in change_logits.items():
            if name in targets[0]:
                losses[f"{name}_cls"] = self._seg_loss(logits, targets, name)

    def _seg_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        task_name: str,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss for a segmentation task."""
        labels = torch.stack(
            [t[task_name]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t[task_name]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")
        return (loss * valid).sum() / valid.sum()

    def _balanced_binary_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Balanced binary loss with per-sample balancing.

        For each sample, the loss is the mean over its change points plus the
        mean over its no-change points (each group divided by its own point
        count). Every sample with any valid points contributes equally.
        """
        labels = torch.stack(
            [t["binary"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["binary"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)

        change_mask = (valid & (labels == 2)).flatten(1).float()  # (B, H*W)
        nochange_mask = (valid & (labels == 1)).flatten(1).float()
        loss_flat = loss.flatten(1)  # (B, H*W)

        pos_count = change_mask.sum(dim=1)  # (B,)
        neg_count = nochange_mask.sum(dim=1)
        has_pos = pos_count > 0
        has_neg = neg_count > 0

        pos_mean = (loss_flat * change_mask).sum(dim=1) / pos_count.clamp(min=1)
        neg_mean = (loss_flat * nochange_mask).sum(dim=1) / neg_count.clamp(min=1)

        sample_loss = pos_mean * has_pos + neg_mean * has_neg  # (B,)
        has_any = has_pos | has_neg
        return sample_loss[has_any].mean()

    def _timestamp_ce(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        key: str,
    ) -> torch.Tensor:
        """Masked cross-entropy over the T timesteps for the start/end boundary.

        ``logits`` is (B, T, H, W); the target is the per-pixel timestep index
        (B, H, W). Loss is averaged over valid (change) pixels only.
        """
        idx = torch.stack(
            [t["timestamps"][key].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["timestamps"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, idx, reduction="none")  # (B, H, W)
        return (loss * valid).sum() / valid.sum()
