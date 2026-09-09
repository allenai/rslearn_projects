"""Transition-segmentation LCC model (single combined head).

This variant keeps the dual-pass temporal backbone and the start/end timestamp
heads of ``model_20260611``/``model_20260610``, but collapses the three separate
binary/src/dst segmentation heads into ONE multiclass segmentation head over
land-cover *transition groups*.

The dataset is unchanged: the combined transition label is derived at train time
from the existing ``label_binary``/``label_src``/``label_dst`` rasters by the
``TransitionLabelBuilder`` transform below (which runs after the existing
``FrequentOptionSamplerV2`` + ``Flip``).

Classes (10):
- 0 ``no_change`` (binary == 1)
- 1..9 semantic transition groups; a change pixel (binary == 2) is mapped via its
  ``(src_id, dst_id)`` pair. Any change pixel whose pair is not listed, whose
  ``src``/``dst`` is 0, or whose ``dst`` is not visible (the sampler zeros
  ``target/dst/valid`` when post-change imagery is unavailable) is masked out.

Everything for this experiment (transform, model, task) lives in this one file.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage
from rslearn.train.tasks.multi_task import MetricWrapper
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.train.tasks.task import Task
from rslearn.train.transforms.transform import Transform
from torchmetrics import MetricCollection
from typing_extensions import override

from .model import StageSpec, _make_decoder
from .model_20260610 import _make_simple_decoder
from .model_20260611 import DualPassTemporalChangeModel
from .tasks_20260610 import LCCMultiTaskV2, TimestampBoundaryAccuracy
from .transforms import (
    ANNOTATION_KEY,
    FREQUENT_KEY_PREFIX,
    NUM_FREQUENT,
    NUM_QUARTERLY,
    OUTPUT_KEY,
    QUARTERLY_KEY,
)
from .transforms_20260610 import _change_index
from .transforms_20260611 import _build_quarterly_stack

# Land cover category ids (must match prepare.CATEGORY_NAMES / config class_names).
BARE = 1
CROPS = 3
GRASSLAND = 5
SHRUB = 7
TREE = 9
URBAN = 10
WATER = 11
WETLAND = 12

NUM_CATEGORIES = 13

# Binary label convention (matches prepare.py).
BIN_NO_CHANGE = 1
BIN_CHANGE = 2

# Ordered list of (group_name, [(src_id, dst_id), ...]). The group index in this
# list is offset by 1 (class 0 is reserved for no_change).
TRANSITION_GROUPS: list[tuple[str, list[tuple[int, int]]]] = [
    (
        "deforestation",
        [
            (TREE, GRASSLAND),
            (TREE, CROPS),
            (TREE, BARE),
            (TREE, URBAN),
            (SHRUB, GRASSLAND),
        ],
    ),
    (
        "urban_expansion",
        [
            (CROPS, URBAN),
            (BARE, URBAN),
            (GRASSLAND, URBAN),
            (WATER, URBAN),
            (URBAN, URBAN),
            (SHRUB, URBAN),
        ],
    ),
    (
        "construction_mining",
        [
            (CROPS, BARE),
            (GRASSLAND, BARE),
        ],
    ),
    (
        "from_water",
        [
            (WATER, BARE),
            (WATER, GRASSLAND),
        ],
    ),
    (
        "to_water",
        [
            (TREE, WATER),
            (BARE, WATER),
            (GRASSLAND, WATER),
            (CROPS, WATER),
        ],
    ),
    (
        "urban_erosion",
        [
            (URBAN, CROPS),
            (URBAN, BARE),
            (URBAN, GRASSLAND),
        ],
    ),
    (
        "new_crop_field",
        [
            (BARE, CROPS),
            (GRASSLAND, CROPS),
            (WATER, CROPS),
        ],
    ),
    (
        "wetland_loss",
        [
            (WETLAND, WATER),
        ],
    ),
    (
        "forest_regrowth",
        [
            (GRASSLAND, TREE),
        ],
    ),
]

TRANSITION_CLASS_NAMES: list[str] = ["no_change"] + [
    name for name, _ in TRANSITION_GROUPS
]
NUM_TRANSITION_CLASSES = len(TRANSITION_CLASS_NAMES)

# Map (src_id, dst_id) -> transition class index (1..9).
PAIR_TO_GROUP: dict[tuple[int, int], int] = {}
for _group_idx, (_name, _pairs) in enumerate(TRANSITION_GROUPS, start=1):
    for _pair in _pairs:
        PAIR_TO_GROUP[_pair] = _group_idx


def _build_pair_lut() -> torch.Tensor:
    """Build a flat lookup table mapping ``src * NUM_CATEGORIES + dst`` to a class.

    Unmapped pairs are -1. Because every mapped pair has ``src >= 1`` and
    ``dst >= 1``, pairs with a zero (nodata) source or destination never collide
    with a real group.
    """
    lut = torch.full((NUM_CATEGORIES * NUM_CATEGORIES,), -1, dtype=torch.long)
    for (src_id, dst_id), group_idx in PAIR_TO_GROUP.items():
        lut[src_id * NUM_CATEGORIES + dst_id] = group_idx
    return lut


# Canonical (src_id, dst_id) pair for each transition group, chosen as the pair
# with the highest annotation count. Used at prediction time to fill the legacy
# 49-band src/dst outputs from the single transition argmax.
GROUP_TO_PAIR: dict[int, tuple[int, int]] = {
    1: (TREE, GRASSLAND),  # deforestation
    2: (CROPS, URBAN),  # urban_expansion
    3: (GRASSLAND, BARE),  # construction_mining
    4: (WATER, BARE),  # from_water
    5: (TREE, WATER),  # to_water
    6: (URBAN, CROPS),  # urban_erosion
    7: (BARE, CROPS),  # new_crop_field
    8: (WETLAND, WATER),  # wetland_loss
    9: (GRASSLAND, TREE),  # forest_regrowth
}

# Per-group representative src/dst class ids, indexed by group (0 = no_change ->
# id 0, never used at change pixels). Module-level so they are built once.
_GROUP_SRC_LUT = torch.tensor(
    [0] + [GROUP_TO_PAIR[g][0] for g in range(1, NUM_TRANSITION_CLASSES)],
    dtype=torch.long,
)
_GROUP_DST_LUT = torch.tensor(
    [0] + [GROUP_TO_PAIR[g][1] for g in range(1, NUM_TRANSITION_CLASSES)],
    dtype=torch.long,
)


class FrequentOptionSampler(Transform):
    """Frequent-option sampler for the transition model.

    Self-contained copy of ``transforms_20260611.FrequentOptionSamplerV2`` with two
    changes:

    1. Option selection prefers options whose destination is observable, i.e. the
       last frequent image timestamp is on/after ``post_change``. If no option
       qualifies, it falls back to a random option (train) / the first option
       (val/test). This avoids wasting change points whose destination land cover
       is not yet visible.
    2. It never masks dst. For the combined transition task, masking dst would drop
       the entire transition label at change pixels; option selection handles
       destination visibility instead, so the label is always kept.
    """

    def __init__(self, deterministic: bool = False) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, pick deterministically (for val/test): the first
                destination-visible option, else the first option overall.
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
                f"Expected {ANNOTATION_KEY!r} in input_dict; is LCCMultiTask in use?"
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

        # Prefer options whose destination is visible (last frequent image on/after
        # post_change); fall back to all options if none qualify.
        eligible = [
            i
            for i, opt in enumerate(frequent_options)
            if opt.timestamps and max(ts[1] for ts in opt.timestamps) >= post_change
        ]
        candidates = eligible if eligible else list(range(len(frequent_options)))
        if self.deterministic:
            opt_idx = candidates[0]
        else:
            opt_idx = random.choice(candidates)
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

        # NOTE: unlike FrequentOptionSamplerV2, dst is never masked here. Option
        # selection above already prefers destination-visible options, and for the
        # combined transition task masking dst would drop the whole label.

        return input_dict, target_dict


class TransitionLabelBuilder(Transform):
    """Derive the combined ``transition`` target from binary/src/dst labels.

    Reads the per-pixel ``binary``/``src``/``dst`` targets (already produced by the
    sub-tasks and, at train time, already flipped by ``Flip``) and writes a single
    ``transition`` target with the 10-class scheme. Should run AFTER
    ``FrequentOptionSamplerV2`` (which sets the dst-visibility mask) and AFTER
    ``Flip``.
    """

    def __init__(self) -> None:
        """Initialize the transition label builder."""
        super().__init__()
        self.register_buffer("pair_lut", _build_pair_lut(), persistent=False)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build ``target_dict['transition']`` from binary/src/dst targets."""
        for key in ("binary", "src", "dst"):
            if key not in target_dict:
                raise KeyError(
                    f"Expected {key!r} target; are the binary/src/dst sub-tasks "
                    f"configured in LCCTransitionMultiTask?"
                )

        binary = target_dict["binary"]["classes"].get_hw_tensor().long()
        binary_valid = target_dict["binary"]["valid"].get_hw_tensor() > 0
        src = target_dict["src"]["classes"].get_hw_tensor().long()
        dst = target_dict["dst"]["classes"].get_hw_tensor().long()
        dst_valid = target_dict["dst"]["valid"].get_hw_tensor() > 0

        h, w = binary.shape
        classes = torch.zeros((h, w), dtype=torch.long)
        valid = torch.zeros((h, w), dtype=torch.float32)

        # No-change points -> class 0.
        no_change = binary_valid & (binary == BIN_NO_CHANGE)
        valid[no_change] = 1.0

        # Change points -> mapped group, only when dst is visible and the pair is
        # in our table.
        src_idx = src.clamp(0, NUM_CATEGORIES - 1)
        dst_idx = dst.clamp(0, NUM_CATEGORIES - 1)
        group = self.pair_lut[src_idx * NUM_CATEGORIES + dst_idx]  # (H, W), -1 if none
        change_ok = (binary == BIN_CHANGE) & dst_valid & (group >= 0)
        classes[change_ok] = group[change_ok]
        valid[change_ok] = 1.0

        target_dict["transition"] = {
            "classes": RasterImage(image=classes[None, None, :, :]),
            "valid": RasterImage(image=valid[None, None, :, :]),
        }
        return input_dict, target_dict


class TransitionDualPassModel(DualPassTemporalChangeModel):
    """Dual-pass temporal model with a single transition head + start/end heads."""

    def __init__(
        self,
        encoder: Any,
        num_classes: int = NUM_TRANSITION_CLASSES,
        num_timesteps: int = 20,
        num_pass1: int = 10,
        embedding_dim: int = 768,
        temporal_dim: int | None = None,
        decoder_stages: list[StageSpec] | None = None,
        simple_decoder: bool = False,
        temporal_depth: int = 1,
        temporal_heads: int = 8,
        dim_feedforward: int = 2048,
        transition_loss_weight: float = 1.0,
    ):
        """Initialize the transition dual-pass model.

        Args mirror ``DualPassTemporalChangeModel`` except the three seg heads are
        replaced by a single ``num_classes``-way transition head.

        Args:
            encoder: the OlmoEarth encoder (token_pooling=False).
            num_classes: number of transition classes (default 10).
            num_timesteps: total input timesteps across both passes.
            num_pass1: timesteps routed to the first encoder pass.
            embedding_dim: per-token encoder embedding size.
            temporal_dim: temporal transformer / decoder dim; defaults to
                ``embedding_dim``.
            decoder_stages: conv decoder spec (required unless simple_decoder).
            simple_decoder: use a 1x1 conv + PixelShuffle decoder instead.
            temporal_depth: number of temporal transformer layers.
            temporal_heads: number of attention heads.
            dim_feedforward: temporal transformer FFN hidden size.
            transition_loss_weight: multiplier on the transition loss.
        """
        # Build the module graph directly (skip the parent decoders we don't need).
        nn.Module.__init__(self)
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.temporal_dim = temporal_dim if temporal_dim is not None else embedding_dim
        self.num_timesteps = num_timesteps
        self.num_pass1 = num_pass1
        self.num_classes = num_classes
        self.transition_loss_weight = transition_loss_weight

        self.input_proj = (
            nn.Linear(embedding_dim, self.temporal_dim)
            if self.temporal_dim != embedding_dim
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_depth
        )

        if simple_decoder:
            self.decoder_transition = _make_simple_decoder(
                self.temporal_dim, num_classes, encoder.patch_size
            )
        else:
            if decoder_stages is None:
                raise ValueError(
                    "decoder_stages must be specified when simple_decoder is False"
                )
            self.decoder_transition = _make_decoder(
                self.temporal_dim, decoder_stages, num_classes
            )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(self.temporal_dim, 1)
        self.end_head = nn.Linear(self.temporal_dim, 1)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Dual-pass features -> temporal transformer -> transition + ts heads."""
        feature = self._per_timestep_features(context)  # (B, C, H, W, T)
        b, c, h, w, t = feature.shape

        x = feature.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
        x = self.input_proj(x)
        x = self._add_temporal_pos(x)
        x = self.temporal_encoder(x)
        x = x.reshape(b, h, w, t, self.temporal_dim).permute(0, 4, 1, 2, 3)

        seg_feat = self._pool_time(x)  # (B, C, H, W)
        logits_transition = self.decoder_transition(seg_feat)

        xt = x.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
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
            losses["transition_cls"] = (
                self.transition_loss_weight
                * self._balanced_transition_loss(logits_transition, targets)
            )
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            outputs.append(
                {
                    "transition": F.softmax(logits_transition[i], dim=0),
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)

    def _balanced_transition_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Per-sample, per-class balanced cross-entropy over valid pixels.

        Generalizes ``_balanced_binary_loss``: within each sample, the loss is the
        mean over the classes present (each averaged over its own pixels), so the
        dominant ``no_change`` class does not swamp the rare transition groups.
        Samples contribute equally regardless of point count.
        """
        labels = torch.stack(
            [t["transition"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["transition"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)
        loss_flat = loss.flatten(1)  # (B, N)
        labels_flat = labels.flatten(1)
        valid_flat = valid.flatten(1)

        sample_losses: list[torch.Tensor] = []
        for b in range(loss_flat.shape[0]):
            v = valid_flat[b]
            if not v.any():
                continue
            lb = labels_flat[b][v]
            ls = loss_flat[b][v]
            per_class = [ls[lb == cls].mean() for cls in lb.unique()]
            sample_losses.append(torch.stack(per_class).mean())

        return torch.stack(sample_losses).mean()


class LCCTransitionMultiTask(LCCMultiTaskV2):
    """MultiTask that loads binary/src/dst rasters but trains a single transition head.

    The binary/src/dst sub-tasks are kept only to load the three label rasters
    (and inject the annotation sidecar). The combined ``transition`` target is
    built by ``TransitionLabelBuilder`` and supervised by ``TransitionDualPassModel``.
    Metrics and outputs cover only ``transition`` + the start/end timestamp heads.
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        annotations_path: str,
        num_classes: int = NUM_TRANSITION_CLASSES,
        class_names: list[str] | None = None,
        soft_binary: bool = False,
    ):
        """Create a new LCCTransitionMultiTask.

        Args:
            tasks: sub-tasks used only to load label rasters (binary/src/dst).
            input_mapping: per-task raw-input remapping for those sub-tasks.
            annotations_path: path to the lcc_annotations.json sidecar.
            num_classes: number of transition classes (default 10).
            class_names: class names for metrics; defaults to TRANSITION_CLASS_NAMES.
            soft_binary: how to write the binary change bands in ``process_output``.
                If False (default), the no_change/change bands are hard 1/0 from the
                overall transition argmax (the original behavior). If True, they are
                soft softmax probabilities (no_change = transition[0], change =
                1 - no_change), matching the soft binary outputs of model.py / tasks.py.
        """
        super().__init__(
            tasks=tasks,
            input_mapping=input_mapping,
            annotations_path=annotations_path,
        )
        self.soft_binary = soft_binary
        self._transition_task = SegmentationTask(
            num_classes=num_classes,
            class_names=class_names or TRANSITION_CLASS_NAMES,
            enable_accuracy_metric=True,
            enable_confusion_matrix=True,
            enable_f1_metric=True,
            report_metric_per_class=True,
            metric_kwargs={"average": "macro"},
        )

    @override
    def get_metrics(self) -> MetricCollection:
        """Metrics for the transition head plus start/end timestamp accuracy."""
        metrics = {
            f"transition/{name}": MetricWrapper("transition", metric)
            for name, metric in self._transition_task.get_metrics().items()
        }
        collection = MetricCollection(metrics)
        collection.add_metrics(
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
        return collection

    @override
    def process_output(
        self, raw_output: Any, metadata: Any
    ) -> npt.NDArray[np.uint8]:
        """Emit the legacy 49-band output_change layout from the transition head.

        This keeps the prediction dataset config and postprocessing unchanged. The
        single transition prediction is mapped back to the old binary/src/dst bands:

        Band layout (matches postprocess.OUTPUT_BANDS):
        0..2   = binary: nodata (always 0), no_change, change. By default
                 (``soft_binary=False``) the no_change/change bands are hard 1/0
                 based on whether the overall transition argmax is the no_change
                 class. When ``soft_binary=True`` they are soft probabilities
                 (no_change = softmax prob of the transition no_change class,
                 change = ``1 - no_change``, the total probability of all transition
                 groups), matching the soft binary outputs of model.py / tasks.py.
        3..15  = src: one-hot at the argmax group's representative source class
        16..28 = dst: one-hot at the argmax group's representative destination class
        29..48 = per-timestep change-window membership reconstructed from the
                 start/end distributions (start-cdf * end-reverse-cdf).
        """
        transition = raw_output["transition"].float()  # (num_classes, H, W)
        _, h, w = transition.shape
        device = transition.device

        if self.soft_binary:
            # Soft binary: no_change band is the softmax prob of the no_change class
            # (index 0); change band is 1 - no_change (the summed probability of all
            # transition groups). This mirrors the soft binary outputs of model.py /
            # tasks.py so postprocess.py can threshold a graded change score.
            no_change_prob = transition[0]  # (H, W)
            change_prob = (1.0 - no_change_prob).clamp(0, 1)  # (H, W)
            binary = torch.stack(
                [
                    torch.zeros((h, w), device=device),
                    no_change_prob,
                    change_prob,
                ],
                dim=0,
            )  # (3, H, W)
        else:
            # Hard binary: change iff the overall argmax is a transition group (not
            # no_change). The no_change/change bands are 1.0/0.0 (scaled to 255 below).
            is_change = transition.argmax(dim=0) != 0  # (H, W)
            binary = torch.stack(
                [
                    torch.zeros((h, w), device=device),
                    (~is_change).float(),
                    is_change.float(),
                ],
                dim=0,
            )  # (3, H, W)

        # Per-pixel best transition group (1..num_classes-1) -> representative pair.
        group = transition[1:].argmax(dim=0) + 1  # (H, W)
        src_id = _GROUP_SRC_LUT.to(device)[group]  # (H, W)
        dst_id = _GROUP_DST_LUT.to(device)[group]
        src = torch.zeros(NUM_CATEGORIES, h, w, device=device)
        dst = torch.zeros(NUM_CATEGORIES, h, w, device=device)
        src.scatter_(0, src_id.unsqueeze(0), 1.0)
        dst.scatter_(0, dst_id.unsqueeze(0), 1.0)

        timestamps = raw_output["timestamps"]
        start_cdf = timestamps["start"].float().cumsum(dim=0)
        end_rev = timestamps["end"].float().flip(0).cumsum(dim=0).flip(0)
        membership = (start_cdf * end_rev).clamp(0, 1)  # (T, H, W)

        stacked = torch.cat([binary, src, dst, membership], dim=0)
        return (stacked * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    @override
    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: dict[str, Any],
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize only the transition head (binary/src/dst are not predicted)."""
        cur_target = target_dict["transition"] if target_dict else None
        images = self._transition_task.visualize(
            input_dict, cur_target, output["transition"]
        )
        return {f"transition_{label}": image for label, image in images.items()}
