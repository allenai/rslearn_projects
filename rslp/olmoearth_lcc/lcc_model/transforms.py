"""Transforms that build the 22-image ``sentinel2_l2a`` stack for the LCC model.

The stack is 16 quarterly mosaics followed by 6 frequent images. A frequent layer
holds the least-cloudy mosaic of each of the 6 most recent 7-day periods with
imagery in a 90-day block.

- ``StackSampler`` (train/val): picks one of the materialized frequent options,
  takes the 16 most recent quarterly mosaics before it, and emits the 22-image
  ``sentinel2_l2a`` stack plus per-pixel start/end timestep-index targets (the
  ``TS_START_KEY`` / ``TS_END_KEY`` segmentation-style tasks, unless
  ``emit_timestep_targets=False``). It also gates the post-change targets on the
  dates of each positive point (see ``StackSampler``).
- ``PredictStackBuilder`` (predict): same stack from the single
  ``sentinel2_frequent_0`` layer, with no annotation or targets.
- ``mark_negative_points_none``: labels negative (no-change) points as "none" for
  the change-category heads.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform
from typing_extensions import override

from .timestamp_encoding import DAYS_NODATA, UNIX_EPOCH

QUARTERLY_KEY = "sentinel2_quarterly"
FREQUENT_KEY_PREFIX = "sentinel2_frequent_"
ANNOTATION_KEY = "_lcc_annotation"
INPUT_KEY = "sentinel2_l2a"

# Task names of the per-pixel start/end timestep-index targets. Each is a
# segmentation-style ``{"classes", "valid"}`` target over the input timesteps, valid
# only at change points.
TS_START_KEY = "ts_start"
TS_END_KEY = "ts_end"

NUM_QUARTERLY = 16
NUM_FREQUENT = 6
# Maximum number of frequent option layers materialized by the prepare script.
NUM_FREQUENT_OPTIONS = 8

# Duration of each quarterly mosaic period (see prepare.py), also the spacing of the
# fake timestamps assigned to padding (duplicated) images.
QUARTERLY_PERIOD = timedelta(days=90)

# Change-category targets that receive a "none" label at negative points.
CHANGE_CATEGORY_TASKS = ("pre_change", "post_change")

# Optional per-pixel label rasters that the training configs pass through to
# StackSampler (see prepare.py): each positive point's first change dates
# (pre_change, post_change, first-noticeable and stop dates, uint16 days since
# 1970-01-01, DAYS_NODATA where unset), and its optional second change's (the same
# dates plus its src / dst / pre_change / post_change classes, 0 = nodata).
LABEL_TS_START_KEY = "label_ts_start"
LABEL_TS_END_KEY = "label_ts_end"
LABEL_FIRST_NOTICEABLE_KEY = "label_first_noticeable"
LABEL_STOP_DATE_KEY = "label_stop_date"
CHANGE2_PREFIX = "label_change2_"
# Second-change class rasters, keyed by the target they replace.
CHANGE2_CLASS_TARGETS = ("src", "dst", "pre_change", "post_change")
# Targets gated on the post-change date (masked while the change is in progress).
POST_TARGETS = ("dst", "post_change")


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


def mark_negative_points_none(target_dict: dict[str, Any]) -> None:
    """Label negative (no-change) points as "none" for the change-category heads.

    At a negative point there is by definition no change over the annotated time
    range, so the pre/post change-category heads should be trained to predict
    class 1 ("none") there instead of being masked out. The label rasters only
    mark negatives in the binary layer (change-category rasters are nodata
    there), so this derives the negative mask from binary class 1 (no_change) and
    sets classes/valid at those pixels for any change-category target present.
    Positive points and unlabeled pixels are untouched.
    """
    if "binary" not in target_dict:
        return
    binary_classes = target_dict["binary"]["classes"].get_hw_tensor().long()
    binary_valid = target_dict["binary"]["valid"].get_hw_tensor() > 0
    neg_mask = binary_valid & (binary_classes == 1)
    if not neg_mask.any():
        return
    for name in CHANGE_CATEGORY_TASKS:
        if name not in target_dict:
            continue
        classes = target_dict[name]["classes"].get_hw_tensor().clone()
        valid = target_dict[name]["valid"].get_hw_tensor().clone()
        classes[neg_mask] = 1  # class 1 = "none"
        valid[neg_mask] = 1
        target_dict[name]["classes"] = RasterImage(image=classes[None, None, :, :])
        target_dict[name]["valid"] = RasterImage(image=valid[None, None, :, :])


def _days_since_epoch(date: datetime) -> float:
    """Fractional days since 1970-01-01 of a timezone-aware datetime."""
    return (date - UNIX_EPOCH) / timedelta(days=1)


def _pop_label(input_dict: dict[str, Any], key: str) -> torch.Tensor | None:
    """Pop an optional passthrough label raster as an HW long tensor."""
    raster: RasterImage | None = input_dict.pop(key, None)
    return None if raster is None else raster.get_hw_tensor().long()


def _mask_targets(target_dict: dict[str, Any], masked: torch.Tensor) -> None:
    """Set valid = 0 at the masked pixels of every target with a valid raster."""
    if not masked.any():
        return
    for target in target_dict.values():
        if not isinstance(target, dict) or "valid" not in target:
            continue
        valid = target["valid"].get_hw_tensor().clone().float()
        valid[masked] = 0.0
        target["valid"] = RasterImage(image=valid[None, None, :, :])


def gate_change_targets(
    input_dict: dict[str, Any],
    target_dict: dict[str, Any],
    end: datetime,
    pre_change: datetime,
    post_change: datetime,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate the change targets on each positive point's dates.

    With ``end`` the end of the latest frequent image, per pixel:
    - which change applies: the second change if the point has one and end >= its
      first-noticeable date, else the first. For the second change, src / dst /
      pre_change / post_change use its classes (masked where 0) and ts_start /
      ts_end its dates.
    - dst and post_change are masked while end < the applying change's post_change
      (the change is still in progress), and ts_end then targets that date, which is
      after end, i.e. the last timestep.
    - everything (every target with a valid raster, binary included) is masked when
      end < the first change's first-noticeable date, or end > the applying
      change's stop date.

    The per-pixel dates come from the passthrough label rasters (removed from
    input_dict). Without label_ts_start / label_ts_end, the dates come from the
    ts_start / ts_end days targets if present (PerPixelTimestepTask configs), else
    the window's pre_change / post_change. Without the first-noticeable, stop-date or
    change2 rasters, nothing is masked by them and no point has a second change.
    ts_start / ts_end days targets (PerPixelTimestepTask configs) are rewritten to
    the per-pixel dates.

    Args:
        input_dict: the input dict, holding the optional label rasters.
        target_dict: the target dict whose targets are rewritten.
        end: the end of the latest frequent image of the sampled stack.
        pre_change: the window's pre_change (from the annotations sidecar).
        post_change: the window's post_change (from the annotations sidecar).
        shape: the (height, width) of the targets.

    Returns:
        (ts_start_days, ts_end_days, masked), HW tensors: the per-pixel ts_start and
        ts_end target dates in days since 1970-01-01, and where all targets are
        masked.
    """

    def days_label(key: str, target_key: str | None, window: datetime) -> torch.Tensor:
        days = _pop_label(input_dict, key)
        if days is None and target_key and "days" in target_dict.get(target_key, {}):
            target = target_dict[target_key]
            days = torch.where(
                target["valid"].get_hw_tensor() > 0,
                target["days"].get_hw_tensor().long(),
                DAYS_NODATA,
            )
        window_days = int(_days_since_epoch(window) // 1)
        if days is None:
            return torch.full(shape, window_days, dtype=torch.long)
        return torch.where(days == DAYS_NODATA, window_days, days)

    def optional_days(key: str) -> torch.Tensor:
        days = _pop_label(input_dict, key)
        return (
            torch.full(shape, DAYS_NODATA, dtype=torch.long) if days is None else days
        )

    start_days = days_label(LABEL_TS_START_KEY, TS_START_KEY, pre_change)
    post_days = days_label(LABEL_TS_END_KEY, TS_END_KEY, post_change)
    first_noticeable = optional_days(LABEL_FIRST_NOTICEABLE_KEY)
    stop = optional_days(LABEL_STOP_DATE_KEY)
    change2_start = optional_days(f"{CHANGE2_PREFIX}ts_start")
    change2_post = optional_days(f"{CHANGE2_PREFIX}ts_end")
    change2_noticeable = optional_days(f"{CHANGE2_PREFIX}first_noticeable")
    change2_stop = optional_days(f"{CHANGE2_PREFIX}stop_date")
    change2_classes = {
        name: _pop_label(input_dict, f"{CHANGE2_PREFIX}{name}")
        for name in CHANGE2_CLASS_TARGETS
    }

    end_days = _days_since_epoch(end)
    second = (change2_noticeable != DAYS_NODATA) & (end_days >= change2_noticeable)
    applying_stop = torch.where(second, change2_stop, stop)
    masked = ((first_noticeable != DAYS_NODATA) & (end_days < first_noticeable)) | (
        (applying_stop != DAYS_NODATA) & (end_days > applying_stop)
    )
    ts_start_days = torch.where(second, change2_start, start_days)
    ts_end_days = torch.where(second, change2_post, post_days)
    post_ok = end_days >= ts_end_days

    for name in CHANGE2_CLASS_TARGETS:
        if name not in target_dict:
            continue
        classes = target_dict[name]["classes"].get_hw_tensor().clone()
        valid = target_dict[name]["valid"].get_hw_tensor().clone().float()
        second_classes = change2_classes[name]
        if second_classes is None:
            valid = torch.where(second, 0.0, valid)
        else:
            classes = torch.where(second, second_classes.to(classes.dtype), classes)
            valid = torch.where(second, (second_classes > 0).float(), valid)
        if name in POST_TARGETS:
            valid = torch.where(post_ok, valid, 0.0)
        target_dict[name]["classes"] = RasterImage(image=classes[None, None, :, :])
        target_dict[name]["valid"] = RasterImage(image=valid[None, None, :, :])

    for key, days in ((TS_START_KEY, ts_start_days), (TS_END_KEY, ts_end_days)):
        if "days" not in target_dict.get(key, {}):
            continue
        target = target_dict[key]
        valid = target["valid"].get_hw_tensor().float()
        new_days = torch.where(valid > 0, days, target["days"].get_hw_tensor().long())
        valid = valid * (new_days != DAYS_NODATA).float()
        target["days"] = RasterImage(image=new_days[None, None, :, :])
        target["valid"] = RasterImage(image=valid[None, None, :, :])

    _mask_targets(target_dict, masked)
    return ts_start_days, ts_end_days, masked


class StackSampler(Transform):
    """Pick a frequent option, take 16 quarterly, build sentinel2_l2a + idx targets.

    Builds the 22-image stack (16 quarterly + 6 frequent) in chronological order
    and emits the per-pixel start/end timestep-index targets consumed by the
    start/end heads.

    With ``emit_timestep_targets=False``, the start/end targets come from rslearn's
    ``PerPixelTimestepTask``, which reads them from the ``label_ts_start`` /
    ``label_ts_end`` day rasters; only the ``ts_end`` days are rewritten per pixel.

    The targets are gated on each positive point's dates (see gate_change_targets):
    the point's second change applies once the sampled images reach its
    first-noticeable date; dst and post_change are only trained once the images
    reach the applying change's post_change (until then ts_end targets the last
    timestep); and nothing is trained at a point for images ending before its
    first-noticeable date or after the applying change's stop date.
    """

    def __init__(
        self,
        deterministic: bool = False,
        emit_timestep_targets: bool = True,
    ) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test). Option 2 is
                a mid-range temporal context rather than the hardest
                immediate-detection case.
            emit_timestep_targets: whether to compute the ``ts_start`` / ``ts_end``
                timestep-index targets from the window's annotation dates.
        """
        super().__init__()
        self.deterministic = deterministic
        self.emit_timestep_targets = emit_timestep_targets

    def _set_timestep_targets(
        self,
        target_dict: dict[str, Any],
        combined_ts: list[tuple[datetime, datetime]],
        ts_start_days: torch.Tensor,
        ts_end_days: torch.Tensor,
        quarterly: RasterImage,
        masked: torch.Tensor | None = None,
    ) -> None:
        """Compute start/end timestamp index targets over the chronological steps.

        The start uses the per-pixel ts_start_days and the end the per-pixel
        ts_end_days of the applying change (see gate_change_targets). Both are masked
        where masked is True.
        """
        centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in combined_ts]

        H, W = quarterly.image.shape[2], quarterly.image.shape[3]
        start_map = torch.zeros(H, W, dtype=torch.long)
        end_map = torch.zeros(H, W, dtype=torch.long)

        if "binary" in target_dict:
            binary_classes = target_dict["binary"]["classes"].get_hw_tensor()
            change_mask = binary_classes == 2
            valid_mask = change_mask.float()
        else:
            change_mask = torch.ones(H, W, dtype=torch.bool)
            valid_mask = torch.ones(H, W, dtype=torch.float32)
        for idx_map, days_map, is_start in (
            (start_map, ts_start_days, True),
            (end_map, ts_end_days, False),
        ):
            for days in days_map[change_mask].unique().tolist():
                idx = _change_index(
                    centers, UNIX_EPOCH + timedelta(days=days), is_start=is_start
                )
                idx_map[change_mask & (days_map == days)] = idx

        if masked is not None:
            valid_mask[masked] = 0.0
        for key, idx_map, valid in (
            (TS_START_KEY, start_map, valid_mask.clone()),
            (TS_END_KEY, end_map, valid_mask.clone()),
        ):
            target_dict[key] = {
                "classes": RasterImage(image=idx_map[None, None, :, :]),
                "valid": RasterImage(image=valid[None, None, :, :]),
            }

    @override
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
        for i in range(NUM_FREQUENT_OPTIONS):
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

        # Quarterly images end where the frequent images begin.
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
        input_dict[INPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Gate the targets on each point's dates: which change applies, whether it is
        # complete (dst / post_change are only predictable once the imagery shows
        # them), and whether the point is masked (before first noticeable, after a
        # stop date).
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        ts_start_days, ts_end_days, masked = gate_change_targets(
            input_dict,
            target_dict,
            latest_freq_ts,
            pre_change,
            post_change,
            (combined_img.shape[2], combined_img.shape[3]),
        )

        if self.emit_timestep_targets:
            self._set_timestep_targets(
                target_dict,
                combined_ts,
                ts_start_days,
                ts_end_days,
                quarterly,
                masked,
            )

        # Train the change-category heads to predict "none" at negative points.
        mark_negative_points_none(target_dict)

        return input_dict, target_dict


class PredictStackBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is one ``sentinel2_frequent_0`` layer with six
    weekly least-cloudy mosaics from [T - 90d, T], and no annotation sidecar. The
    quarterly layer already ends at T - 90d, so this takes the last 16 quarterly
    images as-is and concatenates 16 quarterly + 6 frequent = 22 into
    ``sentinel2_l2a``, matching the 22-timestep stack the model is trained on.
    """

    def __init__(self) -> None:
        """Initialize the prediction stack builder."""
        super().__init__()

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last 16 quarterly + 6 frequent into sentinel2_l2a."""
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
        indices = list(range(max(0, T - NUM_QUARTERLY), T))
        q_img, q_ts = _build_quarterly_stack(quarterly, indices)

        combined_img = torch.cat([q_img, frequent.image], dim=1)
        combined_ts = q_ts + (frequent.timestamps or [])
        input_dict[INPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        return input_dict, target_dict
