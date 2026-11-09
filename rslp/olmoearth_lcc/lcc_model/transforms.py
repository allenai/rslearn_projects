"""Transforms that build the ``sentinel2_l2a`` stack for the LCC model.

The stack is 16 quarterly mosaics (configurable) followed by 6 frequent images. A frequent layer
holds the least-cloudy mosaic of each of the 6 most recent 7-day periods with
imagery in a 90-day block.

- ``StackSampler`` (train/val): picks one of the materialized frequent options,
  takes the 16 most recent quarterly mosaics before it, and emits the 22-image
  ``sentinel2_l2a`` stack plus per-pixel start/end timestep-index targets (the
  ``TS_START_KEY`` / ``TS_END_KEY`` segmentation-style tasks, unless
  ``emit_timestep_targets=False``). It can optionally drop quarterly mosaics at
  train time as augmentation. It also gates the post-change targets on the dates
  of each positive point (see ``StackSampler``).
- ``PredictStackBuilder`` (predict): same stack from the single
  ``sentinel2_frequent_0`` layer, with no annotation or targets.
- ``mark_negative_points_none``: labels negative (no-change) points as "none" for
  the change-category heads.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
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

# Annual OlmoEarth v1.3 embeddings (see the README's "Annual embeddings" section).
# The window layer holds every year in EMBEDDING_YEARS as one raster with
# len(EMBEDDING_YEARS) * EMBEDDING_DIMS int8 bands (year-major), -128 = nodata. The
# samplers replace it with the K most recent eligible years as a (EMBEDDING_DIMS, K,
# H, W) RasterImage under the same key, oldest first, one timestamp per year.
EMBEDDING_KEY = "v13_embeddings"
EMBEDDING_YEARS = list(range(2017, 2026))
EMBEDDING_DIMS = 128
EMBEDDING_NODATA = -128

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
# StackSampler (see prepare.py): each positive point's pre_change, post_change and
# transition dates (uint16 days since 1970-01-01, DAYS_NODATA where unset), and its
# transition land cover and change category classes (0 = nodata).
LABEL_TS_START_KEY = "label_ts_start"
LABEL_TS_END_KEY = "label_ts_end"
LABEL_TRANSITION_DATE_KEY = "label_transition_date"
LABEL_TRANSITION_DST_KEY = "label_transition_dst"
LABEL_TRANSITION_POST_CHANGE_KEY = "label_transition_post_change"


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


def select_embedding_years(
    embeddings: RasterImage | None,
    boundary: datetime,
    num_years: int,
    drop_latest: bool = False,
    image_hw: tuple[int, int] | None = None,
) -> RasterImage:
    """Pick the ``num_years`` most recent annual embeddings that end by ``boundary``.

    A year Y is eligible when it ends (Jan 1 of Y+1) on or before ``boundary``, the
    start of the earliest frequent image, so no embedding overlaps the frequent
    block. ``drop_latest`` skips the most recent eligible year, simulating the lag
    before a year's embeddings are published. The K slots are the consecutive years
    ending at the latest selected year; slots for years outside EMBEDDING_YEARS (or
    a missing layer) are filled with EMBEDDING_NODATA.

    Args:
        embeddings: the window's embedding raster (len(EMBEDDING_YEARS) *
            EMBEDDING_DIMS bands, one timestep), or None if the window lacks it.
        boundary: start of the earliest frequent image.
        num_years: K, the number of year slots to return.
        drop_latest: whether to skip the most recent eligible year.
        image_hw: (H, W) to use when ``embeddings`` is None.

    Returns:
        (EMBEDDING_DIMS, K, H, W) float RasterImage, oldest year first, with
        timestamps [(Jan 1 Y, Jan 1 Y+1)] per slot.
    """
    tz = boundary.tzinfo or timezone.utc
    # Jan 1 of boundary.year is always <= boundary, so the previous year is the latest
    # one that has ended.
    latest = boundary.year - 1
    if drop_latest:
        latest -= 1
    years = list(range(latest - num_years + 1, latest + 1))

    if embeddings is not None:
        H, W = embeddings.image.shape[-2:]
        all_years = embeddings.image.reshape(
            len(EMBEDDING_YEARS), EMBEDDING_DIMS, H, W
        ).float()
    else:
        assert image_hw is not None
        H, W = image_hw
        all_years = None
    out = torch.full(
        (EMBEDDING_DIMS, num_years, H, W), float(EMBEDDING_NODATA), dtype=torch.float32
    )
    for k, year in enumerate(years):
        if all_years is not None and year in EMBEDDING_YEARS:
            out[:, k] = all_years[EMBEDDING_YEARS.index(year)]
    timestamps = [
        (datetime(y, 1, 1, tzinfo=tz), datetime(y + 1, 1, 1, tzinfo=tz)) for y in years
    ]
    return RasterImage(image=out, timestamps=timestamps)


def _build_quarterly_stack(
    quarterly: RasterImage,
    valid_indices: list[int],
    num_quarterly: int = NUM_QUARTERLY,
) -> tuple[torch.Tensor, list[tuple[datetime, datetime]]]:
    """Build exactly ``num_quarterly`` quarterly images and their timestamps.

    Uses the (already-trimmed) ``valid_indices`` in chronological order. When
    fewer than ``num_quarterly`` are available, prepends copies of the earliest
    selected image with progressively-older fake timestamps so the encoder sees
    distinct timesteps (it rejects duplicate timestamps within a modality when
    ``token_pooling=False``).
    """
    assert quarterly.timestamps is not None
    q_ts = [quarterly.timestamps[i] for i in valid_indices]

    img_parts: list[torch.Tensor] = []
    if valid_indices:
        img_parts.append(quarterly.image[:, valid_indices, :, :])

    if len(valid_indices) < num_quarterly:
        deficit = num_quarterly - len(valid_indices)
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


def gate_post_change_targets(
    input_dict: dict[str, Any],
    target_dict: dict[str, Any],
    end: datetime,
    post_change: datetime,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Gate the dst and post_change targets on each positive point's dates.

    With ``end`` the end of the latest frequent image, per pixel:

    - end >= post_change: dst and post_change keep their labels.
    - else, if the point has a transition and end >= transition_date: dst and
      post_change use the transition's land cover and change category (each masked
      if unset).
    - otherwise (the change is still in progress): dst and post_change are masked.

    The per-pixel dates come from the passthrough label_ts_end and
    label_transition_* rasters, which are removed from input_dict. Without
    label_ts_end, the post_change dates come from the ts_end days target if present
    (PerPixelTimestepTask configs), else the window's post_change. Without the
    transition rasters, no point has a transition.

    A ts_end days target (PerPixelTimestepTask configs) is rewritten to the returned
    per-pixel dates.

    Args:
        input_dict: the input dict, holding the optional label rasters.
        target_dict: the target dict whose dst/post_change targets are rewritten.
        end: the end of the latest frequent image of the sampled stack.
        post_change: the window's post_change (from the annotations sidecar).
        shape: the (height, width) of the targets.

    Returns:
        HW long tensor of the per-pixel ts_end target date in days since
        1970-01-01: the post_change date if end >= post_change or the point has no
        transition, else the transition date. While a change is in progress this
        date is after end, so it maps to the last timestep.
    """
    post_days = _pop_label(input_dict, LABEL_TS_END_KEY)
    trans_days = _pop_label(input_dict, LABEL_TRANSITION_DATE_KEY)
    trans_dst = _pop_label(input_dict, LABEL_TRANSITION_DST_KEY)
    trans_post_change = _pop_label(input_dict, LABEL_TRANSITION_POST_CHANGE_KEY)

    if post_days is None and "days" in target_dict.get(TS_END_KEY, {}):
        ts_end = target_dict[TS_END_KEY]
        post_days = torch.where(
            ts_end["valid"].get_hw_tensor() > 0,
            ts_end["days"].get_hw_tensor().long(),
            DAYS_NODATA,
        )
    window_days = int(_days_since_epoch(post_change) // 1)
    if post_days is None:
        post_days = torch.full(shape, window_days, dtype=torch.long)
    else:
        post_days = torch.where(post_days == DAYS_NODATA, window_days, post_days)

    end_days = _days_since_epoch(end)
    post_ok = end_days >= post_days
    if trans_days is None:
        has_transition = torch.zeros_like(post_ok)
        trans_days = torch.full_like(post_days, DAYS_NODATA)
    else:
        has_transition = trans_days != DAYS_NODATA
    trans_ok = ~post_ok & has_transition & (end_days >= trans_days)

    for name, trans_classes in (
        ("dst", trans_dst),
        ("post_change", trans_post_change),
    ):
        if name not in target_dict:
            continue
        classes = target_dict[name]["classes"].get_hw_tensor().clone()
        valid = target_dict[name]["valid"].get_hw_tensor().clone().float()
        if trans_classes is not None:
            classes = torch.where(trans_ok, trans_classes.to(classes.dtype), classes)
            valid = torch.where(trans_ok, (trans_classes > 0).float(), valid)
        else:
            valid = torch.where(trans_ok, 0.0, valid)
        valid = torch.where(post_ok | trans_ok, valid, 0.0)
        target_dict[name]["classes"] = RasterImage(image=classes[None, None, :, :])
        target_dict[name]["valid"] = RasterImage(image=valid[None, None, :, :])

    ts_end_days = torch.where(has_transition & ~post_ok, trans_days, post_days)
    if "days" in target_dict.get(TS_END_KEY, {}):
        ts_end = target_dict[TS_END_KEY]
        valid = ts_end["valid"].get_hw_tensor().float()
        days = torch.where(
            valid > 0, ts_end_days, ts_end["days"].get_hw_tensor().long()
        )
        valid = valid * (days != DAYS_NODATA).float()
        ts_end["days"] = RasterImage(image=days[None, None, :, :])
        ts_end["valid"] = RasterImage(image=valid[None, None, :, :])
    return ts_end_days


class StackSampler(Transform):
    """Pick a frequent option, take N quarterly, build sentinel2_l2a + idx targets.

    Builds the stack (``num_quarterly`` quarterly + 6 frequent, 16 + 6 = 22 by
    default) in chronological order and emits the per-pixel start/end
    timestep-index targets consumed by the start/end heads.

    With ``embedding_years > 0`` it also replaces the window's annual embedding
    raster (``EMBEDDING_KEY``) with the K most recent years that end before the
    chosen option's earliest frequent image (see select_embedding_years). At train
    time ``embedding_lag_drop_prob`` is the probability of skipping the most recent
    eligible year, so the model also sees the lag before a year is published. With
    ``timestep_targets_include_embeddings``, the start/end targets index the
    combined [K embedding years + stack] timeline (for models that keep the
    embedding tokens in the breakpoint scan and timestep heads).

    With ``quarterly_dropout > 0`` (train only), each available quarterly image is
    independently dropped before sampling (keeping at least ``min_keep``), so
    across epochs the model sees different mosaic compositions of the same
    window and learns invariance to composition.

    With ``emit_timestep_targets=False``, the start/end targets come from rslearn's
    ``PerPixelTimestepTask``, which reads them from the ``label_ts_start`` /
    ``label_ts_end`` day rasters; only the ``ts_end`` days are rewritten per pixel.

    The dst and post_change targets are gated on each positive point's dates (see
    gate_post_change_targets): they are only trained once the sampled images reach
    post_change, or the point's transition, whose labels they then use. ts_end
    targets the transition date in that interval, and while the change is still in
    progress it targets the next date (transition or post_change), i.e. the last
    timestep.
    """

    def __init__(
        self,
        deterministic: bool = False,
        quarterly_dropout: float = 0.0,
        min_keep: int = 8,
        emit_timestep_targets: bool = True,
        num_quarterly: int = NUM_QUARTERLY,
        embedding_years: int = 0,
        embedding_lag_drop_prob: float = 0.0,
        timestep_targets_include_embeddings: bool = False,
    ) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test) and
                disable quarterly dropout. Option 2 is a mid-range temporal
                context rather than the hardest immediate-detection case.
            quarterly_dropout: probability of dropping each quarterly image.
            min_keep: minimum number of quarterly images to keep under dropout.
            emit_timestep_targets: whether to compute the ``ts_start`` / ``ts_end``
                timestep-index targets from the window's annotation dates.
            num_quarterly: number of quarterly images in the stack.
            embedding_years: number of annual embedding years K to provide (0 = the
                embeddings are not used).
            embedding_lag_drop_prob: train-time probability of skipping the most
                recent eligible embedding year.
            timestep_targets_include_embeddings: index the start/end targets over
                [K embedding years + stack] instead of the stack alone.
        """
        super().__init__()
        self.deterministic = deterministic
        self.quarterly_dropout = quarterly_dropout
        self.min_keep = min_keep
        self.emit_timestep_targets = emit_timestep_targets
        self.num_quarterly = num_quarterly
        self.embedding_years = embedding_years
        self.embedding_lag_drop_prob = embedding_lag_drop_prob
        self.timestep_targets_include_embeddings = timestep_targets_include_embeddings
        if timestep_targets_include_embeddings and embedding_years <= 0:
            raise ValueError(
                "timestep_targets_include_embeddings requires embedding_years > 0"
            )

    def _drop_quarterly(self, quarterly: RasterImage) -> RasterImage:
        """Randomly drop quarterly images (train-time augmentation)."""
        if (
            self.deterministic
            or self.quarterly_dropout <= 0
            or quarterly.timestamps is None
        ):
            return quarterly
        T = quarterly.image.shape[1]
        keep = [i for i in range(T) if random.random() >= self.quarterly_dropout]
        if len(keep) < min(self.min_keep, T):
            keep = sorted(random.sample(range(T), min(self.min_keep, T)))
        if len(keep) == T:
            return quarterly
        return RasterImage(
            image=quarterly.image[:, keep, :, :],
            timestamps=[quarterly.timestamps[i] for i in keep],
        )

    def _set_timestep_targets(
        self,
        target_dict: dict[str, Any],
        combined_ts: list[tuple[datetime, datetime]],
        ts_start_days: torch.Tensor,
        ts_end_days: torch.Tensor,
        quarterly: RasterImage,
    ) -> None:
        """Compute start/end timestamp index targets over the chronological steps.

        The start uses the per-pixel ts_start_days (each point's pre_change), and
        the end the per-pixel ts_end_days (see gate_post_change_targets).
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

        for key, idx_map in ((TS_START_KEY, start_map), (TS_END_KEY, end_map)):
            target_dict[key] = {
                "classes": RasterImage(image=idx_map[None, None, :, :]),
                "valid": RasterImage(image=valid_mask.clone()[None, None, :, :]),
            }

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample a frequent option, take N quarterly, produce sentinel2_l2a."""
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
        quarterly = self._drop_quarterly(quarterly)

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
        # Take the most recent num_quarterly candidates (consecutive, no skipping).
        valid_indices = valid_indices[-self.num_quarterly :]
        q_img, q_ts = _build_quarterly_stack(
            quarterly, valid_indices, self.num_quarterly
        )

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + chosen_frequent.timestamps
        input_dict[INPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # The embedding raster is passed through only when the model uses it.
        raw_embeddings = input_dict.pop(EMBEDDING_KEY, None)
        target_ts = combined_ts
        if self.embedding_years > 0:
            drop_latest = (
                not self.deterministic
                and random.random() < self.embedding_lag_drop_prob
            )
            emb = select_embedding_years(
                raw_embeddings,
                earliest_freq_ts,
                self.embedding_years,
                drop_latest=drop_latest,
                image_hw=(combined_img.shape[2], combined_img.shape[3]),
            )
            input_dict[EMBEDDING_KEY] = emb
            if self.timestep_targets_include_embeddings:
                assert emb.timestamps is not None
                target_ts = list(emb.timestamps) + combined_ts

        # The model can't predict the destination land cover or post-change category
        # before the imagery shows them, so gate those targets on the dates.
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        ts_end_days = gate_post_change_targets(
            input_dict,
            target_dict,
            latest_freq_ts,
            post_change,
            (combined_img.shape[2], combined_img.shape[3]),
        )

        # Each point's pre_change, falling back to the window's where unset (and for
        # datasets without the label_ts_start raster).
        window_start_days = int(_days_since_epoch(pre_change) // 1)
        ts_start_days = _pop_label(input_dict, LABEL_TS_START_KEY)
        if ts_start_days is None:
            ts_start_days = torch.full_like(ts_end_days, window_start_days)
        else:
            ts_start_days = torch.where(
                ts_start_days == DAYS_NODATA, window_start_days, ts_start_days
            )

        if self.emit_timestep_targets:
            self._set_timestep_targets(
                target_dict, target_ts, ts_start_days, ts_end_days, quarterly
            )

        # Train the change-category heads to predict "none" at negative points.
        mark_negative_points_none(target_dict)

        return input_dict, target_dict


class PredictStackBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is one ``sentinel2_frequent_0`` layer with six
    weekly least-cloudy mosaics from [T - 90d, T], and no annotation sidecar. The
    quarterly layer already ends at T - 90d, so this takes the last
    ``num_quarterly`` (16) quarterly images as-is and concatenates them with the 6
    frequent images into ``sentinel2_l2a``, matching the stack the model is trained
    on. With ``embedding_years > 0`` it selects the annual embeddings the same way
    as StackSampler (without the lag drop).
    """

    def __init__(self, num_quarterly: int = NUM_QUARTERLY, embedding_years: int = 0) -> None:
        """Initialize the prediction stack builder.

        Args:
            num_quarterly: number of quarterly images in the stack.
            embedding_years: number of annual embedding years K (0 = not used).
        """
        super().__init__()
        self.num_quarterly = num_quarterly
        self.embedding_years = embedding_years

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last N quarterly + 6 frequent into sentinel2_l2a."""
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
        indices = list(range(max(0, T - self.num_quarterly), T))
        q_img, q_ts = _build_quarterly_stack(quarterly, indices, self.num_quarterly)

        combined_img = torch.cat([q_img, frequent.image], dim=1)
        combined_ts = q_ts + (frequent.timestamps or [])
        input_dict[INPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        raw_embeddings = input_dict.pop(EMBEDDING_KEY, None)
        if self.embedding_years > 0:
            if not frequent.timestamps:
                raise ValueError("Frequent layer must have timestamps")
            input_dict[EMBEDDING_KEY] = select_embedding_years(
                raw_embeddings,
                min(ts[0] for ts in frequent.timestamps),
                self.embedding_years,
                image_hw=(combined_img.shape[2], combined_img.shape[3]),
            )

        return input_dict, target_dict
