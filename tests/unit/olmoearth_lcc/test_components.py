"""Tests for the LCC training glue: StackSampler targets, balanced head, metrics."""

from datetime import datetime, timedelta, timezone

import pytest
import torch
from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.per_pixel_timestep import (
    DateToTimestepMode,
    PerPixelTimestepHead,
)
from rslearn.train.tasks.segmentation import SegmentationHead

from rslp.olmoearth_lcc.lcc_model.components import BalancedSegmentationHead
from rslp.olmoearth_lcc.lcc_model.tasks import TimestepToleranceAccuracy
from rslp.olmoearth_lcc.lcc_model.transforms import (
    ANNOTATION_KEY,
    FREQUENT_KEY_PREFIX,
    INPUT_KEY,
    NUM_FREQUENT,
    NUM_QUARTERLY,
    QUARTERLY_KEY,
    TS_END_KEY,
    TS_START_KEY,
    PredictStackBuilder,
    StackSampler,
)

CROP = 8
T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _stack_inputs() -> tuple[dict, dict]:
    """Inputs with 16 quarterly + one full frequent option and a binary target."""
    q_ts = [
        (T0 + timedelta(days=90 * i), T0 + timedelta(days=90 * i + 1))
        for i in range(NUM_QUARTERLY)
    ]
    f0 = q_ts[-1][1] + timedelta(days=10)
    f_ts = [
        (f0 + timedelta(days=15 * i), f0 + timedelta(days=15 * i + 1))
        for i in range(NUM_FREQUENT)
    ]
    input_dict = {
        QUARTERLY_KEY: RasterImage(
            image=torch.zeros(3, NUM_QUARTERLY, CROP, CROP), timestamps=q_ts
        ),
        f"{FREQUENT_KEY_PREFIX}0": RasterImage(
            image=torch.zeros(3, NUM_FREQUENT, CROP, CROP), timestamps=f_ts
        ),
        ANNOTATION_KEY: {
            # Change starts around quarterly index 13 and ends at frequent index 1.
            "pre_change": q_ts[13][0] + timedelta(days=1),
            "post_change": f_ts[1][0] + timedelta(hours=1),
            "first_noticeable": f_ts[1][0],
        },
    }
    binary = torch.ones(CROP, CROP, dtype=torch.long)
    binary[2, 3] = 2
    target_dict = {
        "binary": {
            "classes": RasterImage(image=binary[None, None]),
            "valid": RasterImage(image=torch.ones(1, 1, CROP, CROP)),
        }
    }
    return input_dict, target_dict


def test_stack_sampler_timestep_targets() -> None:
    """StackSampler emits segmentation-style ts_start/ts_end timestep targets."""
    input_dict, target_dict = _stack_inputs()
    input_dict, target_dict = StackSampler(deterministic=True)(input_dict, target_dict)

    assert input_dict[INPUT_KEY].image.shape[1] == NUM_QUARTERLY + NUM_FREQUENT
    for key, expected_idx in ((TS_START_KEY, 13), (TS_END_KEY, NUM_QUARTERLY + 1)):
        classes = target_dict[key]["classes"].get_hw_tensor()
        valid = target_dict[key]["valid"].get_hw_tensor()
        assert classes.shape == (CROP, CROP)
        assert classes[2, 3] == expected_idx
        # Only the change point is valid.
        assert valid.sum() == 1 and valid[2, 3] == 1


def test_stack_sampler_per_point_ts_start() -> None:
    """Each change point's ts_start uses its own pre_change from label_ts_start.

    Points without one (nodata) fall back to the window's pre_change.
    """
    input_dict, target_dict = _stack_inputs()
    target_dict["binary"]["classes"].image[0, 0, 5, 6] = 2
    q_ts = input_dict[QUARTERLY_KEY].timestamps
    days = torch.full((CROP, CROP), 65535, dtype=torch.long)
    days[5, 6] = (q_ts[10][0] + timedelta(days=1) - UNIX_EPOCH).days
    input_dict["label_ts_start"] = RasterImage(image=days[None, None])

    input_dict, target_dict = StackSampler(deterministic=True)(input_dict, target_dict)
    assert "label_ts_start" not in input_dict
    classes = target_dict[TS_START_KEY]["classes"].get_hw_tensor()
    assert classes[5, 6] == 10
    assert classes[2, 3] == 13


def test_stack_sampler_without_timestep_targets() -> None:
    """With emit_timestep_targets=False, ts targets come from the day label rasters.

    StackSampler should leave them untouched, and PerPixelTimestepHead should map the
    labeled dates to the same timestep indices that StackSampler would compute.
    """
    input_dict, target_dict = _stack_inputs()
    ann = input_dict[ANNOTATION_KEY]
    for key, date in (
        (TS_START_KEY, ann["pre_change"]),
        (TS_END_KEY, ann["post_change"]),
    ):
        days = torch.full((CROP, CROP), 65535, dtype=torch.long)
        days[2, 3] = (date - UNIX_EPOCH).days
        target_dict[key] = {
            "days": RasterImage(image=days[None, None]),
            "valid": RasterImage(image=(days != 65535).float()[None, None]),
        }

    input_dict, target_dict = StackSampler(
        deterministic=True, emit_timestep_targets=False
    )(input_dict, target_dict)

    num_timesteps = NUM_QUARTERLY + NUM_FREQUENT
    context = ModelContext(inputs=[input_dict], metadatas=[])
    logits = FeatureMaps([torch.zeros(1, num_timesteps, CROP, CROP)])
    for key, mode, expected_idx in (
        (TS_START_KEY, DateToTimestepMode.BEFORE, 13),
        (TS_END_KEY, DateToTimestepMode.AFTER, NUM_QUARTERLY + 1),
    ):
        assert set(target_dict[key]) == {"days", "valid"}
        head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=mode)
        targets = head(logits, context, [target_dict[key]]).outputs[0]["targets"]
        classes = targets["classes"].get_hw_tensor()
        valid = targets["valid"].get_hw_tensor()
        assert classes[2, 3] == expected_idx
        assert valid.sum() == 1 and valid[2, 3] == 1


POINT = (2, 3)
NEGATIVE = (0, 0)
DST, POST_CHANGE = 10, 3
TRANSITION_DST, TRANSITION_POST_CHANGE = 1, 8


def _days(date: datetime) -> int:
    return (date - UNIX_EPOCH).days


def _label(value: int, fill: int = 0) -> RasterImage:
    """A passthrough label raster with value at the positive point."""
    image = torch.full((CROP, CROP), fill, dtype=torch.long)
    image[POINT] = value
    return RasterImage(image=image[None, None])


def _gating_inputs(
    post_change: datetime,
    transition: datetime | None,
    transition_classes: tuple[int, int] = (TRANSITION_DST, TRANSITION_POST_CHANGE),
    with_rasters: bool = True,
) -> tuple[dict, dict]:
    """Stack inputs with dst/post_change targets and the per-pixel date rasters.

    The window's sidecar post_change is left as in _stack_inputs (before the end of
    the frequent images), so the rasters decide the gating unless absent.
    """
    input_dict, target_dict = _stack_inputs()
    target_dict["binary"]["classes"].image[0, 0][NEGATIVE] = 1
    for name, value in (("dst", DST), ("post_change", POST_CHANGE)):
        classes = torch.zeros(CROP, CROP, dtype=torch.long)
        classes[POINT] = value
        target_dict[name] = {
            "classes": RasterImage(image=classes[None, None]),
            "valid": RasterImage(image=(classes > 0).float()[None, None]),
        }
    if with_rasters:
        input_dict["label_ts_end"] = _label(_days(post_change), fill=65535)
        if transition is not None:
            input_dict["label_transition_date"] = _label(_days(transition), fill=65535)
            input_dict["label_transition_dst"] = _label(transition_classes[0])
            input_dict["label_transition_post_change"] = _label(transition_classes[1])
        else:
            input_dict["label_transition_date"] = _label(65535, fill=65535)
            input_dict["label_transition_dst"] = _label(0)
            input_dict["label_transition_post_change"] = _label(0)
    return input_dict, target_dict


def _frequent_ts(input_dict: dict) -> list[tuple[datetime, datetime]]:
    return input_dict[f"{FREQUENT_KEY_PREFIX}0"].timestamps


def _at_point(target_dict: dict, name: str) -> tuple[int, float]:
    classes = target_dict[name]["classes"].get_hw_tensor()
    valid = target_dict[name]["valid"].get_hw_tensor()
    return int(classes[POINT]), float(valid[POINT])


@pytest.mark.parametrize(
    "regime,expected_dst,expected_post_change,expected_end_idx",
    [
        # The images reach post_change: the post labels are trained.
        ("post", (DST, 1.0), (POST_CHANGE, 1.0), NUM_QUARTERLY + 1),
        # Between the transition and post_change: the transition labels.
        (
            "transition",
            (TRANSITION_DST, 1.0),
            (TRANSITION_POST_CHANGE, 1.0),
            NUM_QUARTERLY + 3,
        ),
        # Before the transition (still in progress): masked, and ts_end targets the
        # transition date, which maps to the last timestep.
        (
            "in_progress",
            (DST, 0.0),
            (POST_CHANGE, 0.0),
            NUM_QUARTERLY + NUM_FREQUENT - 1,
        ),
        # No transition and before post_change: masked, ts_end at the last timestep.
        (
            "no_transition",
            (DST, 0.0),
            (POST_CHANGE, 0.0),
            NUM_QUARTERLY + NUM_FREQUENT - 1,
        ),
    ],
)
def test_stack_sampler_date_gating(
    regime: str,
    expected_dst: tuple[int, float],
    expected_post_change: tuple[int, float],
    expected_end_idx: int,
) -> None:
    input_dict, _ = _stack_inputs()
    f_ts = _frequent_ts(input_dict)
    end = f_ts[-1][1]
    later = end + timedelta(days=30)
    post_change, transition = {
        "post": (f_ts[1][0], None),
        "transition": (later, f_ts[3][0]),
        "in_progress": (later, end + timedelta(days=10)),
        "no_transition": (later, None),
    }[regime]
    input_dict, target_dict = _gating_inputs(post_change, transition)

    input_dict, target_dict = StackSampler(deterministic=True)(input_dict, target_dict)
    assert _at_point(target_dict, "dst") == expected_dst
    assert _at_point(target_dict, "post_change") == expected_post_change
    classes = target_dict[TS_END_KEY]["classes"].get_hw_tensor()
    assert classes[POINT] == expected_end_idx
    # The label rasters are consumed, so they don't reach the model.
    assert not any(key.startswith("label_") for key in input_dict)
    # Negative points still train the change-category heads as "none".
    neg_classes = target_dict["post_change"]["classes"].get_hw_tensor()
    neg_valid = target_dict["post_change"]["valid"].get_hw_tensor()
    assert neg_classes[NEGATIVE] == 1 and neg_valid[NEGATIVE] == 1


def test_stack_sampler_masks_unset_transition_categories() -> None:
    input_dict, _ = _stack_inputs()
    f_ts = _frequent_ts(input_dict)
    input_dict, target_dict = _gating_inputs(
        f_ts[-1][1] + timedelta(days=30), f_ts[3][0], transition_classes=(0, 0)
    )
    _, target_dict = StackSampler(deterministic=True)(input_dict, target_dict)
    assert _at_point(target_dict, "dst")[1] == 0.0
    assert _at_point(target_dict, "post_change")[1] == 0.0
    # ts_end still targets the transition date.
    classes = target_dict[TS_END_KEY]["classes"].get_hw_tensor()
    assert classes[POINT] == NUM_QUARTERLY + 3


def test_stack_sampler_falls_back_to_sidecar_post_change() -> None:
    """Without the label rasters, the window's post_change gates the targets."""
    input_dict, _ = _stack_inputs()
    f_ts = _frequent_ts(input_dict)
    input_dict, target_dict = _gating_inputs(f_ts[1][0], None, with_rasters=False)
    _, kept = StackSampler(deterministic=True)(input_dict, target_dict)
    assert _at_point(kept, "dst") == (DST, 1.0)
    assert _at_point(kept, "post_change") == (POST_CHANGE, 1.0)

    input_dict, target_dict = _gating_inputs(f_ts[1][0], None, with_rasters=False)
    input_dict[ANNOTATION_KEY]["post_change"] = f_ts[-1][1] + timedelta(days=30)
    _, masked = StackSampler(deterministic=True)(input_dict, target_dict)
    assert _at_point(masked, "dst")[1] == 0.0
    assert _at_point(masked, "post_change")[1] == 0.0


@pytest.mark.parametrize("use_label_raster", [True, False])
def test_stack_sampler_rewrites_per_pixel_ts_end_days(use_label_raster: bool) -> None:
    """With PerPixelTimestepTask targets, ts_end days move to the transition date.

    The post_change dates come from the label_ts_end passthrough raster, or from
    the ts_end days target itself when the raster is not passed through.
    """
    input_dict, _ = _stack_inputs()
    f_ts = _frequent_ts(input_dict)
    post_change = f_ts[-1][1] + timedelta(days=30)
    transition = f_ts[3][0]
    input_dict, target_dict = _gating_inputs(post_change, transition)
    if not use_label_raster:
        del input_dict["label_ts_end"]
    days = torch.full((CROP, CROP), 65535, dtype=torch.long)
    days[POINT] = _days(post_change)
    target_dict[TS_END_KEY] = {
        "days": RasterImage(image=days[None, None]),
        "valid": RasterImage(image=(days != 65535).float()[None, None]),
    }

    input_dict, target_dict = StackSampler(
        deterministic=True, emit_timestep_targets=False
    )(input_dict, target_dict)
    assert _at_point(target_dict, "dst") == (TRANSITION_DST, 1.0)
    ts_end = target_dict[TS_END_KEY]
    assert ts_end["days"].get_hw_tensor()[POINT] == _days(transition)
    valid = ts_end["valid"].get_hw_tensor()
    assert valid.sum() == 1 and valid[POINT] == 1

    # PerPixelTimestepHead maps it to the transition's timestep.
    num_timesteps = NUM_QUARTERLY + NUM_FREQUENT
    context = ModelContext(inputs=[input_dict], metadatas=[])
    logits = FeatureMaps([torch.zeros(1, num_timesteps, CROP, CROP)])
    head = PerPixelTimestepHead(input_key=INPUT_KEY, mode=DateToTimestepMode.AFTER)
    targets = head(logits, context, [ts_end]).outputs[0]["targets"]
    assert targets["classes"].get_hw_tensor()[POINT] == NUM_QUARTERLY + 3


def test_per_pixel_ts_end_in_progress_maps_to_last_timestep() -> None:
    """With the fallback, a ts_end date after the stack maps to the last timestep."""
    input_dict, _ = _stack_inputs()
    f_ts = _frequent_ts(input_dict)
    post_change = f_ts[-1][1] + timedelta(days=30)
    input_dict, target_dict = _gating_inputs(post_change, None)
    days = torch.full((CROP, CROP), 65535, dtype=torch.long)
    days[POINT] = _days(post_change)
    target_dict[TS_END_KEY] = {
        "days": RasterImage(image=days[None, None]),
        "valid": RasterImage(image=(days != 65535).float()[None, None]),
    }
    input_dict, target_dict = StackSampler(
        deterministic=True, emit_timestep_targets=False
    )(input_dict, target_dict)

    num_timesteps = NUM_QUARTERLY + NUM_FREQUENT
    context = ModelContext(inputs=[input_dict], metadatas=[])
    logits = FeatureMaps([torch.zeros(1, num_timesteps, CROP, CROP)])
    head = PerPixelTimestepHead(
        input_key=INPUT_KEY,
        mode=DateToTimestepMode.AFTER,
        fallback_to_closest_timestep_if_no_match=True,
    )
    targets = head(logits, context, [target_dict[TS_END_KEY]]).outputs[0]["targets"]
    assert targets["classes"].get_hw_tensor()[POINT] == num_timesteps - 1
    assert targets["valid"].get_hw_tensor()[POINT] == 1


def _centers(timestamps: list[tuple[datetime, datetime]]) -> list[datetime]:
    return [t0 + (t1 - t0) / 2 for t0, t1 in timestamps]


def _indexed_image(num: int, offset: float) -> torch.Tensor:
    """Image whose timestep i is filled with offset + i (to track positions)."""
    image = torch.zeros(3, num, CROP, CROP)
    for i in range(num):
        image[:, i] = offset + i
    return image


def _weekly_frequent_ts(start: datetime) -> list[tuple[datetime, datetime]]:
    return [
        (start + timedelta(days=7 * i), start + timedelta(days=7 * i))
        for i in range(NUM_FREQUENT)
    ]


def test_stack_sampler_excludes_quarterly_after_earliest_frequent() -> None:
    """Quarterly images at/after the earliest frequent image are not used."""
    input_dict, target_dict = _stack_inputs()
    q_ts = input_dict[QUARTERLY_KEY].timestamps
    # The frequent images start before the last quarterly mosaic, so it is dropped.
    f0 = q_ts[-1][0] - timedelta(days=5)
    input_dict[f"{FREQUENT_KEY_PREFIX}0"] = RasterImage(
        image=_indexed_image(NUM_FREQUENT, offset=100),
        timestamps=_weekly_frequent_ts(f0),
    )

    input_dict, _ = StackSampler(deterministic=True)(input_dict, target_dict)
    stack = input_dict[INPUT_KEY]
    assert stack.image.shape[1] == NUM_QUARTERLY + NUM_FREQUENT
    centers = _centers(stack.timestamps)
    assert all(a < b for a, b in zip(centers, centers[1:]))
    assert all(c < f0 for c in centers[:NUM_QUARTERLY])
    assert q_ts[-1][0] not in [ts[0] for ts in stack.timestamps]
    assert stack.image[0, NUM_QUARTERLY:, 0, 0].tolist() == [
        100.0 + i for i in range(NUM_FREQUENT)
    ]


def test_predict_stack_builder_short_quarterly() -> None:
    """Prediction pads a short quarterly layer with distinct older timestamps."""
    num_q = NUM_QUARTERLY - 3
    q_ts = [
        (T0 + timedelta(days=90 * i), T0 + timedelta(days=90 * i + 1))
        for i in range(num_q)
    ]
    input_dict = {
        QUARTERLY_KEY: RasterImage(
            image=_indexed_image(num_q, offset=0), timestamps=q_ts
        ),
        f"{FREQUENT_KEY_PREFIX}0": RasterImage(
            image=_indexed_image(NUM_FREQUENT, offset=100),
            timestamps=_weekly_frequent_ts(q_ts[-1][1] + timedelta(days=10)),
        ),
    }
    input_dict, _ = PredictStackBuilder()(input_dict, {})
    stack = input_dict[INPUT_KEY]
    assert stack.image.shape[1] == NUM_QUARTERLY + NUM_FREQUENT
    centers = _centers(stack.timestamps)
    assert all(a < b for a, b in zip(centers, centers[1:]))
    assert stack.image[0, :, 0, 0].tolist() == (
        [0.0] * 3
        + [float(i) for i in range(num_q)]
        + [100.0 + i for i in range(NUM_FREQUENT)]
    )


def test_predict_stack_builder_rejects_wrong_frequent_count() -> None:
    """The prediction frequent layer must have exactly NUM_FREQUENT images."""
    q_ts = [
        (T0 + timedelta(days=90 * i), T0 + timedelta(days=90 * i + 1))
        for i in range(NUM_QUARTERLY)
    ]
    f_ts = _weekly_frequent_ts(q_ts[-1][1] + timedelta(days=10))[:-1]
    input_dict = {
        QUARTERLY_KEY: RasterImage(
            image=_indexed_image(NUM_QUARTERLY, offset=0), timestamps=q_ts
        ),
        f"{FREQUENT_KEY_PREFIX}0": RasterImage(
            image=_indexed_image(NUM_FREQUENT - 1, offset=100), timestamps=f_ts
        ),
    }
    with pytest.raises(ValueError, match="timesteps"):
        PredictStackBuilder()(input_dict, {})


def _seg_targets(labels: torch.Tensor, valid: torch.Tensor) -> list[dict]:
    return [
        {
            "classes": RasterImage(image=labels[i][None, None]),
            "valid": RasterImage(image=valid[i][None, None].float()),
        }
        for i in range(labels.shape[0])
    ]


def test_balanced_head_binary_matches_segmentation_head_outputs() -> None:
    """Outputs match SegmentationHead; the loss balances change vs no-change."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, CROP, CROP)
    labels = torch.ones(2, CROP, CROP, dtype=torch.long)
    labels[:, 0, :] = 2  # one row of change points per sample
    valid = torch.ones(2, CROP, CROP, dtype=torch.bool)
    valid[1, -1, :] = False
    targets = _seg_targets(labels, valid)
    context = ModelContext(inputs=[], metadatas=[])

    balanced = BalancedSegmentationHead()(FeatureMaps([logits]), context, targets)
    plain = SegmentationHead()(FeatureMaps([logits]), context, targets)
    assert torch.allclose(balanced.outputs, plain.outputs)
    assert set(balanced.loss_dict) == {"cls"}

    # Reference: per-sample mean over change points + mean over no-change points.
    per_pixel = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = []
    for i in range(2):
        pos = valid[i] & (labels[i] == 2)
        neg = valid[i] & (labels[i] == 1)
        expected.append(per_pixel[i][pos].mean() + per_pixel[i][neg].mean())
    assert torch.allclose(balanced.loss_dict["cls"], torch.stack(expected).mean())


def test_balanced_head_multiclass() -> None:
    """Per-class means are summed within a sample, then averaged over samples."""
    torch.manual_seed(0)
    num_classes = 8
    logits = torch.randn(3, num_classes, CROP, CROP)
    labels = torch.ones(3, CROP, CROP, dtype=torch.long)
    labels[0, 0, :] = 3  # sample 0: many "none" points, one row of class 3
    labels[0, 1, :2] = 5  # and two points of class 5
    labels[1] = 4  # sample 1: a single class
    valid = torch.zeros(3, CROP, CROP, dtype=torch.bool)
    valid[0] = True
    valid[1, 0, :2] = True
    # Sample 2 has no valid points and must not contribute.
    context = ModelContext(inputs=[], metadatas=[])

    out = BalancedSegmentationHead()(
        FeatureMaps([logits]), context, _seg_targets(labels, valid)
    )

    per_pixel = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = []
    for i in range(2):
        sample_labels = labels[i][valid[i]]
        sample_loss = per_pixel[i][valid[i]]
        expected.append(
            sum(sample_loss[sample_labels == c].mean() for c in sample_labels.unique())
        )
    assert torch.allclose(out.loss_dict["cls"], torch.stack(expected).mean())


def test_balanced_head_no_valid_points() -> None:
    """With no valid points the loss is a differentiable zero."""
    logits = torch.randn(2, 4, CROP, CROP)
    labels = torch.ones(2, CROP, CROP, dtype=torch.long)
    valid = torch.zeros(2, CROP, CROP, dtype=torch.bool)
    context = ModelContext(inputs=[], metadatas=[])

    out = BalancedSegmentationHead()(
        FeatureMaps([logits]), context, _seg_targets(labels, valid)
    )
    assert out.loss_dict["cls"].item() == 0.0
    assert out.loss_dict["cls"].requires_grad


def test_timestep_tolerance_accuracy() -> None:
    """Predictions within the tolerance of the target index count as correct."""
    num_steps = 5
    pred = torch.zeros(num_steps, 2, 2)
    # argmax indices: [[0, 1], [2, 4]]
    pred[0, 0, 0] = 1
    pred[1, 0, 1] = 1
    pred[2, 1, 0] = 1
    pred[4, 1, 1] = 1
    target_idx = torch.tensor([[0, 2], [4, 4]])
    valid = torch.tensor([[1, 1], [1, 0]])
    targets = [
        {
            "classes": RasterImage(image=target_idx[None, None]),
            "valid": RasterImage(image=valid[None, None].float()),
        }
    ]

    exact = TimestepToleranceAccuracy(tolerance=0)
    exact.update([pred], targets)
    assert torch.isclose(exact.compute(), torch.tensor(1 / 3))

    within1 = TimestepToleranceAccuracy(tolerance=1)
    within1.update([pred], targets)
    assert torch.isclose(within1.compute(), torch.tensor(2 / 3))

    within2 = TimestepToleranceAccuracy(tolerance=2)
    within2.update([pred], targets)
    assert torch.isclose(within2.compute(), torch.tensor(1.0))
