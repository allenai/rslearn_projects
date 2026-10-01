"""Tests for the LCC training glue: StackSampler targets, balanced head, metrics."""

from datetime import datetime, timedelta, timezone

import pytest
import torch
from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.segmentation import SegmentationHead

from rslp.olmoearth_lcc.lcc_model.components import BalancedBinarySegmentationHead
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


def test_balanced_binary_head_matches_segmentation_head_outputs() -> None:
    """Outputs match SegmentationHead; the loss balances change vs no-change."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, CROP, CROP)
    labels = torch.ones(2, CROP, CROP, dtype=torch.long)
    labels[:, 0, :] = 2  # one row of change points per sample
    valid = torch.ones(2, CROP, CROP, dtype=torch.bool)
    valid[1, -1, :] = False
    targets = _seg_targets(labels, valid)
    context = ModelContext(inputs=[], metadatas=[])

    balanced = BalancedBinarySegmentationHead()(FeatureMaps([logits]), context, targets)
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
