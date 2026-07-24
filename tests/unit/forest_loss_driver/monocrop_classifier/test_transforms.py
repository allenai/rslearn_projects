from datetime import datetime, timedelta, timezone

import pytest
import torch
from rslearn.train.model_context import RasterImage

from rslp.forest_loss_driver.monocrop_classifier.transforms import (
    PostLossMonthSampler,
)


def _image(num_timesteps: int = 23, reverse: bool = False) -> RasterImage:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    values = list(range(num_timesteps))
    timestamps = [
        (start + timedelta(days=30 * value), start + timedelta(days=30 * (value + 1)))
        for value in values
    ]
    if reverse:
        values.reverse()
        timestamps.reverse()
    tensor = torch.tensor(values, dtype=torch.float32)[None, :, None, None]
    return RasterImage(tensor, timestamps=timestamps)


@pytest.mark.parametrize(
    ("num_post_months", "expected_start"),
    [(1, 0), (6, 5), (12, 11)],
)
def test_fixed_month_slice(
    num_post_months: int,
    expected_start: int,
) -> None:
    inputs = {"sentinel2_l2a": _image(reverse=True)}
    outputs, _ = PostLossMonthSampler(num_post_months)(inputs, {})
    image = outputs["sentinel2_l2a"]
    assert image.image.shape == (1, 12, 1, 1)
    assert image.image.flatten().tolist() == list(
        range(expected_start, expected_start + 12)
    )
    assert image.timestamps is not None
    assert len(image.timestamps) == 12
    assert image.timestamps[0][0] == datetime(
        2024, 1, 1, tzinfo=timezone.utc
    ) + timedelta(days=30 * expected_start)


def test_random_month_slice_is_within_supported_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("random.randint", lambda lower, upper: 7)
    outputs, _ = PostLossMonthSampler()({"sentinel2_l2a": _image()}, {})
    assert outputs["sentinel2_l2a"].image.flatten().tolist() == list(range(6, 18))


def test_default_fixed_month_is_used_when_override_is_missing() -> None:
    outputs, _ = PostLossMonthSampler(default_num_post_months=6)(
        {"sentinel2_l2a": _image()}, {}
    )
    assert outputs["sentinel2_l2a"].image.flatten().tolist() == list(range(5, 17))


def test_fixed_month_is_clamped_to_available_post_months() -> None:
    outputs, _ = PostLossMonthSampler(num_post_months=8)(
        {"sentinel2_l2a": _image(17)}, {}
    )
    assert outputs["sentinel2_l2a"].image.flatten().tolist() == list(range(5, 17))


def test_random_month_uses_available_post_months(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_randint(lower: int, upper: int) -> int:
        calls.append((lower, upper))
        return upper

    monkeypatch.setattr("random.randint", fake_randint)
    outputs, _ = PostLossMonthSampler()({"sentinel2_l2a": _image(17)}, {})
    assert calls == [(1, 6)]
    assert outputs["sentinel2_l2a"].image.flatten().tolist() == list(range(5, 17))


@pytest.mark.parametrize("num_post_months", [0, 13])
def test_invalid_fixed_month(num_post_months: int) -> None:
    with pytest.raises(ValueError, match="num_post_months"):
        PostLossMonthSampler(num_post_months)


def test_requires_12_to_23_timestamped_frames() -> None:
    with pytest.raises(ValueError, match="12-23 timesteps"):
        PostLossMonthSampler(6)({"sentinel2_l2a": _image(11)}, {})

    with pytest.raises(ValueError, match="12-23 timesteps"):
        PostLossMonthSampler(6)({"sentinel2_l2a": _image(24)}, {})

    image = _image()
    image.timestamps = None
    with pytest.raises(ValueError, match="one timestamp range"):
        PostLossMonthSampler(6)({"sentinel2_l2a": image}, {})
