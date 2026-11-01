from datetime import datetime, timedelta

import torch
from rslearn.train.model_context import RasterImage

from rslp.satlas.transforms import RandomConsecutiveTimesteps


def make_image(num_timesteps: int) -> RasterImage:
    # Pixel values encode the timestep index so we can check which ones were kept.
    image = torch.arange(num_timesteps, dtype=torch.float32)[None, :, None, None]
    image = image.expand(2, num_timesteps, 4, 4).clone()
    t0 = datetime(2026, 1, 1)
    timestamps = [
        (t0 + timedelta(days=7 * i), t0 + timedelta(days=7 * (i + 1)))
        for i in range(num_timesteps)
    ]
    return RasterImage(image, timestamps)


def test_keeps_consecutive_run() -> None:
    transform = RandomConsecutiveTimesteps(num_timesteps=6, selectors=["image"])
    starts = set()
    for _ in range(200):
        input_dict, _ = transform({"image": make_image(12)}, {})
        image = input_dict["image"]
        assert image.shape == (2, 6, 4, 4)
        kept = image.image[0, :, 0, 0].long().tolist()
        assert kept == list(range(kept[0], kept[0] + 6))
        assert image.timestamps == make_image(12).timestamps[kept[0] : kept[0] + 6]
        starts.add(kept[0])
    # All 7 possible starts should show up.
    assert starts == set(range(7))


def test_short_image_unchanged() -> None:
    transform = RandomConsecutiveTimesteps(num_timesteps=6, selectors=["image"])
    for num_timesteps in [3, 6]:
        input_dict, _ = transform({"image": make_image(num_timesteps)}, {})
        assert input_dict["image"].shape[1] == num_timesteps


def test_selectors_aligned() -> None:
    transform = RandomConsecutiveTimesteps(num_timesteps=2, selectors=["a", "b"])
    for _ in range(50):
        input_dict, _ = transform({"a": make_image(5), "b": make_image(5)}, {})
        assert torch.equal(input_dict["a"].image, input_dict["b"].image)


def test_skip_missing() -> None:
    transform = RandomConsecutiveTimesteps(
        num_timesteps=2, selectors=["missing", "image"], skip_missing=True
    )
    input_dict, _ = transform({"image": make_image(5)}, {})
    assert input_dict["image"].shape[1] == 2
