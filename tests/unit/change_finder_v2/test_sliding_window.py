"""Tests for sliding-window evaluation of LCC change models."""

import torch
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

from rslp.change_finder_v2.lcc_model.sliding_window import (
    INPUT_KEY,
    SlidingWindowEvalMixin,
    run_sliding_window,
)


def _coord_window(size: int) -> torch.Tensor:
    """A (1, 1, size, size) image whose pixel value encodes its (y, x) position."""
    ys = torch.arange(size).view(size, 1).expand(size, size)
    xs = torch.arange(size).view(1, size).expand(size, size)
    return (ys * size + xs).float().view(1, 1, size, size)


def _make_context(image: torch.Tensor) -> ModelContext:
    """Single-element batch context wrapping one CTHW image under INPUT_KEY."""
    return ModelContext(
        inputs=[{INPUT_KEY: RasterImage(image=image)}],
        metadatas=[None],
    )


def _copy_core(context: ModelContext, targets=None) -> ModelOutput:
    """Core forward that copies the input's spatial content into the output.

    Lets a stitched result be compared against the original full window to
    verify per-tile cropping and placement.
    """
    outputs = []
    for inp in context.inputs:
        tile = inp[INPUT_KEY].image[:, 0]  # (1, h, w)
        outputs.append({"binary": tile.clone()})
    return ModelOutput(outputs=outputs, loss_dict={})


def test_stitch_reconstructs_window_no_overlap() -> None:
    """Non-overlapping tiling copies each crop back to its place exactly."""
    image = _coord_window(128)
    out = run_sliding_window(
        _copy_core, _make_context(image), None, crop_size=64, overlap=0
    )
    assert out.outputs[0]["binary"].shape == (1, 128, 128)
    torch.testing.assert_close(out.outputs[0]["binary"], image[:, 0])


def test_stitch_reconstructs_window_with_overlap() -> None:
    """With overlap, duplicated pixels are averaged; identical values are preserved."""
    image = _coord_window(128)
    out = run_sliding_window(
        _copy_core, _make_context(image), None, crop_size=64, overlap=16
    )
    torch.testing.assert_close(out.outputs[0]["binary"], image[:, 0])


def test_overlap_count_normalization() -> None:
    """A constant prediction stays constant after overlap averaging."""

    def const_core(context: ModelContext, targets=None) -> ModelOutput:
        outputs = []
        for inp in context.inputs:
            h, w = inp[INPUT_KEY].image.shape[-2:]
            outputs.append({"binary": torch.full((3, h, w), 0.25)})
        return ModelOutput(outputs=outputs, loss_dict={})

    image = _coord_window(128)
    out = run_sliding_window(
        const_core, _make_context(image), None, crop_size=64, overlap=16
    )
    torch.testing.assert_close(
        out.outputs[0]["binary"], torch.full((3, 128, 128), 0.25)
    )


def test_probabilities_sum_to_one() -> None:
    """Averaging softmax tiles yields a valid distribution over channels."""

    def softmax_core(context: ModelContext, targets=None) -> ModelOutput:
        outputs = []
        for inp in context.inputs:
            h, w = inp[INPUT_KEY].image.shape[-2:]
            logits = torch.randn(5, h, w)
            outputs.append(
                {
                    "binary": torch.softmax(logits, dim=0),
                    "timestamps": {"start": torch.softmax(torch.randn(4, h, w), dim=0)},
                }
            )
        return ModelOutput(outputs=outputs, loss_dict={})

    image = _coord_window(128)
    out = run_sliding_window(
        softmax_core, _make_context(image), None, crop_size=48, overlap=8
    )
    assert out.outputs[0]["binary"].shape == (5, 128, 128)
    torch.testing.assert_close(
        out.outputs[0]["binary"].sum(dim=0), torch.ones(128, 128)
    )
    # Nested timestamps output is stitched too.
    torch.testing.assert_close(
        out.outputs[0]["timestamps"]["start"].sum(dim=0), torch.ones(128, 128)
    )


def test_losses_are_mean_over_tiles() -> None:
    """Per-tile losses are averaged (not summed) across tiles."""
    counter = {"i": 0}

    def loss_core(context: ModelContext, targets=None) -> ModelOutput:
        outputs = []
        for inp in context.inputs:
            h, w = inp[INPUT_KEY].image.shape[-2:]
            outputs.append({"binary": torch.zeros(1, h, w)})
        loss = torch.tensor(float(counter["i"]))
        counter["i"] += 1
        return ModelOutput(outputs=outputs, loss_dict={"a": loss})

    image = _coord_window(128)
    out = run_sliding_window(
        loss_core, _make_context(image), [{}], crop_size=64, overlap=0
    )
    num_tiles = counter["i"]
    assert num_tiles == 4
    expected = sum(range(num_tiles)) / num_tiles
    torch.testing.assert_close(out.loss_dict["a"], torch.tensor(expected))


def test_targets_are_cropped_per_tile() -> None:
    """Targets are cropped to each tile so the core sees matching-size targets."""
    seen_sizes = []

    def target_core(context: ModelContext, targets=None) -> ModelOutput:
        outputs = []
        for i, inp in enumerate(context.inputs):
            h, w = inp[INPUT_KEY].image.shape[-2:]
            target_hw = targets[i]["binary"]["classes"].image.shape[-2:]
            seen_sizes.append((h, w, *target_hw))
            outputs.append({"binary": torch.zeros(1, h, w)})
        return ModelOutput(outputs=outputs, loss_dict={})

    image = _coord_window(128)
    targets = [{"binary": {"classes": RasterImage(image=torch.zeros(1, 1, 128, 128))}}]
    run_sliding_window(
        target_core, _make_context(image), targets, crop_size=64, overlap=0
    )
    # Every tile saw an input and a target of the same (cropped) spatial size.
    assert all(h == th and w == tw for (h, w, th, tw) in seen_sizes)
    assert all(h == 64 and w == 64 for (h, w, _, _) in seen_sizes)


class _DummyModel(SlidingWindowEvalMixin, torch.nn.Module):
    """Minimal model exercising the mixin dispatch."""

    def __init__(self, eval_crop_size=None, eval_overlap=0):
        super().__init__()
        self.eval_crop_size = eval_crop_size
        self.eval_overlap = eval_overlap
        self.core_calls = 0

    def _forward_core(self, context: ModelContext, targets=None) -> ModelOutput:
        self.core_calls += 1
        return _copy_core(context, targets)


def test_mixin_tiles_in_eval_mode() -> None:
    """Eval mode tiles a large window into multiple core calls and stitches."""
    model = _DummyModel(eval_crop_size=64)
    model.eval()
    image = _coord_window(128)
    out = model(_make_context(image))
    assert model.core_calls == 4
    torch.testing.assert_close(out.outputs[0]["binary"], image[:, 0])


def test_mixin_passthrough_in_training_mode() -> None:
    """Training mode runs a single core call without tiling."""
    model = _DummyModel(eval_crop_size=64)
    model.train()
    image = _coord_window(128)
    model(_make_context(image))
    assert model.core_calls == 1


def test_mixin_passthrough_when_disabled_or_small() -> None:
    """No tiling when eval_crop_size is None or the input already fits."""
    disabled = _DummyModel(eval_crop_size=None)
    disabled.eval()
    disabled(_make_context(_coord_window(128)))
    assert disabled.core_calls == 1

    small = _DummyModel(eval_crop_size=64)
    small.eval()
    small(_make_context(_coord_window(64)))
    assert small.core_calls == 1
