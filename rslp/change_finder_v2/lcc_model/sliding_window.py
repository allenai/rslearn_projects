"""Sliding-window evaluation for LCC change models.

Lets a model trained at a small ``crop_size`` be evaluated over a larger window
(e.g. 128x128) by tiling the window into overlapping crops of ``crop_size``,
running the model's core forward on each tile, and stitching the per-tile
outputs back into a full-resolution prediction. Overlapping pixels are merged by
averaging the (softmax/sigmoid) probability maps, and the per-tile losses are
accumulated and mean-reduced.

This keeps every model in its native spatial regime while producing metrics over
an identical window for all configs, which is what makes cross-config (e.g. size
ablation) comparisons valid: without it, each config's metrics are computed over
a different input extent and are not directly comparable.

The behavior is opt-in via ``SlidingWindowEvalMixin``: a model sets
``eval_crop_size`` (and optionally ``eval_overlap``) and implements
``_forward_core`` (its normal single-pass forward). Tiling only triggers in eval
mode when the input is larger than ``eval_crop_size``; training and inputs that
already fit are passed straight through to ``_forward_core``. The mixin adds no
parameters, so existing checkpoints load unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from rslearn.train.all_crops_dataset import get_window_crop_options
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

INPUT_KEY = "sentinel2_l2a"

CoreForward = Callable[[ModelContext, "list[dict[str, Any]] | None"], ModelOutput]


def _crop_nested(obj: Any, col_off: int, row_off: int, crop_w: int, crop_h: int) -> Any:
    """Recursively crop RasterImage leaves to the given spatial window.

    Dicts are recursed into; RasterImage tensors are sliced on their last two
    (height, width) dimensions; all other leaves are returned unchanged.
    """
    if isinstance(obj, RasterImage):
        cropped = obj.image[
            ..., row_off : row_off + crop_h, col_off : col_off + crop_w
        ]
        return RasterImage(image=cropped, timestamps=obj.timestamps)
    if isinstance(obj, dict):
        return {
            k: _crop_nested(v, col_off, row_off, crop_w, crop_h)
            for k, v in obj.items()
        }
    return obj


def _zeros_full(sample: Any, full_h: int, full_w: int) -> Any:
    """Build a full-window zero buffer matching the structure of one tile output.

    Tensor leaves keep their leading (e.g. channel/time) dims and get the last
    two dims replaced by ``(full_h, full_w)``; dicts are recursed into.
    """
    if isinstance(sample, torch.Tensor):
        shape = tuple(sample.shape[:-2]) + (full_h, full_w)
        return torch.zeros(shape, dtype=sample.dtype, device=sample.device)
    if isinstance(sample, dict):
        return {k: _zeros_full(v, full_h, full_w) for k, v in sample.items()}
    raise TypeError(f"unexpected output leaf type {type(sample)}")


def _accumulate_into(acc: Any, src: Any, col_off: int, row_off: int) -> None:
    """Add tile output ``src`` into the full-window accumulator ``acc`` in place.

    ``acc`` leaves are full-window ``(..., H, W)``; ``src`` leaves are tile-sized
    ``(..., h, w)`` and are added at the given spatial offset.
    """
    if isinstance(src, torch.Tensor):
        h, w = src.shape[-2], src.shape[-1]
        acc[..., row_off : row_off + h, col_off : col_off + w] += src
        return
    if isinstance(src, dict):
        for k, v in src.items():
            _accumulate_into(acc[k], v, col_off, row_off)
        return
    raise TypeError(f"unexpected output leaf type {type(src)}")


def _divide_by_count(acc: Any, count: torch.Tensor) -> Any:
    """Divide every accumulator leaf by the per-pixel ``count`` map (broadcast)."""
    if isinstance(acc, torch.Tensor):
        return acc / count
    if isinstance(acc, dict):
        return {k: _divide_by_count(v, count) for k, v in acc.items()}
    raise TypeError(f"unexpected output leaf type {type(acc)}")


def _exceeds_crop(context: ModelContext, crop_size: int) -> bool:
    """Return whether the main raster input is larger than ``crop_size``."""
    image = context.inputs[0][INPUT_KEY].image
    return image.shape[-2] > crop_size or image.shape[-1] > crop_size


def run_sliding_window(
    core_forward: CoreForward,
    context: ModelContext,
    targets: list[dict[str, Any]] | None,
    crop_size: int,
    overlap: int,
) -> ModelOutput:
    """Run ``core_forward`` over overlapping tiles and stitch the results.

    The window is divided into ``crop_size`` tiles (stride ``crop_size - overlap``)
    using the same logic as rslearn's all-crops sliding-window inference. Each
    tile runs the full batch through ``core_forward``; the per-tile probability
    maps are summed into a full-window buffer and divided by a per-pixel coverage
    count (so overlaps are averaged), and the per-tile losses are mean-reduced.

    Args:
        core_forward: the model's normal single-pass forward.
        context: the full-window model context.
        targets: optional full-window targets, cropped per tile for the loss.
        crop_size: side length of each tile (the model's native crop size).
        overlap: number of pixels shared between adjacent tiles.

    Returns:
        a ModelOutput with full-window stitched outputs and mean per-tile losses.
    """
    inputs = context.inputs
    batch_size = len(inputs)
    image = inputs[0][INPUT_KEY].image  # (C, T, H, W)
    full_h, full_w = image.shape[-2], image.shape[-1]

    crop_bounds = get_window_crop_options(
        (crop_size, crop_size), (overlap, overlap), (0, 0, full_w, full_h)
    )
    num_tiles = len(crop_bounds)

    acc_outputs: list[Any] | None = None
    count = torch.zeros((full_h, full_w), dtype=image.dtype, device=image.device)
    loss_accum: dict[str, torch.Tensor] = {}

    for col_off, row_off, col_end, row_end in crop_bounds:
        tile_w = col_end - col_off
        tile_h = row_end - row_off

        sub_inputs = [
            _crop_nested(inp, col_off, row_off, tile_w, tile_h) for inp in inputs
        ]
        sub_context = ModelContext(
            inputs=sub_inputs,
            metadatas=context.metadatas,
            context_dict=dict(context.context_dict),
        )
        sub_targets = None
        if targets is not None:
            sub_targets = [
                _crop_nested(t, col_off, row_off, tile_w, tile_h) for t in targets
            ]

        out = core_forward(sub_context, sub_targets)

        if acc_outputs is None:
            acc_outputs = [
                _zeros_full(out.outputs[i], full_h, full_w)
                for i in range(batch_size)
            ]
        for i in range(batch_size):
            _accumulate_into(acc_outputs[i], out.outputs[i], col_off, row_off)
        count[row_off:row_end, col_off:col_end] += 1

        for k, v in out.loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

    assert acc_outputs is not None
    count = count.clamp(min=1)
    outputs = [_divide_by_count(acc, count) for acc in acc_outputs]
    loss_dict = {k: v / num_tiles for k, v in loss_accum.items()}

    return ModelOutput(outputs=outputs, loss_dict=loss_dict)


class SlidingWindowEvalMixin:
    """Adds optional sliding-window inference over large eval windows.

    Subclasses implement ``_forward_core`` (the normal single-pass forward) and
    set ``eval_crop_size`` / ``eval_overlap`` in ``__init__``. When
    ``eval_crop_size`` is set and the model is in eval mode, inputs larger than
    ``eval_crop_size`` are tiled into overlapping crops, each run through
    ``_forward_core``, and the per-tile outputs stitched back together. Training
    and already-small inputs go straight to ``_forward_core``.
    """

    eval_crop_size: int | None = None
    eval_overlap: int = 0

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Dispatch to sliding-window inference in eval mode, else single pass."""
        if (
            self.eval_crop_size is not None
            and not self.training
            and _exceeds_crop(context, self.eval_crop_size)
        ):
            return run_sliding_window(
                self._forward_core,
                context,
                targets,
                self.eval_crop_size,
                self.eval_overlap,
            )
        return self._forward_core(context, targets)

    def _forward_core(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Run the model's normal single-pass forward (implemented by subclass)."""
        raise NotImplementedError
