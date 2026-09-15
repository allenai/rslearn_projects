"""Per-pixel land-cover model used for training."""

from __future__ import annotations

from typing import Any

import torch
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
)


class PerPixelModelWrapper(torch.nn.Module):
    """Run a spatial model independently on each pixel of each input sample.

    The wrapped model receives one 1x1 sample for every pixel in the original
    batch. Its BHW x C x 1 x 1 segmentation output is reshaped back to
    B x C x H x W so rslearn's normal segmentation losses and metrics can be
    used at the outer level.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Create a new per-pixel wrapper.

        Args:
            model: rslearn model to run on each 1x1 pixel sample.
        """
        super().__init__()
        self.model = model

    def _get_hw(self, inputs: list[dict[str, Any]]) -> tuple[int, int]:
        """Get and validate spatial dimensions from RasterImage inputs."""
        height_width: tuple[int, int] | None = None
        for input_dict in inputs:
            for value in input_dict.values():
                if not isinstance(value, RasterImage):
                    continue
                cur_hw = (value.image.shape[-2], value.image.shape[-1])
                if height_width is None:
                    height_width = cur_hw
                elif cur_hw != height_width:
                    raise ValueError(
                        "all RasterImage inputs must have matching spatial shapes, "
                        f"got {cur_hw} and {height_width}"
                    )
        if height_width is None:
            raise ValueError(
                "PerPixelModelWrapper requires at least one RasterImage input"
            )
        return height_width

    def _slice_value(self, value: Any, row: int, col: int) -> Any:
        """Slice RasterImage/tensor leaves to one pixel, preserving structure."""
        if isinstance(value, RasterImage):
            return RasterImage(
                value.image[..., row : row + 1, col : col + 1].contiguous(),
                timestamps=value.timestamps,
            )
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return value[..., row : row + 1, col : col + 1].contiguous()
        if isinstance(value, dict):
            return {k: self._slice_value(v, row, col) for k, v in value.items()}
        return value

    def _flatten_inputs(
        self,
        inputs: list[dict[str, Any]],
        height: int,
        width: int,
    ) -> list[dict[str, Any]]:
        """Flatten B spatial samples into BHW one-pixel samples."""
        pixel_inputs = []
        for input_dict in inputs:
            for row in range(height):
                for col in range(width):
                    pixel_inputs.append(
                        {
                            k: self._slice_value(v, row, col)
                            for k, v in input_dict.items()
                        }
                    )
        return pixel_inputs

    def _flatten_targets(
        self,
        targets: list[dict[str, Any]],
        height: int,
        width: int,
    ) -> list[dict[str, Any]]:
        """Flatten B target dicts into BHW one-pixel target dicts."""
        pixel_targets = []
        for target_dict in targets:
            for row in range(height):
                for col in range(width):
                    pixel_targets.append(
                        {
                            k: self._slice_value(v, row, col)
                            for k, v in target_dict.items()
                        }
                    )
        return pixel_targets

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Run the wrapped model independently on every spatial pixel."""
        height, width = self._get_hw(context.inputs)
        batch_size = len(context.inputs)
        pixel_inputs = self._flatten_inputs(context.inputs, height, width)
        pixel_targets = (
            self._flatten_targets(targets, height, width)
            if targets is not None
            else None
        )
        pixel_metadatas = [
            metadata for metadata in context.metadatas for _ in range(height * width)
        ]

        pixel_context = ModelContext(
            inputs=pixel_inputs,
            metadatas=pixel_metadatas,
            context_dict=context.context_dict,
        )
        pixel_output = self.model(pixel_context, pixel_targets)
        outputs = pixel_output.outputs
        if not isinstance(outputs, torch.Tensor):
            raise ValueError("wrapped model output must be a tensor")
        if outputs.ndim != 4 or outputs.shape[-2:] != (1, 1):
            raise ValueError(
                "wrapped model output must have shape BHW x C x 1 x 1, "
                f"got {tuple(outputs.shape)}"
            )
        expected_pixels = batch_size * height * width
        if outputs.shape[0] != expected_pixels:
            raise ValueError(
                f"wrapped model produced {outputs.shape[0]} outputs for "
                f"{expected_pixels} pixels"
            )

        channels = outputs.shape[1]
        reshaped_outputs = outputs[:, :, 0, 0].reshape(
            batch_size, height, width, channels
        )
        reshaped_outputs = reshaped_outputs.permute(0, 3, 1, 2).contiguous()
        return ModelOutput(
            outputs=reshaped_outputs,
            loss_dict=pixel_output.loss_dict,
            metadata=pixel_output.metadata,
        )
