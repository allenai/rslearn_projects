"""DualPassTemporalChangeModel: two encoder passes feeding one temporal decoder.

This combines the original dual-forward-pass backbone usage (see
``DualPassChangeModel``) with the temporal-attention decoder and start/end
timestamp heads of ``TemporalChangeModel`` (model_20260610).

- Input: ``sentinel2_l2a`` with ``num_timesteps`` timesteps (e.g. 16 quarterly +
  4 frequent = 20), built by ``FrequentOptionSamplerV2``.
- The stack is split into pass1 (first ``num_pass1``) and pass2 (the rest). Two
  passes are required because the OlmoEarth encoder is limited to 12 timesteps
  per forward pass.
- Each pass runs the shared encoder with ``token_pooling=False`` so per-timestep
  tokens are preserved, giving ``(B, C, H, W, T_pass)`` features.
- The per-timestep tokens from both passes are concatenated along the time axis,
  so the temporal transformer (and the start/end heads) attend over all
  ``T1 + T2`` chronological tokens.

Everything downstream of feature extraction (temporal transformer, decoders,
start/end heads, losses, outputs) is inherited unchanged from
``TemporalChangeModel``.
"""

from __future__ import annotations

from typing import Any

import torch
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata

from .model import INPUT_KEY
from .model_20260610 import TemporalChangeModel


class DualPassTemporalChangeModel(TemporalChangeModel):
    """Two-pass encoder with a shared temporal-attention decoder."""

    def __init__(
        self,
        num_pass1: int = 10,
        num_timesteps: int = 20,
        **kwargs: Any,
    ):
        """Initialize the dual-pass temporal change model.

        Args:
            num_pass1: number of timesteps routed to the first encoder pass; the
                remaining ``num_timesteps - num_pass1`` go to the second pass.
                Both must be <= 12 (the encoder's per-pass timestep limit).
            num_timesteps: total number of input timesteps across both passes.
            kwargs: forwarded to ``TemporalChangeModel.__init__`` (notably
                ``encoder``).
        """
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.num_pass1 = num_pass1

    def _run_encoder(
        self, raster_images: list[RasterImage], metadatas: list[SampleMetadata]
    ) -> torch.Tensor:
        """Run OlmoEarth on a list of RasterImages, return per-timestep features.

        With ``token_pooling=False`` the encoder emits one token per timestep, so
        the returned feature map is ``(B, C, H, W, T_pass)``.
        """
        inputs = [{INPUT_KEY: img} for img in raster_images]
        sub_context = ModelContext(inputs=inputs, metadatas=metadatas)
        feature_maps = self.encoder(sub_context)
        return feature_maps.feature_maps[0]

    def _per_timestep_features(self, context: ModelContext) -> torch.Tensor:
        """Split into two passes, encode each, concat tokens along time.

        Returns per-timestep features ``(B, C, H, W, T1 + T2)`` in chronological
        order (pass1 tokens followed by pass2 tokens).
        """
        pass1_images: list[RasterImage] = []
        pass2_images: list[RasterImage] = []
        n = self.num_pass1
        for inp in context.inputs:
            combined: RasterImage = inp[INPUT_KEY]
            p1_img = combined.image[:, :n, :, :]
            p2_img = combined.image[:, n:, :, :]
            p1_ts = combined.timestamps[:n] if combined.timestamps else None
            p2_ts = combined.timestamps[n:] if combined.timestamps else None
            pass1_images.append(RasterImage(image=p1_img, timestamps=p1_ts))
            pass2_images.append(RasterImage(image=p2_img, timestamps=p2_ts))

        feat1 = self._run_encoder(pass1_images, context.metadatas)
        feat2 = self._run_encoder(pass2_images, context.metadatas)
        return torch.cat([feat1, feat2], dim=-1)
