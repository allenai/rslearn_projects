"""FeatureDifferenceChangeModel: add an explicit pass difference to the decoder input.

Variant of ``DualPassChangeModel`` whose decoder input is
``cat([feat1, feat2, feat2 - feat1])`` (3 * embedding_dim channels) instead of the
default ``cat([feat1, feat2])``. The explicit ``feat2 - feat1`` difference channels
give the per-task conv decoders a direct change-detection signal.
"""

from __future__ import annotations

from typing import Any

import torch

from .model import DualPassChangeModel, StageSpec, _make_decoder


class FeatureDifferenceChangeModel(DualPassChangeModel):
    """DualPass change model with an added feature-difference channel group."""

    def __init__(
        self,
        *args: Any,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timestamps: int = 20,
        **kwargs: Any,
    ):
        """Initialize and rebuild the decoders for the 3-group concat input.

        Args mirror ``DualPassChangeModel``; the only difference is the decoder
        input dimension (``3 * embedding_dim`` rather than ``2 * embedding_dim``).
        """
        super().__init__(
            *args,
            embedding_dim=embedding_dim,
            decoder_stages=decoder_stages,
            num_classes_binary=num_classes_binary,
            num_classes_src=num_classes_src,
            num_classes_dst=num_classes_dst,
            num_timestamps=num_timestamps,
            **kwargs,
        )

        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        # Decoder input is [feat1, feat2, feat2 - feat1] -> 3 * embedding_dim.
        concat_dim = embedding_dim * 3
        self.decoder_binary = _make_decoder(
            concat_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(concat_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(concat_dim, decoder_stages, num_classes_dst)
        self.decoder_timestamps = _make_decoder(
            concat_dim, decoder_stages, num_timestamps
        )

    def _combine_features(
        self, feat1: torch.Tensor, feat2: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate the two passes plus their difference along channels."""
        return torch.cat([feat1, feat2, feat2 - feat1], dim=1)
