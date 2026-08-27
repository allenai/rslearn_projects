"""RegularizedDecoderChangeModel: add dropout to the per-task conv decoders.

Variant of ``DualPassChangeModel`` that rebuilds the per-task decoders with a
``nn.Dropout2d`` after each conv stage, to curb overfitting (the base model reaches
very low train loss). Features and decoder input dim are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .model import DualPassChangeModel, StageSpec


def _make_decoder_reg(
    in_dim: int, stages: list[StageSpec], num_classes: int, dropout: float
) -> nn.Sequential:
    """Like ``model._make_decoder`` but with ``Dropout2d(dropout)`` per stage.

    Each stage is a list of (out_channels, kernel_size) convs (each followed by
    ReLU); a 2x bilinear upsample precedes every stage after the first, and a
    spatial dropout is applied at the end of each stage. A final 1x1 conv produces
    ``num_classes`` channels.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for i, stage in enumerate(stages):
        if i > 0:
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        for out_ch, k in stage:
            layers.append(nn.Conv2d(prev, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU(inplace=True))
            prev = out_ch
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
    layers.append(nn.Conv2d(prev, num_classes, kernel_size=1))
    return nn.Sequential(*layers)


class RegularizedDecoderChangeModel(DualPassChangeModel):
    """DualPass change model with dropout-regularized conv decoders."""

    def __init__(
        self,
        *args: Any,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timestamps: int = 20,
        dropout: float = 0.2,
        **kwargs: Any,
    ):
        """Initialize and rebuild the decoders with per-stage dropout.

        Args mirror ``DualPassChangeModel`` plus ``dropout`` (the Dropout2d
        probability applied after each decoder stage).
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

        concat_dim = embedding_dim * 2
        self.decoder_binary = _make_decoder_reg(
            concat_dim, decoder_stages, num_classes_binary, dropout
        )
        self.decoder_src = _make_decoder_reg(
            concat_dim, decoder_stages, num_classes_src, dropout
        )
        self.decoder_dst = _make_decoder_reg(
            concat_dim, decoder_stages, num_classes_dst, dropout
        )
        self.decoder_timestamps = _make_decoder_reg(
            concat_dim, decoder_stages, num_timestamps, dropout
        )
