"""DualPassChangeModel: two OlmoEarth forward passes + per-task conv decoders.

Architecture:
- Input: sentinel2_l2a with 20 timesteps (produced by FrequentOptionSampler)
- Split into pass1 (first 10) and pass2 (last 10)
- Shared OlmoEarth encoder processes each pass separately
- Features concatenated along channel dim → 1536ch at 1/4 res
- Per-task convolutional decoder: 1536→768 1x1, 768→512 3x3, upsample 2x,
  512→256 3x3, upsample 2x, 256→128 3x3, 128→logits 1x1 (full res)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)

INPUT_KEY = "sentinel2_l2a"
NUM_PASS1 = 10


def _make_decoder(concat_dim: int, num_classes: int) -> nn.Sequential:
    """Build a per-task convolutional decoder.

    Takes (B, concat_dim, H/4, W/4) and produces (B, num_classes, H, W).
    """
    return nn.Sequential(
        nn.Conv2d(concat_dim, 768, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(768, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, num_classes, kernel_size=1),
    )


class DualPassChangeModel(nn.Module):
    """Two-pass encoder with per-task convolutional decoders.

    The model expects ``sentinel2_l2a`` with 20 timesteps. It splits into
    pass1 (first 10) and pass2 (last 10), runs the shared OlmoEarth encoder
    on each, concatenates features, then runs per-task conv decoders that
    upsample from 1/4 to full resolution.
    """

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timestamps: int = 20,
        embedding_dim: int = 768,
    ):
        """Initialize the LCC multi-task model."""
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timestamps = num_timestamps

        concat_dim = embedding_dim * 2

        self.decoder_binary = _make_decoder(concat_dim, num_classes_binary)
        self.decoder_src = _make_decoder(concat_dim, num_classes_src)
        self.decoder_dst = _make_decoder(concat_dim, num_classes_dst)
        self.decoder_timestamps = _make_decoder(concat_dim, num_timestamps)

        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst

    def _run_encoder(
        self, raster_images: list[RasterImage], metadatas: list[SampleMetadata]
    ) -> torch.Tensor:
        """Run OlmoEarth on a list of RasterImages, return BCHW feature tensor."""
        inputs = [{"sentinel2_l2a": img} for img in raster_images]
        sub_context = ModelContext(inputs=inputs, metadatas=metadatas)
        feature_maps = self.encoder(sub_context)
        return feature_maps.feature_maps[0]

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Forward pass with two encoder calls and per-task decoders.

        Args:
            context: ModelContext with ``sentinel2_l2a`` RasterImage (20 timesteps).
            targets: optional target dicts with "binary", "src", "dst", "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses.
        """
        pass1_images: list[RasterImage] = []
        pass2_images: list[RasterImage] = []
        for inp in context.inputs:
            combined: RasterImage = inp[INPUT_KEY]
            p1_img = combined.image[:, :NUM_PASS1, :, :]
            p2_img = combined.image[:, NUM_PASS1:, :, :]
            p1_ts = combined.timestamps[:NUM_PASS1] if combined.timestamps else None
            p2_ts = combined.timestamps[NUM_PASS1:] if combined.timestamps else None
            pass1_images.append(RasterImage(image=p1_img, timestamps=p1_ts))
            pass2_images.append(RasterImage(image=p2_img, timestamps=p2_ts))

        feat1 = self._run_encoder(pass1_images, context.metadatas)
        feat2 = self._run_encoder(pass2_images, context.metadatas)

        # Concatenate along channel dim: (B, 1536, H/4, W/4)
        concat = torch.cat([feat1, feat2], dim=1)

        # Per-task decoders: (B, 1536, H/4, W/4) -> (B, C, H, W)
        logits_binary = self.decoder_binary(concat)
        logits_src = self.decoder_src(concat)
        logits_dst = self.decoder_dst(concat)
        logits_ts = self.decoder_timestamps(concat)

        losses: dict[str, torch.Tensor] = {}
        outputs: list[dict[str, Any]] = [{} for _ in context.inputs]

        if targets is not None:
            losses["binary_cls"] = self._balanced_binary_loss(logits_binary, targets)
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            losses["timestamps_bce"] = self._timestamp_loss(logits_ts, targets)

        for i in range(len(context.inputs)):
            outputs[i] = {
                "binary": F.softmax(logits_binary[i], dim=0),
                "src": F.softmax(logits_src[i], dim=0),
                "dst": F.softmax(logits_dst[i], dim=0),
                "timestamps": torch.sigmoid(logits_ts[i]),
            }

        return ModelOutput(outputs=outputs, loss_dict=losses)

    def _seg_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        task_name: str,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss for a segmentation task."""
        labels = torch.stack(
            [t[task_name]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t[task_name]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")
        return (loss * valid).sum() / valid.sum()

    def _balanced_binary_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Balanced binary loss: equal weight to positive and negative points.

        When both change and no-change points are present in a batch, the loss
        from each group contributes 50/50 regardless of count imbalance.
        """
        labels = torch.stack(
            [t["binary"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["binary"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")

        change_mask = valid & (labels == 2)
        nochange_mask = valid & (labels == 1)

        if change_mask.any() and nochange_mask.any():
            pos_loss = (loss * change_mask).sum() / change_mask.sum()
            neg_loss = (loss * nochange_mask).sum() / nochange_mask.sum()
            return (pos_loss + neg_loss) / 2
        else:
            return (loss * valid).sum() / valid.sum()

    def _timestamp_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Compute masked BCE loss for timestamp binary classifications.

        targets["timestamps"]["classes"] is a RasterImage with shape (num_ts, 1, H, W)
        targets["timestamps"]["valid"] is a RasterImage with shape (1, 1, H, W)
        """
        # classes: (B, num_ts, H, W) - multi-channel binary targets
        classes = torch.stack(
            [t["timestamps"]["classes"].image[:, 0, :, :] for t in targets], dim=0
        ).float()
        valid = torch.stack(
            [t["timestamps"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Expand valid mask to match timestamp channels
        valid_expanded = valid.unsqueeze(1).expand_as(logits)

        loss = F.binary_cross_entropy_with_logits(logits, classes, reduction="none")
        return (loss * valid_expanded).sum() / valid_expanded.sum()
