"""DualPassChangeModel: two OlmoEarth forward passes + per-task conv decoders.

Architecture:
- Input: sentinel2_l2a with 20 timesteps (produced by FrequentOptionSampler)
- Split into pass1 (first 10) and pass2 (last 10)
- Shared OlmoEarth encoder processes each pass separately
- Features concatenated along channel dim → (2*embedding_dim)ch at 1/patch_size res
- Per-task convolutional decoder defined by configurable stages: each stage is a
  list of (out_channels, kernel_size) convs, with a 2x upsample inserted before
  every stage after the first and a 1x1 head producing the per-task logits.
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

# A stage is a list of (out_channels, kernel_size) conv specs.
StageSpec = list[tuple[int, int]]


def _make_decoder(
    in_dim: int, stages: list[StageSpec], num_classes: int
) -> nn.Sequential:
    """Build a per-task convolutional decoder from explicit stage specs.

    Each stage is a list of (out_channels, kernel_size) convs (each followed by
    ReLU). A 2x bilinear upsample precedes every stage after the first, so the
    output is at 2^(len(stages)-1) times the input resolution. A final 1x1 conv
    produces num_classes channels.

    Takes (B, in_dim, H, W) and produces (B, num_classes, H*2^(n-1), W*2^(n-1))
    where n = len(stages).
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
    layers.append(nn.Conv2d(prev, num_classes, kernel_size=1))
    return nn.Sequential(*layers)


class DualPassChangeModel(nn.Module):
    """Two-pass encoder with per-task convolutional decoders.

    The model expects ``sentinel2_l2a`` with 20 timesteps. It splits into
    pass1 (first 10) and pass2 (last 10), runs the shared OlmoEarth encoder
    on each, concatenates features, then runs per-task conv decoders that
    upsample the 1/patch_size features back to full resolution. The decoder
    shape (channels, conv layers, and number of upsamples) is set by
    ``decoder_stages`` and must be configured to match the encoder: the number
    of upsamples (``len(decoder_stages) - 1``) should equal ``log2(patch_size)``
    and ``embedding_dim`` should match the encoder's embedding size.
    """

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timestamps: int = 20,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 1.0,
    ):
        """Initialize the LCC multi-task model.

        Args:
            encoder: the shared OlmoEarth encoder.
            num_classes_binary: number of classes for the binary change task.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_timestamps: number of timestamp outputs.
            embedding_dim: per-pass encoder embedding size. The decoder input is
                ``2 * embedding_dim`` (the two passes concatenated). Must match the
                encoder (e.g. 768 for BASE, 192 for TINY, 128 for NANO).
            decoder_stages: per-task decoder definition (see ``_make_decoder``). The
                number of 2x upsamples is ``len(decoder_stages) - 1`` and must equal
                ``log2(encoder.patch_size)`` so outputs are full resolution. Required;
                must be specified in the model config.
            binary_loss_weight: multiplier applied to the binary change loss before
                it is summed with the other task losses.
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timestamps = num_timestamps
        self.binary_loss_weight = binary_loss_weight

        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        concat_dim = embedding_dim * 2

        self.decoder_binary = _make_decoder(
            concat_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(concat_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(concat_dim, decoder_stages, num_classes_dst)
        self.decoder_timestamps = _make_decoder(
            concat_dim, decoder_stages, num_timestamps
        )

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
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
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
        """Balanced binary loss with per-sample balancing.

        For each sample, the loss is the mean over its change points plus the
        mean over its no-change points (each group divided by its own point
        count). If a sample only has one of the two groups, its loss is just the
        mean over that group. Every sample with any valid points contributes
        equally to the final loss (mean over such samples), regardless of how
        many points it has.
        """
        labels = torch.stack(
            [t["binary"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["binary"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)

        change_mask = (valid & (labels == 2)).flatten(1).float()  # (B, H*W)
        nochange_mask = (valid & (labels == 1)).flatten(1).float()
        loss_flat = loss.flatten(1)  # (B, H*W)

        pos_count = change_mask.sum(dim=1)  # (B,)
        neg_count = nochange_mask.sum(dim=1)
        has_pos = pos_count > 0
        has_neg = neg_count > 0

        pos_mean = (loss_flat * change_mask).sum(dim=1) / pos_count.clamp(min=1)
        neg_mean = (loss_flat * nochange_mask).sum(dim=1) / neg_count.clamp(min=1)

        # Per-sample loss: sum of whichever group means are present. Where a
        # group is absent its mean is zeroed out so it does not contribute.
        sample_loss = pos_mean * has_pos + neg_mean * has_neg  # (B,)
        has_any = has_pos | has_neg
        return sample_loss[has_any].mean()

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
