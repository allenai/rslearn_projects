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

from .sliding_window import SlidingWindowEvalMixin
from .timestamp_encoding import timestamps_to_days

INPUT_KEY = "sentinel2_l2a"
NUM_PASS1 = 10
DEBUG_PRINT_BINARY_CHANGE_ERRORS = False
DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_POSITIVE = 0.9
DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_NEGATIVE = 0.5

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


class DualPassChangeModel(SlidingWindowEvalMixin, nn.Module):
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
        num_classes_pre_change: int = 7,
        num_classes_post_change: int = 12,
        num_classes_same_change: int = 6,
        num_timestamps: int = 20,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 1.0,
        num_passes: int = 2,
        num_pass1: int = NUM_PASS1,
        eval_crop_size: int | None = None,
        eval_overlap: int = 0,
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
            num_passes: number of encoder passes (1 or 2). With 1, all images are
                encoded in a single pass and the decoder input is ``embedding_dim``.
                With 2, the stack is split at ``num_pass1`` into a historical and a
                recent pass whose features are concatenated (``2 * embedding_dim``).
            num_pass1: number of leading images in pass1 (only used when
                ``num_passes == 2``); the remaining images form pass2.
            eval_crop_size: if set, in eval mode the model tiles inputs larger
                than this size into overlapping ``eval_crop_size`` crops, runs
                each through the normal forward, and stitches the predictions
                back together (see ``SlidingWindowEvalMixin``). Used to evaluate
                every config over an identical window for comparable metrics.
            eval_overlap: pixels shared between adjacent sliding-window tiles.
        """
        super().__init__()
        if num_passes not in (1, 2):
            raise ValueError(f"num_passes must be 1 or 2, got {num_passes}")
        self.eval_crop_size = eval_crop_size
        self.eval_overlap = eval_overlap
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timestamps = num_timestamps
        self.binary_loss_weight = binary_loss_weight
        self.num_passes = num_passes
        self.num_pass1 = num_pass1

        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        concat_dim = embedding_dim * num_passes

        self.decoder_binary = _make_decoder(
            concat_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(concat_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(concat_dim, decoder_stages, num_classes_dst)
        self.decoder_pre_change = _make_decoder(
            concat_dim, decoder_stages, num_classes_pre_change
        )
        self.decoder_post_change = _make_decoder(
            concat_dim, decoder_stages, num_classes_post_change
        )
        self.decoder_same_change = _make_decoder(
            concat_dim, decoder_stages, num_classes_same_change
        )
        self.decoder_timestamps = _make_decoder(
            concat_dim, decoder_stages, num_timestamps
        )

        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst
        self.num_classes_pre_change = num_classes_pre_change
        self.num_classes_post_change = num_classes_post_change
        self.num_classes_same_change = num_classes_same_change

    def _run_encoder(
        self, raster_images: list[RasterImage], metadatas: list[SampleMetadata]
    ) -> torch.Tensor:
        """Run OlmoEarth on a list of RasterImages, return BCHW feature tensor."""
        inputs = [{"sentinel2_l2a": img} for img in raster_images]
        sub_context = ModelContext(inputs=inputs, metadatas=metadatas)
        feature_maps = self.encoder(sub_context)
        return feature_maps.feature_maps[0]

    def _combine_features(
        self, feat1: torch.Tensor, feat2: torch.Tensor
    ) -> torch.Tensor:
        """Combine the two encoder-pass features into the decoder input.

        Default concatenates along the channel dim, giving ``2 * embedding_dim``
        channels. Subclasses may override to change the decoder input (e.g. add a
        difference channel group or apply cross-pass attention); the decoders must
        be built with a matching input dimension.
        """
        return torch.cat([feat1, feat2], dim=1)

    def _forward_core(
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
        if self.num_passes == 1:
            full_images: list[RasterImage] = [inp[INPUT_KEY] for inp in context.inputs]
            concat = self._run_encoder(full_images, context.metadatas)
        else:
            pass1_images: list[RasterImage] = []
            pass2_images: list[RasterImage] = []
            for inp in context.inputs:
                combined: RasterImage = inp[INPUT_KEY]
                p1_img = combined.image[:, : self.num_pass1, :, :]
                p2_img = combined.image[:, self.num_pass1 :, :, :]
                p1_ts = (
                    combined.timestamps[: self.num_pass1]
                    if combined.timestamps
                    else None
                )
                p2_ts = (
                    combined.timestamps[self.num_pass1 :]
                    if combined.timestamps
                    else None
                )
                pass1_images.append(RasterImage(image=p1_img, timestamps=p1_ts))
                pass2_images.append(RasterImage(image=p2_img, timestamps=p2_ts))

            feat1 = self._run_encoder(pass1_images, context.metadatas)
            feat2 = self._run_encoder(pass2_images, context.metadatas)

            # Combine the two passes into the decoder input. Default concatenates
            # along the channel dim: (B, 2*embedding_dim, H/4, W/4). Subclasses may
            # override.
            concat = self._combine_features(feat1, feat2)

        # Per-task decoders: (B, 1536, H/4, W/4) -> (B, C, H, W)
        logits_binary = self.decoder_binary(concat)
        logits_src = self.decoder_src(concat)
        logits_dst = self.decoder_dst(concat)
        change_logits = {
            "pre_change": self.decoder_pre_change(concat),
            "post_change": self.decoder_post_change(concat),
            "same_change": self.decoder_same_change(concat),
        }
        logits_ts = self.decoder_timestamps(concat)

        losses: dict[str, torch.Tensor] = {}
        outputs: list[dict[str, Any]] = [{} for _ in context.inputs]

        if targets is not None:
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            for name, logits_cat in change_logits.items():
                if name in targets[0]:
                    losses[f"{name}_cls"] = self._seg_loss(logits_cat, targets, name)
            losses["timestamps_bce"] = self._timestamp_loss(logits_ts, targets)

        for i in range(len(context.inputs)):
            ts_image = context.inputs[i].get(INPUT_KEY)
            timestep_days = (
                timestamps_to_days(ts_image.timestamps)
                if isinstance(ts_image, RasterImage) and ts_image.timestamps is not None
                else None
            )
            outputs[i] = {
                "binary": F.softmax(logits_binary[i], dim=0),
                "src": F.softmax(logits_src[i], dim=0),
                "dst": F.softmax(logits_dst[i], dim=0),
                **{
                    name: F.softmax(change_logits[name][i], dim=0)
                    for name in change_logits
                },
                "timestamps": torch.sigmoid(logits_ts[i]),
                "timestep_days": timestep_days,
            }

        if (
            DEBUG_PRINT_BINARY_CHANGE_ERRORS
            and targets is not None
            and not self.training
        ):
            self._debug_print_binary_change_errors(context, targets, outputs)

        return ModelOutput(outputs=outputs, loss_dict=losses)

    def _debug_print_binary_change_errors(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> None:
        """Print windows with binary change mistakes at the debug threshold."""
        for output, target, metadata in zip(
            outputs, targets, context.metadatas, strict=True
        ):
            labels = target["binary"]["classes"].get_hw_tensor().long()
            valid = target["binary"]["valid"].get_hw_tensor() > 0
            binary_probs = output["binary"]
            change_prob = binary_probs[2]

            positive = valid & (labels == 2)
            negative = valid & (labels == 1)
            false_negative = positive & ~(
                change_prob >= DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_NEGATIVE
            )
            false_positive = negative & (
                change_prob >= DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_POSITIVE
            )

            num_false_negative = int(false_negative.sum().item())
            num_false_positive = int(false_positive.sum().item())
            if num_false_negative == 0 and num_false_positive == 0:
                continue

            num_positive = int(positive.sum().item())
            num_negative = int(negative.sum().item())
            binary_probs_cpu = binary_probs.detach().cpu()
            crop_x0, crop_y0, _, _ = metadata.crop_bounds

            def sample_error_pixel(mask: torch.Tensor) -> dict[str, Any] | None:
                error_pixels = mask.nonzero(as_tuple=False)
                if error_pixels.shape[0] == 0:
                    return None
                sample_idx = int(
                    torch.randint(
                        error_pixels.shape[0],
                        (1,),
                        device=error_pixels.device,
                    ).item()
                )
                row, col = error_pixels[sample_idx].detach().cpu().tolist()
                row = int(row)
                col = int(col)
                return {
                    "row": row,
                    "col": col,
                    "window_x": crop_x0 + col,
                    "window_y": crop_y0 + row,
                    "change_prob": round(float(binary_probs_cpu[2, row, col]), 6),
                    "binary_probs": [
                        round(float(prob), 6)
                        for prob in binary_probs_cpu[:, row, col].tolist()
                    ],
                }

            sample_false_negative = sample_error_pixel(false_negative)
            sample_false_positive = sample_error_pixel(false_positive)

            print(
                "[LCC binary error debug] "
                f"window={metadata.window_group}/{metadata.window_name} "
                f"crop={metadata.crop_idx + 1}/{metadata.num_crops_in_window} "
                f"crop_bounds={metadata.crop_bounds} "
                f"positive_pixels={num_positive} "
                f"negative_pixels={num_negative} "
                f"false_negative_pixels={num_false_negative} "
                f"false_positive_pixels={num_false_positive} "
                f"threshold_fp={DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_POSITIVE} "
                f"threshold_fp={DEBUG_BINARY_CHANGE_THRESHOLD_FALSE_NEGATIVE} "
                f"sample_false_negative={sample_false_negative} "
                f"sample_false_positive={sample_false_positive}",
                flush=True,
            )

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
