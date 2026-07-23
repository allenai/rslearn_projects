"""Contrast-based decoders for the single-pass LCC change model.

The baseline ``SinglePassChangeModel`` mean-pools the per-timestep OlmoEarth tokens
over time before the binary/src/dst decoders, so the change head can key on static
appearance ("looks like a place where bare->urban happens") rather than an actual
temporal transition. The 2026-07-10 FP investigation showed exactly that failure
mode (confident bare->urban on unchanged desert, FP transition profile mirroring
the training positive-pair prior, ~7% of high-score pixels predicting src==dst).

``FableChangeModel`` keeps the encoder, the src/dst/start/end machinery, and the
losses of the baseline, but replaces the binary pathway so that change evidence is
a function of temporal *contrast* only:

- ``binary_mode="centered"``: binary evidence is decoded from temporally-centered
  residual statistics (std over time and max |residual|). The temporal mean -- the
  static appearance -- is only visible to the src/dst heads. A pixel whose feature
  never varies over time produces constant (appearance-independent) evidence.
- ``binary_mode="breakpoint"``: a learned changepoint scan. For every split t the
  before-mean A_t and after-mean B_t are compared by a shared scorer; evidence is
  pooled over splits. src is decoded from the split-attention-weighted before
  feature and dst from the weighted after feature, so "change" can only be
  expressed as before-vs-after dissimilarity at some breakpoint and src/dst look
  at the correct sides of it.
- ``binary_mode="hybrid"``: concatenation of both evidence feature sets.
- ``binary_mode="mean"``: the baseline pathway (control; equivalent to
  ``SinglePassChangeModel``) but usable with the extras below.

Orthogonal options:

- ``season_embed``: adds a small month-of-year (sin/cos) embedding to each
  timestep token before temporal aggregation, so seasonal differences between
  mosaics are explainable by covariates instead of being read as change.
- ``FableSampler`` (transforms below): train-time quarterly-dropout augmentation,
  randomly removing quarterly mosaics so the model learns invariance to mosaic
  composition (the FP investigation found predictions flipping 0.07 <-> 0.93 from
  composition alone).
- ``NegativeWindowMaxScore``: a val metric tracking the mean over negative-only
  crops of the max change probability over ALL pixels (labeled or not) -- a proxy
  for in-the-wild hallucination that per-labeled-pixel AUROC cannot see.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage
from torchmetrics import Metric
from typing_extensions import override

from .model_singlepass import (
    INPUT_KEY,
    QUARTERLY_KEY,
    SinglePassChangeModel,
    SinglePassSampler,
    StageSpec,
    _make_decoder,
)
from .timestamp_encoding import timestamps_to_days

# Fixed logit for the (never-supervised) nodata channel of the binary head.
NODATA_LOGIT = -10.0


class FableChangeModel(SinglePassChangeModel):
    """Single-pass LCC model with contrast-based binary decoders.

    Inherits the encoder handling, src/dst conv decoders, start/end timestamp
    heads, and all loss functions from ``SinglePassChangeModel``; only the binary
    evidence pathway (and, in breakpoint mode, the src/dst decoder inputs) differ.
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
        num_timesteps: int = 20,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 2.0,
        binary_mode: str = "breakpoint",
        contrast: str = "signed_abs",
        evidence_pool: str = "max",
        scorer_hidden: int = 256,
        evidence_stages: list[StageSpec] | None = None,
        season_embed: bool = False,
    ):
        """Initialize the model.

        Args:
            encoder: the OlmoEarth encoder (token_pooling=False).
            num_classes_binary: number of binary classes (3: nodata/no_change/change).
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_timesteps: expected number of input timesteps.
            embedding_dim: per-token encoder embedding size.
            decoder_stages: conv decoder stages for the src/dst heads (as baseline).
            binary_loss_weight: multiplier for the binary change loss.
            binary_mode: "mean", "centered", "breakpoint", or "hybrid".
            contrast: breakpoint contrast features, "signed_abs" (cat of B-A and
                |B-A|) or "abs" (|B-A| only, metric-style symmetric comparison).
            evidence_pool: how per-split hidden features are pooled over splits,
                "max" or "lse" (logsumexp).
            scorer_hidden: hidden channels of the shared split/residual scorer.
            evidence_stages: conv decoder stages for the 1-channel evidence head.
                Must contain the same number of stages (upsamples) as
                decoder_stages. Defaults to [[(256,3)],[(128,3)],[(64,3)]].
            season_embed: add month-of-year embeddings to tokens before temporal
                aggregation.
        """
        super().__init__(
            encoder=encoder,
            num_classes_binary=num_classes_binary,
            num_classes_src=num_classes_src,
            num_classes_dst=num_classes_dst,
            num_classes_pre_change=num_classes_pre_change,
            num_classes_post_change=num_classes_post_change,
            num_classes_same_change=num_classes_same_change,
            num_timesteps=num_timesteps,
            embedding_dim=embedding_dim,
            decoder_stages=decoder_stages,
            binary_loss_weight=binary_loss_weight,
        )
        if binary_mode not in ("mean", "centered", "breakpoint", "hybrid"):
            raise ValueError(f"unknown binary_mode {binary_mode!r}")
        if contrast not in ("signed_abs", "abs"):
            raise ValueError(f"unknown contrast {contrast!r}")
        if evidence_pool not in ("max", "lse"):
            raise ValueError(f"unknown evidence_pool {evidence_pool!r}")
        self.binary_mode = binary_mode
        self.contrast = contrast
        self.evidence_pool = evidence_pool
        self.season_embed = season_embed

        if evidence_stages is None:
            evidence_stages = [[(256, 3)], [(128, 3)], [(64, 3)]]

        if binary_mode == "mean":
            # Keep the baseline decoder_binary from the parent.
            pass
        else:
            # The parent's mean-input binary decoder is unused; drop it so its
            # parameters are not trained/checkpointed.
            del self.decoder_binary

            contrast_dim = (
                2 * embedding_dim if contrast == "signed_abs" else embedding_dim
            )
            if binary_mode in ("breakpoint", "hybrid"):
                self.split_proj = nn.Sequential(
                    nn.Conv2d(contrast_dim, scorer_hidden, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(scorer_hidden, scorer_hidden, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                # Scalar per-split score for the split-attention over breakpoints
                # (used to pick the before/after aggregates for src/dst).
                self.split_score = nn.Conv2d(scorer_hidden, 1, kernel_size=1)
            if binary_mode in ("centered", "hybrid"):
                # Residual statistics are std over T and max |residual| (2C).
                self.centered_proj = nn.Sequential(
                    nn.Conv2d(2 * embedding_dim, scorer_hidden, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(scorer_hidden, scorer_hidden, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
            evidence_in = scorer_hidden * (2 if binary_mode == "hybrid" else 1)
            self.evidence_decoder = _make_decoder(evidence_in, evidence_stages, 1)

        if season_embed:
            self.month_mlp = nn.Sequential(
                nn.Linear(2, embedding_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim // 4, embedding_dim),
            )

    def _month_embedding(
        self, context: ModelContext, T: int, ref: torch.Tensor
    ) -> torch.Tensor:
        """Build (B, C, 1, 1, T) month-of-year embeddings from input timestamps."""
        rows = []
        for input_dict in context.inputs:
            image = input_dict.get(INPUT_KEY)
            feats = torch.zeros(T, 2)
            if isinstance(image, RasterImage) and image.timestamps is not None:
                for t, (t0, t1) in enumerate(image.timestamps[:T]):
                    mid = t0 + (t1 - t0) / 2
                    frac = (mid.month - 1 + (mid.day - 1) / 31.0) / 12.0
                    feats[t, 0] = math.sin(2 * math.pi * frac)
                    feats[t, 1] = math.cos(2 * math.pi * frac)
            rows.append(feats)
        months = torch.stack(rows, dim=0).to(device=ref.device, dtype=ref.dtype)
        emb = self.month_mlp(months)  # (B, T, C)
        return emb.permute(0, 2, 1)[:, :, None, None, :]  # (B, C, 1, 1, T)

    def _breakpoint_features(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Changepoint scan over the T timesteps.

        Args:
            feature: (B, C, H, W, T) per-timestep tokens.

        Returns:
            (evidence_feat, src_feat, dst_feat): pooled per-split hidden features
            (B, hidden, H, W) and the split-attention-weighted before/after
            aggregates (B, C, H, W).
        """
        B, C, H, W, T = feature.shape
        S = T - 1
        cums = feature.cumsum(dim=-1)  # (B, C, H, W, T)
        total = cums[..., -1:]
        counts = torch.arange(1, T, device=feature.device, dtype=feature.dtype)
        before = cums[..., :-1] / counts  # (B, C, H, W, S) mean of images [0, t]
        after = (total - cums[..., :-1]) / (T - counts)  # mean of images (t, T)

        diff = after - before
        if self.contrast == "signed_abs":
            con = torch.cat([diff, diff.abs()], dim=1)  # (B, 2C, H, W, S)
        else:
            con = diff.abs()

        # Shared scorer over all splits: fold S into the batch dimension.
        con = con.permute(0, 4, 1, 2, 3).reshape(B * S, con.shape[1], H, W)
        hidden = self.split_proj(con)  # (B*S, hidden, H, W)
        scores = self.split_score(hidden)  # (B*S, 1, H, W)
        hidden = hidden.reshape(B, S, -1, H, W)
        scores = scores.reshape(B, S, H, W)

        if self.evidence_pool == "max":
            evidence_feat = hidden.max(dim=1).values
        else:
            evidence_feat = torch.logsumexp(hidden, dim=1)

        w = F.softmax(scores, dim=1)  # (B, S, H, W)
        w = w.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, S)
        src_feat = (before * w).sum(dim=-1)
        dst_feat = (after * w).sum(dim=-1)
        return evidence_feat, src_feat, dst_feat

    def _centered_features(self, feature: torch.Tensor) -> torch.Tensor:
        """Temporally-centered residual statistics -> (B, hidden, H, W)."""
        residual = feature - feature.mean(dim=-1, keepdim=True)
        stats = torch.cat(
            [residual.std(dim=-1), residual.abs().max(dim=-1).values], dim=1
        )
        return self.centered_proj(stats)

    @override
    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Forward pass with the contrast-based binary pathway."""
        token_feature_maps = self.encoder(context)
        feature = token_feature_maps.feature_maps[0]  # (B, C, H, W, T)

        if self.season_embed:
            feature = feature + self._month_embedding(
                context, feature.shape[-1], feature
            )

        mean_feat = feature.mean(dim=-1)  # (B, C, H, W)

        src_in = mean_feat
        dst_in = mean_feat
        if self.binary_mode == "mean":
            logits_binary = self.decoder_binary(mean_feat)
        else:
            parts = []
            if self.binary_mode in ("breakpoint", "hybrid"):
                bp_feat, src_in, dst_in = self._breakpoint_features(feature)
                parts.append(bp_feat)
            if self.binary_mode in ("centered", "hybrid"):
                parts.append(self._centered_features(feature))
            evidence = self.evidence_decoder(torch.cat(parts, dim=1))  # (B,1,H',W')
            logits_binary = torch.cat(
                [
                    torch.full_like(evidence, NODATA_LOGIT),
                    torch.zeros_like(evidence),
                    evidence,
                ],
                dim=1,
            )

        logits_src = self.decoder_src(src_in)
        logits_dst = self.decoder_dst(dst_in)
        change_logits = self._change_category_logits(mean_feat)

        # Per-token start/end timestamp heads (identical to the baseline).
        xt = feature.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
        start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)
        end_logits = self.end_head(xt).squeeze(-1).permute(0, 3, 1, 2)
        scale = self.encoder.patch_size
        start_logits = F.interpolate(
            start_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )
        end_logits = F.interpolate(
            end_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )

        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            self._add_change_category_losses(change_logits, targets, losses)
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            ts_image = context.inputs[i].get(INPUT_KEY)
            timestep_days = (
                timestamps_to_days(ts_image.timestamps)
                if isinstance(ts_image, RasterImage) and ts_image.timestamps is not None
                else None
            )
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    **{
                        name: F.softmax(change_logits[name][i], dim=0)
                        for name in change_logits
                    },
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                    "timestep_days": timestep_days,
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)


class FableSampler(SinglePassSampler):
    """SinglePassSampler with train-time quarterly-dropout augmentation.

    Before the standard sampling logic, each available quarterly image is
    independently dropped with probability ``quarterly_dropout`` (keeping at
    least ``min_keep``), so across epochs the model sees different mosaic
    compositions of the same window. This targets the composition instability
    found in the FP investigation (same window flipping 0.07 <-> 0.93 based on
    which mosaics enter the stack). No-op when deterministic=True.
    """

    def __init__(
        self,
        deterministic: bool = False,
        quarterly_dropout: float = 0.0,
        min_keep: int = 8,
    ) -> None:
        """Initialize the sampler.

        Args:
            deterministic: see SinglePassSampler; also disables the dropout.
            quarterly_dropout: probability of dropping each quarterly image.
            min_keep: minimum number of quarterly images to keep.
        """
        super().__init__(deterministic=deterministic)
        self.quarterly_dropout = quarterly_dropout
        self.min_keep = min_keep

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Randomly drop quarterly images, then run the standard sampler."""
        if (
            not self.deterministic
            and self.quarterly_dropout > 0
            and QUARTERLY_KEY in input_dict
        ):
            quarterly: RasterImage = input_dict[QUARTERLY_KEY]
            if quarterly.timestamps is not None:
                T = quarterly.image.shape[1]
                keep = [
                    i for i in range(T) if random.random() >= self.quarterly_dropout
                ]
                if len(keep) < min(self.min_keep, T):
                    keep = sorted(random.sample(range(T), min(self.min_keep, T)))
                if len(keep) < T:
                    input_dict[QUARTERLY_KEY] = RasterImage(
                        image=quarterly.image[:, keep, :, :],
                        timestamps=[quarterly.timestamps[i] for i in keep],
                    )
        return super().forward(input_dict, target_dict)


class NegativeWindowMaxScore(Metric):
    """Mean over negative-only samples of the max change probability anywhere.

    For each sample whose labeled pixels contain no change point, this records
    the maximum predicted change probability over ALL pixels of the crop
    (labeled or not). Per-labeled-pixel metrics like the balanced AUROC cannot
    see hallucinations on the unlabeled 99.99% of each window; this metric is a
    direct proxy for the in-the-wild false-positive density. Lower is better.
    """

    def __init__(self) -> None:
        """Initialize accumulators."""
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @override
    def update(self, preds: list[torch.Tensor], targets: list[dict[str, Any]]) -> None:
        for pred, target in zip(preds, targets):
            label = target["classes"].get_hw_tensor().long()
            valid = target["valid"].get_hw_tensor() > 0
            has_pos = bool((valid & (label == 2)).any())
            has_neg = bool((valid & (label == 1)).any())
            if has_pos or not has_neg:
                continue
            self.total += pred[2].max().double()
            self.count += 1

    @override
    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return (self.total / self.count).float()
