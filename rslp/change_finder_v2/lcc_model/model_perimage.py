"""Per-image OlmoEarth LCC change model with a temporal decoder.

Unlike ``SinglePassChangeModel`` (which runs one OlmoEarth pass over all 20
timesteps so the encoder attends across time), this model applies OlmoEarth to
each image independently -- there is NO cross-time attention inside the encoder.
All temporal reasoning is pushed into a small temporal stack on top:

1. A learned temporal positional embedding plus ``num_temporal_selfattn_layers``
   (default 1) temporal self-attention layers over the T per-pixel tokens. This
   is required -- without it the per-image tokens carry no cross-time context and
   the start/end timestamp heads have nothing to work with.
2. A temporal decoder that pools over T for the segmentation heads: either a
   learned-query cross-attention pool (``decoder_type="attn"``) or a GRU
   (``decoder_type="rnn"``). The pooled feature is upsampled to full resolution
   by the same conv decoders used by the single-pass model.
3. Per-token linear start/end heads over T (same as the single-pass model),
   upsampled to full resolution.

Everything else (sampler, task, metrics, and the five losses) is reused from
``model_singlepass`` -- this module only introduces the model class.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

from .model_singlepass import (  # noqa: F401  (re-exported for config convenience)
    BalancedBinaryMetric,
    SinglePassChangeModel,
    SinglePassMultiTask,
    SinglePassPredictBuilder,
    SinglePassSampler,
    TimestampBoundaryAccuracy,
    _make_decoder,
)

INPUT_KEY = "sentinel2_l2a"


class PerImageChangeModel(SinglePassChangeModel):
    """OlmoEarth applied per-image, with a temporal self-attention + decoder stack.

    Subclasses ``SinglePassChangeModel`` only to reuse its five loss functions
    (``_seg_loss``, ``_balanced_binary_loss``, ``_timestamp_ce``); the parent
    ``__init__`` is bypassed with a direct ``nn.Module.__init__`` call.
    """

    def __init__(
        self,
        encoder: OlmoEarth,
        decoder_type: str,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timesteps: int = 20,
        embedding_dim: int = 768,
        decoder_stages: list[list[tuple[int, int]]] | None = None,
        num_temporal_selfattn_layers: int = 1,
        attn_num_heads: int = 8,
        rnn_bidirectional: bool = True,
        binary_loss_weight: float = 2.0,
    ):
        """Initialize the per-image LCC model.

        Args:
            encoder: the OlmoEarth encoder. Must be configured with
                ``token_pooling=True`` so each per-image pass returns a single
                ``(B, C, h, w)`` feature map.
            decoder_type: temporal decoder for the segmentation heads, "attn" or
                "rnn".
            num_classes_binary: number of classes for the binary change task.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_timesteps: expected number of input timesteps (e.g. 20).
            embedding_dim: per-token encoder embedding size (768 for BASE).
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size)
                so outputs are full resolution. Required.
            num_temporal_selfattn_layers: number of temporal self-attention layers
                over the T tokens (per pixel) applied before the heads. Default 1.
            attn_num_heads: number of heads for self-attention and the attn decoder.
            rnn_bidirectional: whether the GRU decoder is bidirectional.
            binary_loss_weight: multiplier applied to the binary change loss.
        """
        nn.Module.__init__(self)
        if decoder_type not in ("attn", "rnn"):
            raise ValueError(
                f"decoder_type must be 'attn' or 'rnn', got {decoder_type!r}"
            )
        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.decoder_type = decoder_type
        self.binary_loss_weight = binary_loss_weight
        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst

        # Learned temporal positional embedding + optional self-attention over T.
        self.temporal_pos = nn.Parameter(
            torch.randn(1, num_timesteps, embedding_dim) * 0.02
        )
        if num_temporal_selfattn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_num_heads,
                dim_feedforward=embedding_dim * 4,
                batch_first=True,
            )
            self.temporal_selfattn: nn.Module | None = nn.TransformerEncoder(
                layer, num_layers=num_temporal_selfattn_layers
            )
        else:
            self.temporal_selfattn = None

        # Temporal pooling for the segmentation heads.
        if decoder_type == "attn":
            self.time_query = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
            self.time_attn = nn.MultiheadAttention(
                embedding_dim, attn_num_heads, batch_first=True
            )
            self.time_proj: nn.Module | None = None
        else:
            self.time_gru = nn.GRU(
                embedding_dim,
                embedding_dim,
                batch_first=True,
                bidirectional=rnn_bidirectional,
            )
            gru_out = embedding_dim * 2 if rnn_bidirectional else embedding_dim
            self.time_proj = (
                nn.Linear(gru_out, embedding_dim) if gru_out != embedding_dim else None
            )

        # Segmentation decoders consume the temporally-pooled feature.
        self.decoder_binary = _make_decoder(
            embedding_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(embedding_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(embedding_dim, decoder_stages, num_classes_dst)

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

    def _encode_per_image(self, context: ModelContext) -> torch.Tensor:
        """Run OlmoEarth on each timestep independently.

        Builds ``B*T`` single-timestep contexts (one per image), runs a single
        encoder pass, and returns the per-image token maps as ``(B, C, h, w, T)``.
        """
        inputs = context.inputs
        b = len(inputs)

        per_image_inputs: list[dict[str, Any]] = []
        per_image_metadatas: list[Any] = []
        t_expected = self.num_timesteps
        for i in range(b):
            raster = inputs[i][INPUT_KEY]
            assert isinstance(raster, RasterImage)
            image = raster.image  # (C, T, H, W)
            t = image.shape[1]
            if t != t_expected:
                raise ValueError(f"Expected {t_expected} timesteps, got {t}")
            timestamps = raster.timestamps
            for j in range(t):
                ts = [timestamps[j]] if timestamps is not None else None
                per_image_inputs.append(
                    {
                        INPUT_KEY: RasterImage(
                            image=image[:, j : j + 1, :, :], timestamps=ts
                        )
                    }
                )
                per_image_metadatas.append(context.metadatas[i])

        per_context = ModelContext(
            inputs=per_image_inputs, metadatas=per_image_metadatas
        )
        feature_maps = self.encoder(per_context)
        feat = feature_maps.feature_maps[0]  # (B*T, C, h, w)
        c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
        feat = feat.reshape(b, t_expected, c, h, w).permute(0, 2, 3, 4, 1)
        return feat  # (B, C, h, w, T)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Per-image encode, temporal self-attention, temporal decoder, heads.

        Args:
            context: ModelContext with a ``sentinel2_l2a`` RasterImage per sample
                (each ``(C, num_timesteps, H, W)``).
            targets: optional target dicts with "binary", "src", "dst",
                "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses (same layout as the
            single-pass model).
        """
        feature = self._encode_per_image(context)  # (B, C, h, w, T)
        b, c, h, w, t = feature.shape

        # Per-pixel temporal tokens with positional embedding + self-attention.
        tokens = feature.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
        tokens = tokens + self.temporal_pos
        if self.temporal_selfattn is not None:
            tokens = self.temporal_selfattn(tokens)

        # Temporal pooling for segmentation.
        if self.decoder_type == "attn":
            q = self.time_query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.time_attn(q, tokens, tokens)  # (N, 1, C)
            pooled = pooled.squeeze(1)
        else:
            out, _ = self.time_gru(tokens)  # (N, T, gru_out)
            pooled = out.mean(dim=1)
            if self.time_proj is not None:
                pooled = self.time_proj(pooled)
        seg_feat = pooled.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, h, w)

        logits_binary = self.decoder_binary(seg_feat)
        logits_src = self.decoder_src(seg_feat)
        logits_dst = self.decoder_dst(seg_feat)

        # Per-token timestamp logits over T, upsampled to full resolution.
        xt = tokens.reshape(b, h, w, t, c)
        start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)  # (B,T,h,w)
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
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)
