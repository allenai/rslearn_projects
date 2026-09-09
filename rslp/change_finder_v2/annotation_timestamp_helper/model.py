"""Model for point-level annotation timestamp prediction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.component import TokenFeatureMaps
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)

from .constants import MODEL_INPUT_KEY, NUM_CROP_MONTHS, TIMESTAMP_HEADS

FRAMES_PER_PASS = 12
NUM_PASSES = 5


class AnnotationTimestampModel(nn.Module):
    """Five OlmoEarth passes plus temporal attention over center frame tokens."""

    def __init__(
        self,
        encoder: OlmoEarth,
        embedding_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_key: str = MODEL_INPUT_KEY,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.input_key = input_key

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, NUM_CROP_MONTHS, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.heads = nn.ModuleDict(
            {head: nn.Linear(embedding_dim, 1) for head in TIMESTAMP_HEADS}
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _run_encoder(
        self, raster_images: list[RasterImage], metadatas: list[SampleMetadata]
    ) -> torch.Tensor:
        """Run OlmoEarth and return BNC center-token features for one year."""
        inputs = [{MODEL_INPUT_KEY: img} for img in raster_images]
        sub_context = ModelContext(inputs=inputs, metadatas=metadatas)
        feature_maps = self.encoder(sub_context)
        if not isinstance(feature_maps, TokenFeatureMaps):
            raise ValueError(
                "AnnotationTimestampModel requires OlmoEarth token_pooling=False"
            )
        feat = feature_maps.feature_maps[0]
        if feat.dim() != 5:
            raise ValueError(f"expected BCHWN token feature map, got {feat.shape}")
        _, _, h, w, n = feat.shape
        if n != FRAMES_PER_PASS:
            raise ValueError(
                f"expected {FRAMES_PER_PASS} token frames per pass, got {n}"
            )
        center = feat[:, :, h // 2, w // 2, :]
        return center.permute(0, 2, 1)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Run the model and compute timestamp classification losses."""
        frame_dates_by_sample: list[list[str]] = []
        for inp in context.inputs:
            combined: RasterImage = inp[self.input_key]
            if combined.timestamps is None:
                frame_dates_by_sample.append([])
            else:
                frame_dates_by_sample.append(
                    [
                        (ts[0] + (ts[1] - ts[0]) / 2).date().isoformat()
                        for ts in combined.timestamps
                    ]
                )

        pass_tokens: list[torch.Tensor] = []
        for pass_idx in range(NUM_PASSES):
            start = pass_idx * FRAMES_PER_PASS
            end = start + FRAMES_PER_PASS
            raster_images: list[RasterImage] = []
            for inp in context.inputs:
                combined = inp[self.input_key]
                ts = combined.timestamps[start:end] if combined.timestamps else None
                raster_images.append(
                    RasterImage(
                        image=combined.image[:, start:end, :, :],
                        timestamps=ts,
                    )
                )
            pass_tokens.append(self._run_encoder(raster_images, context.metadatas))

        tokens = torch.cat(pass_tokens, dim=1)
        tokens = tokens + self.pos_embedding
        tokens = self.norm(self.temporal_encoder(tokens))

        logits = {
            head: self.heads[head](tokens).squeeze(-1) for head in TIMESTAMP_HEADS
        }
        outputs: list[dict[str, Any]] = [
            {
                **{head: F.softmax(logits[head][i], dim=0) for head in TIMESTAMP_HEADS},
                "frame_dates": frame_dates_by_sample[i],
            }
            for i in range(len(context.inputs))
        ]

        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            for head in TIMESTAMP_HEADS:
                classes = torch.stack([target[head]["class"] for target in targets])
                valid = torch.stack([target[head]["valid"] for target in targets])
                per_sample = F.cross_entropy(logits[head], classes, reduction="none")
                if valid.sum() > 0:
                    losses[f"{head}_cls"] = (per_sample * valid).sum() / valid.sum()
                else:
                    losses[f"{head}_cls"] = logits[head].sum() * 0

        return ModelOutput(outputs=outputs, loss_dict=losses)
