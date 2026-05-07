"""Model for pre-training Swin backbone on OlmoEarth dataset.

This model interleaves temporal self-attention between each SwinB resolution stage,
allowing information exchange across timesteps at every spatial scale. Cross-attention
temporal pooling then produces per-resolution output features for the UNet decoder.

The patch embedding is a two-stage projection: 1x1 per-pixel conv followed by a
strided 4x4 conv, replacing the standard single-conv Swin patch embed.
"""

import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.unet import UNetDecoder
from rslearn.train.model_context import ModelContext
from torchvision.models.swin_transformer import Swin_V2_B_Weights
from torchvision.ops.misc import Permute

from rslp.swin_pretrain.model_crossattn import (
    CrossAttentionTemporalPool,
    _get_1d_sincos_pos_embed,
)


class TemporalSelfAttention(nn.Module):
    """Self-attention across timesteps at each spatial position.

    For each spatial patch, tokens from different timesteps attend to each other.

    Inputs:  x of shape [B, T, C, H, W]
    Outputs: x of shape [B, T, C, H, W]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Create new TemporalSelfAttention."""
        super().__init__()
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention.

        Args:
            x: input tensor [B, T, C, H, W].

        Returns:
            output tensor [B, T, C, H, W] with temporal information mixed.
        """
        B, T, C, H, W = x.shape
        N = H * W

        pe_time = _get_1d_sincos_pos_embed(C, T, x.device)  # [T, C]

        # [B, T, H*W, C] with temporal PE
        tokens = rearrange(x, "b t c h w -> b t (h w) c")
        tokens = tokens + pe_time.view(1, T, 1, C)

        # Group by spatial position: [B*N, T, C]
        tokens = rearrange(tokens, "b t n c -> (b n) t c")
        normed = self.norm(tokens)

        out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.drop(out)
        tokens = tokens + self.mlp(tokens)

        # Back to [B, T, C, H, W]
        tokens = rearrange(tokens, "(b n) t c -> b t c n", b=B, n=N)
        return tokens.view(B, T, C, H, W)


TEMPORAL_ATTN_LAYERS = {1, 3, 5, 7}
ENCODER_CHANNELS = [128, 256, 512, 1024]


class Model(FeatureExtractor):
    """Swin model with interleaved temporal attention at each resolution."""

    def __init__(
        self,
        input_channels: int = 12,
        target_resolution_factor: int | None = 1,
        unet_out_channels: int | None = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Create new Model."""
        super().__init__()
        self.target_resolution_factor = target_resolution_factor

        swin = torchvision.models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)

        # Replace patch embedding with two-stage projection.
        swin.features[0] = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(128),
        )
        self.swin_features = swin.features

        self.temporal_self_attns = nn.ModuleDict(
            {
                str(layer_idx): TemporalSelfAttention(
                    embed_dim=c,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for layer_idx, c in zip(sorted(TEMPORAL_ATTN_LAYERS), ENCODER_CHANNELS)
            }
        )
        self.temporal_poolers = nn.ModuleDict(
            {
                str(layer_idx): CrossAttentionTemporalPool(
                    embed_dim=c,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for layer_idx, c in zip(sorted(TEMPORAL_ATTN_LAYERS), ENCODER_CHANNELS)
            }
        )

        if self.target_resolution_factor is not None:
            self.unet = UNetDecoder(
                in_channels=[[4, 128], [8, 256], [16, 512], [32, 1024]],
                out_channels=unet_out_channels,
                conv_layers_per_resolution=2,
                target_resolution_factor=target_resolution_factor,
            )

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Extract features with interleaved temporal attention.

        Args:
            context: the model context. Input dicts must include "image" key
                containing a RasterImage with the time series.

        Returns:
            FeatureMaps after temporal attention pooling and optional UNet.
        """
        images = torch.stack(
            [inp["image"].image for inp in context.inputs], dim=0
        )  # [B, C, T, H, W]
        B = images.shape[0]
        T = images.shape[2]

        # Flatten timesteps into batch: [B*T, C, H, W]
        x = rearrange(images, "b c t h w -> (b t) c h w")

        pooled_features: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.swin_features):
            print(f"start layer {layer_idx}: {x.shape}")
            x = layer(x)  # channels-last: [B*T, H, W, C]
            print(f"end layer {layer_idx}: {x.shape}")

            key = str(layer_idx)
            if key in self.temporal_self_attns:
                # Convert to channels-first and reshape for temporal ops
                print(f"layer {key} starting shape: {x.shape}")
                x_chw = x.permute(0, 3, 1, 2)  # [B*T, C, H, W]
                x_btchw = rearrange(x_chw, "(b t) c h w -> b t c h w", b=B, t=T)

                # Apply temporal self-attention.
                x_btchw = self.temporal_self_attns[key](x_btchw)
                print("temporal self attention result", x_btchw.shape)

                # Apply temporal cross-attention to get temporally pooled output features.
                pooled_features.append(self.temporal_poolers[key](x_btchw))
                print("pooling result", pooled_features[-1].shape)

                # Reshape the self attention result back for continuing through Swin
                # (channels-last).
                x = rearrange(x_btchw, "b t c h w -> (b t) c h w")
                x = x.permute(0, 2, 3, 1)  # [B*T, H, W, C]
                print(f"layer {key} ending shape: {x.shape}")

        if self.target_resolution_factor is None:
            return FeatureMaps(pooled_features)

        return self.unet(FeatureMaps(pooled_features), context)
