"""Model for pre-training Swin backbone on Helios dataset.

This model uses a cross-attention mechanism for temporal pooling instead of max
pooling. Temporal pooling is needed since the Swin component is applied on each image
in the time series.
"""

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from rslearn.models.swin import Swin
from rslearn.models.unet import UNetDecoder


def _get_1d_sincos_pos_embed(
    embed_dim: int, length: int, device: torch.device
) -> torch.Tensor:
    """Return [length, embed_dim] 1D sinusoidal PE."""
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2 for sin/cos."
    pos = torch.arange(length, device=device, dtype=torch.float32)  # [L]
    dim = torch.arange(embed_dim // 2, device=device, dtype=torch.float32)  # [D/2]
    freqs = 1.0 / (10000 ** (dim / (embed_dim // 2)))
    angles = pos[:, None] * freqs[None, :]  # [L, D/2]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)  # [L, D]
    return emb  # [L, D]


def _get_2d_sincos_pos_embed(
    embed_dim: int, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """Return [h*w, embed_dim] 2D sinusoidal PE (sum of 1D encodings over x/y)."""
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2 for 2D sin/cos."
    half_dim = embed_dim // 2
    pe_h = _get_1d_sincos_pos_embed(half_dim, h, device)  # [H, D/2]
    pe_w = _get_1d_sincos_pos_embed(half_dim, w, device)  # [W, D/2]
    # meshgrid then concat -> sum as in ViT-style 2D pe
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    pe = torch.cat([pe_h[y], pe_w[x]], dim=2)  # [H, W, D]
    pe = pe.view(h * w, embed_dim)  # [HW, D]
    return pe


class CrossAttentionTemporalPool(nn.Module):
    """Temporal pooling via cross-attention with one dst token per spatial patch.

    Inputs:  x of shape [B, T, C, H, W]
    Outputs: pooled of shape [B, C, H, W]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Create a new crossAttentionTemporalPool.

        Args:
            embed_dim: the embedding dimension of the inputs.
            num_heads: number of attention heads.
            mlp_ratio: ratio of MLP layer embedding dimension over embed_dim.
            dropout: how much to dropout, default to not dropout.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
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

    @torch.no_grad()
    def _make_pos(
        self, T: int, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial [HW, C] and temporal [T, C] sinusoidal embeddings."""
        pe_spatial = _get_2d_sincos_pos_embed(self.embed_dim, H, W, device)  # [HW, C]
        pe_time = _get_1d_sincos_pos_embed(self.embed_dim, T, device)  # [T, C]
        return pe_spatial, pe_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass.

        Args;
            x: the input feature tensor, [B, T, C, H, W]

        Returns;
            an output feature tensor [B, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert C == self.embed_dim, f"Channel dim {C} != embed_dim {self.embed_dim}"

        device = x.device
        N = H * W

        # Positional encodings
        pe_spatial, pe_time = self._make_pos(T, H, W, device)  # [N, C], [T, C]

        # Prepare src tokens: one per (timestep, patch)
        # Add spatial+time PE to each src token
        x_tokens = rearrange(x, "b t c h w -> b t (h w) c")  # [B, T, N, C]
        x_tokens = x_tokens + pe_time.view(1, T, 1, C) + pe_spatial.view(1, 1, N, C)
        x_src = rearrange(x_tokens, "b t n c -> (b n) t c")  # [B*N, T, C]
        x_src = self.norm_kv(x_src)  # pre-norm for KV

        # Prepare dst queries: one per spatial patch; use only spatial PE (content-free query)
        q = pe_spatial.view(1, N, C).expand(B, N, C)  # [B, N, C]
        q = rearrange(q, "b n c -> (b n) 1 c")  # [B*N, 1, C]
        q = self.norm_q(q)

        # Cross-attention: Q (1 token) attends over src sequence of length T for each patch
        out, _ = self.attn(q, x_src, x_src, need_weights=False)  # [B*N, 1, C]
        out = q + self.drop(out)  # residual
        out = out + self.mlp(out)  # MLP block
        out = out.squeeze(1)  # [B*N, C]

        # Back to [B, C, H, W]
        out = rearrange(out, "(b n) c -> b c n", b=B, n=N)
        out = out.view(B, C, H, W)
        return out


class Model(torch.nn.Module):
    """Model for pre-training."""

    def __init__(
        self,
        target_resolution_factor: int | None = 1,
        unet_out_channels: int | None = 128,
        cross_attn_heads: int = 8,
        cross_attn_mlp_ratio: float = 4.0,
        cross_attn_dropout: float = 0.0,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.target_resolution_factor = target_resolution_factor
        # Currently this model can only handle one input image (Sentinel-2).
        self.encoder = Swin(
            arch="swin_v2_b",
            pretrained=True,
            input_channels=12,
            output_layers=[1, 3, 5, 7],
        )
        encoder_channels = [128, 256, 512, 1024]
        self.temporal_poolers = nn.ModuleList(
            [
                CrossAttentionTemporalPool(
                    embed_dim=c,
                    num_heads=cross_attn_heads,
                    mlp_ratio=cross_attn_mlp_ratio,
                    dropout=cross_attn_dropout,
                )
                for c in encoder_channels
            ]
        )

        if self.target_resolution_factor is not None:
            self.unet = UNetDecoder(
                in_channels=[[4, 128], [8, 256], [16, 512], [32, 1024]],
                out_channels=unet_out_channels,
                conv_layers_per_resolution=2,
                target_resolution_factor=target_resolution_factor,
            )

    def forward(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Compute outputs from the wrapped module.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process.
        """
        # Apply the image encoder on each image in the time series.
        images = torch.stack([inp["image"] for inp in inputs], dim=0)
        image_channels = 12
        batch_size = len(inputs)
        n_images = images.shape[1] // image_channels
        # Reshape images to B*T x C x H x W.
        images = rearrange(
            images, "b (t c) h w -> (b t) c h w", t=n_images, c=image_channels
        )
        # Now add "image" key expected by encoder.
        batched_inputs = [{"image": image} for image in images]
        # Encoder provides one feature map per resolution.
        encoder_feats = self.encoder(batched_inputs)
        all_features = [
            rearrange(feat_map, "(b t) c h w -> b t c h w", b=batch_size, t=n_images)
            for feat_map in encoder_feats
        ]

        # Compute pooled features using cross attention.
        pooled_features = [
            pool(feat_map)
            for pool, feat_map in zip(self.temporal_poolers, all_features)
        ]

        if self.target_resolution_factor is None:
            return pooled_features

        hr_features = self.unet(pooled_features, None)
        return [hr_features]
