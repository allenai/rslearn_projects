"""Unit tests for rslp.swin_pretrain.model_crossattn."""

import torch

from rslp.swin_pretrain.model_crossattn import CrossAttentionTemporalPool


class TestCrossAttentionTemporalPool:
    """Test CrossAttentionTemporalPool."""

    def test_single_timestep(self) -> None:
        embed_dim = 64
        size = 8
        pool = CrossAttentionTemporalPool(embed_dim=embed_dim)
        x = torch.zeros((1, 1, embed_dim, size, size))
        result = pool(x)
        assert result.shape == (1, embed_dim, size, size)

    def test_multiple_timesteps(self) -> None:
        embed_dim = 64
        size = 8
        pool = CrossAttentionTemporalPool(embed_dim=embed_dim)
        x = torch.zeros((1, 4, embed_dim, size, size))
        result = pool(x)
        assert result.shape == (1, embed_dim, size, size)
