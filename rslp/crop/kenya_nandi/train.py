"""Custom crop training code."""

from typing import Any

import torch
import torch.nn.functional as F
from rslearn.models.pooling_decoder import PoolingDecoder


class SegmentationPoolingDecoder(PoolingDecoder):
    """Like rslearn.models.pooling_decoder."""

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Extend PoolingDecoder forward to upsample the output to a segmentation mask.

        This only works when all of the pixels have the same segmentation target.
        """
        output_probs = super().forward(features, inputs)
        # BC -> BCHW
        h, w = inputs[0]["sentinel2_l2a"].shape[1:3]
        return output_probs[:, :, None, None].repeat([1, 1, h, w])


class MultiSegmentationPoolingDecoder(torch.nn.Module):
    """This one pools features at every 3x3 patch followed by fully connected layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "mean",
    ) -> None:
        """Initialize a PoolingDecoder.

        Args:
            in_channels: input channels (channels in the last feature map passed to
                this module)
            out_channels: channels for the output flat feature vector
            mode: either mean or max
        """
        super().__init__()
        self.mode = mode
        self.output_layer = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        assert self.mode in ["mean", "max"]

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Compute the segmentation mask via windowed pooling plus output layer."""
        # Only use last feature map.
        features = features[-1]

        if self.mode == "mean":
            pool_func = F.avg_pool2d
        elif self.mode == "max":
            pool_func = F.max_pool2d

        features = pool_func(
            input=features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return self.output_layer(features)
