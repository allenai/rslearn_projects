"""Model for pre-training Swin backbone on Helios dataset."""

from typing import Any

import torch
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.swin import Swin
from rslearn.models.unet import UNetDecoder


class Model(torch.nn.Module):
    """Model for pre-training."""

    def __init__(
        self,
        target_resolution_factor: int | None = 1,
        unet_out_channels: int | None = 128,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.target_resolution_factor = target_resolution_factor
        self.backbone = SimpleTimeSeries(
            encoder=Swin(
                arch="swin_v2_b",
                pretrained=True,
                input_channels=12,
                output_layers=[1, 3, 5, 7],
            ),
            image_channels=12,
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
        features = self.backbone(inputs)
        if self.target_resolution_factor is None:
            return features

        hr_features = self.unet(features, None)
        return [hr_features]
