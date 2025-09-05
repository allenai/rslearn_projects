"""Model for pre-training Swin backbone on Helios dataset."""

from typing import Any

import torch
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.swin import Swin
from rslearn.models.unet import UNetDecoder


class Model(torch.nn.Module):
    """Model for pre-training."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.backbone = SimpleTimeSeries(
            encoder=Swin(
                arch="swin_v2_b",
                pretrained=True,
                input_channels=12,
                output_layers=[1, 3, 5, 7],
            ),
            image_channels=12,
        )
        self.unet = UNetDecoder(
            in_channels=[[4, 128], [8, 256], [16, 512], [32, 1024]],
            out_channels=128,
            conv_layers_per_resolution=2,
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
        hr_features = self.unet(features, None)
        return [hr_features]
