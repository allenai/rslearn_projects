"""Model for pre-training Swin backbone on OlmoEarth dataset."""

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.swin import Swin
from rslearn.models.unet import UNetDecoder
from rslearn.train.model_context import ModelContext


class Model(FeatureExtractor):
    """Model for pre-training."""

    def __init__(
        self,
        target_resolution_factor: int | None = 1,
        unet_out_channels: int | None = 128,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.target_resolution_factor = target_resolution_factor
        # Currently this model can only handle one input image (Sentinel-2).
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

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Extract features from the input images.

        Args:
            context: the model context. Input dicts must include "image" key.

        Returns:
            FeatureMaps from the backbone, optionally passed through UNet decoder.
        """
        features = self.backbone(context)
        if self.target_resolution_factor is None:
            return features
        return self.unet(features, context)
