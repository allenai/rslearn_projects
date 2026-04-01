"""Custom crop training code."""

from typing import Any

from rslearn.models.component import FeatureMaps
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.train.model_context import ModelContext


class SegmentationPoolingDecoder(PoolingDecoder):
    """Like rslearn.models.pooling_decoder.SegmentationPoolingDecoder.

    Produces a per-pixel output by copying the pooled vector to every spatial location.
    This only makes sense for very small windows where the output is the same everywhere.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_key: str = "sentinel2_l2a",
        **kwargs: Any,
    ):
        """Create a new SegmentationPoolingDecoder.

        Args:
            in_channels: input channels.
            out_channels: output channels.
            image_key: key in context.inputs to derive spatial dimensions from.
            kwargs: other arguments to pass to PoolingDecoder.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.image_key = image_key

    def forward(self, intermediates: Any, context: ModelContext) -> Any:
        """Extend PoolingDecoder forward to upsample the output to a segmentation mask."""
        output_probs = super().forward(intermediates, context)
        # Get H, W from the input image (RasterImage with CTHW layout).
        h, w = context.inputs[0][self.image_key].image.shape[-2:]
        feat_map = output_probs.feature_vector[:, :, None, None].repeat([1, 1, h, w])
        return FeatureMaps([feat_map])
