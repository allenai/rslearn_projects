"""Model components for quantized OlmoEarth embedding inference.

The quantization scheme matches the power-based int8 scheme used for AlphaEarth
Foundations embeddings (Brown et al. 2025, section S8.1) and by
olmoearth_pretrain.evals.embedding_transforms: values are compressed with a signed
square root, scaled to the int8 range, and rounded. The value -128 is never produced
by the quantizer; it is reserved as the nodata value in the output GeoTIFFs.
"""

from typing import Any

import torch
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput

QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5

# Reserved nodata value for the int8 output rasters. quantize_embeddings clamps to
# [-127, 127] so it never emits this value.
NODATA_VALUE = -128


def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Quantize float embeddings to int8 using the power-based scheme.

    The values are expected to be roughly in [-1, 1] (e.g. components of unit-norm
    embedding vectors); values outside that range saturate at -127/127.

    Args:
        embeddings: float tensor of any shape.

    Returns:
        int8 tensor of the same shape, with values in [-127, 127].
    """
    sat = embeddings.abs().pow(1.0 / QUANTIZE_POWER) * embeddings.sign()
    return (sat * QUANTIZE_SCALE).clamp(-127, 127).round().to(torch.int8)


def dequantize_embeddings(quantized: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 embeddings back to float32.

    Args:
        quantized: int8 tensor produced by quantize_embeddings.

    Returns:
        float32 tensor approximating the original embeddings.
    """
    rescaled = quantized.float() / QUANTIZE_SCALE
    return rescaled.abs().pow(QUANTIZE_POWER) * rescaled.sign()


class QuantizedEmbeddingHead(Predictor):
    """Head that L2-normalizes and int8-quantizes a feature map.

    Like rslearn.train.tasks.embedding.EmbeddingHead, but the output is an int8
    tensor suitable for writing to an int8 raster layer. Use with EmbeddingTask.
    """

    def __init__(self, l2_normalize: bool = True, epsilon: float = 1e-8):
        """Create a new QuantizedEmbeddingHead.

        Args:
            l2_normalize: whether to L2-normalize each spatial position's embedding
                vector (across the channel dimension) before quantization. The
                power-based quantization scheme assumes values roughly in [-1, 1], so
                this should be enabled unless the model already outputs unit-norm
                embeddings.
            epsilon: minimum norm to avoid division by zero.
        """
        super().__init__()
        self.l2_normalize = l2_normalize
        self.epsilon = epsilon

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Return the quantized feature map along with a dummy loss.

        Args:
            intermediates: output from the previous model component, which must be a
                FeatureMaps consisting of a single BCHW feature map.
            context: the model context.
            targets: the targets (ignored).

        Returns:
            model output with the int8-quantized feature map along with a dummy loss.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to QuantizedEmbeddingHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "input to QuantizedEmbeddingHead must have one feature map, "
                f"but got {len(intermediates.feature_maps)}"
            )

        features = intermediates.feature_maps[0]
        if self.l2_normalize:
            features = features / features.norm(dim=1, keepdim=True).clamp(
                min=self.epsilon
            )

        return ModelOutput(
            outputs=quantize_embeddings(features),
            loss_dict={"loss": 0},
        )
