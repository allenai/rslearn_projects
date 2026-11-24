"""Model components for quantized OlmoEarth embedding inference.

Quantization follows the AlphaEarth Foundations int8 scheme (Brown et al. 2025,
section S8.1): signed square root, scaled to the int8 range, and rounded. The quantizer
never produces -128, which is reserved as nodata.
"""

import time
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput

from rslp.log_utils import get_logger

logger = get_logger(__name__)

QUANTIZE_POWER = 2.0
QUANTIZE_SCALE = 127.5

# Coordinates with a larger magnitude saturate at +/-127.
QUANTIZE_CLIP_THRESHOLD = (127.0 / QUANTIZE_SCALE) ** QUANTIZE_POWER

NODATA_VALUE = -128


class NonFiniteEmbeddingError(Exception):
    """A model output had a non-finite coordinate."""


def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Quantize float embeddings to int8 using the power-based scheme.

    Values are expected to be roughly in [-1, 1]; values outside saturate at +/-127.

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
    """Head that int8-quantizes a feature map.

    Like rslearn.train.tasks.embedding.EmbeddingHead, but the output is an int8
    tensor suitable for writing to an int8 raster layer. Use with EmbeddingTask.
    """

    def __init__(
        self,
        l2_normalize: bool = False,
        epsilon: float = 1e-8,
        log_every_n_batches: int = 200,
        fail_on_non_finite: bool = True,
    ):
        """Create a new QuantizedEmbeddingHead.

        Args:
            l2_normalize: L2-normalize each position's vector across the channel
                dimension. This discards magnitude.
            epsilon: minimum norm to avoid division by zero.
            log_every_n_batches: how often to log the quantization diagnostic.
            fail_on_non_finite: raise on a non-finite coordinate, so the block is
                retried. If False, those pixels are written as nodata instead.
        """
        super().__init__()
        self.l2_normalize = l2_normalize
        self.epsilon = epsilon
        self.log_every_n_batches = log_every_n_batches
        self.fail_on_non_finite = fail_on_non_finite
        self._batches = 0

    def _log_diagnostics(self, features: torch.Tensor) -> None:
        """Log how much the quantizer clips and how well it round-trips."""
        log = self._batches % self.log_every_n_batches == 0
        self._batches += 1
        if not log:
            return
        with torch.no_grad():
            clipped = (features.abs() > QUANTIZE_CLIP_THRESHOLD).float().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                dequantize_embeddings(quantize_embeddings(features)), features, dim=1
            )
            logger.info(
                "quantize diagnostics: coord std %.4f, max |coord| %.4f, "
                "clipped fraction %.5f, round trip cosine mean %.6f min %.6f",
                features.std().item(),
                features.abs().max().item(),
                clipped,
                cos.mean().item(),
                cos.min().item(),
            )

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
            raise TypeError("input to QuantizedEmbeddingHead must be a FeatureMaps")
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
        self._log_diagnostics(features)

        # NaN casts to int8 0, which would read as real data.
        quantized = quantize_embeddings(features)
        non_finite = ~torch.isfinite(features).all(dim=1, keepdim=True)
        if non_finite.any():
            count, total = int(non_finite.sum()), non_finite.numel()
            if self.fail_on_non_finite:
                raise NonFiniteEmbeddingError(
                    f"{count} of {total} pixels have non-finite embeddings"
                )
            logger.warning(
                "%d of %d pixels have non-finite embeddings; writing them as nodata",
                count,
                total,
            )
            quantized = quantized.masked_fill(non_finite, NODATA_VALUE)

        return ModelOutput(outputs=quantized, loss_dict={"loss": 0})


class PredictHeartbeat(Callback):
    """Log prediction progress at a fixed interval, to tell a slow job from a hung one."""

    def __init__(
        self, every_n_batches: int = 50, every_n_seconds: float = 120.0
    ) -> None:
        """Set the heartbeat interval; whichever bound is reached first logs a line.

        Args:
            every_n_batches: log at least this often in batches.
            every_n_seconds: log at least this often in seconds.
        """
        super().__init__()
        self.every_n_batches = every_n_batches
        self.every_n_seconds = every_n_seconds
        self._started = 0.0
        self._last_logged = 0.0
        self._last_batch = 0

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset the clock at the start of prediction.

        Args:
            trainer: the Lightning trainer.
            pl_module: the Lightning module.
        """
        self._started = self._last_logged = time.time()
        self._last_batch = 0

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log a heartbeat once either interval has elapsed.

        Args:
            trainer: the Lightning trainer.
            pl_module: the Lightning module.
            outputs: the batch's predictions, unused.
            batch: the batch, unused.
            batch_idx: index of the batch just finished.
            dataloader_idx: which dataloader, unused.
        """
        now = time.time()
        batches_since = batch_idx - self._last_batch
        if (
            batches_since < self.every_n_batches
            and now - self._last_logged < self.every_n_seconds
        ):
            return
        elapsed = now - self._started
        window = now - self._last_logged
        logger.info(
            "predict heartbeat: batch %d, %.1f min elapsed, %.2f batch/s recent, "
            "%.2f batch/s overall",
            batch_idx,
            elapsed / 60,
            batches_since / window if window > 0 else 0.0,
            (batch_idx + 1) / elapsed if elapsed > 0 else 0.0,
        )
        self._last_logged = now
        self._last_batch = batch_idx
