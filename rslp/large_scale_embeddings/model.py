"""Model components for quantized OlmoEarth embedding inference.

The quantization scheme matches the power-based int8 scheme used for AlphaEarth
Foundations embeddings (Brown et al. 2025, section S8.1) and by
olmoearth_pretrain.evals.embedding_transforms: values are compressed with a signed
square root, scaled to the int8 range, and rounded. The value -128 is never produced
by the quantizer; it is reserved as the store's nodata value.
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

# Coordinates above this saturate. AEF's unit-L2 vectors sit far below it; LayerNorm
# output (per-coordinate std ~ 1) lands on top of it, hence output_scale.
QUANTIZE_CLIP_THRESHOLD = (127.0 / QUANTIZE_SCALE) ** QUANTIZE_POWER

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
    """Head that scales and int8-quantizes a feature map.

    Like rslearn.train.tasks.embedding.EmbeddingHead, but the output is an int8
    tensor suitable for writing to an int8 raster layer. Use with EmbeddingTask.

    L2 normalization is off because it discards magnitude, which carries signal, so
    `output_scale` is what satisfies the quantizer's [-1, 1] assumption instead.
    """

    def __init__(
        self,
        l2_normalize: bool = False,
        output_scale: float = 1.0,
        epsilon: float = 1e-8,
        log_every_n_batches: int = 200,
    ):
        """Create a new QuantizedEmbeddingHead.

        Args:
            l2_normalize: L2-normalize each position's vector across the channel
                dimension. Lossy; prefer output_scale.
            output_scale: divide features by this before quantizing. Record it in the
                store's `geoemb:quantization` scale so a reader can undo it.
            epsilon: minimum norm to avoid division by zero.
            log_every_n_batches: how often to log the clipped fraction.
        """
        super().__init__()
        self.l2_normalize = l2_normalize
        self.output_scale = output_scale
        self.epsilon = epsilon
        self.log_every_n_batches = log_every_n_batches
        self._batches = 0

    def _log_clipping(self, features: torch.Tensor) -> None:
        """Log the clipped fraction, so a bad output_scale is visible not silent."""
        self._batches += 1
        if self._batches % self.log_every_n_batches != 1:
            return
        with torch.no_grad():
            clipped = (features.abs() > QUANTIZE_CLIP_THRESHOLD).float().mean().item()
            std = features.std().item()
            largest = features.abs().max().item()
        logger.info(
            "quantize diagnostics: coord std %.4f, max |coord| %.4f, "
            "clipped fraction %.5f (threshold %.4f)",
            std,
            largest,
            clipped,
            QUANTIZE_CLIP_THRESHOLD,
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
        if self.output_scale != 1.0:
            features = features / self.output_scale
        self._log_clipping(features)

        return ModelOutput(
            outputs=quantize_embeddings(features),
            loss_dict={"loss": 0},
        )


class PredictHeartbeat(Callback):
    """Log prediction progress periodically, so a slow job is not mistaken for a hung one.

    Lightning's progress bar reaches a container log only every 20-30 minutes, so
    silence between flushes cannot be used as a stall signal. A timestamped line at a
    known interval can: if the interval passes with no line, the job really is stuck.
    """

    def __init__(
        self, every_n_batches: int = 50, every_n_seconds: float = 120.0
    ) -> None:
        """Set the heartbeat interval.

        Whichever bound is reached first triggers a line, since batch time varies by
        orders of magnitude with crop size and modality count.

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
        """Mark the start so the first line reports a real rate.

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
