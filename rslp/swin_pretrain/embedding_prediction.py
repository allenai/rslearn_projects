"""Task and head for predicting multi-channel embeddings (e.g. OlmoEarth-v1-Base)."""

from typing import Any, Literal

import torch
from rslearn.models.component import FeatureMaps, IntermediateComponent, Predictor
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.train.tasks.task import BasicTask
from torchmetrics import Metric, MetricCollection


class EmbeddingPredictionTask(BasicTask):
    """A per-pixel multi-channel prediction task for embedding targets."""

    def __init__(
        self,
        nodata_value: float | None = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new EmbeddingPredictionTask.

        Args:
            nodata_value: if set, pixels where ALL channels equal this value are masked.
            kwargs: other arguments to pass to BasicTask.
        """
        super().__init__(**kwargs)
        self.nodata_value = nodata_value

    def process_inputs(
        self,
        raw_inputs: dict[str, Any],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process raw inputs into targets compatible with EmbeddingPredictionHead."""
        if not load_targets:
            return {}, {}

        assert isinstance(raw_inputs["targets"], RasterImage)
        # CTHW -> CHW (single timestep)
        labels = raw_inputs["targets"].single_ts_to_chw_tensor().float()

        # Valid mask: a pixel is invalid if all channels equal nodata_value.
        if self.nodata_value is not None:
            all_nodata = (labels == self.nodata_value).all(dim=0)
            valid = (~all_nodata).float()  # HW
        else:
            valid = torch.ones(labels.shape[1:], dtype=torch.float32)

        return {}, {
            "values": RasterImage(labels[:, None, :, :]),
            "valid": RasterImage(valid[None, None, :, :]),
        }

    def get_metrics(self) -> MetricCollection:
        """Get metrics for this task."""
        return MetricCollection(
            {"l1": EmbeddingL1Metric()},
        )


class EmbeddingPredictionHead(Predictor):
    """Head for multi-channel embedding prediction."""

    def __init__(
        self,
        loss_mode: Literal["l1", "l2", "cosine"] = "l1",
    ) -> None:
        """Initialize a new EmbeddingPredictionHead.

        Args:
            loss_mode: loss function to use -- "l1", "l2", or "cosine".
        """
        super().__init__()
        if loss_mode not in ("l1", "l2", "cosine"):
            raise ValueError(f"invalid loss mode {loss_mode}")
        self.loss_mode = loss_mode

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute outputs and loss from predicted embeddings.

        Args:
            intermediates: FeatureMaps with one BCHW feature map.
            context: the model context.
            targets: dicts with "values" (CTHW) and "valid" (1,1,H,W) keys.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to EmbeddingPredictionHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError("input must have exactly one feature map")

        logits = intermediates.feature_maps[0]  # BCHW

        losses = {}
        if targets:
            labels = torch.stack(
                [t["values"].single_ts_to_chw_tensor() for t in targets]
            )  # BCHW
            mask = torch.stack(
                [t["valid"].single_ts_to_chw_tensor() for t in targets]
            )  # B1HW

            if self.loss_mode == "l1":
                scores = torch.abs(logits - labels).mean(dim=1, keepdim=True)  # B1HW
            elif self.loss_mode == "l2":
                scores = torch.square(logits - labels).mean(dim=1, keepdim=True)  # B1HW
            elif self.loss_mode == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(
                    logits, labels, dim=1
                )  # BHW
                scores = (1 - cos_sim).unsqueeze(1)  # B1HW

            mask_total = mask.sum()
            if mask_total == 0:
                losses["embedding"] = (scores * mask).mean()
            else:
                losses["embedding"] = (scores * mask).sum() / mask_total

        return ModelOutput(outputs=logits, loss_dict=losses)


class SpatialPool(IntermediateComponent):
    """Adaptive average pool applied to each feature map in a FeatureMaps."""

    def __init__(self, output_size: int) -> None:
        """Initialize SpatialPool.

        Args:
            output_size: target spatial size for adaptive average pooling.
        """
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply adaptive average pool to each feature map.

        Args:
            intermediates: the previous output, which must be a FeatureMaps.
            context: the model context.

        Returns:
            the pooled feature maps.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to SpatialPool must be FeatureMaps")
        return FeatureMaps(
            [self.pool(feat_map) for feat_map in intermediates.feature_maps]
        )


class EmbeddingL1Metric(Metric):
    """Mean L1 error over valid embedding pixels."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize EmbeddingL1Metric."""
        super().__init__(**kwargs)
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: list[Any] | torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> None:
        """Update with a batch of predictions and targets."""
        if not isinstance(preds, torch.Tensor):
            preds = torch.stack(preds)
        labels = torch.stack([t["values"].single_ts_to_chw_tensor() for t in targets])
        mask = torch.stack(
            [t["valid"].single_ts_to_chw_tensor() for t in targets]
        )  # B1HW
        l1 = torch.abs(preds - labels).mean(dim=1, keepdim=True)  # B1HW
        self.total_error += (l1 * mask).sum()
        self.total_count += mask.sum().long()

    def compute(self) -> torch.Tensor:
        """Compute the mean L1 error."""
        if self.total_count == 0:
            return torch.tensor(0.0)
        return self.total_error / self.total_count
