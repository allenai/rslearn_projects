"""Training components for the change finder module."""

import random
from typing import Any

import numpy.typing as npt
import torch
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.train.tasks.task import Task
from rslearn.train.transforms.transform import Transform
from torchmetrics import Metric, MetricCollection
from typing_extensions import override


class ChangeFinderNormalize(Transform):
    """Wraps OlmoEarthNormalize to remap arbitrarily-named sentinel2 inputs to sentinel2_l2a."""

    def __init__(
        self,
        modality_names: list[str],
        band_names: list[str],
        **kwargs: Any,
    ) -> None:
        """Create a new ChangeFinderNormalize."""
        super().__init__()
        self.modality_names = modality_names
        self.normalize = OlmoEarthNormalize(
            band_names={"sentinel2_l2a": band_names},
            **kwargs,
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply OlmoEarthNormalize Sentinel-2 normalization on each configured image key."""
        for name in self.modality_names:
            if name not in input_dict:
                continue
            tmp_input = {"sentinel2_l2a": input_dict[name]}
            tmp_input, _ = self.normalize(tmp_input, {})
            input_dict[name] = tmp_input["sentinel2_l2a"]
        return input_dict, target_dict


class ChangeFinderTransform(Transform):
    """Select a random triplet from the four year-offset inputs and assign a class label.

    Class 0 triplet [1y, 0y, 6y]: anchor1=y1, query=y0, anchor2=y6
    Class 1 triplet [0y, 6y, 5y]: anchor1=y0, query=y6, anchor2=y5
    """

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pick one of the two possible triplets."""
        y0 = input_dict.pop("sentinel2_y0")
        y1 = input_dict.pop("sentinel2_y1")
        y5 = input_dict.pop("sentinel2_y5")
        y6 = input_dict.pop("sentinel2_y6")

        if random.random() < 0.5:
            anchor1, query, anchor2 = y1, y0, y6
            label = 0
        else:
            anchor1, query, anchor2 = y0, y6, y5
            label = 1

        input_dict["anchor1"] = anchor1
        input_dict["query"] = query
        input_dict["anchor2"] = anchor2
        target_dict["change"] = {"class_id": torch.tensor(label, dtype=torch.long)}

        return input_dict, target_dict


class ChangeFinderGapTransform(Transform):
    """Symmetric gap-parameterized triplet sampling from all available years.

    Enumerates all valid (close_anchor, query, far_anchor) triplets where
    |query - close| = 1 and |query - far| = gap, plus the reversed anchor
    ordering. This prevents the model from exploiting anchor position bias.
    """

    def __init__(self, gap: int = 6, num_years: int = 7) -> None:
        """Create a new ChangeFinderGapTransform."""
        super().__init__()
        self.gap = gap
        self.num_years = num_years

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Forward pass, see class docstring."""
        years: dict[int, Any] = {}
        for i in range(self.num_years):
            key = f"sentinel2_y{i}"
            if key in input_dict:
                years[i] = input_dict.pop(key)

        triplets: list[tuple[Any, Any, Any, int]] = []
        for q in years:
            for c in years:
                if abs(q - c) != 1:
                    continue
                for f in years:
                    if abs(q - f) != self.gap or f == c:
                        continue
                    if not (min(c, f) < q < max(c, f)):
                        continue
                    triplets.append((years[c], years[q], years[f], 0))
                    triplets.append((years[f], years[q], years[c], 1))

        anchor1, query, anchor2, label = random.choice(triplets)
        input_dict["anchor1"] = anchor1
        input_dict["query"] = query
        input_dict["anchor2"] = anchor2
        target_dict["change"] = {"class_id": torch.tensor(label, dtype=torch.long)}

        return input_dict, target_dict


def _merge_timestamps(a: RasterImage, b: RasterImage) -> list[tuple] | None:
    ts_a = a.timestamps
    ts_b = b.timestamps
    if ts_a is not None and ts_b is not None:
        return ts_a + ts_b
    return None


class ChangeFinderModel(torch.nn.Module):
    """Dual-encoder model that classifies whether a query is closer to anchor1 or anchor2.

    For each forward pass the model:
    1. Assembles pair_a = cat(anchor1, query) and pair_b = cat(anchor2, query) along T.
    2. Runs a shared OlmoEarth encoder on each pair independently.
    3. Pools each encoder output to a feature vector.
    4. Combines the two vectors and classifies or scores them.
    """

    def __init__(
        self,
        encoder: FeatureExtractor,
        in_channels: int = 768,
        num_fc_layers: int = 2,
        fc_channels: int = 512,
        pool_mode: str = "max",
        combine_mode: str = "concat",
        loss_mode: str = "ce",
    ):
        """Create a new ChangeFinderModel."""
        super().__init__()
        self.encoder = encoder
        self.pool_mode = pool_mode
        self.combine_mode = combine_mode
        self.loss_mode = loss_mode

        if pool_mode == "attn":
            self.attn_pool = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1),
            )

        if loss_mode == "ce":
            layers: list[torch.nn.Module] = []
            prev = 2 * in_channels
            for _ in range(num_fc_layers):
                layers.append(torch.nn.Linear(prev, fc_channels))
                layers.append(torch.nn.ReLU(inplace=True))
                prev = fc_channels
            layers.append(torch.nn.Linear(prev, 2))
            self.classifier = torch.nn.Sequential(*layers)
        elif loss_mode == "margin":
            self.scorer = torch.nn.Linear(in_channels, 1)
        else:
            raise ValueError(f"unknown loss_mode: {loss_mode}")

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        """Spatial pooling: B x C x H x W -> B x C."""
        if self.pool_mode == "max":
            return torch.amax(features, dim=(2, 3))
        elif self.pool_mode == "mean":
            return features.mean(dim=(2, 3))
        elif self.pool_mode == "attn":
            b, c, h, w = features.shape
            flat = features.permute(0, 2, 3, 1).reshape(b, h * w, c)  # B x HW x C
            weights = self.attn_pool(flat).squeeze(-1)  # B x HW
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)  # B x HW x 1
            return (flat * weights).sum(dim=1)  # B x C
        else:
            raise ValueError(f"unknown pool_mode: {self.pool_mode}")

    def _combine(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Combine pair features: B x C each -> B x 2C."""
        if self.combine_mode == "concat":
            return torch.cat([feat_a, feat_b], dim=1)
        elif self.combine_mode == "diff":
            return torch.cat([feat_a - feat_b, feat_a * feat_b], dim=1)
        else:
            raise ValueError(f"unknown combine_mode: {self.combine_mode}")

    def _encode_pair(
        self, pair_images: list[RasterImage], context: ModelContext
    ) -> torch.Tensor:
        """Run OlmoEarth on a list of RasterImages (one per batch element) and pool."""
        temp_inputs = [{"sentinel2_l2a": img} for img in pair_images]
        temp_context = ModelContext(inputs=temp_inputs, metadatas=context.metadatas)
        feature_maps: FeatureMaps = self.encoder(temp_context)
        features = feature_maps.feature_maps[-1]  # B x C x H x W
        return self._pool(features)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compare per-pair features to classify which pair is further apart in time."""
        batch_size = len(context.inputs)

        pair_a: list[RasterImage] = []
        pair_b: list[RasterImage] = []
        for i in range(batch_size):
            inp = context.inputs[i]
            a1: RasterImage = inp["anchor1"]
            q: RasterImage = inp["query"]
            a2: RasterImage = inp["anchor2"]

            pair_a.append(
                RasterImage(
                    torch.cat([a1.image, q.image], dim=1),
                    timestamps=_merge_timestamps(a1, q),
                )
            )
            pair_b.append(
                RasterImage(
                    torch.cat([a2.image, q.image], dim=1),
                    timestamps=_merge_timestamps(a2, q),
                )
            )

        feat_a = self._encode_pair(pair_a, context)
        feat_b = self._encode_pair(pair_b, context)

        loss_dict: dict[str, torch.Tensor] = {}

        if self.loss_mode == "ce":
            combined = self._combine(feat_a, feat_b)
            logits = self.classifier(combined)
        else:
            score_a = self.scorer(feat_a).squeeze(-1)  # B
            score_b = self.scorer(feat_b).squeeze(-1)  # B
            logits = torch.stack([score_a, score_b], dim=1)  # B x 2

        outputs = [{"change": logits[i]} for i in range(batch_size)]

        if targets is not None:
            labels = torch.stack([t["change"]["class_id"] for t in targets]).to(
                logits.device
            )
            if self.loss_mode == "ce":
                loss_dict["ce_loss"] = torch.nn.functional.cross_entropy(logits, labels)
            else:
                # label=0 means pair A is close → score_a should be higher → target=+1
                # label=1 means pair B is close → score_b should be higher → target=-1
                target = 1 - 2 * labels.float()
                loss_dict["margin_loss"] = torch.nn.functional.margin_ranking_loss(
                    logits[:, 0], logits[:, 1], target, margin=1.0
                )

        return ModelOutput(outputs=outputs, loss_dict=loss_dict)


class ChangeFinderAccuracy(Metric):
    """Binary classification accuracy for the change finder task."""

    def __init__(self, **kwargs: Any):
        """Create a new ChangeFinderAccuracy."""
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, outputs: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update the correctly classified and total example counts."""
        for output, target in zip(outputs, targets):
            pred = output["change"].detach().argmax()
            label = target["change"]["class_id"]
            self.correct += (pred == label).long()
            self.total += 1

    def compute(self) -> torch.Tensor:
        """Compute the accuracy."""
        return self.correct.float() / self.total.clamp(min=1)


class ChangeFinderTask(Task):
    """Task that passes through inputs (all passthrough) and provides accuracy metrics."""

    @override
    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | Any],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    @override
    def process_output(self, raw_output: Any, metadata: SampleMetadata) -> Any:
        return raw_output

    @override
    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        return {}

    @override
    def get_metrics(self) -> MetricCollection:
        return MetricCollection({"accuracy": ChangeFinderAccuracy()})


class ChangeFinderLightningModule(RslearnLightningModule):
    """Lightning module for change finder training."""

    pass
