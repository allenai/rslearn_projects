"""Tasks, heads, and metrics for training the forest loss driver model.

HierarchicalForestLossTask and HierarchicalClassificationHead train the model on the
leaves of ACA's label hierarchy (see rslp.forest_loss_driver.hierarchy). ForestLossTask
and ForestLossLightningModule are kept for the older configs that use the flat
categories.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
import torch
import wandb
from PIL import Image, ImageDraw
from rslearn.models.component import FeatureVector, Predictor
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.metrics import ConfusionMatrixMetric
from rslearn.train.model_context import ModelContext, ModelOutput, SampleMetadata
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils import Feature, STGeometry
from torchmetrics import Metric, MetricCollection

from rslp.forest_loss_driver import hierarchy

CATEGORY_MAPPING = {
    "agriculture-generic": "agriculture",
    "agriculture-small": "agriculture",
    "agriculture-mennonite": "agriculture",
    "agriculture-rice": "agriculture",
    "coca": "agriculture",
    "flood": "river",
}


class ForestLossTask(ClassificationTask):
    """Forest loss task.

    It is a classification task but just adds some additional pre-processing because of
    the format of the labels where the labels are hierarchical but we want to remap
    them to a particular flat set.
    """

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        This is modified to do category remapping.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        if not load_targets:
            return {}, {}

        data = raw_inputs["targets"]
        for feat in data:
            for property_name, property_value in self.filters:
                if feat.properties.get(property_name) != property_value:
                    continue
            if self.property_name not in feat.properties:
                continue

            class_name = feat.properties[self.property_name]
            if class_name in CATEGORY_MAPPING:
                class_name = CATEGORY_MAPPING[class_name]
            if class_name not in self.classes:
                continue
            class_id = self.classes.index(class_name)

            return {}, {
                "class": torch.tensor(class_id, dtype=torch.int64),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise Exception(
                f"no feature found providing class label for window {metadata['window_name']}"
            )

        return {}, {
            "class": torch.tensor(0, dtype=torch.int64),
            "valid": torch.tensor(0, dtype=torch.float32),
        }


class ForestLossLightningModule(RslearnLightningModule):
    """Lightning module extended with val / test confusion matrix reporting."""

    def on_validation_epoch_start(self) -> None:
        """Initialize val confusion matrix."""
        self.probs: list = []
        self.y_true: list = []

    def on_val_forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        model_outputs: dict[str, Any],
    ) -> None:
        """Hook to run after the forward pass of the model during validation.

        Args:
            inputs: The input batch.
            targets: The target batch.
            model_outputs: The output of the model, with keys "outputs" and "loss_dict", and possibly other keys.
        """
        for output, target in zip(model_outputs["outputs"], targets):
            if not target["class"]["valid"]:
                continue
            self.probs.append(output["class"].cpu().numpy())
            self.y_true.append(target["class"]["class"].cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        """Submit the val confusion matrix."""
        self.logger.experiment.log(
            {
                "val_cm": wandb.plot.confusion_matrix(
                    probs=np.stack(self.probs),
                    y_true=np.stack(self.y_true),
                    class_names=self.task.tasks["class"].classes,
                )
            }
        )

    def on_test_epoch_start(self) -> None:
        """Initialize test confusion matrix."""
        self.probs = []
        self.y_true = []

    def on_test_forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        model_outputs: dict[str, Any],
    ) -> None:
        """Hook to run after the forward pass of the model during testing.

        Args:
            inputs: The input batch.
            targets: The target batch.
            model_outputs: The output of the model, with keys "outputs" and "loss_dict", and possibly other keys.
        """
        for output, target in zip(model_outputs["outputs"], targets):
            if not target["class"]["valid"]:
                continue
            self.probs.append(output["class"].cpu().numpy())
            self.y_true.append(target["class"]["class"].cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Submit the test confusion matrix."""
        self.logger.experiment.log(
            {
                "test_cm": wandb.plot.confusion_matrix(
                    probs=np.stack(self.probs),
                    y_true=np.stack(self.y_true),
                    class_names=self.task.tasks["class"].classes,
                )
            }
        )


def _aggregation_matrix(groups: list[int | None], num_groups: int) -> torch.Tensor:
    """Matrix that sums leaf probabilities into groups (Level 1/2 nodes, etc.).

    Args:
        groups: the group index of each leaf, or None if the leaf is not in any group.
        num_groups: the number of groups.

    Returns:
        a (num_leaves, num_groups) tensor.
    """
    matrix = torch.zeros((len(groups), num_groups), dtype=torch.float32)
    for leaf_idx, group_idx in enumerate(groups):
        if group_idx is not None:
            matrix[leaf_idx, group_idx] = 1
    return matrix


# Name -> (group names, leaf to group aggregation matrix) for the levels at which the
# leaf predictions are evaluated. The legacy level is the old flat categories.
EVAL_LEVELS: dict[str, tuple[tuple[str, ...], torch.Tensor]] = {
    "level1": (
        hierarchy.LEVEL1_NODES,
        _aggregation_matrix(
            list(hierarchy.LEAF_TO_LEVEL1), len(hierarchy.LEVEL1_NODES)
        ),
    ),
    "level2": (
        hierarchy.LEVEL2_NODES,
        _aggregation_matrix(
            list(hierarchy.LEAF_TO_LEVEL2), len(hierarchy.LEVEL2_NODES)
        ),
    ),
    "legacy": (
        hierarchy.LEGACY_CLASSES,
        _aggregation_matrix(
            [
                hierarchy.LEGACY_CLASSES.index(hierarchy.LEAF_TO_LEGACY[leaf])
                if leaf in hierarchy.LEAF_TO_LEGACY
                else None
                for leaf in hierarchy.LEAVES
            ],
            len(hierarchy.LEGACY_CLASSES),
        ),
    ),
}


def _stack_valid(
    preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack the predictions and targets of the valid examples.

    Returns:
        a tuple (preds, allowed, legacy_class) of the valid examples, with shapes
            (N, num_leaves), (N, num_leaves), and (N,).
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.stack(preds)
    allowed = torch.stack([target["allowed"] for target in targets])
    legacy_class = torch.stack([target["legacy_class"] for target in targets])
    valid = torch.stack([target["valid"] > 0 for target in targets])
    return preds[valid], allowed[valid] > 0, legacy_class[valid]


def _scores_at_level(
    level: str, preds: torch.Tensor, allowed: torch.Tensor, legacy_class: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the predicted scores and true class at an evaluation level.

    Only examples whose label determines the true class at that level are kept: at the
    leaf level, examples with a single allowed leaf; at Level 1/2, examples whose
    allowed leaves are all under one node; at the legacy level, examples with a legacy
    flat category.

    Args:
        level: "leaf" or one of EVAL_LEVELS.
        preds: (N, num_leaves) leaf probabilities.
        allowed: (N, num_leaves) boolean allowed leaves.
        legacy_class: (N,) legacy class index, or -1 if none.

    Returns:
        a tuple (scores, true class) with shapes (M, num_classes) and (M,).
    """
    if level == "leaf":
        determined = allowed.sum(dim=1) == 1
        return preds[determined], allowed[determined].float().argmax(dim=1)

    matrix = EVAL_LEVELS[level][1].to(preds.device)
    scores = preds @ matrix
    if level == "legacy":
        determined = legacy_class >= 0
        return scores[determined], legacy_class[determined]

    node_counts = allowed.float() @ matrix
    determined = (node_counts > 0).sum(dim=1) == 1
    return scores[determined], node_counts[determined].argmax(dim=1)


class HierarchicalAccuracy(Metric):
    """Accuracy of leaf predictions against labels that are sets of allowed leaves.

    Modes:
    - set: the predicted leaf is one of the allowed leaves.
    - leaf: exact accuracy on examples with a single allowed leaf.
    - level1, level2, legacy: the leaf probabilities are summed into the nodes (or
      legacy categories) at that level, and the top one is compared with the label on
      examples whose label determines that level.
    """

    def __init__(self, mode: str) -> None:
        """Create a new HierarchicalAccuracy.

        Args:
            mode: "set", "leaf", or one of EVAL_LEVELS.
        """
        super().__init__()
        if mode not in ("set", "leaf", *EVAL_LEVELS):
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self, preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update the metric with a batch of predictions and targets."""
        preds, allowed, legacy_class = _stack_valid(preds, targets)
        if self.mode == "set":
            pred_leaf = preds.argmax(dim=1, keepdim=True)
            self.correct += allowed.gather(1, pred_leaf).sum()
            self.total += len(preds)
            return
        scores, labels = _scores_at_level(self.mode, preds, allowed, legacy_class)
        self.correct += (scores.argmax(dim=1) == labels).sum()
        self.total += len(labels)

    def compute(self) -> torch.Tensor:
        """Compute the accuracy."""
        return self.correct / self.total.clamp(min=1)


class HierarchicalConfusionMatrix(Metric):
    """Confusion matrix of leaf predictions at a level of the hierarchy.

    The examples are selected as in HierarchicalAccuracy.
    """

    def __init__(self, mode: str) -> None:
        """Create a new HierarchicalConfusionMatrix.

        Args:
            mode: "leaf" or one of EVAL_LEVELS.
        """
        super().__init__()
        if mode == "leaf":
            class_names = hierarchy.LEAVES
        elif mode in EVAL_LEVELS:
            class_names = EVAL_LEVELS[mode][0]
        else:
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.confusion_matrix = ConfusionMatrixMetric(
            num_classes=len(class_names), class_names=list(class_names)
        )

    def update(
        self, preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update the metric with a batch of predictions and targets."""
        preds, allowed, legacy_class = _stack_valid(preds, targets)
        scores, labels = _scores_at_level(self.mode, preds, allowed, legacy_class)
        self.confusion_matrix.update(scores, labels)

    def compute(self) -> Any:
        """Compute the confusion matrix."""
        return self.confusion_matrix.compute()

    def reset(self) -> None:
        """Reset the metric."""
        super().reset()
        self.confusion_matrix.reset()


class HierarchicalForestLossTask(ClassificationTask):
    """Classification over the leaves of the forest loss driver label hierarchy.

    The label feature provides the allowed leaves as a list (written by
    rslp.forest_loss_driver.create_dataset). It is converted to these targets:
    - allowed: multi-hot (num_leaves,) tensor of the allowed leaves.
    - class: the leaf index if there is a single allowed leaf, otherwise -1.
    - legacy_class: index of the legacy flat category in LEGACY_CLASSES, or -1.
    - valid: whether the example has a label.

    Labels that allow every leaf carry no information and are treated as invalid.
    """

    def __init__(
        self,
        property_name: str = "leaf_labels",
        classes: list[str] | None = None,
        legacy_property_name: str = "new_label",
        output_property_name: str = "new_label",
        **kwargs: Any,
    ) -> None:
        """Create a new HierarchicalForestLossTask.

        Args:
            property_name: the label feature property containing the list of allowed
                leaves.
            classes: the leaves, which must match hierarchy.LEAVES (defaults to it).
            legacy_property_name: the label feature property containing the legacy
                flat category, used for the legacy metrics.
            output_property_name: the property to write the predicted leaf under.
            kwargs: other arguments to pass to ClassificationTask.
        """
        if classes is None:
            classes = list(hierarchy.LEAVES)
        if list(classes) != list(hierarchy.LEAVES):
            raise ValueError("classes must match hierarchy.LEAVES")
        super().__init__(property_name=property_name, classes=classes, **kwargs)
        self.legacy_property_name = legacy_property_name
        self.output_property_name = output_property_name

    def process_inputs(
        self,
        raw_inputs: dict[str, Any],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert the label feature into the targets."""
        if not load_targets:
            return {}, {}

        for feat in raw_inputs["targets"]:
            properties = feat.properties or {}
            leaves = properties.get(self.property_name)
            if not leaves:
                continue
            unknown = set(leaves) - set(self.classes)
            if unknown:
                raise ValueError(
                    f"unknown leaves {unknown} in window {metadata.window_name}"
                )
            if len(set(leaves)) == len(self.classes):
                continue

            allowed = torch.zeros(len(self.classes), dtype=torch.float32)
            for leaf in leaves:
                allowed[self.classes.index(leaf)] = 1
            class_id = self.classes.index(leaves[0]) if len(set(leaves)) == 1 else -1
            legacy = properties.get(self.legacy_property_name)
            legacy_class = (
                hierarchy.LEGACY_CLASSES.index(legacy)
                if legacy in hierarchy.LEGACY_CLASSES
                else -1
            )
            return {}, {
                "class": torch.tensor(class_id, dtype=torch.int64),
                "allowed": allowed,
                "legacy_class": torch.tensor(legacy_class, dtype=torch.int64),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise ValueError(f"no leaf labels found for window {metadata.window_name}")
        return {}, {
            "class": torch.tensor(-1, dtype=torch.int64),
            "allowed": torch.zeros(len(self.classes), dtype=torch.float32),
            "legacy_class": torch.tensor(-1, dtype=torch.int64),
            "valid": torch.tensor(0, dtype=torch.float32),
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> list[Feature]:
        """Write the predicted leaf along with the Level 1/2 and legacy categories.

        The Level 1/2 nodes and legacy category are the ones with the highest summed
        leaf probability, so they can differ from the nodes containing the top leaf.
        """
        if not isinstance(raw_output, torch.Tensor) or len(raw_output.shape) != 1:
            raise ValueError("expected output to be a Tensor with one dimension")
        probs = raw_output.detach().float().cpu()
        properties: dict[str, Any] = {
            self.output_property_name: self.classes[probs.argmax().item()],
        }
        for level, (names, matrix) in EVAL_LEVELS.items():
            level_name = "legacy_category" if level == "legacy" else level
            properties[level_name] = names[(probs @ matrix).argmax().item()]
        if self.prob_property is not None:
            properties[self.prob_property] = probs.tolist()
        feature = Feature(
            STGeometry(
                metadata.projection,
                shapely.Point(metadata.crop_bounds[0], metadata.crop_bounds[1]),
                None,
            ),
            properties,
        )
        return [feature]

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the image with the allowed leaves and predicted leaf."""
        image = super(ClassificationTask, self).visualize(
            input_dict, target_dict, output
        )["image"]
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        if target_dict is None:
            raise ValueError("target_dict is required for visualization")
        allowed = {
            leaf
            for leaf, flag in zip(self.classes, target_dict["allowed"].tolist())
            if flag > 0
        }
        label = " + ".join(hierarchy.covering_nodes(allowed))
        text = f"Label: {label}\nOutput: {self.classes[output.argmax()]}"
        box = draw.textbbox(xy=(0, 0), text=text, font_size=12)
        draw.rectangle(xy=box, fill=(0, 0, 0))
        draw.text(xy=(0, 0), text=text, font_size=12, fill=(255, 255, 255))
        return {"image": np.array(image)}

    def get_metrics(self) -> MetricCollection:
        """Get the set, leaf, level, and legacy accuracies and confusion matrices."""
        metrics: dict[str, Metric] = {
            f"{mode}_accuracy": HierarchicalAccuracy(mode)
            for mode in ("set", "leaf", *EVAL_LEVELS)
        }
        if self.enable_confusion_matrix:
            for mode in ("leaf", "level2"):
                metrics[f"{mode}_confusion_matrix"] = HierarchicalConfusionMatrix(mode)
        # Compute groups would merge metrics whose states happen to match after the
        # first batch, but these metrics only differ in which examples they count.
        return MetricCollection(metrics, compute_groups=False)


class HierarchicalClassificationHead(Predictor):
    """Classification head with a marginal likelihood loss over the allowed leaves.

    The loss is -log(sum of the probabilities of the allowed leaves), which reduces to
    cross entropy when a single leaf is allowed.
    """

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute the leaf probabilities and the loss.

        Args:
            intermediates: a FeatureVector with (BatchSize, NumLeaves) logits.
            context: the model context.
            targets: target dicts from HierarchicalForestLossTask.

        Returns:
            the leaf probabilities and loss dict.
        """
        if not isinstance(intermediates, FeatureVector):
            raise ValueError(
                "the input to HierarchicalClassificationHead must be a FeatureVector"
            )
        logits = intermediates.feature_vector
        outputs = torch.nn.functional.softmax(logits, dim=1)

        losses = {}
        if targets:
            allowed = torch.stack([target["allowed"] for target in targets]) > 0
            valid = torch.stack([target["valid"] for target in targets])
            # Invalid examples have no allowed leaves; allow all of them so their
            # (masked out) loss stays finite.
            allowed = allowed | (valid[:, None] <= 0)
            allowed_logits = logits.masked_fill(~allowed, float("-inf"))
            loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(
                allowed_logits, dim=1
            )
            losses["cls"] = torch.mean(loss * valid)

        return ModelOutput(outputs=outputs, loss_dict=losses)
