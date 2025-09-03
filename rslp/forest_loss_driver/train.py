"""Remaps categories for Amazon Conservation task to consolidated category set."""

from typing import Any

import numpy as np
import torch
import wandb
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils import Feature

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
