"""Remaps categories for Amazon Conservation task to consolidated category set."""

import os
from typing import Any

import numpy as np
import torch
import wandb
from PIL import Image
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils import Feature

CATEGORIES = [
    "agriculture",
    "mining",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
    "none",
]

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

        This is modified to do category remapping, and also mark category invalid if it
        is not in CATEGORIES.

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
            if class_name not in CATEGORIES:
                continue
            class_id = CATEGORIES.index(class_name)

            return {}, {
                "class": torch.tensor(class_id, dtype=torch.int64),
                "valid": torch.tensor(1, dtype=torch.float32),
            }

        if not self.allow_invalid:
            raise Exception("no feature found providing class label")

        return {}, {
            "class": torch.tensor(0, dtype=torch.int64),
            "valid": torch.tensor(0, dtype=torch.float32),
        }


class ForestLossLightningModule(RslearnLightningModule):
    """Lightning module extended with val / test confusion matrix reporting."""

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Customized training step.

        Args:
            batch: the batch contents.
            batch_idx: the batch index.
            dataloader_idx: the index of the dataloader creating this batch.
        """
        # Copied over from RslearnLightningModule.training_step.
        inputs, targets, _ = batch
        batch_size = len(inputs)
        _, loss_dict = self(inputs, targets)
        train_loss = sum(loss_dict.values())
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss",
            train_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        # Temporary stuff to visualize the inputs.
        if False:
            for image_idx, inp in enumerate(inputs):
                target = targets[image_idx]["class"]
                if not target["valid"]:
                    continue
                category = CATEGORIES[target["class"]]
                pre_im = np.clip(
                    inp["image"][0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    * 255,
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(pre_im).save(
                    f"/home/favyenb/vis/vis/inp_{batch_idx}_{image_idx}_{category}_pre.png"
                )
                post_im = np.clip(
                    inp["image"][9:12, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    * 255,
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(post_im).save(
                    f"/home/favyenb/vis/vis/inp_{batch_idx}_{image_idx}_{category}_post.png"
                )

        return train_loss

    def on_validation_epoch_start(self) -> None:
        """Initialize val confusion matrix."""
        self.probs: list = []
        self.y_true: list = []

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute val performance and also record for confusion matrix.

        Args:
            batch: the batch contents.
            batch_idx: the batch index.
            dataloader_idx: the index of the dataloader creating this batch.
        """
        # Code below is copied from RslearnLightningModule.validation_step.
        inputs, targets, _ = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        val_loss = sum(loss_dict.values())
        self.log_dict(
            {"val_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, batch_size=batch_size, on_epoch=True)

        # Now we hook in part to compute confusion matrix.
        for output, target in zip(outputs, targets):
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
                    class_names=CATEGORIES,
                )
            }
        )

    def on_test_epoch_start(self) -> None:
        """Initialize test confusion matrix."""
        self.probs = []
        self.y_true = []

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute test performance and also record for confusion matrix.

        Args:
            batch: the batch contents.
            batch_idx: the batch index.
            dataloader_idx: the index of the dataloader creating this batch.
        """
        # Code below is copied from RslearnLightningModule.test_step.
        inputs, targets, metadatas = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        test_loss = sum(loss_dict.values())
        self.log_dict(
            {"test_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_loss", test_loss, batch_size=batch_size, on_step=False, on_epoch=True
        )
        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size, on_epoch=True)

        if self.visualize_dir:
            for idx, (inp, target, output, metadata) in enumerate(
                zip(inputs, targets, outputs, metadatas)
            ):
                images = self.task.visualize(inp, target, output)
                for image_suffix, image in images.items():
                    out_fname = os.path.join(
                        self.visualize_dir,
                        f"{metadata['window_name']}_{metadata['bounds'][0]}_{metadata['bounds'][1]}_{image_suffix}.png",
                    )
                    Image.fromarray(image).save(out_fname)

        # Now we hook in part to compute confusion matrix.
        for output, target in zip(outputs, targets):
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
                    class_names=CATEGORIES,
                )
            }
        )
