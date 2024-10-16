"""Segmentation confusion matrix."""

import os
from typing import Any

import numpy as np
import wandb
from PIL import Image
from rslearn.train.lightning_module import RslearnLightningModule

from .config import CATEGORIES


class CMLightningModule(RslearnLightningModule):
    """Lightning module extended with test segmentation confusion matrix."""

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
                        f'{metadata["window_name"]}_{metadata["bounds"][0]}_{metadata["bounds"][1]}_{image_suffix}.png',
                    )
                    Image.fromarray(image).save(out_fname)

        # Now we hook in part to compute confusion matrix.
        for output, target in zip(outputs, targets):
            # cur_probs is CxN array of valid probabilities, N=H*W.
            cur_probs = output["segment"][:, target["segment"]["valid"] > 0]
            # cur_labels is N array of labels.
            cur_labels = target["segment"]["classes"][target["segment"]["valid"] > 0]
            # Make sure probs is list of NxC arrays.
            self.probs.append(cur_probs.cpu().numpy().transpose(1, 0))
            self.y_true.append(cur_labels.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        """Submit the test confusion matrix."""
        self.logger.experiment.log(
            {
                "test_cm": wandb.plot.confusion_matrix(
                    probs=np.concatenate(self.probs, axis=0),
                    y_true=np.concatenate(self.y_true, axis=0),
                    class_names=CATEGORIES,
                )
            }
        )
