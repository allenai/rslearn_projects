"""Segmentation confusion matrix."""

from typing import Any

import numpy as np
import wandb
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import ModelContext, ModelOutput
from typing_extensions import override

from .config import CATEGORIES


class CMLightningModule(RslearnLightningModule):
    """Lightning module extended with test segmentation confusion matrix."""

    @override
    def on_test_epoch_start(self) -> None:
        """Initialize test confusion matrix."""
        self.probs: list = []
        self.y_true: list = []

    @override
    def on_test_forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]],
        model_outputs: ModelOutput,
    ) -> None:
        """Save the probabilities and labels for confusion matrix logging."""
        for output, target in zip(model_outputs.outputs, targets):
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
