from typing import Any

import numpy as np
import torch
from rslearn.train.model_context import ModelContext, ModelOutput
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.segmentation import SegmentationTask

from rslp.maldives_ecosystem_mapping.train import CMLightningModule


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, context: ModelContext, targets: list[dict[str, Any]] | None = None
    ) -> ModelOutput:
        return ModelOutput(outputs=context.inputs, loss_dict={"loss": 0})


def test_cm_module_step() -> None:
    # Make sure that test_step creates correct confusion matrix.
    model = TestModel()
    task = MultiTask(
        tasks={
            "segment": SegmentationTask(num_classes=2),
        },
        input_mapping=None,
    )
    lm = CMLightningModule(model, task)

    # Three examples:
    # 1. Valid, target is cls0, output is cls0.
    # 2. Valid, target is cls1, output is cls1.
    # 3. Invalid, target is cls0, output is cls1.
    # So the confusion matrix should be perfect since invalid target is ignored.
    targets = [
        {
            "segment": {
                "classes": torch.zeros((8, 8), dtype=torch.int32),
                "valid": torch.ones((8, 8), dtype=torch.int32),
            }
        },
        {
            "segment": {
                "classes": torch.ones((8, 8), dtype=torch.int32),
                "valid": torch.ones((8, 8), dtype=torch.int32),
            }
        },
        {
            "segment": {
                "classes": torch.zeros((8, 8), dtype=torch.int32),
                "valid": torch.zeros((8, 8), dtype=torch.int32),
            }
        },
    ]
    output1 = torch.zeros((2, 8, 8), dtype=torch.float32)
    output2 = torch.zeros((2, 8, 8), dtype=torch.float32)
    output3 = torch.zeros((2, 8, 8), dtype=torch.float32)
    output1[:, 0] = 1
    output2[:, 1] = 1
    output3[:, 1] = 1
    outputs = [
        {
            "segment": output1,
        },
        {
            "segment": output2,
        },
        {
            "segment": output3,
        },
    ]
    batch: tuple = (outputs, targets, [])

    lm.on_test_epoch_start()
    lm.test_step(batch, 0, 0)
    # Can't seem to monkeypatch lm.logger, so instead just make sure the probs is the
    # right shape.
    probs = np.concatenate(lm.probs, axis=0)
    assert probs.shape == (64 * 2, 2)
