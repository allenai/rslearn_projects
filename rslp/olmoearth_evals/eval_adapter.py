"""Adapter for evaluation tasks."""

import os
from typing import Any

import torch
from rslearn.train.transforms.transform import Transform

import rslp.olmoearth_evals.aef as aef
import rslp.olmoearth_evals.anysat as anysat
import rslp.olmoearth_evals.clay as clay
import rslp.olmoearth_evals.croma as croma
import rslp.olmoearth_evals.dinov3 as dinov3
import rslp.olmoearth_evals.galileo as galileo
import rslp.olmoearth_evals.olmoearth as olmoearth
import rslp.olmoearth_evals.panopticon as panopticon
import rslp.olmoearth_evals.presto as presto
import rslp.olmoearth_evals.prithvi as prithvi
import rslp.olmoearth_evals.satlaspretrain as satlaspretrain
import rslp.olmoearth_evals.terramind as terramind

modules_by_model_id = {
    "anysat": anysat,
    "clay": clay,
    "croma": croma,
    "croma_large": croma,
    "dinov3": dinov3,
    "galileo": galileo,
    "olmoearth": olmoearth,
    "olmoearth_nano": olmoearth,
    "olmoearth_tiny": olmoearth,
    "olmoearth_large": olmoearth,
    "olmoearth_random": olmoearth,
    "panopticon": panopticon,
    "presto": presto,
    "prithvi": prithvi,
    "satlaspretrain": satlaspretrain,
    "terramind": terramind,
    "terramind_large": terramind,
    "aef": aef,
}


class EvalAdapterModel(torch.nn.Module):
    """Adapter model for evaluation tasks in the OlmoEarth model paper.

    This model provides a common interface to OlmoEarth and several baselines. It is
    only intended to be used for specific evaluation tasks, since it does not afford
    flexibility for fine-grained customization that the model config normally provides.

    It also has lots of hardcoded constants for checkpoints. It is really just for
    internal Ai2 use.
    """

    def __init__(
        self,
        input_size: int,
        input_modalities: list[str],
        task_type: str,
        task_name: str,
        task_channels: int = 1,
        task_timesteps: int = 1,
    ):
        """Create a new EvalAdapterModel.

        Args:
            input_size: height and width of the input in pixels.
            input_modalities: subset of ["sentinel2", "sentinel1", "landsat"].
            task_type: either "segment", "segment_small", "regress", or "detect".
            task_name: the name of the task, like "pastis".
            task_channels: how many output channels there are. For example, this is the
                number of classes for segmentation and detection tasks. For regression,
                it should be 1 which is also the default value.
            task_timesteps: number of input timesteps.
        """
        super().__init__()
        model_id = os.environ["EVAL_ADAPTER_MODEL_ID"]
        self.model = modules_by_model_id[model_id].get_model(
            input_size=input_size,
            input_modalities=input_modalities,
            task_type=task_type,
            task_name=task_name,
            task_channels=task_channels,
            task_timesteps=task_timesteps,
        )

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply the sequence of modules on the inputs, including shared trunk.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            dict with keys "outputs" and "loss_dict".
        """
        return self.model(inputs, targets)


class EvalAdapterNormalize(Transform):
    """Normalization for evaluation tasks."""

    def __init__(
        self,
        input_size: int,
        input_modalities: list[str],
        task_type: str,
        task_name: str,
        task_channels: int = 1,
        task_timesteps: int = 1,
    ):
        """Create a new EvalAdapterNormalize.

        This has the same arguments as EvalAdapterModel.
        """
        super().__init__()
        model_id = os.environ["EVAL_ADAPTER_MODEL_ID"]
        self.transform = modules_by_model_id[model_id].get_transform(
            input_size=input_size,
            input_modalities=input_modalities,
            task_type=task_type,
            task_name=task_name,
            task_channels=task_channels,
            task_timesteps=task_timesteps,
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Normalize the input_dict."""
        return self.transform(input_dict, target_dict)
