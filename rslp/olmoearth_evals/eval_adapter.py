"""Adapter for evaluation tasks."""

import os
from typing import Any

import torch
from rslearn.models.embedding_cache import EmbeddingCache
from rslearn.models.multitask import MultiTaskModel
from rslearn.train.model_context import ModelContext, ModelOutput
from rslearn.train.transforms.crop import Crop
from rslearn.train.transforms.pad import Pad
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

# Task key used in MultiTask for eval configs; target rasters live under target/<key>/...
EVAL_TASK_KEY = "eval_task"


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
        use_embeddings: bool = False,
    ):
        """Create a new EvalAdapterModel.

        Args:
            input_size: height and width of the input in pixels.
            input_modalities: subset of ["sentinel2", "sentinel1", "landsat"].
            task_type: "segment", "segment_small", "regress", "per_pixel_regress",
                "classify", or "detect".
            task_name: the name of the task, like "pastis".
            task_channels: how many output channels there are. For example, this is the
                number of classes for segmentation and detection tasks. For regression,
                it should be 1 which is also the default value.
            task_timesteps: number of input timesteps.
            use_embeddings: add an EmbeddingsCache after the encoder so that the
                does not get gradients (even if its unfrozen) and the embeddings are
                cached for faster training (although data loading will still happen).
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

        if use_embeddings:
            assert isinstance(self.model, MultiTaskModel)
            self.model.encoder = torch.nn.ModuleList(
                [EmbeddingCache(encoder=list(self.model.encoder))]
            )

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Apply the sequence of modules on the inputs, including shared trunk.

        Args:
            context: the model context.
            targets: optional list of target dicts

        Returns:
            ModelOutput with outputs and loss_dict.
        """
        return self.model(context, targets)


class EvalAdapterNormalize(Transform):
    """Normalization for evaluation tasks.

    Optionally applies pad_to and crop_to transforms before the model-specific
    normalization. These are configured via the EVAL_ADAPTER_PAD_TO and
    EVAL_ADAPTER_CROP_TO environment variables.
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
        """Create a new EvalAdapterNormalize.

        Args:
            input_size: height and width of the input in pixels.
            input_modalities: subset of ["sentinel2", "sentinel1", "landsat"].
            task_type: "segment", "segment_small", "regress", "per_pixel_regress",
                "classify", or "detect".
            task_name: the name of the task, like "pastis".
            task_channels: how many output channels there are. For example, this is the
                number of classes for segmentation and detection tasks. For regression,
                it should be 1 which is also the default value.
            task_timesteps: number of input timesteps.
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

        # Read pad_to and crop_to from env vars.
        pad_to: int | None = None
        pad_to_env = os.environ.get("EVAL_ADAPTER_PAD_TO")
        if pad_to_env:
            pad_to = int(pad_to_env)

        crop_to_size: int | None = None
        crop_to_offset: tuple[int, int] | None = None
        crop_to_env = os.environ.get("EVAL_ADAPTER_CROP_TO")
        if crop_to_env:
            parts = [int(x) for x in crop_to_env.split(",")]
            if parts[3] - parts[1] != parts[2] - parts[0]:
                raise ValueError(
                    f"EVAL_ADAPTER_CROP_TO must specify a square region, "
                    f"got width={parts[2] - parts[0]} height={parts[3] - parts[1]}"
                )
            crop_to_size = parts[2] - parts[0]
            crop_to_offset = (parts[0], parts[1])

        # Build selectors to transform: input modalities + target rasters when present.
        # For segment/segment_small: target/eval_task/classes, target/eval_task/valid.
        # For per_pixel_regress: target/eval_task/values, target/eval_task/valid.
        # classify/regress have no target rasters (targets are vectors or scalars).
        image_selectors = list(input_modalities)
        if pad_to is not None or crop_to_size is not None:
            if task_type in ("segment", "segment_small"):
                image_selectors.extend(
                    [
                        f"target/{EVAL_TASK_KEY}/classes",
                        f"target/{EVAL_TASK_KEY}/valid",
                    ]
                )
            elif task_type == "per_pixel_regress":
                image_selectors.extend(
                    [
                        f"target/{EVAL_TASK_KEY}/values",
                        f"target/{EVAL_TASK_KEY}/valid",
                    ]
                )
            elif task_type not in ("classify", "regress"):
                raise ValueError(
                    f"pad_to and crop_to require task_type one of classify, regress, "
                    f"segment, segment_small, per_pixel_regress; got task_type={task_type!r}"
                )

        self.pad_transform: torch.nn.Module | None = None
        if pad_to is not None:
            self.pad_transform = Pad(
                size=pad_to, mode="center", image_selectors=image_selectors
            )

        self.crop_transform: torch.nn.Module | None = None
        if crop_to_size is not None:
            self.crop_transform = Crop(
                crop_size=crop_to_size,
                offset=crop_to_offset,
                image_selectors=image_selectors,
            )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply pad_to, crop_to, then model-specific normalization."""
        if self.pad_transform:
            input_dict, target_dict = self.pad_transform(input_dict, target_dict)
        if self.crop_transform:
            input_dict, target_dict = self.crop_transform(input_dict, target_dict)
        return self.transform(input_dict, target_dict)
