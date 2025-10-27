"""Evaluation adapter for OlmoEarth."""

from typing import Any

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.module_wrapper import EncoderModuleWrapper
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.normalize import Normalize
from torch import nn

from rslp.nandi.train import SegmentationPoolingDecoder


class Identity(nn.Identity):
    """Identity which takes two arguments."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize an Identity."""
        super().__init__()

    def forward(self, features: list[torch.Tensor], inputs: Any) -> list[torch.Tensor]:
        """Compute flat output vector from multi-scale feature map.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            unchanged features
        """
        return features


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate AEF model, which mirrors the OlmoEarth model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[4, 768]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                    num_channels={4: 512, 2: 256, 1: 128},
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                SegmentationPoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                FasterRCNN(
                    downsample_factors=[4],
                    num_channels=768,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                )
            ]
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                ClassificationHead(),
            ]
        )
    elif task_type == "regress":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                RegressionHead(),
            ]
        )
    else:
        raise NotImplementedError

    return MultiTaskModel(
        encoder=[EncoderModuleWrapper(module=Identity())],
        decoders=decoders,
    )


def get_transform(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate OlmoEarth transform."""
    modules: list[torch.nn.Module] = []

    if (len(input_modalities) != 1) or (input_modalities[0] != "gse"):
        raise ValueError("GSE model only works with gse input")

    modules.append(Normalize(mean=0, std=16384))

    return Sequential(*modules)
