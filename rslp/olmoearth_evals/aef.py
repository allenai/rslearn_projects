"""Evaluation adapter for OlmoEarth."""

from typing import Any

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.module_wrapper import EncoderModuleWrapper
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
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


class ConvSameShapeDecoder(nn.Module):
    """Multiple Conv layers.

    A UNet expects upsampling, but the GSE embeddings have the same
    resolutions as the tasks. To remain comparable to the UNet, we stack
    consecutive convolution layers.
    """

    def __init__(
        self,
        num_channels: list[int],
        in_channels: int,
        out_channels: int | None,
        kernel_size: int = 3,
    ):
        """Multiple conv layers."""
        super().__init__()
        num_channels.insert(0, in_channels)
        if out_channels is not None:
            num_channels.append(out_channels)

        layers = []
        for i in range(len(num_channels) - 1):
            layers.extend(
                [
                    torch.nn.Conv2d(
                        in_channels=num_channels[i],
                        out_channels=num_channels[i + 1],
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    torch.nn.ReLU(inplace=True),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(
        self, in_features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Compute output from multi-scale feature map.

        Args:
            in_features: list of feature maps.
            inputs: original inputs (ignored).

        Returns:
            output image
        """
        # Reverse the features since we will pass them in from lowest resolution to highest.
        if len(in_features) != 1:
            raise ValueError("Expecting a single GSE layer.")
        return [self.layers(in_features[0])]


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
                ConvSameShapeDecoder(
                    in_channels=64,
                    out_channels=task_channels,
                    # keep the channels constant?
                    num_channels=[64, 64, 64, 64, 64],
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                SegmentationPoolingDecoder(
                    in_channels=64,
                    out_channels=task_channels,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                FasterRCNN(
                    downsample_factors=[1],
                    num_channels=64,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                )
            ]
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=64,
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
                    in_channels=64,
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

    if (len(input_modalities) != 1) or ("gse" not in input_modalities[0]):
        # we use '"gse" not in' instead of == because
        # the layer name can be a slight variant of gse, e.g. 'gsegood'
        raise ValueError("GSE model only works with gse input")

    modules.append(Normalize(mean=0, std=16384))

    return Sequential(*modules)
