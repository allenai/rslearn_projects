"""Evaluation adapter for OlmoEarth."""

import os

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.feature_center_crop import FeatureCenterCrop
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.select_bands import SelectBands

from rslp.nandi.train import SegmentationPoolingDecoder

from .constants import LANDSAT_BANDS, SENTINEL1_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate OlmoEarth model."""
    model_id = os.environ["EVAL_ADAPTER_MODEL_ID"]
    if model_id == "olmoearth":
        embedding_size = 768
        checkpoint_path = "/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200"
    elif model_id == "olmoearth_tiny":
        embedding_size = 192
        checkpoint_path = "/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000"
    elif model_id == "olmoearth_nano":
        embedding_size = 128
        checkpoint_path = "/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000"
    else:
        raise ValueError(f"unknown olmoearth model ID {model_id}")

    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[4, embedding_size]],
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
                    in_channels=embedding_size,
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
                    num_channels=embedding_size,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                )
            ]
        )
    elif task_type == "classify":
        if task_name == "nandi":
            decoders = dict(
                eval_task=[
                    FeatureCenterCrop(
                        sizes=[[1, 1]],
                    ),
                    PoolingDecoder(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                    ),
                    ClassificationHead(),
                ]
            )
        else:
            decoders = dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=embedding_size,
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
                    in_channels=embedding_size,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                RegressionHead(),
            ]
        )
    else:
        raise NotImplementedError

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=OlmoEarth(
                        checkpoint_path=checkpoint_path,
                        selector=["encoder"],
                        patch_size=4,
                        embedding_size=embedding_size,
                    ),
                    image_channels=12 * 4,
                    image_key="sentinel2_l2a",
                    groups=[[0], [1]],
                ),
            ],
            decoders=dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=embedding_size * 2,
                        out_channels=task_channels,
                        num_conv_layers=1,
                        num_fc_layers=1,
                    ),
                    ClassificationHead(),
                ]
            ),
        )

    return MultiTaskModel(
        encoder=[
            OlmoEarth(
                checkpoint_path=checkpoint_path,
                selector=["encoder"],
                patch_size=4,
                embedding_size=embedding_size,
            ),
        ],
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
    band_names: dict = {}

    if "sentinel2" in input_modalities:
        wanted_sentinel2 = [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ]
        sentinel2_indexes = [SENTINEL2_BANDS.index(band) for band in wanted_sentinel2]
        modules.append(
            SelectBands(
                band_indices=sentinel2_indexes,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="sentinel2_l2a",
            ),
        )
        band_names["sentinel2_l2a"] = wanted_sentinel2

    if "landsat" in input_modalities:
        wanted_landsat = [
            "B8",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B9",
            "B10",
            "B11",
        ]
        landsat_indexes = [LANDSAT_BANDS.index(band) for band in wanted_landsat]
        modules.append(
            SelectBands(
                band_indices=landsat_indexes,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="landsat",
            ),
        )
        band_names["landsat"] = wanted_landsat

    if "sentinel1" in input_modalities:
        wanted_sentinel1 = ["vv", "vh"]
        sentinel1_indexes = [SENTINEL1_BANDS.index(band) for band in wanted_sentinel1]
        modules.append(
            SelectBands(
                band_indices=sentinel1_indexes,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="sentinel1",
            ),
        )
        band_names["sentinel1"] = wanted_sentinel1

    modules.append(
        OlmoEarthNormalize(
            band_names=band_names,
        )
    )

    return Sequential(*modules)
