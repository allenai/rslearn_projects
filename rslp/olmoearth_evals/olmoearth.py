"""Evaluation adapter for OlmoEarth."""

import json
import os

import torch
from rslearn.models.conv import Conv
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.feature_center_crop import FeatureCenterCrop
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.olmoearth_pretrain.model import EMBEDDING_SIZES, ModelID, OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.models.upsample import Upsample
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.select_bands import SelectBands

from rslp.log_utils import get_logger
from rslp.nandi.train import SegmentationPoolingDecoder

from .constants import LANDSAT_BANDS, SENTINEL1_BANDS, SENTINEL2_BANDS

logger = get_logger(__name__)


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
    model_config_env = os.environ.get("EVAL_ADAPTER_MODEL_CONFIG")
    model_config: dict[str, str] = (
        json.loads(model_config_env) if model_config_env else {}
    )
    if model_id in ["olmoearth", "olmoearth_random"]:
        olmoearth_model_id = ModelID.OLMOEARTH_V1_BASE
    elif model_id == "olmoearth_nano":
        olmoearth_model_id = ModelID.OLMOEARTH_V1_NANO
    elif model_id == "olmoearth_tiny":
        olmoearth_model_id = ModelID.OLMOEARTH_V1_TINY
    elif model_id == "olmoearth_large":
        olmoearth_model_id = ModelID.OLMOEARTH_V1_LARGE
    else:
        raise ValueError(f"unknown olmoearth model ID {model_id}")

    embedding_size = EMBEDDING_SIZES[olmoearth_model_id]
    decoder_type = model_config.get("decoder", "default")
    logger.info(
        f"olmoearth: using decoder_type={decoder_type} embedding_size={embedding_size}"
    )

    if task_type == "segment":
        if decoder_type == "singleconv":
            decoders = dict(
                eval_task=[
                    Upsample(scale_factor=4),
                    Conv(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                        kernel_size=1,
                        activation=torch.nn.Identity(),
                    ),
                    SegmentationHead(),
                ]
            )
        else:
            decoders = dict(
                eval_task=[
                    UNetDecoder(
                        in_channels=[[4, embedding_size]],
                        out_channels=task_channels,
                        conv_layers_per_resolution=1,
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
                        model_id=olmoearth_model_id,
                        patch_size=4,
                        random_initialization=model_id == "olmoearth_random",
                        use_legacy_timestamps=False,
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
                model_id=olmoearth_model_id,
                patch_size=4,
                random_initialization=model_id == "olmoearth_random",
                use_legacy_timestamps=False,
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
