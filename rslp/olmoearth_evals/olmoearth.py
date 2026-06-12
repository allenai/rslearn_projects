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
from rslearn.models.reshape_feature_maps import ReshapeFeatureMaps
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

# Patch size used by the OlmoEarth encoder. Encoder output is downsampled by
# this factor, so decoders must upsample/reshape by the same factor to recover
# pixel resolution.
PATCH_SIZE = 4


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate OlmoEarth model.

    Args:
        input_size: height and width of the input in pixels.
        input_modalities: subset of ["sentinel2", "sentinel1", "landsat"].
        task_type: the task type string.
        task_name: the name of the task.
        task_channels: number of output channels.
        task_timesteps: number of input timesteps.
    """
    model_id = os.environ["EVAL_ADAPTER_MODEL_ID"]

    # Extract various customizable options from the model config, if provided.
    model_config_env = os.environ.get("EVAL_ADAPTER_MODEL_CONFIG")
    model_config: dict[str, str] = (
        json.loads(model_config_env) if model_config_env else {}
    )
    decoder_type = model_config.get("decoder", "default")
    checkpoint_path = model_config.get("checkpoint_path")
    use_legacy_timestamps = (
        model_config.get("use_legacy_timestamps", "true").lower() == "true"
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
    logger.info(
        f"olmoearth: using decoder_type={decoder_type} embedding_size={embedding_size}"
        f" checkpoint_path={checkpoint_path} use_legacy_timestamps={use_legacy_timestamps}"
    )

    def _make_encoder() -> OlmoEarth:
        if checkpoint_path is not None:
            return OlmoEarth(
                checkpoint_path=checkpoint_path,
                patch_size=PATCH_SIZE,
                use_legacy_timestamps=use_legacy_timestamps,
            )
        return OlmoEarth(
            model_id=olmoearth_model_id,
            patch_size=PATCH_SIZE,
            random_initialization=model_id == "olmoearth_random",
            use_legacy_timestamps=use_legacy_timestamps,
        )

    if task_type == "segment":
        if decoder_type == "singleconv":
            decoders = dict(
                eval_task=[
                    Upsample(scale_factor=PATCH_SIZE),
                    Conv(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                        kernel_size=1,
                        activation=torch.nn.Identity(),
                    ),
                    SegmentationHead(),
                ]
            )
        elif decoder_type == "singleconv_reshape":
            # Mirrors olmoearth_pretrain's per-patch linear probe: project each
            # patch embedding to (task_channels * PATCH_SIZE * PATCH_SIZE) logits,
            # then depth-to-space reshape up to pixel resolution.
            decoders = dict(
                eval_task=[
                    Conv(
                        in_channels=embedding_size,
                        out_channels=task_channels * PATCH_SIZE * PATCH_SIZE,
                        kernel_size=1,
                        activation=torch.nn.Identity(),
                    ),
                    ReshapeFeatureMaps(upscale_factor=PATCH_SIZE),
                    SegmentationHead(),
                ]
            )
        elif decoder_type == "unet_with_batchnorm":
            decoders = dict(
                eval_task=[
                    UNetDecoder(
                        in_channels=[[PATCH_SIZE, embedding_size]],
                        out_channels=task_channels,
                        conv_layers_per_resolution=2,
                        num_channels={4: 256, 2: 128, 1: 64},
                        use_batch_norm=True,
                    ),
                    SegmentationHead(),
                ]
            )
        elif decoder_type == "default":
            decoders = dict(
                eval_task=[
                    UNetDecoder(
                        in_channels=[[PATCH_SIZE, embedding_size]],
                        out_channels=task_channels,
                        conv_layers_per_resolution=1,
                        num_channels={4: 512, 2: 256, 1: 128},
                    ),
                    SegmentationHead(),
                ]
            )
        else:
            raise ValueError(f"invalid decoder type {decoder_type}")
    elif task_type == "segment_small":
        if decoder_type == "default":
            decoders = dict(
                eval_task=[
                    SegmentationPoolingDecoder(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                    ),
                    SegmentationHead(),
                ]
            )
        else:
            raise ValueError(f"invalid decoder type {decoder_type}")
    elif task_type == "detect":
        if decoder_type == "default":
            decoders = dict(
                eval_task=[
                    FasterRCNN(
                        downsample_factors=[PATCH_SIZE],
                        num_channels=embedding_size,
                        num_classes=task_channels,
                        anchor_sizes=[[32]],
                    )
                ]
            )
        else:
            raise ValueError(f"invalid decoder type {decoder_type}")
    elif task_type == "classify":
        if decoder_type == "default":
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
        elif decoder_type == "center":
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
        elif decoder_type in ("singleconv", "singleconv_reshape"):
            # Linear-probe analog for classification: global max pool over the
            # patch grid followed by a single linear layer. singleconv_reshape
            # produces the same decoder since there is no spatial output to
            # reshape up to.
            decoders = dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                    ),
                    ClassificationHead(),
                ]
            )
        else:
            raise ValueError(f"invalid decoder type {decoder_type}")
    elif task_type == "regress":
        if decoder_type == "default":
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
        elif decoder_type == "center":
            decoders = dict(
                eval_task=[
                    FeatureCenterCrop(
                        sizes=[[1, 1]],
                    ),
                    PoolingDecoder(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                    ),
                    RegressionHead(),
                ]
            )
        elif decoder_type in ("singleconv", "singleconv_reshape"):
            # Linear-probe analog for regression: global max pool over the
            # patch grid followed by a single linear layer. singleconv_reshape
            # produces the same decoder since there is no spatial output to
            # reshape up to.
            decoders = dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=embedding_size,
                        out_channels=task_channels,
                    ),
                    RegressionHead(),
                ]
            )
        else:
            raise ValueError(f"invalid decoder type {decoder_type}")
    else:
        raise NotImplementedError

    return MultiTaskModel(
        encoder=[_make_encoder()],
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
