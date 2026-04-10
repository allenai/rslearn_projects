"""Integration tests for the Landsat vessel prediction pipeline."""

import json
import pathlib
import sys
import tempfile
from datetime import datetime
from typing import Any, BinaryIO

import pytest
import yaml
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_landsat import LandsatOliTirsItem
from rslearn.utils.geometry import STGeometry
from shapely.geometry import box
from upath import UPath

from rslp.landsat_vessels.config import (
    ATTRIBUTE_MODEL_CONFIG,
    AWS_DATASET_CONFIG,
    CLASSIFY_MODEL_CONFIG,
    DETECT_MODEL_CONFIG,
)
from rslp.landsat_vessels.predict_pipeline import predict_pipeline
from tests.integration.vessels.helpers import (
    WGS84_ITEM_BOUNDS,
    apply_common_config_patches,
    create_tiny_detect_config,
    get_projection_and_bounds,
    write_synthetic_geotiff,
)

FAKE_LANDSAT_SCENE_ID = "LC08_L1TP_162042_20241103_20241103_02_RT"

# __init__.py re-exports predict_pipeline as a function, shadowing the module name.
# We get the module directly here so that we can monkeypatch various attributes.
_landsat_mod = sys.modules["rslp.landsat_vessels.predict_pipeline"]

BAND_RESOLUTIONS = {
    "B1": 30,
    "B2": 30,
    "B3": 30,
    "B4": 30,
    "B5": 30,
    "B6": 30,
    "B7": 30,
    "B8": 15,
    "B9": 30,
    "B10": 30,
    "B11": 30,
}


# --- Fake S3 clients ---


class _FakeS3Bucket:
    """Mock for boto3 S3 Bucket used by LandsatOliTirs data source."""

    def download_file(
        self, key: str, fname: str, ExtraArgs: dict | None = None
    ) -> None:
        # .../FAKE_B1.TIF -> B1
        band = key.split("_")[-1].replace(".TIF", "")
        res = BAND_RESOLUTIONS[band]
        proj, bounds = get_projection_and_bounds(res, -res)
        fpath = UPath(fname)
        write_synthetic_geotiff(
            fpath.parent, proj, bounds, dtype="uint8", fname=fpath.name
        )

    def download_fileobj(
        self, key: str, f: BinaryIO, ExtraArgs: dict | None = None
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            self.download_file(key, tmp.name)
            with open(tmp.name, "rb") as src:
                f.write(src.read())

    class objects:
        @staticmethod
        def filter(**kwargs: Any) -> list:
            return []


class _FakeS3Resource:
    def Bucket(self, name: str) -> _FakeS3Bucket:
        return _FakeS3Bucket()


class _FakeS3Client:
    def generate_presigned_url(self, *args: Any, **kwargs: Any) -> str:
        return "https://fake-presigned-url"


def _fake_boto3_resource(service: str, **kwargs: Any) -> _FakeS3Resource:
    return _FakeS3Resource()


def _fake_boto3_client(service: str, **kwargs: Any) -> _FakeS3Client:
    return _FakeS3Client()


# --- Item cache + dataset config ---


def _populate_item_cache(cache_dir: pathlib.Path, scene_id: str) -> None:
    """Write a serialized LandsatOliTirsItem JSON to the metadata cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    geometry = STGeometry(
        projection=WGS84_PROJECTION,
        shp=box(*WGS84_ITEM_BOUNDS),
        time_range=(datetime(2024, 11, 3, 0, 0), datetime(2024, 11, 3, 0, 1)),
    )
    item = LandsatOliTirsItem(
        name=scene_id,
        geometry=geometry,
        blob_path="collection02/level-1/standard/oli-tirs/2024/162/042/FAKE/FAKE_",
        cloud_cover=10.0,
    )
    # The cache is organized by year/path/row so extract this information from the scene ID.
    # LC08_L1TP_162042_20241103_20241103_02_RT -> 2024_162_042
    parts = scene_id.split("_")
    year = parts[3][0:4]
    path = parts[2][0:3]
    row = parts[2][3:6]
    with open(cache_dir / f"{year}_{path}_{row}.json", "w") as f:
        json.dump([item.serialize()], f)


def _write_scene_id_dataset_config(
    original_config_path: str,
    output_path: str,
    metadata_cache_dir: str,
) -> None:
    """Write a modified predict_dataset_config_aws.json with custom cache dir."""
    with open(original_config_path) as f:
        cfg = json.load(f)
    cfg["layers"]["landsat"]["data_source"]["init_args"]["metadata_cache_dir"] = (
        metadata_cache_dir
    )
    cfg["layers"]["landsat_allbands"]["data_source"]["init_args"][
        "metadata_cache_dir"
    ] = metadata_cache_dir
    with open(output_path, "w") as f:
        json.dump(cfg, f)


# --- Tests ---


def _create_tiny_classifier_config(original_path: str, output_path: str) -> None:
    """Patch the Landsat classifier config to use swin_t."""
    with open(original_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg["model"]["init_args"]
    encoder_list = model_args["model"]["init_args"]["encoder"]
    encoder_list[0]["init_args"]["arch"] = "swin_t"
    encoder_list[0]["init_args"]["pretrained"] = False
    decoders = model_args["model"]["init_args"]["decoders"]
    for decoder_list in decoders.values():
        for decoder in decoder_list:
            if "PoolingDecoder" in decoder.get("class_path", ""):
                decoder["init_args"]["in_channels"] = 768

    apply_common_config_patches(cfg)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f)


def _create_tiny_attribute_config(original_path: str, output_path: str) -> None:
    """Patch the Landsat vessel attribute config to use OlmoEarth-v1-Nano."""
    with open(original_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg["model"]["init_args"]
    encoder = model_args["model"]["init_args"]["encoder"][0]
    encoder["init_args"]["model_id"] = "OLMOEARTH_V1_NANO"

    # Nano produces 128-dim embeddings vs Base's 768.
    decoders = model_args["model"]["init_args"]["decoders"]
    for decoder_list in decoders.values():
        for decoder in decoder_list:
            if "PoolingDecoder" in decoder.get("class_path", ""):
                decoder["init_args"]["in_channels"] = 128

    apply_common_config_patches(cfg)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f)


def _patch_model_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Create tiny model configs and monkeypatch the module constants."""
    detect_cfg = str(tmp_path / "detect_config.yaml")
    create_tiny_detect_config(DETECT_MODEL_CONFIG, detect_cfg)
    monkeypatch.setattr(_landsat_mod, "DETECT_MODEL_CONFIG", detect_cfg)

    classify_cfg = str(tmp_path / "classify_config.yaml")
    _create_tiny_classifier_config(CLASSIFY_MODEL_CONFIG, classify_cfg)
    monkeypatch.setattr(_landsat_mod, "CLASSIFY_MODEL_CONFIG", classify_cfg)

    attr_cfg = str(tmp_path / "attribute_config.yaml")
    _create_tiny_attribute_config(ATTRIBUTE_MODEL_CONFIG, attr_cfg)
    monkeypatch.setattr(_landsat_mod, "ATTRIBUTE_MODEL_CONFIG", attr_cfg)

    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path / "rslp"))


def test_predict_pipeline_local_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes end-to-end with synthetic local GeoTIFFs."""
    _patch_model_configs(monkeypatch, tmp_path)

    image_files_dir = UPath(tmp_path / "image_files")
    image_files: dict[str, str] = {}

    for band, res in BAND_RESOLUTIONS.items():
        proj, bounds = get_projection_and_bounds(res, -res)
        write_synthetic_geotiff(
            image_files_dir, proj, bounds, dtype="uint8", fname=f"{band}.tif"
        )
        image_files[band] = str(image_files_dir / f"{band}.tif")

    predict_pipeline(image_files=image_files, scratch_path=str(tmp_path / "scratch"))


def test_predict_pipeline_scene_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes using a fake scene ID with mocked S3 bucket."""
    _patch_model_configs(monkeypatch, tmp_path)

    cache_dir = tmp_path / "landsat_cache"
    _populate_item_cache(cache_dir, FAKE_LANDSAT_SCENE_ID)

    ds_cfg = str(tmp_path / "predict_dataset_config_aws.json")
    _write_scene_id_dataset_config(
        AWS_DATASET_CONFIG,
        ds_cfg,
        metadata_cache_dir=str(cache_dir),
    )
    monkeypatch.setattr(_landsat_mod, "AWS_DATASET_CONFIG", ds_cfg)

    monkeypatch.setattr("boto3.resource", _fake_boto3_resource)
    monkeypatch.setattr("boto3.client", _fake_boto3_client)

    predict_pipeline(
        scene_id=FAKE_LANDSAT_SCENE_ID, scratch_path=str(tmp_path / "scratch")
    )
