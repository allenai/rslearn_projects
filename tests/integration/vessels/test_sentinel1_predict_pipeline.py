"""Integration tests for the Sentinel-1 vessel prediction pipeline."""

import json
import pathlib
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Any, BinaryIO

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.copernicus import CopernicusItem
from rslearn.utils.geometry import STGeometry
from shapely.geometry import box

from rslp.sentinel1_vessels.predict_pipeline import (
    PredictionTask,
    Sentinel1Image,
    predict_pipeline,
)
from tests.integration.vessels.helpers import (
    WGS84_ITEM_BOUNDS,
    apply_common_config_patches,
)

FAKE_S1_SCENE_ID = (
    "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"
)
FAKE_S1_HIST_ID = (
    "S1A_IW_GRDH_1SDV_20240801T003924_20240801T003949_055000_060000_AAAA.SAFE"
)

_s1_mod = sys.modules["rslp.sentinel1_vessels.predict_pipeline"]


# --- S1-specific helpers ---


def _create_tiny_detect_config(original_path: str, output_path: str) -> None:
    """Patch an S1 detection config (SimpleTimeSeries wrapping Swin) to use swin_t."""
    with open(original_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg["model"]["init_args"]
    encoder_list = model_args["model"]["init_args"]["encoder"]
    sts_args = encoder_list[0]["init_args"]
    sts_args["encoder"]["init_args"]["arch"] = "swin_t"
    sts_args["encoder"]["init_args"]["pretrained"] = False
    num_groups = len(encoder_list[0]["init_args"]["groups"])
    encoder_list[1]["init_args"]["in_channels"] = [
        ch * num_groups for ch in [96, 192, 384, 768]
    ]

    apply_common_config_patches(cfg)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f)


def _write_geotiff_with_gcps(
    path: str,
    wgs84_bounds: tuple[float, float, float, float] = WGS84_ITEM_BOUNDS,
    size: int = 128,
    nbands: int = 1,
    dtype: str = "float32",
) -> None:
    """Write a synthetic GeoTIFF using GCPs (no affine transform), like raw S1 GRD."""
    minx, miny, maxx, maxy = wgs84_bounds
    gcps = [
        GroundControlPoint(row=0, col=0, x=minx, y=maxy),
        GroundControlPoint(row=0, col=size, x=maxx, y=maxy),
        GroundControlPoint(row=size, col=0, x=minx, y=miny),
        GroundControlPoint(row=size, col=size, x=maxx, y=miny),
    ]
    gcp_crs = CRS.from_epsg(4326)

    data = np.random.rand(nbands, size, size).astype(dtype)
    profile = {
        "driver": "GTiff",
        "width": size,
        "height": size,
        "count": nbands,
        "dtype": dtype,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.gcps = (gcps, gcp_crs)
        dst.write(data)


# --- Fake S3 clients (for aws_sentinel1 data source used during materialize) ---


class _FakeS3Bucket:
    """Mock for boto3 S3 Bucket used by aws_sentinel1 data source."""

    def download_file(
        self, key: str, fname: str, ExtraArgs: dict | None = None
    ) -> None:
        _write_geotiff_with_gcps(fname)

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


# --- Fake Copernicus Sentinel1 data source ---


def _make_copernicus_item(name: str, time: datetime) -> CopernicusItem:
    return CopernicusItem(
        name=name,
        geometry=STGeometry(
            WGS84_PROJECTION,
            box(*WGS84_ITEM_BOUNDS),
            (time, time + timedelta(seconds=60)),
        ),
        product_uuid="fake-uuid-" + name[:20],
    )


class _FakeCopernicusSentinel1:
    """Replaces rslearn.data_sources.copernicus.Sentinel1 in the pipeline module."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def _get_product(self, scene_id: str, expand_attributes: bool = False) -> dict:
        return {
            "Attributes": [
                {"Name": "orbitDirection", "Value": "ASCENDING"},
            ]
        }

    def get_item_by_name(self, name: str) -> CopernicusItem:
        return _make_copernicus_item(name, datetime(2024, 10, 1))

    def get_items(
        self, geometries: list, query_config: Any
    ) -> list[list[list[CopernicusItem]]]:
        hist_item = _make_copernicus_item(FAKE_S1_HIST_ID, datetime(2024, 8, 1))
        return [[[hist_item]]]


# --- Helpers ---


def _patch_model_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Create a tiny model config and monkeypatch the module constant."""
    detect_cfg = str(tmp_path / "detect_config.yaml")
    _create_tiny_detect_config("data/sentinel1_vessels/config.yaml", detect_cfg)
    monkeypatch.setattr(_s1_mod, "DETECT_MODEL_CONFIG", detect_cfg)
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path / "rslp"))


# --- Tests ---


def test_predict_pipeline_local_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes end-to-end with synthetic GCP-based GeoTIFFs."""
    _patch_model_configs(monkeypatch, tmp_path)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    files: dict[str, str] = {}
    for prefix in ["image", "hist1", "hist2"]:
        for band in ["vv", "vh"]:
            fpath = str(image_dir / f"{prefix}_{band}.tif")
            _write_geotiff_with_gcps(fpath)
            files[f"{prefix}_{band}"] = fpath

    tasks = [
        PredictionTask(
            image=Sentinel1Image(vv=files["image_vv"], vh=files["image_vh"]),
            historical1=Sentinel1Image(vv=files["hist1_vv"], vh=files["hist1_vh"]),
            historical2=Sentinel1Image(vv=files["hist2_vv"], vh=files["hist2_vh"]),
        )
    ]
    predict_pipeline(tasks=tasks, scratch_path=str(tmp_path / "scratch"))


def test_predict_pipeline_scene_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes using a fake scene ID with mocked Copernicus API + S3."""
    _patch_model_configs(monkeypatch, tmp_path)

    # Replace the Copernicus Sentinel1 class used for metadata lookup.
    monkeypatch.setattr(_s1_mod, "Sentinel1", _FakeCopernicusSentinel1)

    # Point to a copy of the dataset config (so writes don't pollute the repo).
    ds_cfg = str(tmp_path / "config.json")
    with open("data/sentinel1_vessels/config.json") as f:
        cfg = json.load(f)
    with open(ds_cfg, "w") as f:
        json.dump(cfg, f)
    monkeypatch.setattr(_s1_mod, "SCENE_ID_DATASET_CONFIG", ds_cfg)

    # Mock boto3 for the aws_sentinel1 data source.
    monkeypatch.setattr("boto3.resource", _fake_boto3_resource)
    monkeypatch.setattr("boto3.client", _fake_boto3_client)

    predict_pipeline(
        tasks=[PredictionTask(scene_id=FAKE_S1_SCENE_ID)],
        scratch_path=str(tmp_path / "scratch"),
    )
