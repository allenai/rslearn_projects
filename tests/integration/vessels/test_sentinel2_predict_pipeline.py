"""Integration tests for the Sentinel-2 vessel prediction pipeline."""

import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime
from typing import BinaryIO

import pytest
import yaml
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2Item
from rslearn.utils.geometry import STGeometry
from shapely.geometry import box
from upath import UPath

from rslp.sentinel2_vessels.predict_pipeline import (
    ImageFile,
    PredictionTask,
    predict_pipeline,
)
from tests.integration.vessels.helpers import (
    WGS84_ITEM_BOUNDS,
    apply_common_config_patches,
    create_tiny_detect_config,
    get_projection_and_bounds,
    write_synthetic_geotiff,
)

FAKE_S2_SCENE_ID = "S2A_MSIL1C_20230101T000000_N0000_R000_T30UYD_20230101T000000"

# __init__.py re-exports predict_pipeline as a function, shadowing the module name.
# We get the module directly here so that we can monkeypatch various attributes.
_s2_mod = sys.modules["rslp.sentinel2_vessels.predict_pipeline"]

_10M_BANDS = ["B02", "B03", "B04", "B08"]
_20M_BANDS = ["B05", "B06", "B07", "B8A", "B11", "B12"]
_60M_BANDS = ["B01", "B09", "B10"]

_BAND_RESOLUTION: dict[str, int] = {}
for _b in _10M_BANDS:
    _BAND_RESOLUTION[_b] = 10
for _b in _20M_BANDS:
    _BAND_RESOLUTION[_b] = 20
for _b in _60M_BANDS:
    _BAND_RESOLUTION[_b] = 60
_BAND_RESOLUTION["TCI"] = 10


# --- Fake GCS clients ---


class _FakeBlob:
    """Mock for google.cloud.storage.Blob."""

    def __init__(self, path: str) -> None:
        self.name = path

    def exists(self) -> bool:
        return True

    def download_to_filename(self, fname: str) -> None:
        # self.name is like bucket_prefix/B02.jp2 so we just get the band name.
        band = os.path.basename(self.name).replace(".jp2", "")
        res = _BAND_RESOLUTION.get(band, 10)
        proj, bounds = get_projection_and_bounds(res, -res)
        nbands = 3 if band == "TCI" else 1
        fpath = UPath(fname)
        write_synthetic_geotiff(
            fpath.parent, proj, bounds, nbands=nbands, fname=fpath.name
        )

    def download_to_file(self, f: BinaryIO) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            self.download_to_filename(tmp.name)
            tmp.seek(0)
            f.write(tmp.read())


class _FakeGCSBucket:
    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(path)

    def list_blobs(self, prefix: str | None = None) -> list[_FakeBlob]:
        return []


class _FakeGCSClient:
    def bucket(self, name: str) -> _FakeGCSBucket:
        return _FakeGCSBucket()


def _fake_create_anonymous_client() -> _FakeGCSClient:
    return _FakeGCSClient()


# --- Item cache + dataset config ---


def _populate_item_cache(cache_dir: pathlib.Path, scene_id: str) -> None:
    """Write a serialized Sentinel2Item JSON to the index cache directory.

    This way, when we call get_items on the GCP Sentinel-2 data source, it will load
    the item from the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    geometry = STGeometry(
        projection=WGS84_PROJECTION,
        shp=box(*WGS84_ITEM_BOUNDS),
        time_range=(datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 0, 1)),
    )
    item = Sentinel2Item(
        name=scene_id,
        geometry=geometry,
        blob_prefix="tiles/30/U/YD/FAKE/GRANULE/FAKE/IMG_DATA/FAKE_",
        cloud_cover=10.0,
    )
    with open(cache_dir / f"{scene_id}.json", "w") as f:
        json.dump(item.serialize(), f)


def _write_scene_id_dataset_config(
    original_config_path: str,
    output_path: str,
    index_cache_dir: str,
) -> None:
    """Write a modified config_predict_gcp.json with harmonize=false and custom cache.

    We set harmonize to False so we don't need to worry about writing a Sentinel-2 XML
    file, which is used when harmonization is enabled to determine which scenes need to
    be modified.
    """
    with open(original_config_path) as f:
        cfg = json.load(f)
    ds = cfg["layers"]["sentinel2"]["data_source"]["init_args"]
    ds["harmonize"] = False
    ds["index_cache_dir"] = index_cache_dir
    ds["use_rtree_index"] = False
    with open(output_path, "w") as f:
        json.dump(cfg, f)


# --- Tests ---


def _create_tiny_attribute_config(original_path: str, output_path: str) -> None:
    """Patch the S2 vessel attribute config: replace SatlasPretrain with Swin+FPN."""
    with open(original_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg["model"]["init_args"]
    model_args["model"]["init_args"]["encoder"] = [
        {
            "class_path": "rslearn.models.swin.Swin",
            "init_args": {
                "arch": "swin_t",
                "pretrained": False,
                "input_channels": 9,
                "output_layers": [1, 3, 5, 7],
            },
        },
        {
            "class_path": "rslearn.models.fpn.Fpn",
            "init_args": {
                "in_channels": [96, 192, 384, 768],
                "out_channels": 128,
            },
        },
    ]

    apply_common_config_patches(cfg)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f)


def _patch_model_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Create tiny model configs and monkeypatch the module constants."""
    detect_cfg = str(tmp_path / "detect_config.yaml")
    create_tiny_detect_config("data/sentinel2_vessels/config.yaml", detect_cfg)
    monkeypatch.setattr(_s2_mod, "DETECT_MODEL_CONFIG", detect_cfg)

    attr_cfg = str(tmp_path / "attribute_config.yaml")
    _create_tiny_attribute_config(
        "data/sentinel2_vessel_attribute/config.yaml", attr_cfg
    )
    monkeypatch.setattr(_s2_mod, "ATTRIBUTE_MODEL_CONFIG", attr_cfg)

    monkeypatch.setattr(_s2_mod, "NUM_DATA_LOADER_WORKERS", 0)
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path / "rslp"))


def test_predict_pipeline_local_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes end-to-end with synthetic local GeoTIFFs."""
    _patch_model_configs(monkeypatch, tmp_path)

    # Create the ImageFile list to pass to the prediction pipeline.
    image_files_dir = UPath(tmp_path / "image_files")
    image_files: list[ImageFile] = []

    for band in _10M_BANDS + _20M_BANDS + _60M_BANDS:
        res = _BAND_RESOLUTION[band]
        proj, bounds = get_projection_and_bounds(res, -res)
        write_synthetic_geotiff(image_files_dir, proj, bounds, fname=f"{band}.tif")
        image_files.append(
            ImageFile(bands=[band], fname=str(image_files_dir / f"{band}.tif"))
        )

    tasks = [PredictionTask(image_files=image_files)]
    predict_pipeline(tasks=tasks, scratch_path=str(tmp_path / "scratch"))


def test_predict_pipeline_scene_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pipeline completes using a fake scene ID with mocked GCS bucket."""
    _patch_model_configs(monkeypatch, tmp_path)

    cache_dir = tmp_path / "s2_cache"
    _populate_item_cache(cache_dir, FAKE_S2_SCENE_ID)

    ds_cfg = str(tmp_path / "config_predict_gcp.json")
    _write_scene_id_dataset_config(
        "data/sentinel2_vessels/config_predict_gcp.json",
        ds_cfg,
        index_cache_dir=str(cache_dir),
    )
    monkeypatch.setattr(_s2_mod, "SCENE_ID_DATASET_CONFIG", ds_cfg)

    monkeypatch.setattr(
        "google.cloud.storage.Client.create_anonymous_client",
        staticmethod(_fake_create_anonymous_client),
    )

    tasks = [PredictionTask(scene_id=FAKE_S2_SCENE_ID)]
    predict_pipeline(tasks=tasks, scratch_path=str(tmp_path / "scratch"))
