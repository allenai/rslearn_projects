"""Integration tests for the NISAR vessel prediction pipeline.

These run the real rslearn materialize and predict path over a synthetic GCOV granule,
with the detector shrunk to swin_t and left untrained. What is being checked is the
plumbing -- that a granule turns into a materialized window, that the model runs over
it, and that detections come back with crops -- not detection quality.
"""

import json
import pathlib

import numpy as np
import pytest
import shapely
import yaml
from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.nisar_vessels import predict_pipeline as pipeline
from rslp.vessels import VesselDetection, VesselDetectionSource
from tests.integration.vessels.helpers import create_tiny_detect_config
from tests.utils.nisar_granule import DUAL_POL_BANDS, write_granule

# Open water in the English Channel. Deliberately not the WGS84_ITEM_BOUNDS the other
# vessel tests share: that box straddles the UTM zone 30/31 boundary at longitude 0, so
# a granule there is reprojected into the neighbouring zone and its window grows, which
# would make the exact window size asserted below meaningless.
GRANULE_CENTER_LON, GRANULE_CENTER_LAT = -3.0, 50.0

# Granule posting, coarser than the 10 m the detector runs at, so the test also covers
# the resampling that a real 20 m GCOV granule needs.
GRANULE_RESOLUTION = 20.0
GRANULE_SIZE = 128

SCENE_ID = "NISAR_L2_GCOV_009_055_A_014_4005_DHDH_A_20260101T000312_20260101T000347"


@pytest.fixture
def granule_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a synthetic GCOV granule centered on GRANULE_CENTER_LON/LAT.

    It is geocoded in the same UTM zone the pipeline will pick from its centroid, which
    is what a real granule over this area would be.
    """
    utm_projection = get_utm_ups_projection(
        GRANULE_CENTER_LON, GRANULE_CENTER_LAT, 1, 1
    )
    centroid = (
        STGeometry(
            WGS84_PROJECTION,
            shapely.Point(GRANULE_CENTER_LON, GRANULE_CENTER_LAT),
            None,
        )
        .to_projection(utm_projection)
        .shp
    )
    half_span = GRANULE_SIZE * GRANULE_RESOLUTION / 2

    rng = np.random.default_rng(0)
    data = {
        # Linear gamma-0 power, in the range open water and vessels actually occupy
        # (roughly -30 to 0 dB), since the model config converts to decibels.
        band: rng.uniform(1e-3, 1.0, size=(GRANULE_SIZE, GRANULE_SIZE)).astype(
            np.float32
        )
        for band in DUAL_POL_BANDS
    }

    path = tmp_path / f"{SCENE_ID}.h5"
    write_granule(
        path,
        data=data,
        epsg_code=utm_projection.crs.to_epsg(),
        x_origin=centroid.x - half_span,
        y_origin=centroid.y + half_span,
        x_resolution=GRANULE_RESOLUTION,
        y_resolution=-GRANULE_RESOLUTION,
        width=GRANULE_SIZE,
        height=GRANULE_SIZE,
    )
    return path


@pytest.fixture
def tiny_detector(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a swin_t detector with no checkpoint to restore, running on the CPU."""
    detect_cfg = tmp_path / "detect_config.yaml"
    create_tiny_detect_config(pipeline.DETECT_MODEL_CONFIG, str(detect_cfg))

    # The config's default DDP strategy is rejected outright on an MPS accelerator, so a
    # developer running this on a Mac would never reach the model at all.
    with detect_cfg.open() as f:
        cfg = yaml.safe_load(f)
    cfg["trainer"]["accelerator"] = "cpu"
    cfg["trainer"]["strategy"] = "auto"
    cfg["trainer"]["devices"] = 1
    with detect_cfg.open("w") as f:
        yaml.dump(cfg, f)

    monkeypatch.setattr(pipeline, "DETECT_MODEL_CONFIG", str(detect_cfg))
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path / "rslp"))


@pytest.fixture
def no_infra(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the near-infrastructure filter at an empty GeoJSON, so nothing is fetched."""
    path = tmp_path / "infra.geojson"
    with path.open("w") as f:
        json.dump({"type": "FeatureCollection", "features": []}, f)
    monkeypatch.setattr(pipeline, "MARINE_INFRA_PATH", str(path))


def test_granule_is_materialized_at_the_detector_resolution(
    granule_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """setup_dataset plus materialize_scenes turn an H5 into a readable 10 m window."""
    ds_path = UPath(tmp_path / "scratch")
    ds_path.mkdir(parents=True)
    task = pipeline.PredictionTask(h5_path=str(granule_path))

    (scene_data,) = pipeline.setup_dataset(ds_path, [task])
    ((scene_idx, window),) = pipeline.materialize_scenes(ds_path, [scene_data])

    assert scene_data.scene_id == SCENE_ID
    assert scene_idx == 0
    assert window.is_layer_completed(pipeline.NISAR_LAYER_NAME)
    # 128 granule pixels at 20 m is 256 window pixels at 10 m. The window is allowed
    # one pixel more per axis: the granule's corner rarely lands exactly on a window
    # pixel boundary, and the bounds are rounded outwards so the whole scene is covered.
    expected_size = int(GRANULE_SIZE * GRANULE_RESOLUTION / pipeline.RESOLUTION)
    minx, miny, maxx, maxy = window.bounds
    assert expected_size <= maxx - minx <= expected_size + 1
    assert expected_size <= maxy - miny <= expected_size + 1

    image = window.data.read_raster(
        pipeline.NISAR_LAYER_NAME, pipeline.BAND_NAMES, GeotiffRasterFormat()
    ).get_chw_array()
    assert image.shape == (len(pipeline.BAND_NAMES), maxy - miny, maxx - minx)
    assert np.isfinite(image).any()


def test_predict_pipeline_end_to_end(
    granule_path: pathlib.Path,
    tmp_path: pathlib.Path,
    tiny_detector: None,
    no_infra: None,
) -> None:
    """A granule goes in, detections and their output files come out."""
    json_path = tmp_path / "out" / "detections.json"
    geojson_path = tmp_path / "out" / "detections.geojson"
    crop_path = tmp_path / "out" / "crops"

    task = pipeline.PredictionTask(
        h5_path=str(granule_path),
        json_path=str(json_path),
        geojson_path=str(geojson_path),
        crop_path=str(crop_path),
    )

    # An untrained detector is not expected to find anything in particular, so the
    # threshold is dropped to 0 to exercise the crop and output paths.
    (detections,) = pipeline.predict_pipeline(
        tasks=[task], score_threshold=0.0, scratch_path=str(tmp_path / "scratch")
    )

    with json_path.open() as f:
        assert json.load(f) == [d.to_dict() for d in detections]
    with geojson_path.open() as f:
        assert len(json.load(f)["features"]) == len(detections)

    for detection in detections:
        assert detection.source == VesselDetectionSource.NISAR
        assert detection.scene_id == SCENE_ID
        assert set(detection.crop_fnames or {}) == {"hh", "hv"}
        for crop_fname in (detection.crop_fnames or {}).values():
            assert crop_fname.exists()


def test_crops_are_written_for_a_detection(
    granule_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A detection's crop is a CROP_WINDOW_SIZE 8-bit PNG per band, keyed hh/hv.

    The crops are sliced back out of the scene window rather than materialized on their
    own, so that read is worth covering directly.
    """
    ds_path = UPath(tmp_path / "scratch")
    ds_path.mkdir(parents=True)
    task = pipeline.PredictionTask(h5_path=str(granule_path))
    (scene_data,) = pipeline.setup_dataset(ds_path, [task])

    minx, miny, maxx, maxy = scene_data.bounds
    detection = VesselDetection(
        source=VesselDetectionSource.NISAR,
        col=(minx + maxx) // 2,
        row=(miny + maxy) // 2,
        projection=scene_data.projection,
        score=0.9,
    )

    crop_fnames = pipeline._write_crops(
        detection, scene_data, UPath(tmp_path / "crops")
    )

    assert set(crop_fnames) == {"hh", "hv"}
    for crop_fname in crop_fnames.values():
        with crop_fname.open("rb") as f:
            image = Image.open(f)
            assert image.mode == "L"
            assert image.size == (
                pipeline.CROP_WINDOW_SIZE,
                pipeline.CROP_WINDOW_SIZE,
            )


def test_crop_at_the_scene_edge_is_still_written(
    granule_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A detection near the edge gets a full-size crop, padded with nodata.

    A crop window running off the scene edge would never materialize on its own, so
    reading from the scene GeoTIFF is what keeps these detections reportable.
    """
    ds_path = UPath(tmp_path / "scratch")
    ds_path.mkdir(parents=True)
    task = pipeline.PredictionTask(h5_path=str(granule_path))
    (scene_data,) = pipeline.setup_dataset(ds_path, [task])

    minx, miny, _, _ = scene_data.bounds
    detection = VesselDetection(
        source=VesselDetectionSource.NISAR,
        col=minx + 2,
        row=miny + 2,
        projection=scene_data.projection,
        score=0.9,
    )

    crop_fnames = pipeline._write_crops(
        detection, scene_data, UPath(tmp_path / "crops")
    )

    assert set(crop_fnames) == {"hh", "hv"}
    for crop_fname in crop_fnames.values():
        assert crop_fname.exists()


def test_large_scene_is_split_into_tiles(
    granule_path: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene larger than the tile size materializes as several bounded windows.

    Peak memory during materialize follows the window size, so a granule bigger than the
    tile must not end up in a single window however large it gets.
    """
    # The synthetic granule is 256px at 10m, so shrink the tile rather than build a
    # multi-gigapixel fixture.
    monkeypatch.setattr(pipeline, "SCENE_TILE_SIZE", 128)
    monkeypatch.setattr(pipeline, "SCENE_TILE_OVERLAP", 16)

    ds_path = UPath(tmp_path / "scratch")
    ds_path.mkdir(parents=True)
    task = pipeline.PredictionTask(h5_path=str(granule_path))
    (scene_data,) = pipeline.setup_dataset(ds_path, [task])

    tiles = pipeline.materialize_scenes(ds_path, [scene_data])

    assert len(tiles) > 1
    for scene_idx, window in tiles:
        assert scene_idx == 0
        assert window.is_layer_completed(pipeline.NISAR_LAYER_NAME)
        # Absorbing a narrow remainder can push one tile past the nominal size, by
        # less than a detector crop.
        limit = 128 + pipeline.PREDICT_CROP_SIZE
        minx, miny, maxx, maxy = window.bounds
        assert maxx - minx <= limit
        assert maxy - miny <= limit

    # Every tile still reads back as real imagery, not an empty sliver.
    for _, window in tiles:
        image = window.data.read_raster(
            pipeline.NISAR_LAYER_NAME, pipeline.BAND_NAMES, GeotiffRasterFormat()
        ).get_chw_array()
        assert image.shape[0] == len(pipeline.BAND_NAMES)
        assert np.isfinite(image).any()
