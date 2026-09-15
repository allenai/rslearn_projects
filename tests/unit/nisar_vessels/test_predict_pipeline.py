"""Unit tests for the pieces of the NISAR prediction pipeline that need no model."""

import json
import pathlib

import numpy as np
import pytest
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.nisar_vessels import predict_pipeline as pipeline
from rslp.nisar_vessels.hdf5 import GranuleGrid
from rslp.vessels import VesselDetection, VesselDetectionSource

# Somewhere in the Pacific, well away from any marine infrastructure.
OPEN_WATER_LON, OPEN_WATER_LAT = -140.0, 30.0


def _grid(epsg_code: int = 32610) -> GranuleGrid:
    return GranuleGrid(
        crs=CRS.from_epsg(epsg_code),
        x_resolution=20.0,
        y_resolution=-20.0,
        x_origin=500000.0,
        y_origin=4200000.0,
        width=40,
        height=30,
    )


# --- PredictionTask ---


def test_scene_id_defaults_to_the_granule_filename() -> None:
    task = pipeline.PredictionTask(h5_path="/shared/granules/NISAR_L2_GCOV_009.h5")

    assert task.get_scene_id() == "NISAR_L2_GCOV_009"


def test_scene_id_override_wins() -> None:
    task = pipeline.PredictionTask(h5_path="/shared/x.h5", scene_id="granule-name")

    assert task.get_scene_id() == "granule-name"


def test_prediction_task_is_frozen() -> None:
    task = pipeline.PredictionTask(h5_path="/shared/x.h5")

    with pytest.raises(AttributeError):
        task.h5_path = "/shared/y.h5"  # type: ignore[misc]


# --- Scene placement ---


def test_scene_data_is_placed_on_the_detector_grid() -> None:
    """A granule's footprint comes back in a UTM/UPS zone at the detector resolution."""
    scene_data = pipeline._get_scene_data("granule", _grid())

    assert scene_data.scene_id == "granule"
    assert scene_data.projection.x_resolution == pipeline.RESOLUTION
    assert scene_data.projection.y_resolution == -pipeline.RESOLUTION
    # The grid is already in UTM zone 10N, so that is the zone the centroid picks too
    # and the bounds are the granule's own extent rescaled to 10 m pixels.
    assert scene_data.projection.crs.to_epsg() == 32610
    minx, miny, maxx, maxy = scene_data.bounds
    assert (maxx - minx, maxy - miny) == (80, 60)


def test_scene_data_reprojects_a_granule_from_another_zone() -> None:
    """A granule geocoded outside the zone its centroid falls in is still placed right.

    GCOV granules are geocoded already, but not necessarily in the zone the training
    windows used, so the footprint is re-derived from the centroid rather than trusted.
    """
    # Same easting/northing, but declared in UTM zone 11N instead of 10N.
    scene_data = pipeline._get_scene_data("granule", _grid(epsg_code=32611))

    assert scene_data.projection.crs.to_epsg() == 32611
    minx, miny, maxx, maxy = scene_data.bounds
    assert (maxx - minx, maxy - miny) == (80, 60)


# --- Crop rendering ---


def test_decibel_stretch_spans_the_display_range() -> None:
    """The bottom of the display range maps to 0 and the top to 255."""
    min_db, max_db = pipeline.CROP_DECIBEL_RANGE
    linear = np.array([[10 ** (min_db / 10), 10 ** (max_db / 10)]], dtype=np.float32)

    assert pipeline._to_uint8(linear).tolist() == [[0, 255]]


def test_decibel_stretch_is_monotonic() -> None:
    linear = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.0]], dtype=np.float32)

    stretched = pipeline._to_uint8(linear)[0]

    assert list(stretched) == sorted(stretched)
    assert stretched.dtype == np.uint8


def test_nodata_becomes_black() -> None:
    """NaN is GCOV's nodata and has to be cleared before the log, not carried through."""
    linear = np.array([[np.nan, np.inf, -np.inf, 0.0]], dtype=np.float32)

    assert pipeline._to_uint8(linear).tolist() == [[0, 0, 0, 0]]


def test_values_above_the_display_range_clip_rather_than_wrap() -> None:
    linear = np.array([[1e6]], dtype=np.float32)

    assert pipeline._to_uint8(linear).tolist() == [[255]]


# --- Detection grouping and filtering ---


@pytest.fixture
def infra_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point the near-infrastructure filter at a one-platform GeoJSON."""
    path = tmp_path / "infra.geojson"
    with path.open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Point",
                            "coordinates": [OPEN_WATER_LON, OPEN_WATER_LAT],
                        },
                    }
                ],
            },
            f,
        )
    monkeypatch.setattr(pipeline, "MARINE_INFRA_PATH", str(path))
    return path


def _detection(task_idx: int, col: int, row: int) -> VesselDetection:
    # Resolution 1 in WGS84 makes col/row degrees, so a detection can be placed at a
    # known longitude and latitude without going through a UTM zone.
    return VesselDetection(
        source=VesselDetectionSource.NISAR,
        col=col,
        row=row,
        projection=Projection(CRS.from_epsg(4326), 1, 1),
        score=0.9,
        metadata={"task_idx": task_idx},
    )


def test_detections_are_grouped_by_task(infra_path: pathlib.Path) -> None:
    tasks = [
        pipeline.PredictionTask(h5_path="/a.h5"),
        pipeline.PredictionTask(h5_path="/b.h5"),
    ]
    # All well clear of the platform in the infra fixture, so none are filtered.
    detections = [
        _detection(0, OPEN_WATER_LON + 4, OPEN_WATER_LAT),
        _detection(1, OPEN_WATER_LON + 5, OPEN_WATER_LAT),
        _detection(1, OPEN_WATER_LON + 6, OPEN_WATER_LAT),
    ]

    by_task = pipeline._build_predictions_and_crops(detections, [], tasks)

    assert [len(group) for group in by_task] == [1, 2]


def test_task_with_no_detections_gets_an_empty_list(
    infra_path: pathlib.Path,
) -> None:
    tasks = [pipeline.PredictionTask(h5_path="/a.h5")]

    assert pipeline._build_predictions_and_crops([], [], tasks) == [[]]


def test_detection_on_marine_infrastructure_is_dropped(
    infra_path: pathlib.Path,
) -> None:
    """A detection sitting on a fixed platform is filtered out, a distant one is kept."""
    tasks = [pipeline.PredictionTask(h5_path="/a.h5")]
    on_platform = _detection(0, OPEN_WATER_LON, OPEN_WATER_LAT)
    far_away = _detection(0, OPEN_WATER_LON + 1, OPEN_WATER_LAT)

    ((kept,),) = pipeline._build_predictions_and_crops(
        [on_platform, far_away], [], tasks
    )

    assert kept is far_away
