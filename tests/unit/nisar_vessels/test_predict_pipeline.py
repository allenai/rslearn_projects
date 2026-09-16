"""Unit tests for the pieces of the NISAR prediction pipeline that need no model."""

import itertools
import json
import pathlib

import numpy as np
import pytest
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from upath import UPath

from rslp.nisar_vessels import predict_pipeline as pipeline
from rslp.nisar_vessels.hdf5 import GranuleGrid
from rslp.utils.nms import distance_nms
from rslp.vessels import VesselDetection, VesselDetectionSource

# Somewhere in the Pacific, well away from any marine infrastructure. Whole degrees so
# detections can be placed by integer column/row against a degree-resolution projection.
OPEN_WATER_LON, OPEN_WATER_LAT = -140, 30


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


def test_scene_id_falls_back_to_the_granule_filename() -> None:
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
    scene_data = pipeline._get_scene_data(
        "granule", _grid(), UPath("/scratch/granule.tif")
    )

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
    scene_data = pipeline._get_scene_data(
        "granule", _grid(epsg_code=32611), UPath("/scratch/granule.tif")
    )

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


# --- Scene tiling ---


def test_small_scene_is_a_single_tile() -> None:
    """A scene that already fits the tile size is not split."""
    bounds = (0, 0, 1000, 800)

    assert pipeline.tile_scene_bounds(bounds, 4096, 64) == [bounds]


def test_tiles_cover_the_whole_scene() -> None:
    """Every pixel of the scene falls inside at least one tile."""
    bounds = (0, 0, 10000, 7000)

    tiles = pipeline.tile_scene_bounds(bounds, 4096, 64)

    assert min(t[0] for t in tiles) == bounds[0]
    assert min(t[1] for t in tiles) == bounds[1]
    assert max(t[2] for t in tiles) == bounds[2]
    assert max(t[3] for t in tiles) == bounds[3]


def test_no_tile_exceeds_the_tile_size() -> None:
    """The tile size is the memory bound, so nothing may come out larger."""
    tiles = pipeline.tile_scene_bounds((0, 0, 10000, 7000), 4096, 64)

    assert tiles
    for minx, miny, maxx, maxy in tiles:
        assert maxx - minx <= 4096
        assert maxy - miny <= 4096


def test_tile_count_grows_with_scene_not_memory() -> None:
    """A scene 4x the area yields ~4x the tiles, each still one tile's worth of memory.

    This is the property the whole change exists for: peak memory follows the tile size,
    not the granule, which is unbounded.
    """
    small = pipeline.tile_scene_bounds((0, 0, 8192, 8192), 4096, 64)
    large = pipeline.tile_scene_bounds((0, 0, 16384, 16384), 4096, 64)

    assert len(large) > len(small)
    assert max(t[2] - t[0] for t in large) == max(t[2] - t[0] for t in small)


def test_adjacent_tiles_overlap() -> None:
    """Neighbouring tiles share a band, so a vessel on a seam is whole in one of them."""
    tiles = pipeline.tile_scene_bounds((0, 0, 10000, 4096), 4096, 64)
    row = sorted({(t[0], t[2]) for t in tiles})

    assert len(row) > 1
    for (_, first_maxx), (second_minx, _) in itertools.pairwise(row):
        assert first_maxx - second_minx >= 64


def test_offset_bounds_are_tiled_from_their_origin() -> None:
    """Scene bounds are negative in y, so tiling must not assume it starts at zero."""
    bounds = (50000, -420000, 58000, -412000)

    tiles = pipeline.tile_scene_bounds(bounds, 4096, 64)

    assert min(t[0] for t in tiles) == bounds[0]
    assert min(t[1] for t in tiles) == bounds[1]
    assert max(t[2] for t in tiles) == bounds[2]
    assert max(t[3] for t in tiles) == bounds[3]


# --- Cross-tile dedup ---


def _scored(task_idx: int, col: int, row: int, score: float) -> VesselDetection:
    return VesselDetection(
        source=VesselDetectionSource.NISAR,
        col=col,
        row=row,
        projection=Projection(CRS.from_epsg(32610), 10, -10),
        score=score,
        metadata={"task_idx": task_idx},
    )


def test_duplicate_across_tiles_keeps_the_best_score() -> None:
    """The same vessel seen from two overlapping tiles collapses to one detection."""
    detections = [_scored(0, 100, 100, 0.6), _scored(0, 102, 101, 0.9)]

    kept = pipeline.dedupe_detections(detections)

    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_distinct_vessels_are_both_kept() -> None:
    detections = [_scored(0, 100, 100, 0.9), _scored(0, 400, 400, 0.8)]

    assert len(pipeline.dedupe_detections(detections)) == 2


def test_same_position_in_different_scenes_is_not_deduped() -> None:
    """Tiles only overlap within a scene, so identical coordinates elsewhere are real."""
    detections = [_scored(0, 100, 100, 0.9), _scored(1, 100, 100, 0.8)]

    assert len(pipeline.dedupe_detections(detections)) == 2


def test_dedupe_of_nothing_is_nothing() -> None:
    assert pipeline.dedupe_detections([]) == []


def test_dedupe_uses_the_same_distance_metric_as_the_merger() -> None:
    """Two vessels a diagonal 13px apart are distinct, as the in-window merger sees it.

    The merger combines crops within a window by Euclidean distance, so cross-tile
    dedup has to agree or the same pair is merged in one place and kept in the other.
    """
    detections = [_scored(0, 100, 100, 0.9), _scored(0, 109, 109, 0.8)]

    assert len(pipeline.dedupe_detections(detections)) == 2


def test_dedupe_keeps_the_higher_score_of_a_seam_pair() -> None:
    """Suppression is by score, not by which tile happened to report first."""
    detections = [_scored(0, 100, 100, 0.4), _scored(0, 103, 102, 0.95)]

    ((kept),) = pipeline.dedupe_detections(detections)

    assert kept.score == 0.95


def test_dedupe_matches_the_merger_on_the_same_input() -> None:
    """Cross-tile suppression and the in-window merger agree, since they share a pass.

    They ran different distance metrics once; pinning them together stops that
    recurring, which would merge a pair in one stage and keep it in the other.
    """
    positions = [(100, 100), (106, 106), (400, 400), (100, 112)]
    scores = [0.9, 0.8, 0.7, 0.6]
    detections = [
        _scored(0, col, row, score)
        for (col, row), score in zip(positions, scores, strict=True)
    ]

    kept = pipeline.dedupe_detections(detections)

    expected = distance_nms(
        np.array(positions, dtype=float),
        np.array(scores, dtype=float),
        pipeline.DEDUPE_DISTANCE_PIXELS,
    )
    assert {(d.col, d.row) for d in kept} == {positions[i] for i in expected}
