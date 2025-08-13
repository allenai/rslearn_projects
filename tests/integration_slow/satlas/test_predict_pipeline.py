"""Test the Satlas prediction pipeline."""

import json
import pathlib
from datetime import UTC, datetime

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

from rslp.satlas.predict_pipeline import (
    PATCH_SIZE,
    Application,
    get_output_fname,
    predict_pipeline,
)


def test_predict_pipeline_point(tmp_path: pathlib.Path) -> None:
    # Test the prediction pipeline runs correctly for a point detection task.
    # Specifically, we apply the marine infrastructure model on a window covering a
    # small portion of the Hornsea 2 Offshore Wind Farm, and verify that the resulting
    # detections include at least one turbine.

    # These are the coordinates of the known wind turbine.
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(1.859, 53.91), None)

    # We get the corresponding UTM window.
    # The window's bounds must all be multiples of PATCH_SIZE.
    projection = get_utm_ups_projection(src_geom.shp.x, src_geom.shp.y, 10, -10)
    dst_geom = src_geom.to_projection(projection)
    start = (
        int(dst_geom.shp.x) // PATCH_SIZE * PATCH_SIZE,
        int(dst_geom.shp.y) // PATCH_SIZE * PATCH_SIZE,
    )
    bounds = (start[0], start[1], start[0] + PATCH_SIZE, start[1] + PATCH_SIZE)

    # The wind farm existed since 2019 so this time range will work.
    time_range = (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 8, 1, tzinfo=UTC),
    )

    # Output path will contain outputs, while scratch path is used as a working
    # directory. Specifically, the scratch path will be populated with an rslearn
    # dataset containing one window matching the projection/bounds that we provide.
    out_path = tmp_path / "out"
    scratch_path = tmp_path / "scratch"
    out_path.mkdir()

    # Apply the pipeline. It will ingest data and apply the model.
    # We disable rtree index so that it doesn't need an hour to create it.
    predict_pipeline(
        application=Application.MARINE_INFRA,
        projection_json=json.dumps(projection.serialize()),
        bounds=bounds,
        time_range=time_range,
        out_path=str(out_path),
        scratch_path=str(scratch_path),
        use_rtree_index=False,
    )

    # Now we verify that the output includes at least one turbine.
    out_fname = get_output_fname(
        Application.MARINE_INFRA, str(out_path), projection, bounds
    )
    with out_fname.open() as f:
        fc = json.load(f)
    turbine_features = [
        feat for feat in fc["features"] if feat["properties"]["category"] == "turbine"
    ]
    assert len(turbine_features) > 0
