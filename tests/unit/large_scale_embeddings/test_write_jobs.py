"""Unit tests for rslp.large_scale_embeddings.write_jobs."""

import json
import pathlib
from datetime import UTC, datetime

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs
from rslp.large_scale_embeddings.write_jobs import get_jobs

# Around Kenya: spans the equator (so both northern and southern UTM zones) and
# starts exactly at the boundary between UTM zones 36 and 37.
WGS84_BOUNDS = (36.0, -2.0, 38.0, 1.0)

TIMESTAMP = datetime(2025, 1, 1, tzinfo=UTC)


def test_get_jobs_wgs84_bounds(tmp_path: pathlib.Path) -> None:
    """Jobs limited by wgs84_bounds cover the right zones and tiles."""
    jobs = get_jobs(
        inputs=EmbeddingInputs.S2,
        timestamp=TIMESTAMP,
        out_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        checkpoint_path="/fake/checkpoint",
        wgs84_bounds=WGS84_BOUNDS,
    )
    assert len(jobs) > 0

    # Kept tiles are filtered by the bounding box of the reprojected user bounds, so
    # they can extend a bit past the exact WGS84 box; allow a margin.
    padded_query_shp = shapely.box(*WGS84_BOUNDS).buffer(0.5)

    seen_epsg_codes = set()
    for job in jobs:
        args = dict(zip(job[0::2], job[1::2]))
        assert args["--inputs"] == "S2"
        assert args["--time_range"] == json.dumps(
            [TIMESTAMP.isoformat(), TIMESTAMP.isoformat()]
        )

        projection = Projection.deserialize(json.loads(args["--projection_json"]))
        seen_epsg_codes.add(projection.crs.to_epsg())

        bounds = json.loads(args["--bounds"])
        tile_geom = STGeometry(projection, shapely.box(*bounds), None).to_projection(
            WGS84_PROJECTION
        )
        assert tile_geom.shp.intersects(padded_query_shp)

    # The bounds span lon 36-38 and lat -2 to 1, so tiles should be limited to UTM
    # zones 36/37 north and south. Zone 36 only touches at lon=36 exactly so it may
    # or may not contribute tiles, but zone 37 must appear in both hemispheres.
    assert seen_epsg_codes <= {32636, 32637, 32736, 32737}
    assert 32637 in seen_epsg_codes
    assert 32737 in seen_epsg_codes
