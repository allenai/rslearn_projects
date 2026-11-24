"""Unit tests for rslp.large_scale_embeddings.write_jobs."""

import json
import pathlib
from datetime import UTC, datetime

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry

from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs
from rslp.large_scale_embeddings.write_jobs import get_jobs

# Around Kenya: spans the equator and starts at the zone 36/37 boundary.
WGS84_BOUNDS = (36.0, -2.0, 38.0, 1.0)

TIMESTAMP = datetime(2025, 1, 1, tzinfo=UTC)


def test_get_jobs_wgs84_bounds(tmp_path: pathlib.Path) -> None:
    """Jobs limited by wgs84_bounds cover the right zones and tiles."""
    jobs = get_jobs(
        inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        timestamp=TIMESTAMP,
        store_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        time_index=0,
        checkpoint_path="/fake/checkpoint",
        wgs84_bounds=WGS84_BOUNDS,
    )
    assert len(jobs) > 0

    # Blocks are filtered by the reprojected bounding box, so allow a margin.
    padded_query_shp = shapely.box(*WGS84_BOUNDS).buffer(0.5)

    seen_epsg_codes = set()
    seen_bounds: list[tuple[int, list[int]]] = []
    for job in jobs:
        args = dict(zip(job[0::2], job[1::2]))
        assert args["--inputs"] == "S2_S1_LANDSAT_DISTILLED"
        assert args["--time_range"] == json.dumps(
            [TIMESTAMP.isoformat(), TIMESTAMP.isoformat()]
        )

        projection = Projection.deserialize(json.loads(args["--projection_json"]))
        seen_epsg_codes.add(projection.crs.to_epsg())

        bounds = json.loads(args["--bounds"])
        seen_bounds.append((projection.crs.to_epsg(), bounds))
        tile_geom = STGeometry(projection, shapely.box(*bounds), None).to_projection(
            WGS84_PROJECTION
        )
        assert tile_geom.shp.intersects(padded_query_shp)

    # Zone 36 only touches lon 36, and every zone uses its northern CRS.
    assert seen_epsg_codes <= {32636, 32637}
    assert 32637 in seen_epsg_codes

    # Pixel y < 0 is north of the equator, y > 0 south of it.
    zone_37_bounds = [b for epsg, b in seen_bounds if epsg == 32637]
    assert any(b[1] < 0 for b in zone_37_bounds), "no tile north of the equator"
    assert any(b[3] > 0 for b in zone_37_bounds), "no tile south of the equator"


def test_get_jobs_geojson(tmp_path: pathlib.Path) -> None:
    """Jobs limited by a GeoJSON file only cover blocks intersecting its features."""
    # A feature spanning UTM zones 11 to 15, and one near Nairobi in zone 37.
    wide_shp = shapely.box(-117.0, 40.0, -93.0, 45.0)
    nairobi_shp = shapely.box(36.6, -1.4, 36.9, -1.2)
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": shapely.geometry.mapping(shp),
            }
            for shp in [wide_shp, nairobi_shp]
        ],
    }
    geojson_fname = tmp_path / "aoi.geojson"
    with geojson_fname.open("w") as f:
        json.dump(feature_collection, f)

    jobs = get_jobs(
        inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        timestamp=TIMESTAMP,
        store_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        time_index=0,
        checkpoint_path="/fake/checkpoint",
        geojson_fname=str(geojson_fname),
    )
    assert len(jobs) > 0

    # Allow a margin for reprojection error.
    padded_query_shp = shapely.union_all([wide_shp, nairobi_shp]).buffer(0.5)

    seen_epsg_codes = set()
    seen_bounds: list[tuple[int, list[int]]] = []
    for job in jobs:
        args = dict(zip(job[0::2], job[1::2]))
        projection = Projection.deserialize(json.loads(args["--projection_json"]))
        seen_epsg_codes.add(projection.crs.to_epsg())

        bounds = json.loads(args["--bounds"])
        seen_bounds.append((projection.crs.to_epsg(), bounds))
        tile_geom = STGeometry(projection, shapely.box(*bounds), None).to_projection(
            WGS84_PROJECTION
        )
        assert tile_geom.shp.intersects(padded_query_shp)

    assert seen_epsg_codes == {32611, 32612, 32613, 32614, 32615, 32637}
    assert any(
        epsg == 32637 and bounds[3] > 0 for epsg, bounds in seen_bounds
    ), "the Nairobi feature should produce a tile extending south of the equator"


def test_get_jobs_carries_the_model_settings(tmp_path: pathlib.Path) -> None:
    """Each job names every setting that changes the embeddings."""
    jobs = get_jobs(
        inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        timestamp=TIMESTAMP,
        store_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        time_index=0,
        checkpoint_path="/fake/checkpoint",
        wgs84_bounds=WGS84_BOUNDS,
        count=1,
    )
    args = dict(zip(jobs[0][0::2], jobs[0][1::2]))
    assert args["--patch_size"] == "2"
    assert args["--location_query_patch_size"] == "1"
    assert args["--window_size"] == "480"
    assert args["--overlap_size"] == "32"
    assert args["--context_pixels"] == "48"


def test_get_jobs_refuses_markers_from_another_job_size(
    tmp_path: pathlib.Path,
) -> None:
    """Marker names hold only the block corner, so job sizes cannot share a directory."""
    import pytest

    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "EPSG:32637_0_0.json").write_text(
        json.dumps({"bounds": [0, 0, 32768, 32768]})
    )
    with pytest.raises(ValueError, match="job_size"):
        get_jobs(
            inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
            timestamp=TIMESTAMP,
            store_path=str(tmp_path / "out"),
            completed_path=str(completed),
            time_index=0,
            checkpoint_path="/fake/checkpoint",
            wgs84_bounds=WGS84_BOUNDS,
            job_size=8192,
        )


def test_blocks_fname_keeps_exactly_the_named_blocks(tmp_path: pathlib.Path) -> None:
    """A block list narrows the enumeration to those blocks, and only those."""
    from rslp.large_scale_embeddings.write_jobs import block_id, enumerate_blocks

    every = enumerate_blocks(job_size=8192, wgs84_bounds=WGS84_BOUNDS)
    chosen = [block_id(*block) for block in every[::3]]
    blocks_fname = tmp_path / "blocks.json"
    blocks_fname.write_text(json.dumps(chosen))
    kept = enumerate_blocks(
        job_size=8192, wgs84_bounds=WGS84_BOUNDS, blocks_fname=str(blocks_fname)
    )
    assert [block_id(*block) for block in kept] == chosen


def test_blocks_fname_rejects_a_block_the_run_does_not_have(
    tmp_path: pathlib.Path,
) -> None:
    """A list built for another job_size must fail rather than shrink the run."""
    import pytest

    from rslp.large_scale_embeddings.write_jobs import enumerate_blocks

    blocks_fname = tmp_path / "blocks.json"
    blocks_fname.write_text(json.dumps(["EPSG:32637_1_1"]))
    with pytest.raises(ValueError, match="does not have"):
        enumerate_blocks(
            job_size=8192, wgs84_bounds=WGS84_BOUNDS, blocks_fname=str(blocks_fname)
        )


def test_a_footprint_covering_the_zone_keeps_every_block(
    tmp_path: pathlib.Path,
) -> None:
    """A footprint wider than the zone keeps exactly the unrestricted blocks."""
    from rslp.large_scale_embeddings.write_jobs import enumerate_blocks

    fname = tmp_path / "wider.geojson"
    fname.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": shapely.geometry.mapping(
                            shapely.box(30.0, -85.0, 48.0, 85.0)
                        ),
                    }
                ],
            }
        )
    )
    restricted = enumerate_blocks(
        job_size=8192, epsg_code=32637, geojson_fname=str(fname)
    )
    unrestricted = enumerate_blocks(job_size=8192, epsg_code=32637)
    assert restricted == unrestricted
