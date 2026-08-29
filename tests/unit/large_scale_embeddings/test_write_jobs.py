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
        store_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        time_index=0,
        checkpoint_path="/fake/checkpoint",
        wgs84_bounds=WGS84_BOUNDS,
    )
    assert len(jobs) > 0

    # Kept tiles are filtered by the bounding box of the reprojected user bounds, so
    # they can extend a bit past the exact WGS84 box; allow a margin.
    padded_query_shp = shapely.box(*WGS84_BOUNDS).buffer(0.5)

    seen_epsg_codes = set()
    seen_bounds: list[tuple[int, list[int]]] = []
    for job in jobs:
        args = dict(zip(job[0::2], job[1::2]))
        assert args["--inputs"] == "S2"
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

    # The bounds span lon 36-38 and lat -2 to 1, so tiles should be limited to UTM
    # zones 36 and 37. Zone 36 only touches at lon=36 exactly so it may or may not
    # contribute tiles. Every zone is stored in its *northern* CRS, so no 327NN code
    # should ever appear even though the bounds reach 2 degrees south.
    assert seen_epsg_codes <= {32636, 32637}
    assert 32637 in seen_epsg_codes

    # Both hemispheres still have to be covered; they now differ by the sign of the
    # pixel row rather than by EPSG code. Pixel y runs southward from a negative origin,
    # so y < 0 is north of the equator and y > 0 is south of it.
    zone_37_bounds = [b for epsg, b in seen_bounds if epsg == 32637]
    assert any(b[1] < 0 for b in zone_37_bounds), "no tile north of the equator"
    assert any(b[3] > 0 for b in zone_37_bounds), "no tile south of the equator"


def test_get_jobs_geojson(tmp_path: pathlib.Path) -> None:
    """Jobs limited by a GeoJSON file only cover tiles intersecting its features.

    One feature spans many UTM zones, which exercises the per-zone clipping (naively
    reprojecting the whole shape into every zone fails or yields meaningless bounds).
    """
    # A wide feature across the US at lat 40-45 (all land), spanning from the
    # midpoint of UTM zone 11 (-117) to the midpoint of UTM zone 15 (-93) so it
    # comfortably covers exactly zones 11 through 15, and a small polygon near
    # Nairobi (~36.8E, -1.3S, in UTM zone 37S).
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
        inputs=EmbeddingInputs.S2,
        timestamp=TIMESTAMP,
        store_path=str(tmp_path / "out"),
        completed_path=str(tmp_path / "completed"),
        time_index=0,
        checkpoint_path="/fake/checkpoint",
        geojson_fname=str(geojson_fname),
    )
    assert len(jobs) > 0

    # Tiles intersect the features in projected coordinates, so allow a margin for
    # reprojection error when checking in WGS84.
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

    # The wide feature yields tiles in exactly zones 11-15 north, and the Nairobi
    # feature in zone 37 south.
    # Nairobi is ~1.3 degrees south, but zone 37 is stored in its northern CRS, so it
    # is 32637 with a positive (southward) pixel row rather than 32737.
    assert seen_epsg_codes == {32611, 32612, 32613, 32614, 32615, 32637}
    # The tile grid is aligned to northing 0, so the Nairobi tile starts exactly at the
    # equator and extends south: y1 is what carries the sign, not y0.
    assert any(epsg == 32637 and bounds[3] > 0 for epsg, bounds in seen_bounds), (
        "the Nairobi feature should produce a tile extending south of the equator"
    )
