import json
import pathlib

import pyarrow.parquet as pq
import pytest
import shapely

from rslp.large_scale_embeddings.export_provenance import export_provenance

CRS = "EPSG:32631"
S2_FOOTPRINT = "POLYGON ((0.4 47.7, 0.8 48.6, 1.2 48.6, 1.1 47.6, 0.4 47.7))"
S1_FOOTPRINT = "POLYGON ((2.9 47.6, 4.4 47.7, 4.1 49.2, 0.7 48.9, 2.9 47.6))"


def _item(name: str, shp: str, when: str) -> dict:
    return {
        "name": name,
        "geometry": {
            "projection": {"crs": "EPSG:4326", "x_resolution": 1, "y_resolution": 1},
            "shp": shp,
            "time_range": [when, when],
        },
    }


def _window_layers() -> dict:
    period = ["2025-12-02T00:00:00+00:00", "2026-01-01T00:00:00+00:00"]
    return {
        "sentinel2_l2a": [
            {
                "time_range": period,
                "items": [
                    _item("S2_A", S2_FOOTPRINT, "2025-12-30T10:57:56+00:00"),
                    _item("S2_B", S2_FOOTPRINT, "2025-12-20T10:57:56+00:00"),
                ],
            }
        ],
        "sentinel1": [
            {
                "time_range": period,
                "items": [_item("S1_A", S1_FOOTPRINT, "2025-12-04T17:39:36+00:00")],
            }
        ],
        "landsat": [],
    }


def _write_block(
    run: pathlib.Path,
    year: int,
    origin: tuple[int, int],
    written: list[list[int]],
    window_names: list[str],
    marker_extra: dict | None = None,
    provenance_extra: dict | None = None,
    with_marker: bool = True,
) -> None:
    x, y = origin
    name = f"{CRS}_{x}_{y}.json"
    projection = {"crs": CRS, "x_resolution": 10, "y_resolution": -10}
    marker = {
        "projection": projection,
        "bounds": [x, y, x + 8192, y + 8192],
        "written": written,
        "window_size": 480,
        "worker": "worker_abc",
        **(marker_extra or {}),
    }
    provenance = {
        "projection": projection,
        "bounds": [x, y, x + 8192, y + 8192],
        "windows": {window: _window_layers() for window in window_names},
        **(provenance_extra or {}),
    }
    (run / f"provenance_{year}").mkdir(parents=True, exist_ok=True)
    (run / f"completed_{year}").mkdir(parents=True, exist_ok=True)
    (run / f"provenance_{year}" / name).write_text(json.dumps(provenance))
    if with_marker:
        (run / f"completed_{year}" / name).write_text(json.dumps(marker))


def test_export_writes_three_tables(tmp_path: pathlib.Path) -> None:
    """Windows, inputs and scenes come out joined by name, with scenes deduplicated."""
    run = tmp_path / "run"
    # Block at pixel (32768, -540672): windows 16_-264 (written) and 17_-264 (no data).
    _write_block(
        run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264", "17_-264"]
    )
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2025, out_dir=str(out), workers=2)

    windows = pq.read_table(out / "windows" / "year=2025").to_pylist()
    assert [(w["window"], w["written"]) for w in windows] == [
        ("16_-264", True),
        ("17_-264", False),
    ]
    first = windows[0]
    assert (first["xmin"], first["xmax"]) == (327680.0, 348160.0)
    # Northing is the negated pixel row, so the northern edge is the smaller row.
    assert (first["ymin"], first["ymax"]) == (5386240.0, 5406720.0)
    assert first["context_pixels"] == 0
    assert first["code_version"] is None
    footprint = shapely.from_wkb(first["geometry"])
    assert footprint.bounds[0] == pytest.approx(first["bbox"]["xmin"])
    assert 0 < footprint.centroid.x < 3 and 48 < footprint.centroid.y < 49

    inputs = pq.read_table(out / "inputs" / "year=2025").to_pylist()
    assert len(inputs) == 2 * 3
    s2 = [r for r in inputs if r["window"] == "16_-264" and r["sensor"] == "sentinel2_l2a"]
    assert [(r["rank"], r["scene_id"]) for r in s2] == [(0, "S2_A"), (1, "S2_B")]
    assert s2[0]["period_start"].isoformat() == "2025-12-02T00:00:00+00:00"

    scenes = pq.read_table(out / "scenes" / "year=2025").to_pylist()
    assert [s["scene_id"] for s in scenes] == ["S1_A", "S2_A", "S2_B"]
    assert scenes[1]["acquired"].isoformat() == "2025-12-30T10:57:56+00:00"


def test_tables_carry_geoparquet_metadata(tmp_path: pathlib.Path) -> None:
    """The geometry tables declare WKB in OGC:CRS84 with a bbox covering column."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2025, out_dir=str(out), workers=1)

    for table in ("windows", "scenes"):
        (part,) = (out / table / "year=2025").iterdir()
        geo = json.loads(pq.read_schema(part).metadata[b"geo"])
        column = geo["columns"]["geometry"]
        assert geo["primary_column"] == "geometry"
        assert column["encoding"] == "WKB"
        assert "crs" not in column, "omitting crs means OGC:CRS84"
        assert column["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
        assert column["geometry_types"] == ["Polygon"]


def test_margin_and_code_version_are_carried(tmp_path: pathlib.Path) -> None:
    """A padded block records its margin and code; its window bounds must match it."""
    run = tmp_path / "run"
    _write_block(
        run,
        2024,
        (32768, -540672),
        [[32768, -540672]],
        ["16_-264"],
        provenance_extra={
            "context_pixels": 48,
            "code_version": "rslp=505ef4b2",
            "window_bounds": {"16_-264": [32720, -540720, 34864, -538576]},
        },
    )
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2024, out_dir=str(out), workers=1)
    (window,) = pq.read_table(out / "windows" / "year=2024").to_pylist()
    assert window["context_pixels"] == 48
    assert window["code_version"] == "rslp=505ef4b2"

    _write_block(
        run,
        2024,
        (32768, -540672),
        [[32768, -540672]],
        ["16_-264"],
        provenance_extra={
            "context_pixels": 48,
            "window_bounds": {"16_-264": [32768, -540672, 34816, -538624]},
        },
    )
    with pytest.raises(ValueError, match="not its crop"):
        export_provenance(run_path=str(run), year=2024, out_dir=str(out), workers=1)


def test_blocks_without_a_marker_are_left_out(tmp_path: pathlib.Path) -> None:
    """A provenance file with no marker may be newer than the block's embeddings."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    _write_block(
        run, 2025, (40960, -540672), [[40960, -540672]], ["20_-264"], with_marker=False
    )
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2025, out_dir=str(out), workers=1)
    windows = pq.read_table(out / "windows" / "year=2025").to_pylist()
    assert [w["block"] for w in windows] == [f"{CRS}_32768_-540672"]
