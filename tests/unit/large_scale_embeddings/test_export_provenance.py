import json
import pathlib
import sys

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
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=2, resolve_sources=False
    )

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
    s2 = [
        r for r in inputs if r["window"] == "16_-264" and r["sensor"] == "sentinel2_l2a"
    ]
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
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )

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
    export_provenance(
        run_path=str(run), year=2024, out_dir=str(out), workers=1, resolve_sources=False
    )
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
        export_provenance(
            run_path=str(run),
            year=2024,
            out_dir=str(out),
            workers=1,
            resolve_sources=False,
        )


def test_blocks_without_a_marker_are_left_out(tmp_path: pathlib.Path) -> None:
    """A provenance file with no marker may be newer than the block's embeddings."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    _write_block(
        run, 2025, (40960, -540672), [[40960, -540672]], ["20_-264"], with_marker=False
    )
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )
    windows = pq.read_table(out / "windows" / "year=2025").to_pylist()
    assert [w["block"] for w in windows] == [f"{CRS}_32768_-540672"]


def test_a_block_deleted_mid_export_is_skipped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker deleted after the listing, to recompute its block, is left out."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    _write_block(run, 2025, (40960, -540672), [[40960, -540672]], ["20_-264"])
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    real_block_rows = module._block_rows
    gone = run / "completed_2025" / f"{CRS}_40960_-540672.json"

    def delete_then_read(marker: dict, provenance: dict, block: str) -> tuple:
        gone.unlink(missing_ok=True)
        return real_block_rows(marker, provenance, block)

    monkeypatch.setattr(module, "_block_rows", delete_then_read)
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )
    windows = pq.read_table(out / "windows" / "year=2025").to_pylist()
    assert [w["block"] for w in windows] == [f"{CRS}_32768_-540672"]


def _record(scene_id: str, providers: dict, **properties: object) -> dict:
    return {
        "id": scene_id,
        "properties": properties,
        "data_providers": {
            name: {
                "id": item_id,
                "collection": "c",
                "assets": [
                    {"name": "B1", "url": f"{prefix}{item_id}_B1.tif"},
                    {"name": "B2", "url": f"{prefix}{item_id}_B2.tif"},
                ],
            }
            for name, (item_id, prefix) in providers.items()
        },
    }


def test_scenes_resolve_to_the_public_providers_the_config_allows(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each scene names the public data behind it, minus the excluded providers.

    Readers have no access to the datasets service, so its scene IDs alone are not
    enough to find the imagery.
    """
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    records = {
        "S2_A": _record(
            "S2_A",
            {
                "aws_sentinel_2_l2a_cogs": ("S2_A_aws", "s3://bucket/a/"),
                "planetary_computer": ("S2_A_pc", "https://pc.example/a/"),
            },
            product_id="S2_A_N0511_ESA",
            cloud_cover=3.5,
        ),
        "S2_B": _record(
            "S2_B", {"planetary_computer": ("S2_B_pc", "https://pc.example/b/")}
        ),
        "S1_A": _record(
            "S1_A",
            {
                "planetary_computer": ("S1_A_rtc", "https://pc.example/s1/"),
                "gcs_mirror": ("S1_A_rtc", "gs://mirror/s1/"),
            },
        ),
    }
    seen: list[list[str]] = []

    def fake_lookup(scene_ids: list[str], api_url: str, token: str) -> dict:
        seen.append(scene_ids)
        return {i: records[i] for i in scene_ids if i in records}

    monkeypatch.setattr(module, "_lookup_records", fake_lookup)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "layers": {
                    "sentinel2_l2a": {
                        "data_source": {
                            "init_args": {
                                "provider_excludes": ["AWS_SENTINEL_2_L2A_COGS"]
                            }
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(module, "DATASET_CONFIG_FNAME", str(config))
    monkeypatch.setenv(module.DATASETS_API_TOKEN_ENV_VAR, "token")
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2025, out_dir=str(out), workers=1)

    assert seen == [["S1_A", "S2_A", "S2_B"]]
    scenes = {
        s["scene_id"]: s
        for s in pq.read_table(out / "scenes" / "year=2025").to_pylist()
    }
    s2 = scenes["S2_A"]
    assert s2["provider"] == "planetary_computer"
    assert s2["product_id"] == "S2_A_N0511_ESA"
    assert s2["cloud_cover"] == 3.5
    assert s2["sources"] == [
        {
            "provider": "planetary_computer",
            "collection": "c",
            "item_id": "S2_A_pc",
            "asset_prefix": "https://pc.example/a/",
        }
    ]
    # Two providers allowed: which one served a read is not recorded, so none is named,
    # but the ID they share still identifies the product.
    s1 = scenes["S1_A"]
    assert s1["provider"] is None
    assert [src["provider"] for src in s1["sources"]] == [
        "gcs_mirror",
        "planetary_computer",
    ]
    assert s1["product_id"] == "S1_A_rtc"
    assert all(s["resolved"] for s in scenes.values())


def test_a_scene_missing_from_the_service_is_marked_unresolved(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    monkeypatch.setattr(module, "_lookup_records", lambda ids, url, token: {})
    monkeypatch.setenv(module.DATASETS_API_TOKEN_ENV_VAR, "token")
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(run_path=str(run), year=2025, out_dir=str(out), workers=1)
    scenes = pq.read_table(out / "scenes" / "year=2025").to_pylist()
    assert scenes and not any(s["resolved"] for s in scenes)
    assert all(s["sources"] == [] for s in scenes)


def test_resolving_sources_needs_a_token(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    monkeypatch.delenv(module.DATASETS_API_TOKEN_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="DATASETS_API_TOKEN"):
        export_provenance(run_path=str(tmp_path), year=2025, out_dir=str(tmp_path))


def test_the_run_config_excludes_aws_for_s2_and_landsat() -> None:
    """The exclusions the scenes table relies on, read from the real config."""
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    excludes = module.provider_excludes(
        "data/large_scale_embeddings/s2_s1_landsat_distilled.json"
    )
    assert excludes["sentinel2_l2a"] == {"aws_sentinel_2_l2a_cogs"}
    assert excludes["landsat"] == {"aws_usgs_landsat"}
    assert excludes["sentinel1"] == set()
