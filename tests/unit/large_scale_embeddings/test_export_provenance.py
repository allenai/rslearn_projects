import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import shapely

from rslp.large_scale_embeddings.export_provenance import (
    _split_antimeridian,
    export_provenance,
)
from rslp.large_scale_embeddings.render_web_pca import UTM_ORIGIN_X, UTM_ORIGIN_Y

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
    crs: str = CRS,
) -> None:
    x, y = origin
    name = f"{crs}_{x}_{y}.json"
    projection = {"crs": crs, "x_resolution": 10, "y_resolution": -10}
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
    """Windows, window_scenes and scenes come out joined by ID, with scenes deduplicated."""
    run = tmp_path / "run"
    # Block at pixel (32768, -540672): windows 16_-264 (written) and 17_-264 (no data).
    _write_block(
        run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264", "17_-264"]
    )
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=2, resolve_sources=False
    )

    windows = sorted(
        pq.read_table(out / "windows" / "2025.parquet").to_pylist(),
        key=lambda w: w["col"],
    )
    # Zone 31, chunk (16, 200) of utm31/embeddings: rslearn row -264 plus 464.
    assert [(w["window_id"], w["written"]) for w in windows] == [
        ("31_16_200", True),
        ("31_17_200", False),
    ]
    assert {w["year"] for w in windows} == {2025}
    first = windows[0]
    assert (first["utm_xmin"], first["utm_xmax"]) == (327680, 348160)
    # Northing is the negated pixel row, so the northern edge is the smaller row.
    assert (first["utm_ymin"], first["utm_ymax"]) == (5386240, 5406720)
    assert first["context_pixels"] == 0
    assert first["inference_crop_size"] == 480
    footprint = shapely.from_wkb(first["geometry"])
    # The float32 covering still contains the footprint.
    bbox = first["bbox"]
    assert bbox["xmin"] <= footprint.bounds[0] and bbox["ymin"] <= footprint.bounds[1]
    assert bbox["xmax"] >= footprint.bounds[2] and bbox["ymax"] >= footprint.bounds[3]
    assert footprint.bounds[0] == pytest.approx(bbox["xmin"], abs=1e-5)
    assert 0 < footprint.centroid.x < 3 and 48 < footprint.centroid.y < 49

    window_scenes = pq.read_table(out / "window_scenes" / "2025.parquet").to_pylist()
    assert len(window_scenes) == 2 * 3
    s2 = [
        r
        for r in window_scenes
        if r["window_id"] == "31_16_200" and r["modality"] == "sentinel2_l2a"
    ]
    assert [(r["mosaic_order"], r["scene_id"]) for r in s2] == [
        (0, "S2_A"),
        (1, "S2_B"),
    ]
    assert s2[0]["period_start"].isoformat() == "2025-12-02"
    assert s2[0]["period_end"].isoformat() == "2026-01-01"

    scenes = pq.read_table(out / "scenes" / "2025.parquet").to_pylist()
    assert [s["scene_id"] for s in scenes] == ["S1_A", "S2_A", "S2_B"]
    assert {s["year"] for s in scenes} == {2025}
    assert scenes[1]["collected_at"].isoformat() == "2025-12-30T10:57:56+00:00"


def test_window_key_is_the_zarr_shard(tmp_path: pathlib.Path) -> None:
    """col and row index the window's shard in utm<NN>/embeddings."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )
    (window,) = pq.read_table(out / "windows" / "2025.parquet").to_pylist()
    span = 2048 * 10
    assert window["utm_xmin"] == UTM_ORIGIN_X + window["col"] * span
    assert window["utm_ymax"] == UTM_ORIGIN_Y - window["row"] * span


def test_window_scenes_are_sorted_by_zone_then_window(tmp_path: pathlib.Path) -> None:
    """A reader finds one window's scenes by row group statistics on its key."""
    run = tmp_path / "run"
    # Listed in marker name order: zone 32 sorts after 31, block -548864 before
    # -540672, so the file order differs from the name order.
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["17_-264"])
    _write_block(run, 2025, (32768, -548864), [[32768, -548864]], ["16_-268"])
    _write_block(
        run,
        2025,
        (32768, -540672),
        [[32768, -540672]],
        ["16_-264"],
        crs="EPSG:32632",
    )
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=2, resolve_sources=False
    )
    rows = pq.read_table(out / "window_scenes" / "2025.parquet").to_pylist()
    keys = [(r["zone"], r["row"], r["col"]) for r in rows]
    assert keys == sorted(keys)
    assert list(dict.fromkeys(r["window_id"] for r in rows)) == [
        "31_16_196",
        "31_17_200",
        "32_16_200",
    ]
    # Within a window, mosaic order is kept.
    s2 = [r["mosaic_order"] for r in rows[:3] if r["modality"] == "sentinel2_l2a"]
    assert s2 == [0, 1]


def test_footprints_past_the_antimeridian_are_split() -> None:
    """An unwrapped footprint becomes two polygons inside ±180, with the same area."""
    unwrapped = shapely.box(178.0, -18.0, 180.7, -16.0)
    split = _split_antimeridian(unwrapped)
    assert split.geom_type == "MultiPolygon"
    assert sorted(p.bounds for p in split.geoms) == [
        pytest.approx((-180.0, -18.0, -179.3, -16.0)),
        pytest.approx((178.0, -18.0, 180.0, -16.0)),
    ]
    assert split.area == pytest.approx(unwrapped.area)
    inside = shapely.box(10, 10, 11, 11)
    assert _split_antimeridian(inside) is inside


def test_tables_carry_geoparquet_metadata(tmp_path: pathlib.Path) -> None:
    """The geometry tables declare WKB in OGC:CRS84 with a bbox covering column."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )

    for table in ("windows", "scenes"):
        geo = json.loads(pq.read_schema(out / table / "2025.parquet").metadata[b"geo"])
        column = geo["columns"]["geometry"]
        assert geo["primary_column"] == "geometry"
        assert column["encoding"] == "WKB"
        assert "crs" not in column, "omitting crs means OGC:CRS84"
        assert column["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
        assert column["geometry_types"] == ["Polygon"]


def test_margin_is_carried(tmp_path: pathlib.Path) -> None:
    """A padded block records its margin; its window bounds must match it."""
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
    (window,) = pq.read_table(out / "windows" / "2024.parquet").to_pylist()
    assert window["context_pixels"] == 48

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
    windows = pq.read_table(out / "windows" / "2025.parquet").to_pylist()
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

    def delete_then_read(
        marker: dict, provenance: dict, block: str, year: int
    ) -> tuple:
        gone.unlink(missing_ok=True)
        return real_block_rows(marker, provenance, block, year)

    monkeypatch.setattr(module, "_block_rows", delete_then_read)
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )
    windows = pq.read_table(out / "windows" / "2025.parquet").to_pylist()
    assert [w["block"] for w in windows] == [f"{CRS}_32768_-540672"]


def _record(
    scene_id: str, providers: dict, collection: str = "c", **properties: object
) -> dict:
    return {
        "id": scene_id,
        "properties": properties,
        "data_providers": {
            name: {
                "id": vendor_id,
                "collection": collection,
                "assets": [
                    {"name": "B1", "url": f"{prefix}{vendor_id}_B1.tif"},
                    {"name": "B2", "url": f"{prefix}{vendor_id}_B2.tif"},
                ],
            }
            for name, (vendor_id, prefix) in providers.items()
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
            collection="sentinel-2-l2a",
            product_id="S2_A_N0511_ESA",
            cloud_cover=3.5,
            satellite="S2A",
        ),
        "S2_B": _record(
            "S2_B", {"planetary_computer": ("S2_B_pc", "https://pc.example/b/")}
        ),
        "S1_A": _record(
            "S1_A", {"planetary_computer": ("S1_A_rtc", "https://pc.example/s1/")}
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
        for s in pq.read_table(out / "scenes" / "2025.parquet").to_pylist()
    }
    s2 = scenes["S2_A"]
    assert (s2["data_provider"], s2["collection"], s2["vendor_id"]) == (
        "planetary_computer",
        "sentinel-2-l2a",
        "S2_A_pc",
    )
    assert s2["stac_item_url"] == (
        "https://planetarycomputer.microsoft.com/api/stac/v1/collections/"
        "sentinel-2-l2a/items/S2_A_pc"
    )
    assert s2["product_id"] == "S2_A_N0511_ESA"
    assert s2["cloud_cover"] == 3.5
    assert s2["satellite"] == "S2A"
    # Only Sentinel-2 has a SAFE name; elsewhere vendor_id already names the product.
    assert scenes["S1_A"]["product_id"] is None
    assert scenes["S1_A"]["vendor_id"] == "S1_A_rtc"
    assert all(s["vendor_id"] for s in scenes.values())


def test_landsat_links_to_the_usgs_catalog() -> None:
    """Landsat read from Google's bucket is catalogued by USGS, not by the service."""
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    vendor_id = "LC08_L1GT_001004_20250404_20250411_02_T2"
    assert module._stac_item_url(
        "gcs_usgs_landsat", "landsat-8-9-c2-l1", vendor_id
    ) == (
        "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/"
        + vendor_id
    )
    assert module._stac_item_url("some_mirror", "c", vendor_id) is None


def test_a_scene_with_two_allowed_providers_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two providers allowed the reader picked one per read, so none is known."""
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    record = _record(
        "S1_A",
        {
            "planetary_computer": ("S1_A_rtc", "https://pc.example/s1/"),
            "gcs_mirror": ("S1_A_rtc", "gs://mirror/s1/"),
        },
    )
    monkeypatch.setattr(
        module,
        "_lookup_records",
        lambda ids, url, token: {"S1_A": record} if "S1_A" in ids else {},
    )
    monkeypatch.setenv(module.DATASETS_API_TOKEN_ENV_VAR, "token")
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    with pytest.raises(ValueError, match="2 allowed providers"):
        export_provenance(
            run_path=str(run), year=2025, out_dir=str(tmp_path / "out"), workers=1
        )


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
    scenes = pq.read_table(out / "scenes" / "2025.parquet").to_pylist()
    assert scenes and not any(s["vendor_id"] or s["data_provider"] for s in scenes)
    assert all(s["collected_at"] for s in scenes)


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


def test_a_period_off_midnight_is_refused() -> None:
    """A period stored as a date must not silently drop a time of day."""
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    assert module._date("2025-12-02T00:00:00+00:00").isoformat() == "2025-12-02"
    with pytest.raises(ValueError, match="not midnight UTC"):
        module._date("2025-12-02T06:00:00+00:00")


def test_every_column_carries_a_description(tmp_path: pathlib.Path) -> None:
    """Each field's description is written as Arrow field metadata."""
    run = tmp_path / "run"
    _write_block(run, 2025, (32768, -540672), [[32768, -540672]], ["16_-264"])
    out = tmp_path / "out"
    export_provenance(
        run_path=str(run), year=2025, out_dir=str(out), workers=1, resolve_sources=False
    )
    for table in ("windows", "window_scenes", "scenes"):
        schema = pq.read_schema(out / table / "2025.parquet")
        for field in schema:
            assert (field.metadata or {}).get(b"description"), (table, field.name)


def test_float32_bbox_always_contains_the_geometry() -> None:
    """Each bound rounds away from the geometry, whichever way float32 rounds it."""
    module = sys.modules["rslp.large_scale_embeddings.export_provenance"]
    rng = np.random.default_rng(0)
    for x0, y0 in rng.uniform([-179, -89], [178, 88], size=(2000, 2)):
        box = shapely.box(x0, y0, x0 + rng.uniform(0.01, 1), y0 + rng.uniform(0.01, 1))
        bbox = module._bbox(box)
        xmin, ymin, xmax, ymax = box.bounds
        assert bbox["xmin"] <= xmin and bbox["ymin"] <= ymin
        assert bbox["xmax"] >= xmax and bbox["ymax"] >= ymax
        assert all(float(np.float32(v)) == v for v in bbox.values())
