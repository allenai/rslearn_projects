"""Tests for the web-mercator reprojection grid and cascade."""

import pathlib

import numpy as np
import pytest

from rslp.large_scale_embeddings import render_web_pca as rw


def test_level_is_the_xyz_zoom() -> None:
    """A level must be a web-map zoom, or the viewer needs a translation table."""
    # The standard XYZ scheme: zoom z is 2**z tiles of 256 px across the world.
    for zoom in (0, 5, 14):
        assert rw.web_size(zoom) == 256 * 2**zoom
    # And the resolution that falls out of that, ~156 km/px at z0 down to ~9.55 m at z14.
    assert rw.web_resolution(0) == pytest.approx(156543.03, abs=0.01)
    assert rw.web_resolution(14) == pytest.approx(9.5546, abs=0.001)


def test_each_level_is_twice_as_coarse() -> None:
    """The whole point of the layout: resolution halves, so ground per shard doubles."""
    for zoom in range(1, 15):
        assert rw.web_resolution(zoom - 1) == pytest.approx(2 * rw.web_resolution(zoom))


def test_shards_tile_the_world_without_gaps() -> None:
    """Adjacent shards must abut exactly; a seam here is a visible seam on the map."""
    zoom = 10
    _, _, max_x, _ = rw.shard_bounds(zoom, 0, 0)
    next_min_x, _, _, _ = rw.shard_bounds(zoom, 0, 1)
    assert max_x == pytest.approx(next_min_x)
    _, min_y, _, _ = rw.shard_bounds(zoom, 0, 0)
    _, _, _, next_max_y = rw.shard_bounds(zoom, 1, 0)
    assert min_y == pytest.approx(next_max_y)
    # The grid starts at the world's corner.
    first = rw.shard_bounds(zoom, 0, 0)
    assert first[0] == pytest.approx(-rw.WEB_HALF_EXTENT)
    assert first[3] == pytest.approx(rw.WEB_HALF_EXTENT)


def test_ground_per_shard_doubles_per_level() -> None:
    """This is what makes object count fall 4x per level, unlike the UTM pyramid."""

    def width(zoom: int) -> float:
        min_x, _, max_x, _ = rw.shard_bounds(zoom, 0, 0)
        return max_x - min_x

    for zoom in range(1, 14):
        assert width(zoom - 1) == pytest.approx(2 * width(zoom))


def test_zones_for_bounds_spans_a_boundary() -> None:
    """A span crossing a zone edge must return both, or half a view goes missing."""
    # Kenya straddles the 36/37 boundary at 36E.
    assert rw.zones_for_bounds(35.0, 38.0) == [36, 37]
    assert rw.zones_for_bounds(-122.4, -122.2) == [10]
    # Clamped at the antimeridian rather than running off the end.
    assert rw.zones_for_bounds(-180.0, -179.0) == [1]
    assert rw.zones_for_bounds(179.0, 180.0) == [60]


def test_parent_shards_collapses_four_into_one() -> None:
    """The cascade's arithmetic: a 2x2 block of shards has one parent."""
    assert rw.parent_shards({(0, 0), (0, 1), (1, 0), (1, 1)}) == {(0, 0)}
    assert rw.parent_shards({(4, 6), (5, 7)}) == {(2, 3)}


def test_multiscales_carries_resolution_not_just_an_index() -> None:
    """Two level conventions exist in this codebase; readers must key off metres."""
    ms = rw.build_multiscales(8, 14)
    assert [m["zoom"] for m in ms] == list(range(8, 15))
    assert all("resolution" in m for m in ms)
    # Coarsest first, so resolution descends.
    res = [m["resolution"] for m in ms]
    assert res == sorted(res, reverse=True)


def test_web_shards_for_source_lands_near_the_source() -> None:
    """A UTM source shard must map to output shards covering the same ground."""
    from pyproj import Transformer

    # A source shard in zone 10, in the UTM shard grid.
    sy, sx = 3000, 20
    got = rw.web_shards_for_source({(sy, sx)}, 10, 12)
    assert got, "a written source shard must produce at least one output shard"
    # Its centre in 3857 must fall inside one of the returned shards.
    span = 2048 * rw.UTM_RES
    cx = rw.UTM_ORIGIN_X + (sx + 0.5) * span
    cy = rw.UTM_ORIGIN_Y - (sy + 0.5) * span
    to_web = Transformer.from_crs("EPSG:32610", "EPSG:3857", always_xy=True)
    wx, wy = to_web.transform(cx, cy)
    hit = [
        (r, c)
        for r, c in got
        if (b := rw.shard_bounds(12, r, c))[0] <= wx <= b[2] and b[1] <= wy <= b[3]
    ]
    assert hit, f"centre {wx},{wy} fell outside every returned shard {sorted(got)}"


def test_downsample_averages_and_ignores_nodata(tmp_path: pathlib.Path) -> None:
    """Averaging must skip nodata, or coastlines darken as they coarsen."""
    import zarr

    store = str(tmp_path / "w.zarr")
    rw.init_web_store(store, years=[2020], min_zoom=12, max_zoom=13)
    g = zarr.open_group(store, mode="a")
    fine = g[rw.web_array_name(13)]

    # A 2x2 block: three valid pixels of 100 and one nodata. The mean of the valid
    # ones is 100; including nodata as zero would give 75.
    block = np.zeros((3, 2, 2), dtype=np.uint8)
    block[:, 0, 0] = 100
    block[:, 0, 1] = 100
    block[:, 1, 0] = 100
    fine[0, :, 0:2, 0:2] = block

    assert rw.downsample_shard(g, 12, 0, 0, 0) > 0
    out = np.asarray(g[rw.web_array_name(12)][0, :, 0, 0])
    assert list(out) == [100, 100, 100]


def test_downsample_of_empty_writes_nothing(tmp_path: pathlib.Path) -> None:
    """An empty region must not be written, so unused shards never materialise."""
    import zarr

    store = str(tmp_path / "w.zarr")
    rw.init_web_store(store, years=[2020], min_zoom=12, max_zoom=13)
    g = zarr.open_group(store, mode="a")
    assert rw.downsample_shard(g, 12, 0, 5, 5) == 0


def test_init_rejects_an_inverted_zoom_range(tmp_path: pathlib.Path) -> None:
    """A silently empty pyramid would be much harder to notice than an error."""
    with pytest.raises(ValueError, match="below min_zoom"):
        rw.init_web_store(
            str(tmp_path / "w.zarr"), years=[2020], min_zoom=10, max_zoom=8
        )


def test_source_url_accepts_both_bucket_url_forms() -> None:
    """run_all hands over a gs:// path; requiring https:// failed the driver on Beaker.

    Both spellings name the same object, so the listing must derive a bucket and prefix
    from either. This is checked on the parse alone -- no network -- because the bug was
    a rejected string, not a bad request.
    """
    import re

    for url in (
        "gs://bucket/a/b/pca_v1.zarr",
        "https://storage.googleapis.com/bucket/a/b/pca_v1.zarr",
        "gs://bucket/a/b/pca_v1.zarr/",
    ):
        m = re.match(
            r"https://storage\.googleapis\.com/([^/]+)/(.+)", url.rstrip("/")
        ) or re.match(r"gs://([^/]+)/(.+)", url.rstrip("/"))
        assert m, f"{url} should parse"
        assert m.group(1) == "bucket"
        assert m.group(2) == "a/b/pca_v1.zarr"

    with pytest.raises(ValueError, match="expected a gs://"):
        rw.source_shard_positions("s3://bucket/a", "utm10")


def test_enumeration_and_writer_agree_on_the_marker_name() -> None:
    """Enumeration skips by name, so a drift here silently rebuilds everything forever."""
    fname = rw.web_marker_fname("gs://b/done", zoom=14, time_index=2, row=7, col=9)
    assert fname.name == rw.web_marker_name(2, 7, 9)
    assert fname.parent.name == "z14"


def test_handover_share_ramps_between_the_two_zooms() -> None:
    """Pure base up to the start zoom, pure overlay from the end zoom, linear between."""
    assert [rw.handover_share(z, 5, 9) for z in (0, 5, 6, 7, 8, 9, 14)] == [
        0.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
    ]
    with pytest.raises(ValueError, match="must be above"):
        rw.handover_share(7, 9, 9)


def test_blend_keeps_either_footprint_and_never_writes_nodata() -> None:
    """A pixel valid in only one scheme keeps that scheme; a mix never becomes nodata."""
    base = np.zeros((3, 1, 3), dtype=np.uint8)
    overlay = np.zeros((3, 1, 3), dtype=np.uint8)
    base[:, 0, 0], overlay[:, 0, 0] = 100, 200  # both valid: mixed
    base[:, 0, 1] = 50  # only base
    overlay[:, 0, 2] = 1  # only overlay
    out = rw.blend_rgb(base, overlay, 0.25)
    assert list(out[:, 0, 0]) == [125, 125, 125]
    assert list(out[:, 0, 1]) == [50, 50, 50]
    assert list(out[:, 0, 2]) == [1, 1, 1]
    assert (
        rw.blend_rgb(
            np.full((3, 1, 1), 1, np.uint8), np.full((3, 1, 1), 1, np.uint8), 0.5
        )
        == 1
    ).all()


def test_compose_copies_or_blends_by_zoom(tmp_path: pathlib.Path) -> None:
    """Shallow zooms copy the base, deep ones the overlay, and the ramp blends them."""
    import json

    import zarr

    stores = {k: str(tmp_path / f"{k}.zarr") for k in ("base", "overlay", "out")}
    for path in stores.values():
        rw.init_web_store(path, years=[2025], min_zoom=5, max_zoom=9)
    for zoom in (5, 7, 9):
        zarr.open_group(stores["base"], mode="a")[rw.web_array_name(zoom)][
            0, :, 0:2, 0:2
        ] = 40
        zarr.open_group(stores["overlay"], mode="a")[rw.web_array_name(zoom)][
            0, :, 0:2, 0:2
        ] = 200

    done = str(tmp_path / "done")
    for zoom in (5, 7, 9):
        rw.compose_web_shard_pipeline(
            base_store_path=stores["base"],
            overlay_store_path=stores["overlay"],
            web_store_path=stores["out"],
            completed_path=done,
            zoom=zoom,
            time_index=0,
            shard_row=0,
            shard_col=0,
            start_zoom=5,
            end_zoom=9,
        )
    out = zarr.open_group(stores["out"], mode="r")
    got = {z: int(out[rw.web_array_name(z)][0, 0, 0, 0]) for z in (5, 7, 9)}
    assert got == {5: 40, 7: 120, 9: 200}
    marker = rw.web_marker_fname(done, 7, 0, 0, 0)
    assert json.loads(marker.read_text())["overlay_share"] == 0.5


def test_gs_listing_uses_credentials_not_anonymous_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CDN-only bucket refuses anonymous listing, so gs:// must go through fsspec."""
    prefix = "bucket/a/pca.zarr/utm10/pca_rgb/c/"

    class FakeGcs:
        def find(self, path: str) -> list[str]:
            assert path == prefix
            return [prefix + "0/0/3/4", prefix + "0/0/3/5", prefix + "zarr.json"]

    monkeypatch.setattr(rw.fsspec, "filesystem", lambda protocol: FakeGcs())

    def no_anonymous(*args: object, **kwargs: object) -> None:
        raise AssertionError("gs:// must not be listed over anonymous https")

    monkeypatch.setattr(rw.urllib.request, "urlopen", no_anonymous)
    assert rw.source_shard_positions("gs://bucket/a/pca.zarr", "utm10") == {
        (3, 4),
        (3, 5),
    }


def _spy_writable_store(monkeypatch: pytest.MonkeyPatch, module: object) -> list[str]:
    """Record every store path a module opens for writing, passing it through unchanged."""
    from rslp.large_scale_embeddings import zarr_store

    opened: list[str] = []

    def spy(
        path: str, storage_options: object = None, *args: object, **kwargs: object
    ) -> object:
        opened.append(path)
        return zarr_store.writable_store(path, storage_options, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "writable_store", spy)
    return opened


def test_web_writers_stamp_cache_headers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every web-store write goes through writable_store, which stamps Cache-Control remotely."""
    import zarr

    opened = _spy_writable_store(monkeypatch, rw)
    stores = {k: str(tmp_path / f"{k}.zarr") for k in ("base", "overlay", "out")}
    for path in stores.values():
        rw.init_web_store(path, years=[2025], min_zoom=6, max_zoom=7)
    assert sorted(opened) == sorted(stores.values())

    for key in ("base", "overlay"):
        zarr.open_group(stores[key], mode="a")[rw.web_array_name(7)][0, :, 0:2, 0:2] = 9
    opened.clear()
    rw.compose_web_shard_pipeline(
        base_store_path=stores["base"],
        overlay_store_path=stores["overlay"],
        web_store_path=stores["out"],
        completed_path=str(tmp_path / "done"),
        zoom=7,
        time_index=0,
        shard_row=0,
        shard_col=0,
        start_zoom=5,
        end_zoom=9,
    )
    rw.render_web_pca_pipeline(
        source_store_path="unused-below-base-zoom",
        web_store_path=stores["out"],
        completed_path=str(tmp_path / "done_web"),
        zoom=6,
        time_index=0,
        shard_row=0,
        shard_col=0,
        zone_numbers=[10],
        base_zoom=7,
    )
    assert opened == [stores["out"], stores["out"]]
