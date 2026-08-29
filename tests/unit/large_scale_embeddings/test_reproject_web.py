"""Tests for the web-mercator reprojection grid and cascade."""

import numpy as np
import pytest

from rslp.large_scale_embeddings import reproject_web as rw


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


def test_downsample_averages_and_ignores_nodata(tmp_path) -> None:
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


def test_downsample_of_empty_writes_nothing(tmp_path) -> None:
    """An empty region must not be written, so unused shards never materialise."""
    import zarr

    store = str(tmp_path / "w.zarr")
    rw.init_web_store(store, years=[2020], min_zoom=12, max_zoom=13)
    g = zarr.open_group(store, mode="a")
    assert rw.downsample_shard(g, 12, 0, 5, 5) == 0


def test_init_rejects_an_inverted_zoom_range(tmp_path) -> None:
    """A silently empty pyramid would be much harder to notice than an error."""
    with pytest.raises(ValueError, match="below min_zoom"):
        rw.init_web_store(str(tmp_path / "w.zarr"), years=[2020], min_zoom=10, max_zoom=8)
