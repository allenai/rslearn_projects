import json
import pathlib
from typing import Any, cast

import numpy as np
import pytest
import zarr

from rslp.olmoearth_lcc.lcc_model import geozarr, geozarr_pyramid
from rslp.olmoearth_lcc.lcc_model.output_bands import NONE_CATEGORY_IDX
from rslp.olmoearth_lcc.lcc_model.tiling import (
    RESOLUTION,
    TILE_SIZE,
    enumerate_zone_tiles,
    get_zone_grid,
    get_zone_projection,
)

BINARY, PRE_LC, POST_LC, PRE_CAT, POST_CAT, PRE_SCORE, POST_SCORE = range(7)


def _pixel_block(pixels: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2x2 change array from four [binary, pre_cat, post_cat] pixels.

    The other bands hold the pixel index so the chosen pixel can be identified.
    """
    change = np.zeros((len(geozarr.CHANGE_BANDS), 2, 2), dtype=np.uint8)
    dates = np.zeros((len(geozarr.CHANGE_DATES_BANDS), 2, 2), dtype=np.uint16)
    for i, (binary, pre_cat, post_cat) in enumerate(pixels):
        row, col = divmod(i, 2)
        change[:, row, col] = [binary, i, i, pre_cat, post_cat, i, i]
        dates[:, row, col] = [1000 + i, 2000 + i]
    return change, dates


@pytest.mark.parametrize(
    "pixels,expected",
    [
        # A real change beats a higher binary change probability.
        ([[250, 1, 1], [40, 1, 5], [0, 1, 1], [0, 1, 1]], 1),
        # Without real changes, the highest binary change wins.
        ([[20, 1, 1], [40, 1, 1], [90, 1, 1], [0, 0, 0]], 2),
        # A valid pixel (binary_change stored as 0) beats nodata.
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]], 3),
        # A pre-only change also counts as a real change.
        ([[200, 1, 1], [0, 1, 1], [30, 4, 1], [0, 1, 1]], 2),
    ],
)
def test_reduce_2x_picks_one_pixel(pixels: list[list[int]], expected: int) -> None:
    change, dates = _pixel_block(pixels)
    reduced_change, reduced_dates = geozarr_pyramid.reduce_2x(change, dates)
    assert reduced_change.shape == (len(geozarr.CHANGE_BANDS), 1, 1)
    # All bands come from the same pixel.
    np.testing.assert_array_equal(
        reduced_change[:, 0, 0], change[:, expected // 2, expected % 2]
    )
    np.testing.assert_array_equal(
        reduced_dates[:, 0, 0], [1000 + expected, 2000 + expected]
    )


def test_reduce_2x_blocks_are_independent() -> None:
    rng = np.random.default_rng(0)
    change = rng.integers(0, 255, (len(geozarr.CHANGE_BANDS), 8, 6), dtype=np.uint8)
    dates = rng.integers(0, 65535, (2, 8, 6), dtype=np.uint16)
    reduced, _ = geozarr_pyramid.reduce_2x(change, dates)
    assert reduced.shape == (len(geozarr.CHANGE_BANDS), 4, 3)
    for row in range(4):
        for col in range(3):
            block_change, block_dates = (
                change[:, 2 * row : 2 * row + 2, 2 * col : 2 * col + 2],
                dates[:, 2 * row : 2 * row + 2, 2 * col : 2 * col + 2],
            )
            single, _ = geozarr_pyramid.reduce_2x(block_change, block_dates)
            np.testing.assert_array_equal(reduced[:, row, col], single[:, 0, 0])
    with pytest.raises(ValueError):
        geozarr_pyramid.reduce_2x(change[:, :7], dates[:, :7])


def test_init_pyramid_store(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "pyramid.zarr")
    geozarr_pyramid.init_pyramid_store(
        store_path, "gs://bucket/lcc.zarr", zone_numbers=[10]
    )
    assert (
        geozarr_pyramid.read_pyramid_source_store(store_path) == "gs://bucket/lcc.zarr"
    )
    assert geozarr_pyramid.read_pyramid_source_store(str(tmp_path / "missing")) is None

    zone_group = zarr.open_group(store_path, path="utm10", mode="r")
    assert zone_group.attrs["source_group"] == "utm10"
    conventions = cast(list[dict[str, Any]], zone_group.attrs["zarr_conventions"])
    assert "multiscales" in [c["name"] for c in conventions]
    multiscales = cast(dict[str, Any], zone_group.attrs["multiscales"])
    layout = multiscales["layout"]
    assert [entry["asset"] for entry in layout] == ["1", "2", "3", "4", "5"]
    assert "derived_from" not in layout[0]
    assert layout[1]["derived_from"] == "1"
    assert layout[1]["transform"]["scale"] == [2.0, 2.0]

    _, origin_px, shape_px = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    for level in geozarr_pyramid.PYRAMID_LEVELS:
        scale = 2**level
        group = zarr.open_group(store_path, path=f"utm10/{level}", mode="r")
        transform = cast(list[float], group.attrs["spatial:transform"])
        assert transform[0] == RESOLUTION * scale
        assert transform[2] == origin_px[0] * RESOLUTION
        assert group.attrs["spatial:shape"] == [
            shape_px[0] // scale,
            shape_px[1] // scale,
        ]
        assert layout[level - 1]["spatial:transform"] == transform
        change = group[geozarr.CHANGE_ARRAY]
        assert isinstance(change, zarr.Array)
        shard = min(2048, TILE_SIZE // scale)
        assert change.shards == (len(geozarr.CHANGE_BANDS), shard, shard)
        assert change.chunks == (len(geozarr.CHANGE_BANDS), 512, 512)


def test_init_pyramid_store_consolidated_metadata(tmp_path: pathlib.Path) -> None:
    store_path = tmp_path / "pyramid.zarr"
    geozarr_pyramid.init_pyramid_store(
        str(store_path), "lcc.zarr", zone_numbers=[10, 11]
    )
    root = json.loads((store_path / "zarr.json").read_text())
    consolidated = root["consolidated_metadata"]["metadata"]
    for zone in ["utm10", "utm11"]:
        own = json.loads((store_path / zone / "zarr.json").read_text())
        assert "multiscales" in own["attributes"]
        assert consolidated[zone]["attributes"] == own["attributes"]
        for level in geozarr_pyramid.PYRAMID_LEVELS:
            level_path = f"{zone}/{level}"
            own = json.loads((store_path / level_path / "zarr.json").read_text())
            assert consolidated[level_path]["attributes"] == own["attributes"]


def test_init_pyramid_store_existing(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "pyramid.zarr")
    geozarr_pyramid.init_pyramid_store(store_path, "a.zarr", zone_numbers=[10])
    with pytest.raises(FileExistsError):
        geozarr_pyramid.init_pyramid_store(store_path, "b.zarr", zone_numbers=[10])
    geozarr_pyramid.init_pyramid_store(
        store_path, "b.zarr", zone_numbers=[10], overwrite=True
    )
    assert geozarr_pyramid.read_pyramid_source_store(store_path) == "b.zarr"


def test_ensure_pyramid_store(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    pyramid_path = str(tmp_path / "pyramid.zarr")
    # The main store must exist.
    with pytest.raises(ValueError):
        geozarr_pyramid.ensure_pyramid_store(pyramid_path, store_path)

    geozarr.init_store(store_path, zone_numbers=[10])
    geozarr_pyramid.init_pyramid_store(pyramid_path, store_path, zone_numbers=[10])
    geozarr_pyramid.ensure_pyramid_store(pyramid_path, store_path)
    # An existing pyramid store for another main store fails.
    other_path = str(tmp_path / "other.zarr")
    geozarr.init_store(other_path, zone_numbers=[10])
    with pytest.raises(ValueError):
        geozarr_pyramid.ensure_pyramid_store(pyramid_path, other_path)


def test_write_tile_pyramid(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "pyramid.zarr")
    geozarr_pyramid.init_pyramid_store(store_path, "lcc.zarr", zone_numbers=[10])
    projection = get_zone_projection(10, RESOLUTION)
    col, row = next(enumerate_zone_tiles(10, RESOLUTION, TILE_SIZE))
    bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        (col + 1) * TILE_SIZE,
        (row + 1) * TILE_SIZE,
    )
    half = TILE_SIZE // 2
    change = np.zeros((len(geozarr.CHANGE_BANDS), half, half), dtype=np.uint8)
    change[PRE_CAT] = NONE_CATEGORY_IDX
    change[POST_CAT] = NONE_CATEGORY_IDX
    # One change pixel in the tile's level-1 array.
    change[:, 100, 200] = [230, 9, 10, NONE_CATEGORY_IDX, 3, 235, 240]
    dates = np.zeros((2, half, half), dtype=np.uint16)
    dates[:, 100, 200] = [19000, 19100]
    geozarr_pyramid.write_tile_pyramid(
        store_path, projection, bounds, [(bounds, change, dates)]
    )

    for level in geozarr_pyramid.PYRAMID_LEVELS:
        scale = 2**level
        size = TILE_SIZE // scale
        level_bounds = tuple(v // scale for v in bounds)
        group = zarr.open_group(store_path, path=f"utm10/{level}", mode="r")
        r0, c0 = geozarr.get_group_offset(group, level_bounds)  # type: ignore[arg-type]
        level_change = group[geozarr.CHANGE_ARRAY]
        assert isinstance(level_change, zarr.Array)
        block = np.asarray(level_change[:, r0 : r0 + size, c0 : c0 + size])
        # The change pixel survives at every level, at its downscaled position.
        factor = scale // 2
        np.testing.assert_array_equal(
            block[:, 100 // factor, 200 // factor],
            [230, 9, 10, NONE_CATEGORY_IDX, 3, 235, 240],
        )
        assert (block[POST_CAT] == 3).sum() == 1
        assert (block[PRE_CAT] == NONE_CATEGORY_IDX).all()

    # The tile writes whole shards only: one shard per tile at the coarse levels.
    for level in [3, 4, 5]:
        chunk_dir = tmp_path / "pyramid.zarr" / "utm10" / str(level) / "change" / "c"
        assert len([p for p in chunk_dir.rglob("*") if p.is_file()]) == 1

    with pytest.raises(ValueError):
        geozarr_pyramid.write_tile_pyramid(
            store_path,
            projection,
            (bounds[0] + 2048, bounds[1], bounds[2] + 2048, bounds[3]),
            [(bounds, change, dates)],
        )
