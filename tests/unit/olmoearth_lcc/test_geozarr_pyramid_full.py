import json
import pathlib

import numpy as np
import pytest
import zarr

from rslp.olmoearth_lcc.lcc_model import geozarr, geozarr_pyramid, geozarr_pyramid_full
from rslp.olmoearth_lcc.lcc_model.output_bands import NONE_CATEGORY_IDX
from rslp.olmoearth_lcc.lcc_model.tiling import (
    RESOLUTION,
    TILE_SIZE,
    enumerate_zone_tiles,
    get_zone_projection,
)

BINARY, PRE_LC, POST_LC, PRE_CAT, POST_CAT, PRE_SCORE, POST_SCORE = range(7)
NONE = NONE_CATEGORY_IDX


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
        # Three "none" pixels outvote one change with a high binary_change.
        ([[250, NONE, 5], [10, NONE, NONE], [20, NONE, NONE], [30, NONE, NONE]], 3),
        # The most common post category wins; then the higher binary_change among them.
        ([[90, NONE, 5], [40, NONE, 7], [60, NONE, 7], [0, NONE, NONE]], 2),
        # A tie between a category and none goes to the category.
        ([[10, NONE, NONE], [5, NONE, 4], [20, NONE, NONE], [7, NONE, 4]], 3),
        # Without a post category, the pre category is the key; post wins over the
        # same number as a pre category (it's a different key), then binary_change.
        ([[10, 3, NONE], [50, 3, NONE], [20, NONE, 3], [0, 0, 0]], 1),
        # A post category counts even when the pixel also has a pre category.
        ([[10, 2, 6], [30, NONE, 6], [99, 2, NONE], [0, 0, 0]], 1),
        # Nodata pixels don't vote: one valid "none" beats three nodata.
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, NONE, NONE]], 3),
        # An all-nodata block stays nodata.
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 0),
    ],
)
def test_reduce_2x_majority(pixels: list[list[int]], expected: int) -> None:
    change, dates = _pixel_block(pixels)
    reduced_change, reduced_dates = geozarr_pyramid_full.reduce_2x_majority(
        change, dates
    )
    assert reduced_change.shape == (len(geozarr.CHANGE_BANDS), 1, 1)
    np.testing.assert_array_equal(
        reduced_change[:, 0, 0], change[:, expected // 2, expected % 2]
    )
    np.testing.assert_array_equal(
        reduced_dates[:, 0, 0], dates[:, expected // 2, expected % 2]
    )


def test_reduce_2x_majority_blocks_are_independent() -> None:
    rng = np.random.default_rng(0)
    change = rng.integers(0, 6, (len(geozarr.CHANGE_BANDS), 8, 6), dtype=np.uint8)
    change[BINARY] = rng.integers(0, 255, (8, 6), dtype=np.uint8)
    dates = rng.integers(0, 65535, (2, 8, 6), dtype=np.uint16)
    reduced, _ = geozarr_pyramid_full.reduce_2x_majority(change, dates)
    assert reduced.shape == (len(geozarr.CHANGE_BANDS), 4, 3)
    for row in range(4):
        for col in range(3):
            single, _ = geozarr_pyramid_full.reduce_2x_majority(
                change[:, 2 * row : 2 * row + 2, 2 * col : 2 * col + 2],
                dates[:, 2 * row : 2 * row + 2, 2 * col : 2 * col + 2],
            )
            np.testing.assert_array_equal(reduced[:, row, col], single[:, 0, 0])
    with pytest.raises(ValueError):
        geozarr_pyramid_full.reduce_2x_majority(change[:, :7], dates[:, :7])


def test_coarse_shard_size() -> None:
    assert geozarr_pyramid_full.coarse_shard_size((28672, 1280)) == 2048
    assert geozarr_pyramid_full.coarse_shard_size((3584, 160)) == 3584
    assert geozarr_pyramid_full.coarse_shard_size((1792, 80)) == 2048
    assert geozarr_pyramid_full.coarse_shard_size((448, 20)) == 512


def _write_source(tmp_path: pathlib.Path, zone: int) -> tuple[str, tuple[int, ...]]:
    """A pyramid store with one tile: a block of post category 3 and one lone change."""
    store_path = str(tmp_path / "pyramid.zarr")
    geozarr_pyramid.init_pyramid_store(store_path, "lcc.zarr", zone_numbers=[zone])
    projection = get_zone_projection(zone, RESOLUTION)
    col, row = next(enumerate_zone_tiles(zone, RESOLUTION, TILE_SIZE))
    bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        (col + 1) * TILE_SIZE,
        (row + 1) * TILE_SIZE,
    )
    half = TILE_SIZE // 2
    change = np.zeros((len(geozarr.CHANGE_BANDS), half, half), dtype=np.uint8)
    change[PRE_CAT] = NONE
    change[POST_CAT] = NONE
    change[BINARY] = 5
    # A 2048 x 2048 (level 1) block of post category 3, i.e. 32 x 32 km.
    change[POST_CAT, :2048, :2048] = 3
    change[BINARY, :2048, :2048] = 200
    # A lone change elsewhere.
    change[:, 5000, 6000] = [230, 9, 10, NONE, 7, 235, 240]
    dates = np.zeros((2, half, half), dtype=np.uint16)
    dates[:, :2048, :2048] = [[[19000]], [[19100]]]
    geozarr_pyramid.write_tile_pyramid(
        store_path, projection, bounds, [(bounds, change, dates)]
    )
    return store_path, bounds


def test_pyramid_full_zone(tmp_path: pathlib.Path) -> None:
    zone = 10
    source, bounds = _write_source(tmp_path, zone)
    dest = str(tmp_path / "pyramid_full.zarr")
    stats_path = str(tmp_path / "stats.json")
    geozarr_pyramid_full.pyramid_full_zone(source, dest, zone, stats_path)
    with pytest.raises(FileExistsError):
        geozarr_pyramid_full.build_pyramid_full_zone(source, dest, zone)

    stats = json.loads(pathlib.Path(stats_path).read_text())
    assert stats["band"] is not None
    assert stats["levels"]["5"]["with_post_category"] > 0

    zone_group = zarr.open_group(dest, path=f"utm{zone:02d}", mode="r")
    layout = zone_group.attrs["multiscales"]["layout"]  # type: ignore[index]
    assert [e["asset"] for e in layout] == [str(x) for x in range(1, 13)]  # type: ignore[index, union-attr]
    assert layout[5]["resampling_method"] == "majority_category_pixel"  # type: ignore[index]
    assert "resampling_method" not in layout[4]  # type: ignore[operator, index]

    for level in range(1, 13):
        scale = 2**level
        group = zarr.open_group(dest, path=f"utm{zone:02d}/{level}", mode="r")
        r0, c0 = geozarr.get_group_offset(group, tuple(v // scale for v in bounds))  # type: ignore[arg-type]
        size = max(TILE_SIZE // scale, 1)
        block = np.asarray(
            group[geozarr.CHANGE_ARRAY][:, r0 : r0 + size, c0 : c0 + size]
        )  # type: ignore[index]
        # The block's category survives every level while it covers the cells.
        cells = max(4096 // scale, 1)
        assert (block[POST_CAT, :cells, :cells] == 3).all(), level
        dates = np.asarray(
            group[geozarr.CHANGE_DATES_ARRAY][:, r0 : r0 + 1, c0 : c0 + 1]
        )  # type: ignore[index]
        assert dates[:, 0, 0].tolist() == [19000, 19100]
        lone = (block[POST_CAT] == 7).sum()
        # max_change_pixel keeps the lone change through level 5; the majority rule
        # drops it right after.
        assert lone == (1 if level <= 5 else 0), level

    # Nothing outside the tile, and level 12 is a single shard.
    level12 = zarr.open_group(dest, path=f"utm{zone:02d}/12", mode="r")
    arr = level12[geozarr.CHANGE_ARRAY]
    assert arr.shards is not None and arr.shards[1] >= arr.shape[1]  # type: ignore[union-attr]
    # The tile is 4 x 4 level-12 pixels, all valid.
    assert int((np.asarray(arr[:]) > 0).any(axis=0).sum()) == 16  # type: ignore[index]

    geozarr_pyramid_full.finalize_pyramid_full(dest, source)
    root = zarr.open_group(dest, mode="r")
    assert root.attrs["coarse_resampling_method"] == "majority_category_pixel"
    assert root.attrs["zones"] == [f"utm{zone:02d}"]
    consolidated = json.loads(
        (tmp_path / "pyramid_full.zarr" / "zarr.json").read_text()
    )
    assert (
        f"utm{zone:02d}/12/change" in consolidated["consolidated_metadata"]["metadata"]
    )
    with pytest.raises(FileExistsError):
        geozarr_pyramid_full.finalize_pyramid_full(dest, source)


def test_pyramid_full_zone_without_data(tmp_path: pathlib.Path) -> None:
    source = str(tmp_path / "pyramid.zarr")
    geozarr_pyramid.init_pyramid_store(source, "lcc.zarr", zone_numbers=[11])
    dest = str(tmp_path / "pyramid_full.zarr")
    stats = geozarr_pyramid_full.build_pyramid_full_zone(source, dest, 11)
    geozarr_pyramid_full.verify_pyramid_full_zone(source, dest, 11)
    assert stats["band"] is None
    group = zarr.open_group(dest, path="utm11/12", mode="r")
    assert not np.asarray(group[geozarr.CHANGE_ARRAY][:]).any()  # type: ignore[index]
