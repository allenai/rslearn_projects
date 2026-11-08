import json
import pathlib
from typing import cast

import numpy as np
import pytest
import zarr
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.olmoearth_lcc.lcc_model import geozarr
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    NONE_CATEGORY_IDX,
    OUTPUT_BANDS,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    SRC_BAND_OFFSET,
    TS_POST_DAYS_BAND,
    TS_PRE_DAYS_BAND,
)
from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    enumerate_zone_tiles,
    get_zone_projection,
)

# 2015-01-01 (the model timestamp epoch) in days since 1970-01-01.
EPOCH_2015_UNIX_DAYS = 16436


def _pixel(
    binary_change: int,
    pre_category: int,
    post_category: int,
    src: int = 9,
    dst: int = 10,
    ts_pre: int = 3000,
    ts_post: int = 3100,
) -> np.ndarray:
    """Build one pixel of the full output_change raster."""
    arr = np.zeros(len(OUTPUT_BANDS), dtype=np.uint16)
    arr[BINARY_CHANGE_BAND - 1] = 255 - binary_change
    arr[BINARY_CHANGE_BAND] = binary_change
    arr[SRC_BAND_OFFSET + src] = 200
    arr[DST_BAND_OFFSET + dst] = 190
    arr[PRE_CHANGE_BAND_OFFSET + pre_category] = 180
    arr[POST_CHANGE_BAND_OFFSET + post_category] = 170
    arr[TS_PRE_DAYS_BAND] = ts_pre
    arr[TS_POST_DAYS_BAND] = ts_post
    return arr


def _window_array(pixels: list[np.ndarray]) -> np.ndarray:
    """Stack pixels into a (len(OUTPUT_BANDS), 1, N) window array."""
    return np.stack(pixels, axis=1)[:, None, :]


def test_geozarr_window_arrays_encoding() -> None:
    new_building = POST_CHANGE_CATEGORY_NAMES.index("new_building")
    arr = _window_array(
        [
            # Change pixel.
            _pixel(200, NONE_CATEGORY_IDX, new_building),
            # No change category and low binary change: details are zeroed.
            _pixel(50, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX),
            # No change category but high binary change: details are kept.
            _pixel(200, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX),
            # No prediction.
            np.zeros(len(OUTPUT_BANDS), dtype=np.uint16),
            # Binary change below GEOZARR_BINARY_ZERO_THRESHOLD is stored as 0.
            _pixel(10, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX),
            # Same with a change category: only binary_change is zeroed.
            _pixel(10, NONE_CATEGORY_IDX, new_building),
        ]
    )
    change, dates = geozarr.geozarr_window_arrays(arr)
    assert change.dtype == np.uint8 and change.shape == (6, 1, 6)
    assert dates.dtype == np.uint16 and dates.shape == (2, 1, 6)

    np.testing.assert_array_equal(
        change[:, 0, 0], [200, 9, 10, NONE_CATEGORY_IDX, new_building, 170]
    )
    np.testing.assert_array_equal(
        dates[:, 0, 0], [3000 + EPOCH_2015_UNIX_DAYS, 3100 + EPOCH_2015_UNIX_DAYS]
    )

    np.testing.assert_array_equal(
        change[:, 0, 1], [50, 0, 0, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX, 0]
    )
    np.testing.assert_array_equal(dates[:, 0, 1], [0, 0])

    np.testing.assert_array_equal(
        change[:, 0, 2], [200, 9, 10, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX, 170]
    )
    assert (dates[:, 0, 2] > 0).all()

    assert not change[:, 0, 3].any()
    assert not dates[:, 0, 3].any()

    np.testing.assert_array_equal(
        change[:, 0, 4], [0, 0, 0, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX, 0]
    )
    np.testing.assert_array_equal(dates[:, 0, 4], [0, 0])

    np.testing.assert_array_equal(
        change[:, 0, 5], [0, 9, 10, NONE_CATEGORY_IDX, new_building, 170]
    )
    assert (dates[:, 0, 5] > 0).all()


def _first_window(zone_number: int) -> tuple[Projection, tuple[int, int, int, int]]:
    """Get the projection and bounds of the first window of a zone's first tile."""
    col, row = next(enumerate_zone_tiles(zone_number, RESOLUTION, TILE_SIZE))
    bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        col * TILE_SIZE + PATCH_SIZE,
        row * TILE_SIZE + PATCH_SIZE,
    )
    return get_zone_projection(zone_number, RESOLUTION), bounds


def test_init_store_consolidated_metadata(tmp_path: pathlib.Path) -> None:
    store_path = tmp_path / "lcc.zarr"
    geozarr.init_store(str(store_path), zone_numbers=[10, 11])
    root = json.loads((store_path / "zarr.json").read_text())
    assert root["attributes"][geozarr.DETAIL_ATTR] == geozarr.DETAIL_COMPACT
    consolidated = root["consolidated_metadata"]["metadata"]
    for zone in ["utm10", "utm11"]:
        own = json.loads((store_path / zone / "zarr.json").read_text())
        assert "spatial:transform" in own["attributes"]
        assert consolidated[zone]["attributes"] == own["attributes"]
    with pytest.raises(FileExistsError):
        geozarr.init_store(str(store_path), zone_numbers=[10])


def test_store_round_trip(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, zone_numbers=[10])
    assert geozarr.read_store_detail(store_path) == geozarr.DETAIL_COMPACT
    assert geozarr.read_store_detail(str(tmp_path / "missing.zarr")) is None
    # An existing store with the same detail mode is reused; another one fails.
    geozarr.ensure_store(store_path)
    with pytest.raises(ValueError):
        geozarr.ensure_store(store_path, detail=geozarr.DETAIL_FULL)

    projection, bounds = _first_window(10)
    rng = np.random.default_rng(0)
    change = rng.integers(0, 255, (6, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    dates = rng.integers(0, 65535, (2, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    geozarr.write_window(store_path, projection, bounds, change, dates)

    group = zarr.open_group(store_path, path="utm10", mode="r")
    assert group.attrs["proj:code"] == "EPSG:32610"
    transform = cast(list[float], group.attrs["spatial:transform"])
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    row = bounds[1] - origin_y
    col = bounds[0] - origin_x
    change_array = group[geozarr.CHANGE_ARRAY]
    dates_array = group[geozarr.CHANGE_DATES_ARRAY]
    x_array = group["x"]
    assert isinstance(change_array, zarr.Array)
    assert isinstance(dates_array, zarr.Array)
    assert isinstance(x_array, zarr.Array)
    window = (slice(None), slice(row, row + PATCH_SIZE), slice(col, col + PATCH_SIZE))
    np.testing.assert_array_equal(change_array[window], change)
    np.testing.assert_array_equal(dates_array[window], dates)
    # Nothing written outside the window reads as nodata.
    below = (
        slice(None),
        slice(row + PATCH_SIZE, row + 2 * PATCH_SIZE),
        slice(col, col + PATCH_SIZE),
    )
    assert not np.asarray(change_array[below]).any()
    # x coordinate of the window's first column is its pixel center easting.
    assert x_array[col] == (bounds[0] + 0.5) * RESOLUTION

    # The window is exactly one shard per array.
    for name in [geozarr.CHANGE_ARRAY, geozarr.CHANGE_DATES_ARRAY]:
        chunk_dir = tmp_path / "lcc.zarr" / "utm10" / name / "c"
        assert len([p for p in chunk_dir.rglob("*") if p.is_file()]) == 1

    # read_window returns the same data; compact stores have no probabilities.
    arrays = geozarr.read_window(store_path, projection, bounds)
    assert set(arrays) == {geozarr.CHANGE_ARRAY, geozarr.CHANGE_DATES_ARRAY}
    np.testing.assert_array_equal(arrays[geozarr.CHANGE_ARRAY], change)
    np.testing.assert_array_equal(arrays[geozarr.CHANGE_DATES_ARRAY], dates)

    # Compact stores have no probability arrays.
    with pytest.raises(KeyError):
        geozarr.write_window(
            store_path,
            projection,
            bounds,
            change,
            dates,
            prob_arrays=geozarr.prob_window_arrays(
                np.zeros((len(OUTPUT_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
            ),
        )


def test_full_store(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, detail=geozarr.DETAIL_FULL, zone_numbers=[10])
    assert geozarr.read_store_detail(store_path) == geozarr.DETAIL_FULL
    geozarr.ensure_store(store_path, detail=geozarr.DETAIL_FULL)
    with pytest.raises(ValueError):
        geozarr.ensure_store(store_path)

    new_building = POST_CHANGE_CATEGORY_NAMES.index("new_building")
    pixel = _pixel(200, NONE_CATEGORY_IDX, new_building)
    arr = np.broadcast_to(
        pixel[:, None, None], (len(OUTPUT_BANDS), PATCH_SIZE, PATCH_SIZE)
    ).copy()
    change, dates = geozarr.geozarr_window_arrays(arr)
    prob_arrays = geozarr.prob_window_arrays(arr)
    assert {name: a.shape[0] for name, a in prob_arrays.items()} == {
        "src_probs": 13,
        "dst_probs": 13,
        "pre_change_probs": 7,
        "post_change_probs": 20,
    }
    projection, bounds = _first_window(10)
    geozarr.write_window(
        store_path, projection, bounds, change, dates, prob_arrays=prob_arrays
    )

    arrays = geozarr.read_window(store_path, projection, bounds)
    post_probs = arrays["post_change_probs"]
    # Band 0 is "none" (the nodata class is not stored).
    assert post_probs.dtype == np.uint8
    assert post_probs[new_building - 1, 0, 0] == 170
    assert arrays["src_probs"][9 - 1, 0, 0] == 200
    assert arrays["pre_change_probs"][NONE_CATEGORY_IDX - 1, 0, 0] == 180

    group = zarr.open_group(store_path, path="utm10", mode="r")
    band_coord = group["post_change_probs_band"]
    assert isinstance(band_coord, zarr.Array)
    assert np.asarray(band_coord[:2]).tolist() == ["none", "vegetation_growth"]


def test_write_window_rejects_southern_crs(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, zone_numbers=[10])
    projection = Projection(CRS.from_epsg(32710), RESOLUTION, -RESOLUTION)
    change = np.zeros((6, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    dates = np.zeros((2, PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    with pytest.raises(ValueError):
        geozarr.write_window(
            store_path, projection, (0, 0, PATCH_SIZE, PATCH_SIZE), change, dates
        )
