"""Tests for rslp.olmoearth_lcc.lcc_model.geozarr_summary."""

import pathlib

import numpy as np
import zarr
from pyproj import Transformer

from rslp.olmoearth_lcc.lcc_model import geozarr_summary as gs


def _days(year: int, month: int) -> int:
    return int(np.datetime64(f"{year}-{month:02d}-15", "D").astype(np.int64))


def test_window_counts_keep_change_pixels_ending_in_window_inside_zone() -> None:
    zone = 33
    transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    first = gs.month_value(2024, 12)
    pre = np.ones((4, 4), np.uint8)
    post = np.ones((4, 4), np.uint8)
    end = np.full((4, 4), _days(2026, 11), np.uint16)
    post[0, 0] = 3  # new_building, counted
    pre[0, 1] = 2  # forest_loss with post none, counted
    post[1, 1] = 3
    end[1, 1] = _days(2024, 6)  # ended before the window: dropped
    # Both none (e.g. only a high binary score): dropped.
    keys, counts = gs.tile_window_counts(
        np.stack([pre, post]), end, 500_000.0, 1_000_000.0, transformer, zone, first
    )
    _, _, cat, month = gs.decode_key(keys, gs.MAX_LEVEL)
    by_cat = np.bincount(cat, weights=counts, minlength=gs.NUM_CATEGORIES)
    assert by_cat[2] == 1  # post new_building
    assert by_cat[0] == 1  # post none (the forest loss)
    assert by_cat[gs.NUM_POST + 1] == 1  # pre forest_loss
    assert by_cat[gs.NUM_POST] == 1  # pre none (the new building)
    assert counts.sum() == 4
    assert set(month.tolist()) == {gs.NUM_MONTHS - 1}

    # A window outside the zone's 6 degree band counts nothing.
    keys, _ = gs.tile_window_counts(
        np.stack([pre, post]), end, 2_000_000.0, 1_000_000.0, transformer, zone, first
    )
    assert keys.size == 0


def test_coarser_levels_sum_exactly() -> None:
    rows = np.array([10, 11, 900, 901])
    cols = np.array([20, 21, 40, 41])
    keys = gs.encode_key(rows, cols, np.array([2, 2, 5, 5]), np.array([0, 0, 3, 3]), 10)
    counts = np.array([5, 7, 1, 2])
    for level in range(9, -1, -1):
        keys, counts = gs.coarsen(keys, counts, level + 1, level)
        assert counts.sum() == 15
    assert keys.size == 2  # one cell at level 0, two (category, month) pairs


def test_fraction_encoding_round_trips_within_five_percent() -> None:
    frac = np.array([3e-10, 1e-6, 0.01, 0.5, 1.0])
    decoded = gs.decode_fraction(gs.encode_fraction(frac))
    assert np.all(np.abs(decoded / frac - 1) < 0.05)
    assert gs.decode_fraction(np.array([0]))[0] == 0


def test_store_chunks_hold_encoded_fractions(tmp_path: pathlib.Path) -> None:
    store = str(tmp_path / "summary.zarr")
    gs.init_store(store, "gs://bucket/lcc.zarr", "2026-11")
    root = zarr.open_group(store, mode="r+")
    assert root.attrs["first_month"] == "2024-12"
    row, col = 1000, 70
    keys = gs.encode_key(
        np.array([row]), np.array([col]), np.array([3]), np.array([23]), 10
    )
    count = np.array([16])
    assert gs.write_level(root, keys, count, 10) == 1
    value = root["10/values"][3, 23, row, col]
    expected = 16 * gs.PIXEL_AREA_M2 / gs.cell_area_m2(np.array([row]), 10)[0]
    assert abs(gs.decode_fraction(np.array([value]))[0] / expected - 1) < 0.05
    assert root["10/values"][3, 22, row, col] == 0
