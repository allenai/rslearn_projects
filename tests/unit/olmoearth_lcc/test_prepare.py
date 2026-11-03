from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.olmoearth_lcc.lcc_model.prepare import (
    CATEGORY_NAMES,
    FREQUENT_BLOCK_DURATION,
    OPTION_MARGIN,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
    QUARTERLY_LOOKBACK,
    QUARTERLY_PERIOD,
    _compute_frequent_block_ends,
    _compute_label_arrays,
    _compute_window_time_range,
    _get_unknown_categories,
    _lonlat_to_pixel,
)

PROJECTION = Projection(CRS.from_epsg(32610), 10, -10)
LON, LAT = -122.3, 47.6
HALF = 4


def _labels_at_point(point: dict[str, Any]) -> dict[str, int]:
    """Rasterize one positive point and get each label layer's value at it."""
    col, row = _lonlat_to_pixel(LON, LAT, PROJECTION, (0, 0, 0, 0))
    bounds = (col - HALF, row - HALF, col + HALF, row + HALF)
    entry = {"positive_points": [{"lon": LON, "lat": LAT, **point}]}
    arrays = _compute_label_arrays(entry, PROJECTION, bounds)
    return {name: int(array[HALF, HALF]) for name, array in arrays.items()}


def _change_labels(point: dict[str, Any]) -> tuple[str, str]:
    """Get the (pre, post) change-category class names rasterized for a point."""
    labels = _labels_at_point(point)
    return (
        PRE_CHANGE_CATEGORY_NAMES[labels["label_pre_change"]],
        POST_CHANGE_CATEGORY_NAMES[labels["label_post_change"]],
    )


@pytest.mark.parametrize(
    "point,expected",
    [
        ({"pre_change_category": "deforestation"}, ("deforestation", "none")),
        ({"post_change_category": "wildfire"}, ("none", "wildfire")),
        ({"post_change_category": "storm_damage"}, ("none", "storm_damage")),
        ({"pre_change_category": "mangrove_loss"}, ("mangrove_loss", "none")),
        ({"pre_change_category": "river_contract"}, ("water_contract", "none")),
        (
            {"pre_change_category": "river_channel_migration"},
            ("water_contract", "none"),
        ),
        (
            {"post_change_category": "river_channel_migration"},
            ("none", "water_expand"),
        ),
        ({"post_change_category": "coastal_erosion"}, ("none", "water_expand")),
        ({"post_change_category": "borrow_pit"}, ("none", "new_road")),
        ({"post_change_category": "land_reclamation"}, ("water_contract", "none")),
        (
            {
                "pre_change_category": "wetland_loss",
                "post_change_category": "land_reclamation",
            },
            ("wetland_loss", "none"),
        ),
        (
            {
                "pre_change_category": "lake_level_drop",
                "post_change_category": "new_crop_field",
            },
            ("water_contract", "new_crop_field"),
        ),
        # The legacy same_change_category field is ignored.
        ({"same_change_category": "wildfire"}, ("nodata", "nodata")),
    ],
)
def test_change_category_labels(
    point: dict[str, Any], expected: tuple[str, str]
) -> None:
    assert _change_labels(point) == expected


def test_mangrove_land_cover() -> None:
    labels = _labels_at_point({"pre_category": "mangrove", "post_category": "bare"})
    assert CATEGORY_NAMES[labels["label_src"]] == "mangrove"
    assert CATEGORY_NAMES[labels["label_dst"]] == "bare"
    assert "label_same_change" not in labels


def test_get_unknown_categories() -> None:
    entries = [
        {
            "positive_points": [
                {
                    "pre_category": "mangrove",
                    "post_category": "lava",
                    "pre_change_category": "river_contract",
                    "post_change_category": "new_volcano",
                },
                {
                    "post_change_category": "land_reclamation",
                    "same_change_category": "flooding",
                },
            ]
        }
    ]
    assert dict(_get_unknown_categories(entries)) == {
        "post_category: lava": 1,
        "post_change_category: new_volcano": 1,
        "same_change_category: flooding": 1,
    }


def test_label_arrays_have_no_change_labels_without_categories() -> None:
    labels = _labels_at_point({"pre_category": "tree", "post_category": "bare"})
    assert labels["label_pre_change"] == 0
    assert labels["label_post_change"] == 0


def test_window_time_range_aligns_quarterly_with_prediction() -> None:
    """Option 0 of an entry anchored at T - OPTION_MARGIN matches prediction at T.

    Prediction uses frequent [T - 90d, T] and quarterly [T - 1890d, T - 90d]. The
    quarterly mosaics are built backwards from the end of the window time range, so
    the end must be a whole number of periods after T - 90d.
    """
    t = datetime(2024, 5, 17, tzinfo=timezone.utc)
    anchor = t - OPTION_MARGIN
    block_ends = _compute_frequent_block_ends(
        anchor, anchor, "some_window", datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    block_starts = [end - FREQUENT_BLOCK_DURATION for end in block_ends]

    # Option 0's frequent block is exactly the prediction frequent block.
    assert block_ends[0] == t
    assert block_starts[0] == t - timedelta(days=90)

    start, end = _compute_window_time_range(block_starts, block_ends)
    assert start == t - timedelta(days=90) - QUARTERLY_LOOKBACK
    assert end >= max(block_ends)
    assert end - max(block_ends) < QUARTERLY_PERIOD
    # Period boundaries (end - k * 90d) include T - 90d and T - 1890d.
    assert (end - (t - timedelta(days=90))) % QUARTERLY_PERIOD == timedelta(0)
    assert (end - start) % QUARTERLY_PERIOD == timedelta(0)
