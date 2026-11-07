from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.olmoearth_lcc.lcc_model.output_bands import (
    LC_CLASS_NAMES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from rslp.olmoearth_lcc.lcc_model.prepare import (
    FREQUENT_BLOCK_DURATION,
    OPTION_MARGIN,
    QUARTERLY_LOOKBACK,
    _compute_frequent_block_ends,
    _compute_label_arrays,
    _compute_window_time_range,
    _get_unknown_categories,
    _lonlat_to_pixel,
    _sidecar_dates_match,
    _validate_positive_point_dates,
    positive_point_dates_valid,
)
from rslp.olmoearth_lcc.lcc_model.timestamp_encoding import DAYS_NODATA, UNIX_EPOCH
from rslp.olmoearth_lcc.lcc_model.transforms import QUARTERLY_PERIOD

PROJECTION = Projection(CRS.from_epsg(32610), 10, -10)
LON, LAT = -122.3, 47.6
HALF = 4


def _labels_at_point(
    point: dict[str, Any], gap: timedelta = timedelta(0)
) -> dict[str, int]:
    """Rasterize one positive point and get each label layer's value at it."""
    col, row = _lonlat_to_pixel(LON, LAT, PROJECTION, (0, 0, 0, 0))
    bounds = (col - HALF, row - HALF, col + HALF, row + HALF)
    entry = {"positive_points": [{"lon": LON, "lat": LAT, **point}]}
    arrays = _compute_label_arrays(entry, PROJECTION, bounds, gap=gap)
    return {name: int(array[HALF, HALF]) for name, array in arrays.items()}


def _days(date: str) -> int:
    return (datetime.fromisoformat(date).replace(tzinfo=timezone.utc) - UNIX_EPOCH).days


DATED_POINT = {
    "pre_change": "2020-01-15",
    "first_date_change_noticeable": "2020-02-01",
    "post_change": "2020-09-01",
    "pre_category": "mangrove",
    "post_category": "crops",
    "pre_change_category": "mangrove_loss",
    "post_change_category": "new_crop_field",
}


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
    assert LC_CLASS_NAMES[labels["label_src"]] == "mangrove"
    assert LC_CLASS_NAMES[labels["label_dst"]] == "bare"
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


def test_transition_label_arrays() -> None:
    gap = timedelta(days=10)
    labels = _labels_at_point(
        {
            **DATED_POINT,
            "transition_date": "2020-04-01",
            "transition_category": "bare",
            "transition_change_category": "site_clearing",
        },
        gap=gap,
    )
    # The transition date gets the gap, like label_ts_end.
    assert labels["label_transition_date"] == _days("2020-04-01") + 10
    assert labels["label_ts_end"] == _days("2020-09-01") + 10
    assert LC_CLASS_NAMES[labels["label_transition_dst"]] == "bare"
    assert (
        POST_CHANGE_CATEGORY_NAMES[labels["label_transition_post_change"]]
        == "site_clearing"
    )


@pytest.mark.parametrize(
    "transition,expected",
    [
        # Aliases apply as for post_change_category.
        ({"transition_change_category": "borrow_pit"}, "new_road"),
        # land_reclamation is not moved to the pre-change head; it is nodata.
        ({"transition_change_category": "land_reclamation"}, "nodata"),
        # A missing transition change category is nodata (masked).
        ({"transition_category": "bare"}, "nodata"),
    ],
)
def test_transition_change_category(transition: dict[str, Any], expected: str) -> None:
    labels = _labels_at_point(
        {**DATED_POINT, "transition_date": "2020-04-01", **transition}
    )
    assert POST_CHANGE_CATEGORY_NAMES[labels["label_transition_post_change"]] == (
        expected
    )
    # The regular post change category is unaffected.
    assert POST_CHANGE_CATEGORY_NAMES[labels["label_post_change"]] == "new_crop_field"


def test_no_transition_labels_without_transition() -> None:
    labels = _labels_at_point(DATED_POINT)
    assert labels["label_transition_date"] == DAYS_NODATA
    assert labels["label_transition_dst"] == 0
    assert labels["label_transition_post_change"] == 0
    # A transition on a point without dates is not rasterized either.
    labels = _labels_at_point(
        {"transition_date": "2020-04-01", "transition_category": "bare"}
    )
    assert labels["label_transition_date"] == DAYS_NODATA
    assert labels["label_transition_dst"] == 0


@pytest.mark.parametrize(
    "transition_date,valid",
    [
        ("2020-02-01", True),
        ("2020-08-31", True),
        ("2020-01-31", False),
        ("2020-09-01", False),
    ],
)
def test_validate_transition_date(transition_date: str, valid: bool) -> None:
    entry = {
        "positive_points": [
            {"lon": LON, "lat": LAT, **DATED_POINT, "transition_date": transition_date}
        ]
    }
    if valid:
        _validate_positive_point_dates(entry)
    else:
        with pytest.raises(ValueError, match="transition_date"):
            _validate_positive_point_dates(entry)
    # filter_v2_jsons omits the entries that would raise.
    assert positive_point_dates_valid(entry) == valid


def test_transition_frequent_option() -> None:
    first = datetime(2020, 2, 1, tzinfo=timezone.utc)
    post = datetime(2020, 9, 1, tzinfo=timezone.utc)
    transition = datetime(2020, 4, 1, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    without = _compute_frequent_block_ends(first, post, "w", cutoff)
    with_transition = _compute_frequent_block_ends(
        first, post, "w", cutoff, transition_date=transition
    )
    assert len(with_transition) == len(without)
    # The deterministic options (including option 2, used for validation) stay,
    # the transition option comes next, and it replaces the last random option.
    assert with_transition[:3] == without[:3]
    assert with_transition[3] == transition + OPTION_MARGIN
    assert with_transition[4:] == without[3:-1]


def test_sidecar_dates_match_transition() -> None:
    old = {
        "pre_change": "2020-01-15",
        "post_change": "2020-09-01T00:00:00+00:00",
        "first_noticeable": "2020-02-01T00:00:00+00:00",
        "stop_date": None,
    }
    # Sidecars written before transition_date existed match windows without one.
    assert _sidecar_dates_match(old, {**old, "transition_date": None})
    assert not _sidecar_dates_match(
        old, {**old, "transition_date": "2020-04-01T00:00:00+00:00"}
    )


def test_unknown_transition_categories() -> None:
    entries = [
        {
            "positive_points": [
                {
                    "transition_category": "lava",
                    "transition_change_category": "land_reclamation",
                },
                {
                    "transition_category": "bare",
                    "transition_change_category": "borrow_pit",
                },
            ]
        }
    ]
    assert dict(_get_unknown_categories(entries)) == {
        "transition_category: lava": 1,
        "transition_change_category: land_reclamation": 1,
    }


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
