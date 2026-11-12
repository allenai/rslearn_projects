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
    "post_category": "agriculture",
    "pre_change_category": "mangrove_loss",
    "post_change_category": "new_agriculture",
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
        ({"pre_change_category": "forest_loss"}, ("forest_loss", "none")),
        ({"post_change_category": "burned"}, ("none", "burned")),
        ({"post_change_category": "storm_damage"}, ("none", "storm_damage")),
        ({"pre_change_category": "mangrove_loss"}, ("mangrove_loss", "none")),
        ({"pre_change_category": "river_contract"}, ("water_to_land", "none")),
        (
            {"pre_change_category": "river_channel_migration"},
            ("water_to_land", "none"),
        ),
        (
            {"post_change_category": "river_channel_migration"},
            ("none", "land_to_water"),
        ),
        ({"post_change_category": "coastal_erosion"}, ("none", "land_to_water")),
        ({"post_change_category": "borrow_pit"}, ("none", "new_mining_or_quarry")),
        ({"post_change_category": "land_reclamation"}, ("water_to_land", "none")),
        # Finer-grained categories train as their coarse class.
        ({"post_change_category": "new_solar"}, ("none", "new_infrastructure")),
        (
            {"post_change_category": "new_other_infrastructure"},
            ("none", "new_infrastructure"),
        ),
        ({"post_change_category": "tree_crop_growth"}, ("none", "vegetation_growth")),
        ({"post_change_category": "wetland_growth"}, ("none", "vegetation_growth")),
        (
            {"post_change_category": "tree_crop_harvest"},
            ("none", "agricultural_activity"),
        ),
        ({"post_change_category": "other_land_to_water"}, ("none", "land_to_water")),
        (
            {"pre_change_category": "small_water_body_level_drop"},
            ("water_to_land", "none"),
        ),
        # ice_loss has no training class: unknown, like any unknown category.
        ({"post_change_category": "ice_loss"}, ("nodata", "nodata")),
        # Names from before the 2026-11-12 renames still map.
        ({"pre_change_category": "deforestation"}, ("forest_loss", "none")),
        ({"pre_change_category": "urban_erosion"}, ("removed_building", "none")),
        ({"post_change_category": "mining"}, ("none", "new_mining_or_quarry")),
        ({"post_change_category": "wildfire"}, ("none", "burned")),
        ({"post_change_category": "water_expand"}, ("none", "land_to_water")),
        ({"post_change_category": "lake_level_rise"}, ("none", "land_to_water")),
        (
            {
                "pre_change_category": "wetland_loss",
                "post_change_category": "land_reclamation",
            },
            ("wetland_loss", "none"),
        ),
        (
            {
                "pre_change_category": "large_water_body_level_drop",
                "post_change_category": "new_agriculture",
            },
            ("water_to_land", "new_agriculture"),
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
    # Without a transition stop date the raster is nodata (= post_change).
    assert labels["label_transition_stop_date"] == DAYS_NODATA
    labels = _labels_at_point(
        {
            **DATED_POINT,
            "transition_date": "2020-04-01",
            "transition_stop_date": "2020-06-01",
            "transition_category": "bare",
        },
        gap=gap,
    )
    assert labels["label_transition_stop_date"] == _days("2020-06-01") + 10


@pytest.mark.parametrize(
    "transition,expected",
    [
        # Aliases apply as for post_change_category.
        ({"transition_change_category": "borrow_pit"}, "new_mining_or_quarry"),
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
    assert POST_CHANGE_CATEGORY_NAMES[labels["label_post_change"]] == "new_agriculture"


@pytest.mark.parametrize(
    "old,new",
    [
        ("crops", "agriculture"),
        ("snow and ice", "snow/ice"),
        ("wetland (herbaceous)", "wetland"),
    ],
)
def test_legacy_land_cover_names(old: str, new: str) -> None:
    labels = _labels_at_point(
        {**DATED_POINT, "pre_category": old, "post_category": old}
    )
    assert LC_CLASS_NAMES[labels["label_src"]] == new
    assert LC_CLASS_NAMES[labels["label_dst"]] == new


def test_no_transition_labels_without_transition() -> None:
    labels = _labels_at_point(DATED_POINT)
    assert labels["label_transition_date"] == DAYS_NODATA
    assert labels["label_transition_stop_date"] == DAYS_NODATA
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


@pytest.mark.parametrize(
    "transition,valid",
    [
        ({"transition_date": "2020-04-01", "transition_stop_date": "2020-04-02"}, True),
        ({"transition_date": "2020-04-01", "transition_stop_date": "2020-08-31"}, True),
        # Strictly between transition_date and post_change.
        (
            {"transition_date": "2020-04-01", "transition_stop_date": "2020-04-01"},
            False,
        ),
        (
            {"transition_date": "2020-04-01", "transition_stop_date": "2020-09-01"},
            False,
        ),
        # A transition stop date needs a transition date.
        ({"transition_stop_date": "2020-06-01"}, False),
    ],
)
def test_validate_transition_stop_date(transition: dict[str, str], valid: bool) -> None:
    entry = {"positive_points": [{"lon": LON, "lat": LAT, **DATED_POINT, **transition}]}
    if valid:
        _validate_positive_point_dates(entry)
    else:
        with pytest.raises(ValueError, match="transition_stop_date"):
            _validate_positive_point_dates(entry)
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
    new: dict[str, str | None] = {**old, "transition_date": None, "image_cutoff": None}
    # Sidecars written before transition_date existed match windows without one.
    assert _sidecar_dates_match(old, new)
    assert not _sidecar_dates_match(
        old, {**new, "transition_date": "2020-04-01T00:00:00+00:00"}
    )


def test_sidecar_dates_match_image_cutoff() -> None:
    old = {
        "pre_change": "2020-01-15",
        "post_change": "2020-09-01T00:00:00+00:00",
        "first_noticeable": "2020-02-01T00:00:00+00:00",
        "stop_date": None,
        "transition_date": None,
    }
    # Sidecars written before image_cutoff existed match the default cutoff (None).
    assert _sidecar_dates_match(old, {**old, "image_cutoff": None})
    cutoff = "2026-11-08T00:00:00+00:00"
    assert not _sidecar_dates_match(old, {**old, "image_cutoff": cutoff})
    assert not _sidecar_dates_match(
        {**old, "image_cutoff": "2026-06-01T00:00:00+00:00"},
        {**old, "image_cutoff": cutoff},
    )


def test_recent_negative_frequent_options_deduplicated() -> None:
    """A negative anchored shortly before the cutoff gets two distinct options.

    Option 1 and every random option fall after the cutoff, so they are all
    clamped to it and kept once.
    """
    cutoff = datetime(2026, 11, 8, tzinfo=timezone.utc)
    anchor = cutoff - timedelta(days=13)
    block_ends = _compute_frequent_block_ends(anchor, anchor, "some_window", cutoff)
    assert block_ends == [anchor + OPTION_MARGIN, cutoff]


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
