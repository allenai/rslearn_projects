from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from rslp.olmoearth_lcc.lcc_model.label_changes import (
    legacy_transition_to_change2,
    missing_date_fields,
    point_date_errors,
)
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
    _entry_has_complete_annotations,
    _get_positive_points_missing_dates,
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


def test_resurfaced_road_trains_as_new_road() -> None:
    assert _change_labels({"post_change_category": "resurfaced_road"}) == (
        "none",
        "new_road",
    )


def test_sidecar_dates_match_excluded_end_ranges() -> None:
    old = {
        "pre_change": "2020-01-15",
        "post_change": "2020-09-01T00:00:00+00:00",
        "first_noticeable": "2020-02-01T00:00:00+00:00",
        "stop_date": None,
        "transition_date": None,
        "image_cutoff": None,
    }
    # Sidecars written before excluded_end_ranges existed match windows without any.
    assert _sidecar_dates_match(old, {**old, "excluded_end_ranges": None})
    ranges = [["2020-05-01T00:00:00+00:00", "2020-09-01T00:00:00+00:00"]]
    assert not _sidecar_dates_match(old, {**old, "excluded_end_ranges": ranges})
    assert _sidecar_dates_match(
        {**old, "excluded_end_ranges": ranges}, {**old, "excluded_end_ranges": ranges}
    )


CHANGE2_POINT = {
    **DATED_POINT,
    "change2_pre_change": "2021-03-01",
    "change2_first_date_change_noticeable": "2021-04-01",
    "change2_post_change": "2021-06-01",
    "change2_pre_category": "agriculture",
    "change2_post_category": "water",
    "change2_post_change_category": "flooding",
    "change2_stop_date": "2022-01-01",
}


def test_change2_label_arrays() -> None:
    labels = _labels_at_point(
        {**CHANGE2_POINT, "stop_date": "2021-01-01"}, gap=timedelta(days=10)
    )
    # The first change, with first-noticeable and stop date rasters (first
    # noticeable and post-change get the gap; pre-change and stop dates don't).
    assert labels["label_ts_start"] == _days("2020-01-15")
    assert labels["label_ts_end"] == _days("2020-09-01") + 10
    assert labels["label_first_noticeable"] == _days("2020-02-01") + 10
    assert labels["label_stop_date"] == _days("2021-01-01")
    assert LC_CLASS_NAMES[labels["label_dst"]] == "agriculture"
    # The second change.
    assert labels["label_change2_ts_start"] == _days("2021-03-01")
    assert labels["label_change2_ts_end"] == _days("2021-06-01") + 10
    assert labels["label_change2_first_noticeable"] == _days("2021-04-01") + 10
    assert labels["label_change2_stop_date"] == _days("2022-01-01")
    assert LC_CLASS_NAMES[labels["label_change2_src"]] == "agriculture"
    assert LC_CLASS_NAMES[labels["label_change2_dst"]] == "water"
    # An unset pre-change category next to a set post one is "none".
    assert PRE_CHANGE_CATEGORY_NAMES[labels["label_change2_pre_change"]] == "none"
    assert POST_CHANGE_CATEGORY_NAMES[labels["label_change2_post_change"]] == (
        "flooding"
    )


def test_no_change2_labels_without_second_change() -> None:
    labels = _labels_at_point(DATED_POINT)
    for name, value in labels.items():
        if name.startswith("label_change2_"):
            assert value in (0, DAYS_NODATA), name
    assert labels["label_stop_date"] == DAYS_NODATA


LEGACY_TRANSITION_POINT = {
    **DATED_POINT,
    "transition_date": "2020-04-01",
    "transition_category": "bare",
    "transition_change_category": "site_clearing",
    "transition_terminal_state": "bare",
    "terminal_state": "agriculture",
    "stop_date": "2021-01-01",
}


def test_legacy_transition_to_change2() -> None:
    converted = legacy_transition_to_change2(LEGACY_TRANSITION_POINT)
    assert converted == {
        "pre_change": "2020-01-15",
        "first_date_change_noticeable": "2020-02-01",
        "post_change": "2020-04-01",
        "pre_category": "mangrove",
        "post_category": "bare",
        "pre_change_category": "mangrove_loss",
        "post_change_category": "site_clearing",
        "terminal_state": "bare",
        "change2_pre_change": "2020-01-15",
        "change2_first_date_change_noticeable": "2020-09-01",
        "change2_post_change": "2020-09-01",
        "change2_pre_category": "mangrove",
        "change2_post_category": "agriculture",
        "change2_pre_change_category": "mangrove_loss",
        "change2_post_change_category": "new_agriculture",
        "change2_stop_date": "2021-01-01",
        "change2_terminal_state": "agriculture",
    }
    # Converted points rasterize like the legacy ones (and are valid).
    assert _labels_at_point(converted) == _labels_at_point(LEGACY_TRANSITION_POINT)
    assert point_date_errors(converted) == []
    # Points without a transition are unchanged.
    assert legacy_transition_to_change2(DATED_POINT) == DATED_POINT
    # The second change is noticeable once the transition stage ends.
    with_stop = legacy_transition_to_change2(
        {**LEGACY_TRANSITION_POINT, "transition_stop_date": "2020-05-01"}
    )
    assert with_stop == {
        **converted,
        "change2_first_date_change_noticeable": "2020-05-01",
    }
    assert point_date_errors(with_stop) == []
    with pytest.raises(ValueError, match="post_transition_pre_change"):
        legacy_transition_to_change2(
            {**LEGACY_TRANSITION_POINT, "post_transition_pre_change": "2020-05-01"}
        )
    with pytest.raises(ValueError, match="both"):
        legacy_transition_to_change2({**CHANGE2_POINT, "transition_date": "2020-04-01"})


def test_legacy_transition_label_arrays() -> None:
    """A transition trains as the first change's end, then the second change."""
    labels = _labels_at_point(LEGACY_TRANSITION_POINT, gap=timedelta(days=10))
    assert labels["label_ts_end"] == _days("2020-04-01") + 10
    assert LC_CLASS_NAMES[labels["label_src"]] == "mangrove"
    assert LC_CLASS_NAMES[labels["label_dst"]] == "bare"
    assert POST_CHANGE_CATEGORY_NAMES[labels["label_post_change"]] == "site_clearing"
    assert labels["label_stop_date"] == DAYS_NODATA
    assert labels["label_change2_ts_start"] == _days("2020-01-15")
    assert labels["label_change2_ts_end"] == _days("2020-09-01") + 10
    assert labels["label_change2_first_noticeable"] == _days("2020-09-01") + 10
    assert labels["label_change2_stop_date"] == _days("2021-01-01")
    assert LC_CLASS_NAMES[labels["label_change2_src"]] == "mangrove"
    assert LC_CLASS_NAMES[labels["label_change2_dst"]] == "agriculture"
    assert PRE_CHANGE_CATEGORY_NAMES[labels["label_change2_pre_change"]] == (
        "mangrove_loss"
    )
    # A transition without a land cover leaves the first change's dst nodata.
    labels = _labels_at_point({**DATED_POINT, "transition_date": "2020-04-01"})
    assert labels["label_dst"] == 0


@pytest.mark.parametrize(
    "fields,match",
    [
        ({}, None),
        # First-noticeable after post-change is allowed.
        ({"first_date_change_noticeable": "2020-10-01"}, None),
        ({"first_date_change_noticeable": "2020-01-15"}, "pre_change must be before"),
        ({"post_change": "2020-01-10"}, "pre_change must be before post_change"),
        ({"stop_date": "2020-09-01"}, "stop_date must be after"),
        ({"stop_date": "2020-09-02"}, None),
        # Second change.
        ({**CHANGE2_POINT, "change2_post_category": None}, "missing change2_post_cat"),
        ({**CHANGE2_POINT, "change2_pre_change": "2021-04-01"}, "change2_pre_change"),
        (
            {**CHANGE2_POINT, "change2_first_date_change_noticeable": "2020-02-01"},
            "after first_date_change_noticeable",
        ),
        # The second change's first-noticeable may be after its post-change.
        ({**CHANGE2_POINT, "change2_first_date_change_noticeable": "2021-07-01"}, None),
        ({**CHANGE2_POINT, "change2_stop_date": "2021-05-01"}, "change2_stop_date"),
        ({**CHANGE2_POINT, "stop_date": "2020-12-01"}, None),
        (
            {**CHANGE2_POINT, "stop_date": "2021-04-01"},
            "stop_date must be before change2_first",
        ),
        (
            {
                "transition_date": "2020-04-01",
                "post_transition_pre_change": "2020-05-01",
            },
            "legacy",
        ),
    ],
)
def test_validate_point_dates(fields: dict[str, Any], match: str | None) -> None:
    point = {k: v for k, v in {**DATED_POINT, **fields}.items() if v is not None}
    entry = {"positive_points": [{"lon": LON, "lat": LAT, **point}]}
    if match is None:
        _validate_positive_point_dates(entry)
    else:
        with pytest.raises(ValueError, match=match):
            _validate_positive_point_dates(entry)
    # filter_v2_jsons omits the entries that would raise.
    assert positive_point_dates_valid(entry) == (match is None)


def test_validate_skips_incomplete_points() -> None:
    """Points without all their dates are left to the incomplete path.

    A legacy transition point without post_change converts to a dated first change
    (ending at the transition date) and a second change without dates, so it stays
    incomplete, as before the conversion.
    """
    point = {**LEGACY_TRANSITION_POINT, "post_change": None}
    assert missing_date_fields(point) == [
        "change2_first_date_change_noticeable",
        "change2_post_change",
    ]
    _validate_positive_point_dates({"positive_points": [point]})
    assert not _entry_has_complete_annotations({"positive_points": [point]})
    mixed = {"positive_points": [{"lon": LON, "lat": LAT, **DATED_POINT}, point]}
    assert _get_positive_points_missing_dates(mixed) == [point]


@pytest.mark.parametrize(
    "window_stop,valid", [("2020-09-02", True), ("2020-09-01", False)]
)
def test_validate_window_stop_date(window_stop: str, valid: bool) -> None:
    entry = {
        "window_stop_date": window_stop,
        "positive_points": [{"lon": LON, "lat": LAT, **DATED_POINT}],
    }
    assert positive_point_dates_valid(entry) == valid


def test_extra_frequent_options() -> None:
    first = datetime(2020, 2, 1, tzinfo=timezone.utc)
    post = datetime(2020, 9, 1, tzinfo=timezone.utc)
    extra = [datetime(2020, 4, 1, tzinfo=timezone.utc)]
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    without = _compute_frequent_block_ends(first, post, "w", cutoff)
    with_extra = _compute_frequent_block_ends(
        first, post, "w", cutoff, extra_dates=extra
    )
    assert len(with_extra) == len(without)
    # The deterministic options (including option 2, used for validation) stay,
    # the extra option comes next, and it replaces the last random option.
    assert with_extra[:3] == without[:3]
    assert with_extra[3] == extra[0] + OPTION_MARGIN
    assert with_extra[4:] == without[3:-1]


def test_frequent_options_first_noticeable_after_post_change() -> None:
    """Option 2 (right after post-change) is skipped; no option ends before fo."""
    post = datetime(2020, 2, 1, tzinfo=timezone.utc)
    first = datetime(2020, 4, 1, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ends = _compute_frequent_block_ends(first, post, "w", cutoff)
    assert ends[0] == first + OPTION_MARGIN
    assert all(end > first for end in ends)
    # Random options reach two years after the later date.
    assert max(ends) <= first + timedelta(days=730)


def test_frequent_options_avoid_excluded_end_ranges() -> None:
    first = datetime(2020, 2, 1, tzinfo=timezone.utc)
    post = datetime(2020, 9, 1, tzinfo=timezone.utc)
    extra = [datetime(2020, 4, 1, tzinfo=timezone.utc)]
    stop = datetime(2020, 5, 1, tzinfo=timezone.utc)
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    without = _compute_frequent_block_ends(first, post, "w", cutoff, extra_dates=extra)
    # No ranges: the options are unchanged.
    assert (
        _compute_frequent_block_ends(
            first, post, "w", cutoff, extra_dates=extra, excluded_end_ranges=[]
        )
        == without
    )
    for name in ("w", "x", "y", "z"):
        ends = _compute_frequent_block_ends(
            first,
            post,
            name,
            cutoff,
            extra_dates=extra,
            excluded_end_ranges=[(stop, post)],
        )
        assert not any(stop < end < post for end in ends)
        # Option 2 (validation) still ends right after post_change.
        assert ends[2] == post + OPTION_MARGIN

    # A deterministic end inside a range moves back to its start.
    ends = _compute_frequent_block_ends(
        first,
        post,
        "w",
        cutoff,
        excluded_end_ranges=[(first, post)],
    )
    assert ends[0] == first
    assert not any(first < end < post for end in ends)


def test_unknown_change2_categories() -> None:
    entries = [
        {
            "positive_points": [
                {**CHANGE2_POINT, "change2_post_category": "lava"},
                {**LEGACY_TRANSITION_POINT, "transition_category": "rock"},
            ]
        }
    ]
    assert dict(_get_unknown_categories(entries)) == {
        "change2_post_category: lava": 1,
        "post_category: rock": 1,
    }
