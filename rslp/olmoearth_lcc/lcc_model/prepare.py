"""Prepare the LCC model dataset: windows, imagery layers, and labels.

This script takes one or more v2 annotation JSONs and creates an rslearn dataset with:
- sentinel2_quarterly: WindowLayerData with quarterly mosaics (90-day periods);
  the least-cloudy mosaic is selected within each period
- sentinel2_frequent_0..7: WindowLayerData with one 90-day block each, holding
  the least-cloudy mosaic of each of the six most recent 7-day periods in that
  block that have imagery (blocks with fewer than six such periods are skipped)
- label_binary, label_src, label_dst: Pre-rasterized point labels
- label_pre_change, label_post_change: Pre-rasterized point labels for the
  pre/post change-category heads (class 1 = "none"). Fine-grained annotation
  categories are mapped to the training classes (see the *_ALIASES dicts).
- label_ts_start, label_ts_end: Pre-rasterized uint16 pre_change / post_change
  dates of each positive point, as days since 1970-01-01 (65535 = nodata)
- label_first_noticeable, label_stop_date: each positive point's
  first_date_change_noticeable and optional stop_date (uint16 days like
  label_ts_start). StackSampler trains nothing at a point for imagery ending before
  its first-noticeable date or after its stop date.
- label_change2_*: the optional second change of each positive point (the change2_*
  fields, or the legacy transition fields converted by
  label_changes.legacy_transition_to_change2), with the same src / dst / pre_change
  / post_change / ts_start / ts_end / first_noticeable / stop_date rasters as the
  first change. StackSampler uses them instead of the first change's for imagery
  ending at or after the second change's first-noticeable date.

The time range for each window covers all annotation-derived frequent blocks and
QUARTERLY_LOOKBACK of preceding quarterly history. The 90-day quarterly periods are
aligned so that one ends exactly at the start of the earliest frequent block, and
they reach back QUARTERLY_LOOKBACK from there, matching the prediction dataset config
(sentinel2_quarterly = [T - 1890d, T - 90d] for reference timestamp T). So for a
window whose earliest frequent block is [T - 90d, T], the first training option sees
the same 22-image stack as prediction at T. Frequent image options can extend up to
post_change + 2 years, so some samples have the change further in the past. No
frequent block extends past the imagery cutoff (IMAGE_CUTOFF, or the group's
entry in GROUP_IMAGE_CUTOFFS) or the entry's optional window_stop_date, and no
option ends where the reference point is masked by its stop dates.
Options whose blocks coincide after clamping to the cutoff are kept only once.

Scene metadata is fetched from the OlmoEarth Datasets API. Required env vars:
- OEDATASETS_API_URL: e.g. https://datasets.olmoearth.allenai.org
- DATASETS_API_TOKEN: bearer token for API auth

Idempotent: existing windows are compared against their annotation entry.
Unchanged entries are skipped; entries whose labels changed get their label
layers rewritten in place (imagery stays valid); entries whose geometry or
dates changed are deleted and recreated (and need re-materialization).

After running this script, use ``rslearn dataset materialize`` to download imagery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import requests
import shapely
import shapely.geometry
from rasterio.enums import Resampling
from rslearn.config import LayerConfig, QueryConfig, SpaceMode
from rslearn.data_sources import Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import retry
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from .label_changes import (
    WINDOW_STOP_DATE,
    Change,
    missing_date_fields,
    point_changes,
    point_date_errors,
)
from .output_bands import (
    LC_CLASS_NAMES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from .timestamp_encoding import DAYS_NODATA, UNIX_EPOCH, parse_date
from .transforms import NUM_FREQUENT, NUM_FREQUENT_OPTIONS, QUARTERLY_PERIOD

COLLECTION = "sentinel-2-l2a"
# Only scenes this provider has are used. The dataset and prediction configs read
# Sentinel-2 from Planetary Computer only (AWS is excluded: some of its
# early-2022 baseline 04.00 scenes are not harmonized), so a scene only AWS has
# would materialize as a blank image, while prediction never sees it.
REQUIRED_PROVIDER = "planetary_computer"

FREQUENT_BLOCK_DURATION = timedelta(days=90)
FREQUENT_PERIOD = timedelta(days=7)
# Deterministic options end this long after the annotation date so the block
# covers that date.
OPTION_MARGIN = timedelta(days=6)

# How far before the earliest frequent block the quarterly layer reaches (20
# periods), matching the duration of sentinel2_quarterly in config_predict.json.
QUARTERLY_LOOKBACK = timedelta(days=1800)

# Annotation was done using imagery up to this date, so no frequent option should
# sample imagery after it (otherwise it could contain unannotated changes).
IMAGE_CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Groups annotated with imagery past IMAGE_CUTOFF, mapped to their own cutoff.
GROUP_IMAGE_CUTOFFS = {
    "phase23": datetime(2026, 11, 8, tzinfo=timezone.utc),
}

WINDOW_SIZE = 128

BIN_NO_CHANGE = 1
BIN_CHANGE = 2

# Annotation categories that train as another class of the same field: the
# finer-grained annotation categories, and the pre-2026-11-12 names of renamed
# categories (so older annotation exports still map). Categories in neither the
# training classes nor these maps (e.g. ice_loss) rasterize as nodata.
LEGACY_LAND_COVER_ALIASES = {
    "crops": "agriculture",
    "snow and ice": "snow/ice",
    "wetland (herbaceous)": "wetland",
}
PRE_CHANGE_CATEGORY_ALIASES = {
    "river_channel_migration": "water_to_land",
    "river_contract": "water_to_land",
    "large_water_body_level_drop": "water_to_land",
    "small_water_body_level_drop": "water_to_land",
    "other_water_to_land": "water_to_land",
    # Renamed 2026-11-12.
    "deforestation": "forest_loss",
    "urban_erosion": "removed_building",
    "water_contract": "water_to_land",
    "lake_level_drop": "water_to_land",
    "reservoir_level_drop": "water_to_land",
    "water_contract_other": "water_to_land",
}
POST_CHANGE_CATEGORY_ALIASES = {
    "tree_growth": "vegetation_growth",
    "wetland_growth": "vegetation_growth",
    "tree_crop_growth": "vegetation_growth",
    "new_reservoir": "new_infrastructure",
    "new_solar": "new_infrastructure",
    "new_wind": "new_infrastructure",
    "new_dam": "new_infrastructure",
    "new_industrial_infrastructure": "new_infrastructure",
    "new_offshore_wind": "new_infrastructure",
    "new_offshore_platform": "new_infrastructure",
    "new_paved_area": "new_infrastructure",
    "new_canal": "new_infrastructure",
    "new_other_infrastructure": "new_infrastructure",
    "other_land_to_water": "land_to_water",
    "river_channel_migration": "land_to_water",
    "river_flooding": "flooding",
    "large_water_body_level_rise": "land_to_water",
    "small_water_body_level_rise": "land_to_water",
    "coastal_erosion": "land_to_water",
    "borrow_pit": "new_mining_or_quarry",
    "tree_crop_harvest": "agricultural_activity",
    "resurfaced_road": "new_road",
    # Renamed 2026-11-12.
    "new_crop_field": "new_agriculture",
    "mining": "new_mining_or_quarry",
    "wildfire": "burned",
    "water_expand": "land_to_water",
    "lake_level_rise": "land_to_water",
    "reservoir_level_rise": "land_to_water",
}
# Post-change annotation categories that train as a pre-change class instead (the
# post field then counts as unset). A pre_change_category on the same point takes
# precedence.
POST_TO_PRE_CHANGE_CATEGORIES = {
    "land_reclamation": "water_to_land",
}

# Maps positive-point annotation field -> (label layer name, class-name list).
CHANGE_CATEGORY_FIELDS = {
    "pre_change_category": ("label_pre_change", PRE_CHANGE_CATEGORY_NAMES),
    "post_change_category": ("label_post_change", POST_CHANGE_CATEGORY_NAMES),
}
CHANGE_CATEGORY_ALIASES = {
    "pre_change_category": PRE_CHANGE_CATEGORY_ALIASES,
    "post_change_category": POST_CHANGE_CATEGORY_ALIASES,
}
# Former annotation field whose categories are now post-change categories. Values
# still found in it are ignored.
LEGACY_SAME_CHANGE_FIELD = "same_change_category"

ANNOTATIONS_SIDECAR_FNAME = "lcc_annotations.json"


def _lonlat_to_pixel(
    lon: float, lat: float, projection: Projection, bounds: tuple[int, ...]
) -> tuple[int, int]:
    """Convert lon/lat to pixel coords within bounds, using floor for snapping."""
    st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None)
    projected = st.to_projection(projection)
    col = math.floor(projected.shp.x) - bounds[0]
    row = math.floor(projected.shp.y) - bounds[1]
    return col, row


def _category_to_id(category: str) -> int:
    """Convert a land cover category name to class ID (1-indexed, 0 = nodata).

    Pre-2026-11-12 names are mapped through LEGACY_LAND_COVER_ALIASES.
    """
    try:
        return LC_CLASS_NAMES.index(LEGACY_LAND_COVER_ALIASES.get(category, category))
    except ValueError:
        return 0


def _change_category_to_id(value: str, class_names: list[str]) -> int:
    """Convert a change-category value to a class ID within ``class_names``.

    Returns 0 (nodata) for empty/unknown values. A set-but-recognized value maps
    to its index in ``class_names`` (>= 2, since class 1 is "none").
    """
    if not value:
        return 0
    try:
        return class_names.index(value)
    except ValueError:
        return 0


def _get_change_categories(pt: dict[str, Any]) -> dict[str, str]:
    """Get a positive point's change categories as training class names.

    Applies CHANGE_CATEGORY_ALIASES and POST_TO_PRE_CHANGE_CATEGORIES. Unset fields
    are "", and unknown values are returned unchanged (they rasterize as nodata).

    Returns:
        map from change-category field (CHANGE_CATEGORY_FIELDS) to class name.
    """
    values = {field: pt.get(field) or "" for field in CHANGE_CATEGORY_FIELDS}
    post_value = values["post_change_category"]
    if post_value in POST_TO_PRE_CHANGE_CATEGORIES:
        values["post_change_category"] = ""
        if not values["pre_change_category"]:
            values["pre_change_category"] = POST_TO_PRE_CHANGE_CATEGORIES[post_value]
    return {
        field: CHANGE_CATEGORY_ALIASES[field].get(value, value)
        for field, value in values.items()
    }


def _get_unknown_categories(entries: list[dict[str, Any]]) -> Counter[str]:
    """Count category values of positive points that have no training class.

    These are unknown land cover or change categories of either change (the second
    change's are counted under its change2_* field names), and any value in the
    legacy same_change_category field. Their field rasterizes as nodata, or as
    "none" for a change field when the other change field is set.

    Returns:
        counts keyed by "field: value".
    """
    counts: Counter[str] = Counter()
    for entry in entries:
        for pt in entry.get("positive_points", []):
            try:
                changes = point_changes(pt)
            except ValueError:
                continue  # reported by _validate_positive_point_dates
            for prefix, change in zip(("", "change2_"), changes):
                if change is None:
                    continue
                fields = change.as_point_fields()
                for field in ("pre_category", "post_category"):
                    value = fields[field]
                    if value and _category_to_id(value) == 0:
                        counts[f"{prefix}{field}: {value}"] += 1
                for field, value in _get_change_categories(fields).items():
                    class_names = CHANGE_CATEGORY_FIELDS[field][1]
                    if value and value not in class_names[2:]:
                        counts[f"{prefix}{field}: {value}"] += 1
            if pt.get(LEGACY_SAME_CHANGE_FIELD):
                counts[
                    f"{LEGACY_SAME_CHANGE_FIELD}: {pt[LEGACY_SAME_CHANGE_FIELD]}"
                ] += 1
    return counts


def _search_oedatasets(
    session: requests.Session,
    api_url: str,
    api_token: str,
    geometry_geojson: dict[str, Any],
    time_range: tuple[datetime, datetime],
    required_provider: str | None = REQUIRED_PROVIDER,
) -> list[dict[str, Any]]:
    """Search OlmoEarth Datasets API for Sentinel-2 scenes.

    Scenes that required_provider (if set) does not have are dropped.

    Returns list of dicts with keys: id, collected_at, cloud_cover,
    geometry_geojson, data_providers.
    """
    url = f"{api_url}/api/v1/items/search"
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    items: list[dict[str, Any]] = []
    offset = 0
    limit = 1000

    while True:
        body = {
            "collection": {"eq": COLLECTION},
            "intersects_geometry": geometry_geojson,
            "collected_at": {
                "gte": time_range[0].isoformat(),
                "lt": time_range[1].isoformat(),
            },
            "limit": limit,
            "offset": offset,
            "sort_by": "collected_at",
            "sort_direction": "desc",
        }
        resp = session.post(url, json=body, headers=headers, timeout=30)
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} for {url}: {resp.text}\nRequest body: {json.dumps(body, default=str)}",
                response=resp,
            )
        records = resp.json()["records"]

        if not records:
            break

        for item in records:
            data_providers = sorted(item.get("data_providers") or {})
            if (
                required_provider is not None
                and required_provider not in data_providers
            ):
                continue
            props = item["properties"]
            cloud_cover = props.get("cloud_cover")
            if cloud_cover is None:
                cloud_cover = 100
            items.append(
                {
                    "id": item["id"],
                    "collected_at": parse_date(props["collected_at"]),
                    "cloud_cover": cloud_cover,
                    "geometry_geojson": props["geometry"],
                    "data_providers": data_providers,
                }
            )

        if len(records) < limit:
            break
        offset += limit

    return items


QUARTERLY_QUERY_CONFIG = QueryConfig(
    space_mode=SpaceMode.MOSAIC,
    # Enough periods to cover the whole window time range, since rslearn keeps the
    # most recent periods when max_matches truncates.
    max_matches=100,
    min_matches=1,
    period_duration=QUARTERLY_PERIOD,
    per_period_mosaic_reverse_time_order=False,
)

# Walks back from the block end one 7-day period at a time, skipping periods with no
# imagery, until NUM_FREQUENT periods have a mosaic; blocks with fewer are
# rejected. Matches sentinel2_frequent_0 in the prediction dataset config.
FREQUENT_QUERY_CONFIG = QueryConfig(
    space_mode=SpaceMode.MOSAIC,
    max_matches=NUM_FREQUENT,
    min_matches=NUM_FREQUENT,
    period_duration=FREQUENT_PERIOD,
    per_period_mosaic_reverse_time_order=False,
)


def _to_rslearn_items(items: list[dict[str, Any]]) -> list[Item]:
    """Convert OlmoEarth Datasets search results to rslearn Items, keeping order."""
    return [
        Item(
            item["id"],
            STGeometry(
                WGS84_PROJECTION,
                shapely.geometry.shape(item["geometry_geojson"]),
                (item["collected_at"], item["collected_at"]),
            ),
        )
        for item in items
    ]


def _build_quarterly_layer_data(
    rslearn_items: list[Item],
    time_range: tuple[datetime, datetime],
    projection: Projection,
    bounds: tuple[int, ...],
) -> WindowLayerData:
    """Build quarterly mosaics using rslearn's match_candidate_items_to_window.

    Matches the MOSAIC + period_duration=90d behavior used in the prediction pipeline.
    Items must be sorted by cloud cover ascending so the MOSAIC matcher sees the
    clearest candidates first within each period.
    """
    window_geom = STGeometry(
        projection,
        shapely.box(bounds[0], bounds[1], bounds[2], bounds[3]),
        time_range,
    )

    matched_groups = match_candidate_items_to_window(
        window_geom, rslearn_items, QUARTERLY_QUERY_CONFIG
    )

    serialized_groups = [
        [gi.serialize() for gi in group.items] for group in matched_groups
    ]
    group_time_ranges: list[tuple[datetime, datetime] | None] = [
        group.request_time_range for group in matched_groups
    ]

    return WindowLayerData(
        layer_name="sentinel2_quarterly",
        serialized_item_groups=serialized_groups,
        group_time_ranges=group_time_ranges,
    )


def _build_frequent_layer_data(
    rslearn_items: list[Item],
    block_start: datetime,
    projection: Projection,
    bounds: tuple[int, ...],
    layer_name: str,
) -> WindowLayerData | None:
    """Build WindowLayerData for one frequent option.

    Selects the least-cloudy mosaic in each of the NUM_FREQUENT most recent
    7-day periods with imagery in the 90-day block starting at block_start, in
    chronological order. Items must be sorted by cloud cover ascending so the
    MOSAIC matcher sees the clearest candidates first within each period. Returns
    None if fewer than NUM_FREQUENT periods have imagery.
    """
    window_geom = STGeometry(
        projection,
        shapely.box(bounds[0], bounds[1], bounds[2], bounds[3]),
        (block_start, block_start + FREQUENT_BLOCK_DURATION),
    )

    matched_groups = match_candidate_items_to_window(
        window_geom, rslearn_items, FREQUENT_QUERY_CONFIG
    )
    if not matched_groups:
        return None

    serialized_groups = [
        [gi.serialize() for gi in group.items] for group in matched_groups
    ]
    group_time_ranges = [group.request_time_range for group in matched_groups]

    return WindowLayerData(
        layer_name=layer_name,
        serialized_item_groups=serialized_groups,
        group_time_ranges=group_time_ranges,
    )


def _compute_frequent_block_ends(
    first_noticeable: datetime,
    post_change: datetime,
    window_name: str,
    cutoff: datetime,
    extra_dates: list[datetime] | None = None,
    excluded_end_ranges: list[tuple[datetime, datetime]] | None = None,
) -> list[datetime]:
    """Compute 90-day frequent-image block ends for training options.

    Each option's block is [end - FREQUENT_BLOCK_DURATION, end]. Every end is
    clamped to ``cutoff`` so no option samples imagery after it, and ends that
    coincide after clamping are kept once (in order), so windows annotated close
    to the cutoff get fewer options rather than duplicate ones.

    Option 2 (the one validation uses) ends right after post_change, only when that
    is more than 5 days after first_noticeable (so not when first_noticeable is at
    or after post_change). For a point with a second change, post_change is the
    second change's, and extra_dates holds the first change's end and, when it
    differs from its post_change, the second change's first-noticeable date: an
    option ending right after each is added after the deterministic options, in
    place of one random option each, so option 2 does not change. Random options
    run up to 2 years after the latest of these dates.

    No end falls strictly inside one of the excluded_end_ranges (start, end): the
    random options are drawn outside them, and any other end inside one moves back
    to its start. Without excluded ranges the options are unchanged.

    Randomness is derived from window_name so results are deterministic per window.
    """
    rng = random.Random(hashlib.sha256(window_name.encode()).hexdigest())

    block_ends: list[datetime] = []

    # Option 0: the block ends right after first_noticeable.
    block_ends.append(first_noticeable + OPTION_MARGIN)

    # Option 1: first_noticeable falls partway through the block.
    block_ends.append(first_noticeable + timedelta(days=rng.randint(15, 60)))

    # Option 2: the block ends right after post_change, if meaningfully different.
    if (post_change - first_noticeable).days > 5:
        block_ends.append(post_change + OPTION_MARGIN)

    # Extra options (first change's end, second change's first-noticeable): the
    # block ends right after each.
    extra = extra_dates or []
    for date in extra:
        block_ends.append(date + OPTION_MARGIN)

    # Remaining random options: later post-change contexts up to 2y after the
    # latest annotated date.
    latest = max([post_change, first_noticeable] + extra)
    random_start = first_noticeable + timedelta(days=60)
    random_end = min(latest + timedelta(days=730), cutoff)
    num_random = NUM_FREQUENT_OPTIONS - len(block_ends)

    excluded = excluded_end_ranges or []

    def is_excluded(end: datetime) -> bool:
        return any(start < end < stop for start, stop in excluded)

    for _ in range(num_random):
        if random_end > random_start:
            days_range = (random_end - random_start).days
            if excluded:
                allowed = [
                    offset
                    for offset in range(days_range + 1)
                    if not is_excluded(random_start + timedelta(days=offset))
                ]
                random_offset = rng.choice(allowed) if allowed else 0
            else:
                random_offset = rng.randint(0, max(days_range, 1))
            block_ends.append(random_start + timedelta(days=random_offset))
        else:
            block_ends.append(random_start)

    block_ends = [min(end, cutoff) for end in block_ends]
    for i, end in enumerate(block_ends):
        # Moving back to a range's start can land in another range.
        while is_excluded(end):
            end = max(start for start, stop in excluded if start < end < stop)
        block_ends[i] = end
    block_ends = list(dict.fromkeys(block_ends))

    return block_ends[:NUM_FREQUENT_OPTIONS]


def _compute_window_time_range(
    block_starts: list[datetime], block_ends: list[datetime]
) -> tuple[datetime, datetime]:
    """Compute the window time range from the frequent blocks.

    rslearn builds period mosaics backwards from the end of the time range, so the
    end is extended (by less than one quarterly period) past the latest block end
    to land on the 90-day grid through the earliest block start. Then one quarterly
    period ends exactly where the earliest frequent block begins, like the
    prediction dataset's quarterly layer (which ends at T - 90d), and the
    quarterly images line up with prediction for the earliest option. The start
    is QUARTERLY_LOOKBACK before the earliest block start.
    """
    earliest_start = min(block_starts)
    num_periods = math.ceil((max(block_ends) - earliest_start) / QUARTERLY_PERIOD)
    window_end = earliest_start + num_periods * QUARTERLY_PERIOD
    window_start = earliest_start - QUARTERLY_LOOKBACK
    return (window_start, window_end)


def _write_label_layer(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    array_hw: np.ndarray,
) -> None:
    """Write a single-band label raster and mark layer complete."""
    band_set = layer_config.band_sets[0]
    chw = array_hw[np.newaxis, :, :].astype(
        band_set.dtype.get_numpy_dtype(), copy=False
    )
    with window.data.open_layer_writer(layer_name) as writer:
        writer.write_raster(
            band_set.bands,
            band_set.instantiate_raster_format(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=chw),
        )
    window.mark_layer_completed(layer_name)


def _date_to_unix_days(date: datetime) -> int:
    """Days since 1970-01-01 (UTC) of a timezone-aware datetime."""
    days = (date - UNIX_EPOCH) // timedelta(days=1)
    if days < 0 or days >= DAYS_NODATA:
        raise ValueError(f"{date} cannot be represented as uint16 days since 1970")
    return days


def _change_category_ids(change: Change) -> tuple[int, int]:
    """(pre, post) change-category class ids of a change, 0 = nodata.

    Only set when at least one of the two is set; an unset one then becomes class
    1 ("none").
    """
    values = _get_change_categories(change.as_point_fields())
    ids = [
        _change_category_to_id(values[field], class_names)
        for field, (_, class_names) in CHANGE_CATEGORY_FIELDS.items()
    ]
    if not any(cid > 0 for cid in ids):
        return 0, 0
    pre_id, post_id = (cid if cid > 0 else 1 for cid in ids)
    return pre_id, post_id


def _days_or_nodata(value: str, gap: timedelta = timedelta(0)) -> int:
    """uint16 days since 1970 of a YYYY-MM-DD string plus gap, DAYS_NODATA if unset."""
    if not value:
        return DAYS_NODATA
    return _date_to_unix_days(parse_date(value) + gap)


def _compute_label_arrays(
    entry: dict[str, Any],
    projection: Projection,
    bounds: tuple[int, ...],
    gap: timedelta = timedelta(0),
) -> dict[str, np.ndarray]:
    """Rasterize point labels into per-layer arrays (no I/O).

    Each positive point's first change fills label_src/dst/pre_change/post_change,
    label_ts_start/ts_end (pre_change/post_change), label_first_noticeable and
    label_stop_date, and its optional second change the label_change2_* rasters
    (see transforms.gate_change_targets). gap is added to the first-noticeable and
    post-change dates, matching the dates the frequent options are built from.
    """
    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]

    def dates() -> np.ndarray:
        return np.full((h, w), DAYS_NODATA, dtype=np.uint16)

    def classes() -> np.ndarray:
        return np.zeros((h, w), dtype=np.uint8)

    binary = classes()
    # Per-change rasters: land cover (src/dst) and change-category classes (0 =
    # nodata, masked), and dates as days since 1970-01-01 (DAYS_NODATA = unset).
    # label_ts_start/label_ts_end are also read by rslearn's PerPixelTimestepTask.
    first = {
        "label_src": classes(),
        "label_dst": classes(),
        "label_pre_change": classes(),
        "label_post_change": classes(),
        "label_ts_start": dates(),
        "label_ts_end": dates(),
        "label_first_noticeable": dates(),
        "label_stop_date": dates(),
    }
    second = {
        f"label_change2_{key[len('label_') :]}": array()
        for key, array in (
            ("label_src", classes),
            ("label_dst", classes),
            ("label_pre_change", classes),
            ("label_post_change", classes),
            ("label_ts_start", dates),
            ("label_ts_end", dates),
            ("label_first_noticeable", dates),
            ("label_stop_date", dates),
        )
    }

    for pt in entry.get("negative_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if 0 <= col < w and 0 <= row < h:
            binary[row, col] = BIN_NO_CHANGE

    for pt in entry.get("positive_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if not (0 <= col < w and 0 <= row < h):
            continue
        binary[row, col] = BIN_CHANGE
        first_change, second_change = point_changes(pt)
        for change, arrays, prefix in (
            (first_change, first, "label_"),
            (second_change, second, "label_change2_"),
        ):
            if change is None:
                continue
            arrays[f"{prefix}src"][row, col] = _category_to_id(change.pre_category)
            arrays[f"{prefix}dst"][row, col] = _category_to_id(change.post_category)
            pre_id, post_id = _change_category_ids(change)
            arrays[f"{prefix}pre_change"][row, col] = pre_id
            arrays[f"{prefix}post_change"][row, col] = post_id
            if change.pre_change and change.post_change:
                arrays[f"{prefix}ts_start"][row, col] = _days_or_nodata(
                    change.pre_change
                )
                arrays[f"{prefix}ts_end"][row, col] = _days_or_nodata(
                    change.post_change, gap
                )
            arrays[f"{prefix}first_noticeable"][row, col] = _days_or_nodata(
                change.first_noticeable, gap
            )
            arrays[f"{prefix}stop_date"][row, col] = _days_or_nodata(change.stop_date)

    return {"label_binary": binary, **first, **second}


def _write_label_layers(
    window: Window,
    layers: dict[str, LayerConfig],
    label_arrays: dict[str, np.ndarray],
) -> None:
    """Write the rasterized label arrays to the window's label layers."""
    for layer_name, array_hw in label_arrays.items():
        _write_label_layer(window, layer_name, layers[layer_name], array_hw)


def _label_layers_match(
    window: Window,
    layers: dict[str, LayerConfig],
    label_arrays: dict[str, np.ndarray],
) -> bool:
    """Check whether the window's existing label rasters equal ``label_arrays``."""
    for layer_name, expected_hw in label_arrays.items():
        if not window.is_layer_completed(layer_name):
            return False
        band_set = layers[layer_name].band_sets[0]
        raster = window.data.read_raster(
            layer_name,
            band_set.bands,
            band_set.instantiate_raster_format(),
            resampling=Resampling.nearest,
        )
        existing = raster.array
        if existing.shape != (1, 1) + expected_hw.shape:
            return False
        if not np.array_equal(existing[0, 0], expected_hw):
            return False
    return True


def _validate_positive_point_dates(entry: dict[str, Any]) -> None:
    """Validate the dates of the entry's positive points and its window stop date.

    Checks label_changes.point_date_errors on every positive point with all its
    dates (label_changes.missing_date_fields; points without them are handled as
    incomplete), and that the optional window_stop_date is after the
    first-noticeable and post-change dates of every such point's first change.

    Raises:
        ValueError: if any check fails.
    """
    window = entry.get("window_name")
    window_stop = entry.get(WINDOW_STOP_DATE)
    for pt in entry.get("positive_points", []):
        if missing_date_fields(pt):
            continue
        errors = point_date_errors(pt)
        if window_stop and not all(
            parse_date(window_stop) > parse_date(pt[field])
            for field in ("first_date_change_noticeable", "post_change")
        ):
            errors.append(
                f"{WINDOW_STOP_DATE} {window_stop} must be after "
                "first_date_change_noticeable and post_change"
            )
        if errors:
            raise ValueError(
                f"Invalid dates for positive point in window {window} "
                f"(lon={pt.get('lon')}, lat={pt.get('lat')}): {'; '.join(errors)}"
            )


def positive_point_dates_valid(entry: dict[str, Any]) -> bool:
    """Check the positive point dates like _validate_positive_point_dates.

    Returns:
        False where _validate_positive_point_dates would raise, otherwise True.
    """
    try:
        _validate_positive_point_dates(entry)
    except ValueError:
        return False
    return True


def _get_stop_date(entry: dict[str, Any]) -> datetime | None:
    """Get the entry's window stop date, if any."""
    value = entry.get(WINDOW_STOP_DATE)
    return parse_date(value) if value else None


def _get_positive_points_missing_dates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Get positive points missing date fields, if the entry has mixed points.

    An entry is mixed when at least one positive point has all its date fields
    (label_changes.missing_date_fields) and at least one doesn't. Returns the incomplete points for mixed entries, otherwise an empty
    list (entries with no dated positive points stay on the "incomplete" path).
    """
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for pt in entry.get("positive_points", []):
        if not missing_date_fields(pt):
            complete.append(pt)
        else:
            incomplete.append(pt)
    if complete and incomplete:
        return incomplete
    return []


def _entry_has_complete_annotations(entry: dict[str, Any]) -> bool:
    """Check that the entry has enough annotation info to create a training window.

    Accepts entries with either:
    - At least one fully-annotated positive point (with dates and categories), OR
    - No positive points but at least one negative point and an anchor_date or
      time_range field.
    """
    for pt in entry.get("positive_points", []):
        if (
            not missing_date_fields(pt)
            and pt.get("pre_category")
            and pt.get("post_category")
        ):
            return True
    if (
        not entry.get("positive_points")
        and entry.get("negative_points")
        and (entry.get("anchor_date") or entry.get("time_range"))
    ):
        return True
    return False


def _negative_only_anchor(entry: dict[str, Any]) -> datetime:
    """Get the anchor date for an entry without positive points.

    The earliest frequent option ends right after this date. Uses the entry's
    anchor_date if set, otherwise the midpoint of its time_range.
    """
    if entry.get("anchor_date"):
        return parse_date(entry["anchor_date"])
    tr = entry["time_range"]
    t_start = parse_date(tr[0])
    t_end = parse_date(tr[1])
    return t_start + (t_end - t_start) / 2


def _get_window_wgs84_bounds(
    projection: Projection, bounds: tuple[int, ...]
) -> shapely.geometry.base.BaseGeometry:
    """Get the WGS84 bounding box for the window."""
    box = shapely.box(bounds[0], bounds[1], bounds[2], bounds[3])
    st = STGeometry(projection, box, time_range=None)
    wgs84 = st.to_projection(WGS84_PROJECTION)
    return wgs84.shp


def _sidecar_dates_match(
    existing_value: dict[str, Any] | None,
    new_value: dict[str, Any],
) -> bool:
    """Check whether the annotation dates in the existing sidecar entry match.

    The frequent-block time ranges (and hence the imagery layer datas) are fully
    determined by these dates plus the window name, so matching dates mean the
    existing imagery layers are still valid.
    """
    if existing_value is None:
        return False
    # Sidecars written before transition_date / image_cutoff / excluded_end_ranges
    # existed lack them, which matches None.
    return all(
        existing_value.get(key) == new_value.get(key)
        for key in (
            "pre_change",
            "post_change",
            "first_noticeable",
            "stop_date",
            "transition_date",
            "image_cutoff",
            "excluded_end_ranges",
        )
    )


@dataclass
class WindowDates:
    """The dates that determine a window's frequent options (see _window_dates)."""

    center_point: dict[str, Any]
    first_noticeable: datetime
    post_change: datetime
    extra_dates: list[datetime]
    cutoff: datetime
    excluded_end_ranges: list[tuple[datetime, datetime]]
    # The annotations sidecar entry: these dates, so a window whose sidecar matches
    # still has valid imagery layers.
    sidecar_value: dict[str, Any]


def _window_dates(
    entry: dict[str, Any], window_group: str, gap: timedelta
) -> WindowDates:
    """The reference point and dates of an entry's frequent options.

    The reference point is the first positive point with all its dates
    (label_changes.missing_date_fields); without one, the window is anchored on its
    negative points.
    """
    ref_point = None
    for pt in entry.get("positive_points", []):
        if not missing_date_fields(pt):
            ref_point = pt
            break

    # With a second change, the reference point's post_change is the second
    # change's, and transition_date (named for the sidecars written before the
    # second change replaced the transition) is the first change's post_change.
    transition_date: datetime | None = None
    extra_dates: list[datetime] = []
    # The reference point's masked intervals: after its first change's stop date
    # (until its second change is noticeable), and after its second change's.
    ref_masked: list[tuple[datetime, datetime | None]] = []
    if ref_point is None:
        if not entry.get("negative_points"):
            raise ValueError("Entry has no positive or negative points")
        center_point = entry["negative_points"][0]
        anchor = _negative_only_anchor(entry)
        post_change = anchor + gap
        first_noticeable = anchor + gap
    else:
        center_point = ref_point
        first_change, second_change = point_changes(ref_point)
        post_change = parse_date(first_change.post_change) + gap
        first_noticeable = parse_date(first_change.first_noticeable) + gap
        if second_change is not None:
            transition_date = post_change
            extra_dates.append(transition_date)
            second_post = parse_date(second_change.post_change) + gap
            second_noticeable = parse_date(second_change.first_noticeable) + gap
            if second_noticeable != second_post:
                extra_dates.append(second_noticeable)
            post_change = second_post
            if first_change.stop_date:
                ref_masked.append(
                    (parse_date(first_change.stop_date), second_noticeable)
                )
            if second_change.stop_date:
                ref_masked.append((parse_date(second_change.stop_date), None))
        elif first_change.stop_date:
            ref_masked.append((parse_date(first_change.stop_date), None))

    stop_date = _get_stop_date(entry)
    image_cutoff = GROUP_IMAGE_CUTOFFS.get(window_group, IMAGE_CUTOFF)
    cutoff = image_cutoff if stop_date is None else min(image_cutoff, stop_date)
    # None for the default cutoff, so sidecars written before this key existed match.
    sidecar_image_cutoff = (
        None if image_cutoff == IMAGE_CUTOFF else image_cutoff.isoformat()
    )

    # No frequent option ends where the reference point is masked (after a stop
    # date). Intervals starting at or after the cutoff are left out, since no
    # option ends past the cutoff anyway (and then the options are drawn as without
    # excluded ranges).
    excluded_end_ranges = [
        (start, cutoff + timedelta(days=1) if stop is None else stop)
        for start, stop in ref_masked
        if start < cutoff
    ]
    # None without ranges, so sidecars written before this key existed match.
    sidecar_excluded_end_ranges = [
        [start.isoformat(), end.isoformat()] for start, end in excluded_end_ranges
    ] or None

    sidecar_value: dict[str, Any]
    if ref_point is None:
        sidecar_value = {
            "pre_change": anchor.isoformat(),
            "post_change": post_change.isoformat(),
            "first_noticeable": first_noticeable.isoformat(),
            "stop_date": None,
            "transition_date": None,
            "image_cutoff": sidecar_image_cutoff,
            "excluded_end_ranges": sidecar_excluded_end_ranges,
        }
    else:
        sidecar_value = {
            "pre_change": ref_point["pre_change"],
            "post_change": post_change.isoformat(),
            "first_noticeable": first_noticeable.isoformat(),
            "stop_date": stop_date.isoformat() if stop_date is not None else None,
            "transition_date": (
                transition_date.isoformat() if transition_date is not None else None
            ),
            "image_cutoff": sidecar_image_cutoff,
            "excluded_end_ranges": sidecar_excluded_end_ranges,
        }

    return WindowDates(
        center_point=center_point,
        first_noticeable=first_noticeable,
        post_change=post_change,
        extra_dates=extra_dates,
        cutoff=cutoff,
        excluded_end_ranges=excluded_end_ranges,
        sidecar_value=sidecar_value,
    )


def _window_block_ends(window_name: str, dates: WindowDates) -> list[datetime]:
    """The frequent block ends of a window (see _compute_frequent_block_ends)."""
    return _compute_frequent_block_ends(
        dates.first_noticeable,
        dates.post_change,
        window_name,
        dates.cutoff,
        extra_dates=dates.extra_dates,
        excluded_end_ranges=dates.excluded_end_ranges,
    )


def _process_entry(
    entry: dict[str, Any],
    ds_path: str,
    gap_days: int = 0,
    existing_sidecar_value: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None, str]:
    """Process one annotation entry: create window, query API, write labels.

    Each call is independent (creates its own Dataset/session) so it can run
    in a separate multiprocessing worker.

    If the window already exists, it is compared against the entry:
    - projection/bounds and sidecar dates match, label rasters equal: skipped.
    - only the label rasters differ: label layers are rewritten in place (the
      imagery layer datas are unaffected since they depend only on the dates).
    - projection/bounds or dates differ: the window directory is deleted and the
      window is recreated from scratch (existing materialized imagery would be
      stale), so it must be materialized again.

    Args:
        entry: the annotation entry to process.
        ds_path: path to the output rslearn dataset.
        gap_days: add this many days to first_noticeable and post_change, so the
            frequent options start later and the model predicts change through
            post_change + gap.
        existing_sidecar_value: the current annotations-sidecar entry for this
            window, if any.

    Returns (sidecar_key, sidecar_value, status) where status is one of
    "created", "recreated", "labels_updated", or "unchanged". sidecar_value is
    None for "unchanged" (the existing sidecar entry should be kept).
    """
    api_url = os.environ["OEDATASETS_API_URL"].rstrip("/")
    api_token = os.environ.get("DATASETS_API_TOKEN", "")
    session = requests.Session()

    dataset = Dataset(UPath(ds_path))

    projection = Projection.deserialize(entry["projection"])
    window_name = entry["window_name"]
    window_group = entry["group"]

    gap = timedelta(days=gap_days)

    dates = _window_dates(entry, window_group, gap)
    center_point = dates.center_point

    # Center 128x128 window on the reference point.
    st = STGeometry(
        WGS84_PROJECTION,
        shapely.Point(center_point["lon"], center_point["lat"]),
        time_range=None,
    )
    projected = st.to_projection(projection)
    center_col = math.floor(projected.shp.x)
    center_row = math.floor(projected.shp.y)
    half = WINDOW_SIZE // 2
    bounds = (
        center_col - half,
        center_row - half,
        center_col + half,
        center_row + half,
    )

    label_arrays = _compute_label_arrays(entry, projection, bounds, gap=gap)

    positive_pixels = []
    for pt in entry.get("positive_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        positive_pixels.append({"col": col, "row": row})

    sidecar_key = f"{window_group}/{window_name}"
    sidecar_value = dates.sidecar_value

    # Decide what to do if the window already exists.
    status = "created"
    window_root = Window.get_window_root(UPath(ds_path), window_group, window_name)
    if (window_root / "metadata.json").exists():
        existing_windows = dataset.load_windows(
            groups=[window_group], names=[window_name]
        )
        existing_window = existing_windows[0] if existing_windows else None
        if (
            existing_window is not None
            and existing_window.projection == projection
            and tuple(existing_window.bounds) == bounds
            and _sidecar_dates_match(existing_sidecar_value, sidecar_value)
        ):
            if _label_layers_match(existing_window, dataset.layers, label_arrays):
                return sidecar_key, None, "unchanged"
            # Only the labels changed: rewrite them in place and keep the
            # existing window metadata, layer datas, and materialized imagery.
            _write_label_layers(existing_window, dataset.layers, label_arrays)
            return sidecar_key, sidecar_value, "labels_updated"
        # Geometry or dates changed: existing imagery is stale, start over.
        window_root.fs.rm(window_root.path, recursive=True)
        status = "recreated"

    block_ends = _window_block_ends(window_name, dates)
    block_starts = [end - FREQUENT_BLOCK_DURATION for end in block_ends]

    window_time_range = _compute_window_time_range(block_starts, block_ends)

    split_hash = hashlib.sha256(f"{window_group}/{window_name}".encode()).hexdigest()
    split = "val" if split_hash[0] in "01" else "train"

    window = Window(
        storage=dataset.storage,
        group=window_group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=window_time_range,
        options=dict(split=split),
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    bounds_wgs84 = _get_window_wgs84_bounds(projection, bounds)
    geojson = json.loads(shapely.to_geojson(bounds_wgs84))

    # Query OlmoEarth Datasets API one year at a time.
    all_items: list[dict[str, Any]] = []
    chunk_start = window_time_range[0]
    while chunk_start < window_time_range[1]:
        chunk_end = min(chunk_start + timedelta(days=365), window_time_range[1])
        chunk_range = (chunk_start, chunk_end)
        chunk_items = retry(
            lambda cr=chunk_range: _search_oedatasets(
                session, api_url, api_token, geojson, cr
            ),
            retry_max_attempts=3,
            retry_backoff=timedelta(seconds=30),
        )
        all_items.extend(chunk_items)
        chunk_start = chunk_end

    # Sort by cloud cover. Scenes the prediction pipeline used for this window (the
    # optional prediction_items entry key, set for windows sampled from
    # predictions) win cloud-cover ties, since the API orders ties arbitrarily;
    # this way the mosaics for the earliest option pick the same scenes as
    # prediction did.
    prediction_items = set(entry.get("prediction_items") or [])
    least_cloudy_items = sorted(
        all_items, key=lambda x: (x["cloud_cover"], x["id"] not in prediction_items)
    )

    rslearn_items = _to_rslearn_items(least_cloudy_items)
    quarterly_data = _build_quarterly_layer_data(
        rslearn_items, window_time_range, projection, bounds
    )
    layer_datas = window.load_layer_datas()
    layer_datas["sentinel2_quarterly"] = quarterly_data

    frequent_idx = 0
    for block_start in block_starts:
        layer_name = f"sentinel2_frequent_{frequent_idx}"
        freq_data = _build_frequent_layer_data(
            rslearn_items, block_start, projection, bounds, layer_name
        )
        if freq_data is not None:
            layer_datas[layer_name] = freq_data
            frequent_idx += 1

    window.save_layer_datas(layer_datas)

    _write_label_layers(window, dataset.layers, label_arrays)

    return sidecar_key, sidecar_value, status


def prepare(
    *,
    v2_json_paths: list[str],
    ds_path: str,
    workers: int = 32,
    gap_days: int = 0,
) -> None:
    """Prepare the LCC model dataset from v2 annotation JSONs.

    Idempotent: existing windows are compared against their annotation entry and
    only reprocessed as much as needed. Unchanged entries are skipped; if only
    the labels changed, the label layers are rewritten in place; if the window
    geometry or annotation dates changed, the window is deleted and recreated
    (and must be materialized again).

    Args:
        v2_json_paths: Paths to the v2 annotation JSONs.
        ds_path: Path to the output rslearn dataset (config.json must exist).
        workers: Number of parallel workers (0 = sequential).
        gap_days: add this many days to first_noticeable and post_change for
            every window, shifting the frequent options later.
    """
    if "OEDATASETS_API_URL" not in os.environ:
        raise RuntimeError("OEDATASETS_API_URL env var must be set")

    # Each element is (source path, index within that file, entry).
    entries: list[tuple[str, int, dict[str, Any]]] = []
    for v2_json_path in v2_json_paths:
        with open(v2_json_path) as f:
            for idx, entry in enumerate(json.load(f)):
                entries.append((v2_json_path, idx, entry))

    ds_upath = UPath(ds_path)

    # Load existing sidecar so we can merge new entries into it.
    sidecar_path = ds_upath / ANNOTATIONS_SIDECAR_FNAME
    if sidecar_path.exists():
        with sidecar_path.open("r") as f:
            annotations_sidecar: dict[str, dict[str, Any]] = json.load(f)
    else:
        annotations_sidecar = {}

    # Filter to complete, non-duplicate entries; existing windows are checked
    # against their entry in the workers.
    pending: list[dict[str, Any]] = []
    skipped_incomplete = 0
    skipped_duplicate_input = 0
    seen_window_keys: set[tuple[str, str]] = set()
    # One warning string per skipped mixed entry (some positive points have all
    # three date fields, some don't).
    mixed_warnings: list[str] = []

    for source_path, source_idx, entry in entries:
        _validate_positive_point_dates(entry)
        missing_date_points = _get_positive_points_missing_dates(entry)
        if missing_date_points:
            lines = [
                f"ERROR: mixed positive points in {source_path} index {source_idx} "
                f"(window {entry.get('group')}/{entry.get('window_name')}): "
                f"{len(missing_date_points)} positive point(s) missing date fields:"
            ]
            for pt in missing_date_points:
                missing_fields = missing_date_fields(pt)
                lines.append(
                    f"  point lon={pt.get('lon')} lat={pt.get('lat')} "
                    f"missing {', '.join(missing_fields)}"
                )
            mixed_warnings.append("\n".join(lines))
            continue
        if not _entry_has_complete_annotations(entry):
            skipped_incomplete += 1
            continue
        window_key = (entry["group"], entry["window_name"])
        if window_key in seen_window_keys:
            skipped_duplicate_input += 1
            continue
        seen_window_keys.add(window_key)
        pending.append(entry)

    print(
        f"{len(pending)} to check/process, "
        f"{skipped_incomplete} incomplete, "
        f"{skipped_duplicate_input} duplicate inputs, "
        f"{len(mixed_warnings)} skipped for mixed positive points"
    )

    unknown_categories = _get_unknown_categories(pending)
    if unknown_categories:
        print("Positive point categories without a training class (ignored):")
        for key, count in unknown_categories.most_common():
            print(f"  {key} ({count} points)")

    kwargs_list = [
        dict(
            entry=entry,
            ds_path=ds_path,
            gap_days=gap_days,
            existing_sidecar_value=annotations_sidecar.get(
                f"{entry['group']}/{entry['window_name']}"
            ),
        )
        for entry in pending
    ]

    status_counts = {"created": 0, "recreated": 0, "labels_updated": 0, "unchanged": 0}
    processed = 0
    with make_pool_and_star_imap_unordered(
        workers, _process_entry, kwargs_list
    ) as outputs:
        for sidecar_key, sidecar_value, status in outputs:
            if sidecar_value is not None:
                annotations_sidecar[sidecar_key] = sidecar_value
            status_counts[status] += 1
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{len(pending)} windows...")

    # Write annotation sidecar
    with sidecar_path.open("w") as f:
        json.dump(annotations_sidecar, f)

    print(
        f"{status_counts['created']} created, "
        f"{status_counts['recreated']} recreated (need re-materialization), "
        f"{status_counts['labels_updated']} labels updated, "
        f"{status_counts['unchanged']} unchanged; "
        f"skipped {skipped_incomplete} incomplete "
        f"+ {skipped_duplicate_input} duplicate inputs "
        f"+ {len(mixed_warnings)} mixed positive points"
    )
    print(f"Wrote annotation sidecar to {sidecar_path}")

    if mixed_warnings:
        print(
            f"\n{len(mixed_warnings)} entries were skipped because some of their "
            "positive points have all of pre_change/post_change/"
            "first_date_change_noticeable and some do not:"
        )
        for warning in mixed_warnings:
            print(warning)


def main() -> None:
    """Prepare LCC model dataset from v2 annotation JSONs."""
    parser = argparse.ArgumentParser(
        description="Prepare LCC model dataset from v2 annotation JSONs."
    )
    parser.add_argument(
        "--v2-json-paths",
        nargs="+",
        required=True,
        help="Path(s) to v2 annotation JSONs.",
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--gap_days",
        type=int,
        default=0,
        help="Add this many days to first_noticeable and post_change.",
    )
    args = parser.parse_args()

    prepare(
        v2_json_paths=args.v2_json_paths,
        ds_path=args.ds_path,
        workers=args.workers,
        gap_days=args.gap_days,
    )


if __name__ == "__main__":
    main()
