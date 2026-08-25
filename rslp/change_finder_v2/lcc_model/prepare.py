"""Prepare the LCC model dataset: windows, imagery layers, and labels.

This script takes one or more v2 annotation JSONs and creates an rslearn dataset with:
- sentinel2_quarterly: WindowLayerData with quarterly mosaics (90-day periods);
  the least-cloudy mosaic is selected within each period
- sentinel2_frequent_0..7: WindowLayerData with four 15-day periods each; the
  least-cloudy mosaic is selected within each period
- label_binary, label_src, label_dst: Pre-rasterized point labels
- label_pre_change, label_post_change, label_same_change: Pre-rasterized point
  labels for the pre/post/same change-category heads (class 1 = "none")

The time range for each window covers all annotation-derived frequent blocks and
enough preceding quarterly history. Frequent image options can extend up to
post_change + 2 years, so some samples have the change further in the past.

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

COLLECTION = "sentinel-2-l2a"

NUM_FREQUENT_OPTIONS = 8
NUM_FREQUENT_PERIODS = 4
FREQUENT_PERIOD_DAYS = 15
FREQUENT_PERIOD = timedelta(days=FREQUENT_PERIOD_DAYS)
FREQUENT_BLOCK_DURATION = NUM_FREQUENT_PERIODS * FREQUENT_PERIOD
FREQUENT_LAST_PERIOD_OFFSET = (NUM_FREQUENT_PERIODS - 1) * FREQUENT_PERIOD

# Annotation was done using imagery up to this date, so no frequent option should
# sample imagery after it (otherwise it could contain unannotated changes).
IMAGE_CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)

WINDOW_SIZE = 128

BIN_NODATA = 0
BIN_NO_CHANGE = 1
BIN_CHANGE = 2

CATEGORY_NAMES = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]

# Change-category names for the pre/post/same_change_category annotation fields.
# Class layout is [nodata, none, <options...>]: class 0 = nodata (masked, no point
# or none of the three fields set); class 1 = "none" (this field unset but a
# sibling field is set); classes >= 2 = the annotated options. Option lists mirror
# the annotation app (annotation_app/static/app.js).
PRE_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "deforestation",
    "urban_erosion",
    "wetland_loss",
    "water_contract",
    "removed_crop_structure",
]

POST_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_crop_field",
    "new_aquafarm",
    "site_clearing",
    "water_expand",
    "mining",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
]

SAME_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "agricultural_activity",
    "wildfire",
    "ice_motion",
    "flooding",
]

# Maps positive-point annotation field -> (label layer name, class-name list).
CHANGE_CATEGORY_FIELDS = {
    "pre_change_category": ("label_pre_change", PRE_CHANGE_CATEGORY_NAMES),
    "post_change_category": ("label_post_change", POST_CHANGE_CATEGORY_NAMES),
    "same_change_category": ("label_same_change", SAME_CHANGE_CATEGORY_NAMES),
}

ANNOTATIONS_SIDECAR_FNAME = "lcc_annotations.json"


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


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
    """Convert category name to class ID (1-indexed, 0 = nodata)."""
    try:
        return CATEGORY_NAMES.index(category)
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


def _search_oedatasets(
    session: requests.Session,
    api_url: str,
    api_token: str,
    geometry_geojson: dict[str, Any],
    time_range: tuple[datetime, datetime],
) -> list[dict[str, Any]]:
    """Search OlmoEarth Datasets API for Sentinel-2 scenes.

    Returns list of dicts with keys: id, collected_at, cloud_cover, geometry_geojson.
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
            props = item["properties"]
            cloud_cover = props.get("cloud_cover")
            if cloud_cover is None:
                cloud_cover = 100
            items.append(
                {
                    "id": item["id"],
                    "collected_at": _parse_date(props["collected_at"]),
                    "cloud_cover": cloud_cover,
                    "geometry_geojson": props["geometry"],
                }
            )

        if len(records) < limit:
            break
        offset += limit

    return items


QUARTERLY_QUERY_CONFIG = QueryConfig(
    space_mode=SpaceMode.MOSAIC,
    max_matches=40,
    min_matches=1,
    period_duration=timedelta(days=90),
    per_period_mosaic_reverse_time_order=False,
)

FREQUENT_QUERY_CONFIG = QueryConfig(
    space_mode=SpaceMode.MOSAIC,
    max_matches=NUM_FREQUENT_PERIODS,
    min_matches=NUM_FREQUENT_PERIODS,
    period_duration=FREQUENT_PERIOD,
    per_period_mosaic_reverse_time_order=False,
)


def _build_quarterly_layer_data(
    items: list[dict[str, Any]],
    time_range: tuple[datetime, datetime],
    projection: Projection,
    bounds: tuple[int, ...],
) -> WindowLayerData:
    """Build quarterly mosaics using rslearn's match_candidate_items_to_window.

    Matches the MOSAIC + period_duration=90d behavior used in the prediction pipeline.
    Items must be sorted by cloud cover ascending so the MOSAIC matcher sees the
    clearest candidates first within each period.
    """
    rslearn_items = [
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
    items: list[dict[str, Any]],
    block_start: datetime,
    projection: Projection,
    bounds: tuple[int, ...],
    layer_name: str,
) -> WindowLayerData | None:
    """Build WindowLayerData for one frequent option.

    Selects one least-cloudy mosaic in each of four 15-day periods. Items must be
    sorted by cloud cover ascending so the MOSAIC matcher sees the clearest
    candidates first within each period.
    """
    rslearn_items = [
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

    window_geom = STGeometry(
        projection,
        shapely.box(bounds[0], bounds[1], bounds[2], bounds[3]),
        (block_start, block_start + FREQUENT_BLOCK_DURATION),
    )

    matched_groups = match_candidate_items_to_window(
        window_geom, rslearn_items, FREQUENT_QUERY_CONFIG
    )
    if len(matched_groups) < NUM_FREQUENT_PERIODS:
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


def _compute_frequent_block_starts(
    first_noticeable: datetime,
    post_change: datetime,
    window_name: str,
) -> list[datetime]:
    """Compute 60-day frequent-image block starts for training options.

    Randomness is derived from window_name so results are deterministic per window.
    """
    rng = random.Random(hashlib.sha256(window_name.encode()).hexdigest())

    block_starts: list[datetime] = []

    # Option 0: first_noticeable starts the last 15-day frequent period.
    block_starts.append(first_noticeable - FREQUENT_LAST_PERIOD_OFFSET)

    # Option 1: first_noticeable starts one of the first three periods.
    notice_period_idx = rng.randrange(NUM_FREQUENT_PERIODS - 1)
    block_starts.append(first_noticeable - notice_period_idx * FREQUENT_PERIOD)

    # Option 2: post_change starts the last period, if it is meaningfully different.
    has_option1 = (post_change - first_noticeable).days > 5
    if has_option1:
        block_starts.append(post_change - FREQUENT_LAST_PERIOD_OFFSET)

    # Remaining random options: sample the last-period start, then derive block start.
    random_start = first_noticeable + timedelta(days=60)
    random_end = post_change + timedelta(days=730)
    num_random = NUM_FREQUENT_OPTIONS - len(block_starts)

    for _ in range(num_random):
        if random_end > random_start:
            days_range = (random_end - random_start).days
            random_offset = rng.randint(0, max(days_range, 1))
            last_period_start = random_start + timedelta(days=random_offset)
        else:
            last_period_start = random_start
        block_starts.append(last_period_start - FREQUENT_LAST_PERIOD_OFFSET)

    # Cap every option so its 60-day frequent block ends on/before IMAGE_CUTOFF;
    # clamping the start is equivalent to clamping the block end.
    max_block_start = IMAGE_CUTOFF - FREQUENT_BLOCK_DURATION
    block_starts = [min(bs, max_block_start) for bs in block_starts]

    return block_starts[:NUM_FREQUENT_OPTIONS]


def _write_label_layer(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    array_hw: np.ndarray,
) -> None:
    """Write a single-band uint8 label raster and mark layer complete."""
    band_set = layer_config.band_sets[0]
    chw = array_hw[np.newaxis, :, :].astype(np.uint8, copy=False)
    with window.data.open_layer_writer(layer_name) as writer:
        writer.write_raster(
            band_set.bands,
            band_set.instantiate_raster_format(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=chw),
        )
    window.mark_layer_completed(layer_name)


def _compute_label_arrays(
    entry: dict[str, Any],
    projection: Projection,
    bounds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Rasterize point labels into per-layer arrays (no I/O)."""
    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]

    binary = np.zeros((h, w), dtype=np.uint8)
    src_label = np.zeros((h, w), dtype=np.uint8)
    dst_label = np.zeros((h, w), dtype=np.uint8)
    # One label raster per change-category field (pre/post/same). Class 0 = nodata
    # (masked); at a positive point with at least one change-category field set,
    # every field's raster is written (unset fields become class 1 = "none").
    change_labels = {
        layer_name: np.zeros((h, w), dtype=np.uint8)
        for layer_name, _ in CHANGE_CATEGORY_FIELDS.values()
    }

    for pt in entry.get("negative_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if 0 <= col < w and 0 <= row < h:
            binary[row, col] = BIN_NO_CHANGE

    for pt in entry.get("positive_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if 0 <= col < w and 0 <= row < h:
            binary[row, col] = BIN_CHANGE
            src_id = _category_to_id(pt.get("pre_category", ""))
            dst_id = _category_to_id(pt.get("post_category", ""))
            if src_id > 0:
                src_label[row, col] = src_id
            if dst_id > 0:
                dst_label[row, col] = dst_id

            # Only train the change-category heads when at least one of the three
            # fields is set; otherwise leave nodata (masked) for all three.
            change_ids = {
                layer_name: _change_category_to_id(pt.get(field, ""), class_names)
                for field, (layer_name, class_names) in CHANGE_CATEGORY_FIELDS.items()
            }
            if any(cid > 0 for cid in change_ids.values()):
                for layer_name, cid in change_ids.items():
                    # Unset (cid == 0) but a sibling is set -> class 1 ("none").
                    change_labels[layer_name][row, col] = cid if cid > 0 else 1

    label_arrays = {
        "label_binary": binary,
        "label_src": src_label,
        "label_dst": dst_label,
    }
    label_arrays.update(change_labels)
    return label_arrays


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
    """Validate date ordering of fully-annotated positive points.

    Expected ordering is pre_change < first_observable <= post_change, where
    first_observable is the first_date_change_noticeable field. Raises ValueError
    on any positive point that violates this.
    """
    for pt in entry.get("positive_points", []):
        if not (
            pt.get("pre_change")
            and pt.get("post_change")
            and pt.get("first_date_change_noticeable")
        ):
            continue
        pre_change = _parse_date(pt["pre_change"])
        post_change = _parse_date(pt["post_change"])
        first_observable = _parse_date(pt["first_date_change_noticeable"])
        if pre_change >= first_observable or first_observable > post_change:
            raise ValueError(
                f"Invalid date ordering for positive point in window "
                f"{entry.get('group')}/{entry.get('window_name')}: "
                f"expected pre_change < first_observable <= post_change, got "
                f"pre_change={pt['pre_change']}, "
                f"first_observable={pt['first_date_change_noticeable']}, "
                f"post_change={pt['post_change']}"
            )


POSITIVE_POINT_DATE_FIELDS = (
    "pre_change",
    "post_change",
    "first_date_change_noticeable",
)


def _get_positive_points_missing_dates(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Get positive points missing date fields, if the entry has mixed points.

    An entry is mixed when at least one positive point has all three date fields
    (pre_change, post_change, first_date_change_noticeable) and at least one
    doesn't. Returns the incomplete points for mixed entries, otherwise an empty
    list (entries with no dated positive points stay on the "incomplete" path).
    """
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for pt in entry.get("positive_points", []):
        if all(pt.get(field) for field in POSITIVE_POINT_DATE_FIELDS):
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
    - No positive points but at least one negative point and a time_range field.
    """
    for pt in entry.get("positive_points", []):
        if (
            pt.get("pre_change")
            and pt.get("post_change")
            and pt.get("first_date_change_noticeable")
            and pt.get("pre_category")
            and pt.get("post_category")
        ):
            return True
    if (
        not entry.get("positive_points")
        and entry.get("negative_points")
        and entry.get("time_range")
    ):
        return True
    return False


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
    return all(
        existing_value.get(key) == new_value[key]
        for key in ("pre_change", "post_change", "first_noticeable")
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

    ref_point = None
    for pt in entry.get("positive_points", []):
        if (
            pt.get("pre_change")
            and pt.get("post_change")
            and pt.get("first_date_change_noticeable")
        ):
            ref_point = pt
            break

    if ref_point is None:
        if not entry.get("negative_points"):
            raise ValueError("Entry has no positive or negative points")
        center_point = entry["negative_points"][0]
        tr = entry["time_range"]
        t_start = _parse_date(tr[0])
        t_end = _parse_date(tr[1])
        midpoint = t_start + (t_end - t_start) / 2
        post_change = midpoint + gap
        first_noticeable = midpoint + gap
    else:
        center_point = ref_point
        post_change = _parse_date(ref_point["post_change"]) + gap
        first_noticeable = _parse_date(ref_point["first_date_change_noticeable"]) + gap

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

    label_arrays = _compute_label_arrays(entry, projection, bounds)

    positive_pixels = []
    for pt in entry.get("positive_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        positive_pixels.append({"col": col, "row": row})

    sidecar_key = f"{window_group}/{window_name}"
    if ref_point is None:
        sidecar_value = {
            "pre_change": midpoint.isoformat(),
            "post_change": post_change.isoformat(),
            "first_noticeable": first_noticeable.isoformat(),
            "positive_pixel_coords": [],
            "is_negative_only": True,
        }
    else:
        sidecar_value = {
            "pre_change": ref_point["pre_change"],
            "post_change": post_change.isoformat(),
            "first_noticeable": first_noticeable.isoformat(),
            "positive_pixel_coords": positive_pixels,
        }

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

    block_starts = _compute_frequent_block_starts(
        first_noticeable, post_change, window_name
    )

    window_end = max(start + FREQUENT_BLOCK_DURATION for start in block_starts)
    window_start = min(start - timedelta(days=16 * 90) for start in block_starts)
    window_time_range = (window_start, window_end)

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

    least_cloudy_items = sorted(all_items, key=lambda x: x["cloud_cover"])

    quarterly_data = _build_quarterly_layer_data(
        least_cloudy_items, window_time_range, projection, bounds
    )
    layer_datas = window.load_layer_datas()
    layer_datas["sentinel2_quarterly"] = quarterly_data

    frequent_idx = 0
    for block_start in block_starts:
        layer_name = f"sentinel2_frequent_{frequent_idx}"
        freq_data = _build_frequent_layer_data(
            least_cloudy_items, block_start, projection, bounds, layer_name
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
                missing_fields = [
                    field for field in POSITIVE_POINT_DATE_FIELDS if not pt.get(field)
                ]
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
