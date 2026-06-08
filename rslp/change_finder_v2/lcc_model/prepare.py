"""Prepare the LCC model dataset: windows, imagery layers, and labels.

This script takes one or more v2 annotation JSONs and creates an rslearn dataset with:
- sentinel2_quarterly: WindowLayerData with quarterly mosaics (90-day periods)
- sentinel2_frequent_0..7: WindowLayerData with four 15-day periods each; the
  least-cloudy mosaic is selected within each period
- label_binary, label_src, label_dst: Pre-rasterized point labels

The time range for each window covers all annotation-derived frequent blocks and
enough preceding quarterly history. Frequent image options can extend up to
post_change + 2 years, so some samples have the change further in the past.

Scene metadata is fetched from the OlmoEarth Datasets API. Required env vars:
- OEDATASETS_API_URL: e.g. https://datasets.olmoearth.allenai.org
- DATASETS_API_TOKEN: bearer token for API auth

Idempotent: existing windows are skipped, so re-running after new annotations
are added only processes and materializes the new entries.

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
from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources import Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import retry
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

COLLECTION = "sentinel-2-l2a"

NUM_FREQUENT_OPTIONS = 8
NUM_FREQUENT_PERIODS = 4
FREQUENT_PERIOD_DAYS = 15
FREQUENT_PERIOD = timedelta(days=FREQUENT_PERIOD_DAYS)
FREQUENT_BLOCK_DURATION = NUM_FREQUENT_PERIODS * FREQUENT_PERIOD
FREQUENT_LAST_PERIOD_OFFSET = (NUM_FREQUENT_PERIODS - 1) * FREQUENT_PERIOD

LABEL_BAND = "label"
RASTER_FORMAT = GeotiffRasterFormat()
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

    return block_starts[:NUM_FREQUENT_OPTIONS]


def _write_label_layer(window: Window, layer_name: str, array_hw: np.ndarray) -> None:
    """Write a single-band uint8 label raster and mark layer complete."""
    raster_dir = window.get_raster_dir(layer_name, [LABEL_BAND])
    chw = array_hw[np.newaxis, :, :].astype(np.uint8, copy=False)
    RASTER_FORMAT.encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=chw),
    )
    window.mark_layer_completed(layer_name)


def _rasterize_labels(
    window: Window,
    entry: dict[str, Any],
    projection: Projection,
    bounds: tuple[int, ...],
) -> None:
    """Rasterize point labels into binary/src/dst layers."""
    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]

    binary = np.zeros((h, w), dtype=np.uint8)
    src_label = np.zeros((h, w), dtype=np.uint8)
    dst_label = np.zeros((h, w), dtype=np.uint8)

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

    _write_label_layer(window, "label_binary", binary)
    _write_label_layer(window, "label_src", src_label)
    _write_label_layer(window, "label_dst", dst_label)


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


def _process_entry(
    entry: dict[str, Any],
    ds_path: str,
) -> tuple[str, dict[str, Any]]:
    """Process one annotation entry: create window, query API, write labels.

    Each call is independent (creates its own Dataset/session) so it can run
    in a separate multiprocessing worker.

    Returns (sidecar_key, sidecar_value) for the annotations sidecar.
    """
    api_url = os.environ["OEDATASETS_API_URL"].rstrip("/")
    api_token = os.environ.get("DATASETS_API_TOKEN", "")
    session = requests.Session()

    dataset = Dataset(UPath(ds_path))

    projection = Projection.deserialize(entry["projection"])
    window_name = entry["window_name"]
    window_group = entry["group"]

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
        post_change = midpoint
        first_noticeable = midpoint
    else:
        center_point = ref_point
        post_change = _parse_date(ref_point["post_change"])
        first_noticeable = _parse_date(ref_point["first_date_change_noticeable"])

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
        all_items, window_time_range, projection, bounds
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

    _rasterize_labels(window, entry, projection, bounds)

    positive_pixels = []
    for pt in entry.get("positive_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        positive_pixels.append({"col": col, "row": row})

    sidecar_key = f"{window_group}/{window_name}"
    if ref_point is None:
        mid_iso = midpoint.isoformat()
        sidecar_value = {
            "pre_change": mid_iso,
            "post_change": mid_iso,
            "first_noticeable": mid_iso,
            "positive_pixel_coords": [],
            "is_negative_only": True,
        }
    else:
        sidecar_value = {
            "pre_change": ref_point["pre_change"],
            "post_change": ref_point["post_change"],
            "first_noticeable": ref_point["first_date_change_noticeable"],
            "positive_pixel_coords": positive_pixels,
        }
    return sidecar_key, sidecar_value


def prepare(
    *,
    v2_json_paths: list[str],
    ds_path: str,
    workers: int = 32,
) -> None:
    """Prepare the LCC model dataset from v2 annotation JSONs.

    Idempotent: windows that already exist are skipped, so re-running after new
    annotations have been added only processes the new entries.

    Args:
        v2_json_paths: Paths to the v2 annotation JSONs.
        ds_path: Path to the output rslearn dataset (config.json must exist).
        workers: Number of parallel workers (0 = sequential).
    """
    if "OEDATASETS_API_URL" not in os.environ:
        raise RuntimeError("OEDATASETS_API_URL env var must be set")

    entries = []
    for v2_json_path in v2_json_paths:
        with open(v2_json_path) as f:
            entries.extend(json.load(f))

    ds_upath = UPath(ds_path)

    # Load existing sidecar so we can merge new entries into it.
    sidecar_path = ds_upath / ANNOTATIONS_SIDECAR_FNAME
    if sidecar_path.exists():
        with sidecar_path.open("r") as f:
            annotations_sidecar: dict[str, dict[str, Any]] = json.load(f)
    else:
        annotations_sidecar = {}

    # Filter to entries that need processing.
    pending: list[dict[str, Any]] = []
    skipped_incomplete = 0
    skipped_existing = 0
    skipped_duplicate_input = 0
    seen_window_keys: set[tuple[str, str]] = set()

    for entry in entries:
        if not _entry_has_complete_annotations(entry):
            skipped_incomplete += 1
            continue
        window_name = entry["window_name"]
        window_group = entry["group"]
        window_key = (window_group, window_name)
        if window_key in seen_window_keys:
            skipped_duplicate_input += 1
            continue
        seen_window_keys.add(window_key)
        window_root = Window.get_window_root(ds_upath, window_group, window_name)
        if (window_root / "metadata.json").exists():
            skipped_existing += 1
            continue
        pending.append(entry)

    print(
        f"{len(pending)} to process, "
        f"{skipped_incomplete} incomplete, {skipped_existing} already exist, "
        f"{skipped_duplicate_input} duplicate inputs"
    )

    kwargs_list = [dict(entry=entry, ds_path=ds_path) for entry in pending]

    created = 0
    with make_pool_and_star_imap_unordered(
        workers, _process_entry, kwargs_list
    ) as outputs:
        for sidecar_key, sidecar_value in outputs:
            annotations_sidecar[sidecar_key] = sidecar_value
            created += 1
            if created % 10 == 0:
                print(f"  Processed {created}/{len(pending)} windows...")

    # Write annotation sidecar
    with sidecar_path.open("w") as f:
        json.dump(annotations_sidecar, f)

    print(
        f"Created {created} windows, "
        f"skipped {skipped_incomplete} incomplete + {skipped_existing} existing "
        f"+ {skipped_duplicate_input} duplicate inputs"
    )
    print(f"Wrote annotation sidecar to {sidecar_path}")


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
    args = parser.parse_args()

    prepare(
        v2_json_paths=args.v2_json_paths,
        ds_path=args.ds_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
