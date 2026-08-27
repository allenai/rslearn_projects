"""Prepare an LCC ablation dataset with FIXED frequent-image options.

This is a variant of ``prepare.py`` used for temporal-input ablations. It builds
the same rslearn dataset (quarterly + frequent + label layers, same sidecar and
idempotency), but replaces the randomized 8-option frequent scheme with a fixed,
deterministic schedule anchored to the first-observable date.

For each window, eight frequent options are created where the start of option ``i``
is::

    first_noticeable + i * 30 days   (i = 0..7)

So option 0 begins its 60-day frequent block exactly at the first-observable date
(unlike ``prepare.py``, where first-observable starts the *last* 15-day subperiod),
option 1 begins one month later, and so on up to seven months later. Every option
is still four 15-day periods (60-day block) and is capped so the block ends on or
before ``IMAGE_CUTOFF``.

Later, modified model configs can select among these fixed options (e.g. fewer
quarterly images or a bitemporal pair) to compare temporal-input decisions.

Scene metadata is fetched from the OlmoEarth Datasets API. Required env vars:
- OEDATASETS_API_URL: e.g. https://datasets.olmoearth.allenai.org
- DATASETS_API_TOKEN: bearer token for API auth

Idempotent: existing windows are skipped. After running this script, use
``rslearn dataset materialize`` to download imagery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from typing import Any

import requests
import shapely
import shapely.geometry
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import retry
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from upath import UPath

from rslp.change_finder_v2.lcc_model.prepare import (
    ANNOTATIONS_SIDECAR_FNAME,
    FREQUENT_BLOCK_DURATION,
    IMAGE_CUTOFF,
    NUM_FREQUENT_OPTIONS,
    WINDOW_SIZE,
    _build_frequent_layer_data,
    _build_quarterly_layer_data,
    _entry_has_complete_annotations,
    _get_window_wgs84_bounds,
    _lonlat_to_pixel,
    _parse_date,
    _rasterize_labels,
    _search_oedatasets,
    _validate_positive_point_dates,
)

# Spacing between consecutive fixed frequent options, in days. Each option shifts
# the 60-day frequent block one "month" later than the previous one.
FREQUENT_OPTION_STEP = timedelta(days=30)


def _compute_frequent_block_starts(first_noticeable: datetime) -> list[datetime]:
    """Compute fixed 60-day frequent-block starts for the ablation.

    Option ``i`` begins ``first_noticeable + i * 30 days`` (i = 0..7), so the
    first option's block starts exactly at the first-observable date. Every start
    is clamped so its 60-day block ends on/before ``IMAGE_CUTOFF`` (this may
    collapse late options to duplicate starts, which is acceptable).
    """
    max_block_start = IMAGE_CUTOFF - FREQUENT_BLOCK_DURATION
    return [
        min(first_noticeable + i * FREQUENT_OPTION_STEP, max_block_start)
        for i in range(NUM_FREQUENT_OPTIONS)
    ]


def _process_entry(
    entry: dict[str, Any],
    ds_path: str,
) -> tuple[str, dict[str, Any]]:
    """Process one annotation entry: create window, query API, write labels.

    Each call is independent (creates its own Dataset/session) so it can run in a
    separate multiprocessing worker.

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
        first_noticeable = midpoint
    else:
        center_point = ref_point
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

    block_starts = _compute_frequent_block_starts(first_noticeable)

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

    _rasterize_labels(window, dataset.layers, entry, projection, bounds)

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
    """Prepare the LCC ablation dataset from v2 annotation JSONs.

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
        _validate_positive_point_dates(entry)
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
    """Prepare LCC ablation dataset from v2 annotation JSONs."""
    parser = argparse.ArgumentParser(
        description="Prepare LCC ablation dataset (fixed frequent options) "
        "from v2 annotation JSONs."
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
