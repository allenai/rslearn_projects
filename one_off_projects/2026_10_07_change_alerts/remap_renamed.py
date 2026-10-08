"""Remap renamed Sentinel-2 items in image layers that failed to materialize.

The olmoearth_datasets Sentinel-2 backfill renames scenes, e.g.
  S2B_MSIL2A_20190813T103029_R108_T30PVV
  -> S2B_MSIL2A_20190813T103029_R108_T30PVV_S20190813T104548
so items prepared before a rename fail to materialize with
"Expected 1 item for X, got 0 from OlmoEarth API", and the whole layer is left
incomplete for that window.

This is a cheaper alternative to fix_incomplete.py (which re-prepares the layers):
for each window with incomplete image layers, it searches the API once over the
window bounds and the time span of the items in those layers, then for every item:
- keeps it if the name still exists,
- otherwise replaces it with the item(s) named "<old name>_..." (several when the
  scene is split into datastrips; they are kept in the same mosaic group, least
  cloudy first),
- otherwise leaves it and counts it as unresolved.
The item groups (so the scenes chosen per period) are unchanged. items.json is
rewritten atomically and only if a name changed, so this is safe to run while
materialize jobs are running (rslearn only marks a layer completed after all its
item groups succeed).

Afterwards, run `rslearn dataset materialize` again (completed layers are skipped).

Usage:
    python remap_renamed.py --ds_path PATH [--groups GROUP,...] [--workers N]
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

import requests
import shapely
import tqdm
from fix_incomplete import get_incomplete_layers
from rslearn.dataset import Dataset
from rslearn.dataset.manage import retry
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from upath import UPath

COLLECTION = "sentinel-2-l2a"
SEARCH_LIMIT = 1000
# Renamed items can have a slightly different collected_at.
TIME_PAD = timedelta(days=2)
# Maximum time span of one search request.
SEARCH_CHUNK = timedelta(days=365)


def search_items(
    session: requests.Session, geojson: dict[str, Any], t0: datetime, t1: datetime
) -> dict[str, dict[str, Any]]:
    """Search the OlmoEarth Datasets API, returning item ID -> item."""
    url = os.environ["OEDATASETS_API_URL"].rstrip("/") + "/api/v1/items/search"
    headers = {"Authorization": f"Bearer {os.environ['DATASETS_API_TOKEN']}"}
    items: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        body = {
            "collection": {"eq": COLLECTION},
            "intersects_geometry": geojson,
            "collected_at": {"gte": t0.isoformat(), "lt": t1.isoformat()},
            "limit": SEARCH_LIMIT,
            "offset": offset,
            "sort_by": "collected_at",
            "sort_direction": "asc",
        }
        resp = session.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        records = resp.json()["records"]
        for record in records:
            items[record["id"]] = record
        if len(records) < SEARCH_LIMIT:
            return items
        offset += SEARCH_LIMIT


def to_serialized(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an API record to the serialized item format in items.json."""
    props = record["properties"]
    return {
        "name": record["id"],
        "geometry": {
            "projection": WGS84_PROJECTION.serialize(),
            "shp": shapely.geometry.shape(props["geometry"]).wkt,
            "time_range": [props["collected_at"], props["collected_at"]],
        },
    }


def remap_window(args: tuple[str, str, str]) -> tuple[str, int, int, list[str]]:
    """Remap the renamed items of the incomplete layers of one window.

    Returns:
        (status, #incomplete layers, #items remapped, unresolved names), where status
        is complete, nochange, remapped or error.
    """
    ds_path, group, name = args
    try:
        window = Dataset(UPath(ds_path)).load_windows(groups=[group], names=[name])[0]
        layers = get_incomplete_layers(window)
        if not layers:
            return "complete", 0, 0, []

        items_fname = window.window_root / "items.json"
        with items_fname.open() as f:
            layer_datas = json.load(f)
        times = [
            datetime.fromisoformat(t)
            for ld in layer_datas
            if ld["layer_name"] in layers
            for item_group in ld["serialized_item_groups"]
            for item in item_group
            for t in item["geometry"]["time_range"]
        ]
        geom = STGeometry(window.projection, shapely.box(*window.bounds), None)
        geojson = shapely.geometry.mapping(geom.to_projection(WGS84_PROJECTION).shp)

        session = requests.Session()
        api: dict[str, dict[str, Any]] = {}
        start, end = min(times) - TIME_PAD, max(times) + TIME_PAD
        while start < end:
            chunk_end = min(start + SEARCH_CHUNK, end)
            api.update(
                retry(
                    lambda s=start, e=chunk_end: search_items(session, geojson, s, e),
                    retry_max_attempts=3,
                    retry_backoff=timedelta(seconds=30),
                )
            )
            start = chunk_end
        by_prefix: dict[str, list[str]] = defaultdict(list)
        for item_id in api:
            by_prefix[item_id.rsplit("_", 1)[0]].append(item_id)

        num_remapped = 0
        unresolved: list[str] = []
        for ld in layer_datas:
            if ld["layer_name"] not in layers:
                continue
            new_groups = []
            for item_group in ld["serialized_item_groups"]:
                new_group = []
                for item in item_group:
                    candidates = by_prefix.get(item["name"])
                    if item["name"] in api:
                        new_group.append(item)
                    elif candidates:
                        candidates.sort(
                            key=lambda c: api[c]["properties"].get("cloud_cover", 100)
                        )
                        new_group.extend(to_serialized(api[c]) for c in candidates)
                        num_remapped += 1
                    else:
                        new_group.append(item)
                        unresolved.append(item["name"])
                new_groups.append(new_group)
            ld["serialized_item_groups"] = new_groups

        if num_remapped:
            tmp_fname = window.window_root / "items.json.tmp"
            with tmp_fname.open("w") as f:
                json.dump(layer_datas, f)
            os.replace(tmp_fname.path, items_fname.path)
        status = "remapped" if num_remapped else "nochange"
        return status, len(layers), num_remapped, unresolved
    except Exception as e:  # noqa: BLE001 (reported per window)
        return "error", 0, 0, [repr(e)[:300]]


def main() -> None:
    """Remap renamed items in the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated window groups to process (default all)",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    groups = args.groups.split(",") if args.groups else None
    windows = dataset.load_windows(groups=groups, workers=args.workers)
    jobs = [(args.ds_path, window.group, window.name) for window in windows]
    statuses: Counter = Counter()
    num_remapped = 0
    unresolved: list[str] = []
    errors: list[str] = []
    with multiprocessing.Pool(args.workers) as pool:
        for status, _, remapped, names in tqdm.tqdm(
            pool.imap_unordered(remap_window, jobs), total=len(jobs), desc="Remapping"
        ):
            statuses[status] += 1
            num_remapped += remapped
            if status == "error":
                errors.extend(names)
            else:
                unresolved.extend(names)

    for error in errors[:5]:
        print(f"error: {error}")
    print(f"unresolved examples: {unresolved[:5]}")
    print(
        f"{statuses['complete']}/{len(windows)} windows complete; "
        f"{dict(statuses)}; remapped {num_remapped} items, "
        f"{len(unresolved)} unresolved, {len(errors)} errors"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
