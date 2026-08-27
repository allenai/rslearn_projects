"""Upload forest loss disagreement events to OlmoEarth Studio.

Creates one task per disagreeing event. The task geometry is a 256x256 box
(at 10m/pixel) centered at the polygon centroid. The original polygon is
uploaded as an annotation on the task.

Task names encode the index, centroid, date, and old/new model predictions.
Safe to re-run: existing tasks with matching names are deleted before
re-creation.

Requires STUDIO_API_KEY environment variable.
"""

import argparse
import json
import math
import os
import time
from typing import Any

import requests
from shapely.geometry import box as shapely_box
from shapely.geometry import shape

DEFAULT_API_URL = "https://olmoearth.allenai.org/api/v1"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
BOX_SIZE_M = 2560


def _get_headers() -> dict[str, str]:
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _api_request(
    method: str,
    url: str,
    retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> requests.Response:
    kwargs.setdefault("headers", _get_headers())
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    for attempt in range(retries):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code < 500:
            return resp
        if attempt < retries - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt))
    return resp


def get_existing_tasks(api_url: str, project_id: str) -> dict[str, str]:
    """Fetch all tasks in the project. Returns {task_name: task_id}."""
    result: dict[str, str] = {}
    offset = 0
    while True:
        resp = _api_request(
            "POST",
            f"{api_url}/tasks/search",
            json={"project_id": {"eq": project_id}, "offset": offset, "limit": 1000},
        )
        resp.raise_for_status()
        records = resp.json()["records"]
        if not records:
            break
        for task in records:
            result[task["name"]] = task["id"]
        offset += len(records)
    return result


def _delete_task(api_url: str, task_id: str) -> None:
    resp = _api_request("DELETE", f"{api_url}/tasks/{task_id}")
    resp.raise_for_status()


def _create_task(
    api_url: str,
    project_id: str,
    name: str,
    geom_wkt: str,
    start_time: str,
    end_time: str,
    attributes: dict,
) -> str:
    resp = _api_request(
        "POST",
        f"{api_url}/tasks",
        json={
            "name": name,
            "project_id": project_id,
            "geom": geom_wkt,
            "start_time": start_time,
            "end_time": end_time,
            "attributes": attributes,
        },
    )
    resp.raise_for_status()
    return resp.json()["records"][0]["id"]


def _create_annotation(api_url: str, task_id: str, geom_wkt: str) -> str:
    resp = _api_request(
        "POST",
        f"{api_url}/annotations",
        json={
            "status": "pending",
            "geom": geom_wkt,
            "task_id": task_id,
        },
    )
    resp.raise_for_status()
    return resp.json()["records"][0]["id"]


def _centroid_box_wkt(lon: float, lat: float) -> str:
    """Create a BOX_SIZE_M x BOX_SIZE_M WKT box centered at (lon, lat)."""
    half_m = BOX_SIZE_M / 2
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    half_lat = half_m / meters_per_deg_lat
    half_lon = half_m / meters_per_deg_lon
    return shapely_box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat).wkt


def build_task_name(
    index: int, pad_width: int, lon: float, lat: float, date: str, old_label: str, new_label: str
) -> str:
    return f"#{str(index + 1).zfill(pad_width)} ({lon:.4f}, {lat:.4f}) {date} old:{old_label} new:{new_label}"


def main():
    parser = argparse.ArgumentParser(description="Upload disagreements to Studio")
    parser.add_argument(
        "--input", default="/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/2026_04_13_forest_loss_disagreement/disagreements.geojson", help="Input GeoJSON"
    )
    parser.add_argument("--project-id", required=True, help="Studio project UUID")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Studio API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Print without uploading")
    args = parser.parse_args()

    with open(args.input) as f:
        fc = json.load(f)

    features = fc["features"]
    print(f"Loaded {len(features)} disagreement features")

    if not args.dry_run:
        print("Fetching existing tasks for idempotent re-run...")
        existing_tasks = get_existing_tasks(args.api_url, args.project_id)
        print(f"Found {len(existing_tasks)} existing tasks in project")
    else:
        existing_tasks = {}

    pad_width = max(len(str(len(features))), 3)

    for i, feat in enumerate(features):
        props = feat["properties"]
        geom = shape(feat["geometry"])
        centroid = geom.centroid
        date = props["oe_start_time"][:10]
        old_label = props["old_model_prediction"]
        new_label = props["new_model_prediction"]

        name = build_task_name(i, pad_width, centroid.x, centroid.y, date, old_label, new_label)

        box_wkt = _centroid_box_wkt(centroid.x, centroid.y)

        if args.dry_run:
            print(f"  [DRY RUN] {name}")
            continue

        if name in existing_tasks:
            _delete_task(args.api_url, existing_tasks[name])

        task_id = _create_task(
            api_url=args.api_url,
            project_id=args.project_id,
            name=name,
            geom_wkt=box_wkt,
            start_time=props["oe_start_time"],
            end_time=props["oe_end_time"],
            attributes={
                "old_model_prediction": old_label,
                "new_model_prediction": new_label,
            },
        )

        _create_annotation(args.api_url, task_id, geom.wkt)

        if (i + 1) % 50 == 0:
            print(f"  Uploaded {i + 1}/{len(features)} tasks")

    print(f"Done. {'Would upload' if args.dry_run else 'Uploaded'} {len(features)} tasks.")


if __name__ == "__main__":
    main()
