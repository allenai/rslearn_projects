"""Take labels from ES Studio and put them back into a local rslearn dataset."""

import argparse
import json
import os
import shutil
from typing import Any

import requests
from upath import UPath

BASE_URL = "https://earth-system-studio.allen.ai/api/v1/"
LABEL_MAP = {
    "Agriculture-Large": "agriculture",
    "Agriculture-Medium": "agriculture",
    "Agriculture-Small": "agriculture",
    "Burned areas (Fire)": "burned",
    "Dry areas (seasonal)": "none",
    "False positives (not forest loss, an alert error)": "none",
    "Flooded (rivers)": "river",
    "Landslides": "landslide",
    "Logging roads": "road",
    "Mining": "mining",
    "Pasture (praderas, ganadería)": "agriculture",
    "Roads": "road",
    "Selective Logging": "logging",
    "Windthrows/blowdowns (Hurricane winds)": "hurricane",
}


def get_headers() -> dict[str, str]:
    """Get the headers to use for HTTP requests."""
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def get_tasks(project_id: str) -> list[dict[str, Any]]:
    """Get tasks all tasks in a project, handling pagination."""
    cur_offset = 0
    tasks: list[dict[str, Any]] = []
    while True:
        response = requests.get(
            BASE_URL + f"projects/{project_id}/tasks?offset={cur_offset}",
            headers=get_headers(),
            timeout=10,
        )
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"got bad API response {response.status_code}")

        json_data = response.json()
        tasks.extend(json_data["items"])

        meta = json_data["meta"]
        if cur_offset != meta["offset"]:
            raise Exception(
                f"requested offset {cur_offset} but got offset {meta['offset']}"
            )
        cur_count = len(json_data["items"])
        if meta["total"] <= cur_offset + cur_count:
            break
        cur_offset += cur_count
    return tasks


def get_annotations(project_id: str) -> dict[str, Any]:
    """Get all annotations for the specified project as GeoJSON."""
    response = requests.get(
        BASE_URL + f"projects/{project_id}/annotations",
        headers=get_headers(),
        timeout=10,
    )
    if response.status_code != 200:
        print(response.text)
        raise Exception(f"got bad API response {response.status_code}")

    return response.json()


def get_label_from_feat(feat: dict[str, Any]) -> str | None:
    """Get the labeled category for this GeoJSON feature (if any)."""
    for metadata_value in feat["properties"]["metadata_values"]:
        if metadata_value["name"] != "tag_name":
            continue
        return metadata_value["tag_name"]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="ES Studio project ID",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="rslearn dataset path",
    )
    args = parser.parse_args()
    ds_path = UPath(args.ds_path)

    tasks = get_tasks(args.project_id)
    project_fc = get_annotations(args.project_id)

    task_by_id = {task["id"]: task for task in tasks}

    for feat in project_fc["features"]:
        properties = feat["properties"]
        task = task_by_id[properties["task_id"]]
        group = task["attributes"]["group"]
        window_name = task["attributes"]["window"]

        label = get_label_from_feat(feat)
        if label not in LABEL_MAP:
            continue

        geojson_fname = (
            ds_path
            / "windows"
            / group
            / window_name
            / "layers"
            / "label"
            / "data.geojson"
        )
        backup_fname = geojson_fname.parent / "data.geojson.bak"

        with geojson_fname.open("rb") as src:
            with backup_fname.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        with geojson_fname.open("r") as f:
            label_fc = json.load(f)
        if len(label_fc["features"]) != 1:
            raise ValueError(
                f"expected geojson file at {geojson_fname} to contain exactly one feature"
            )
        label_fc["features"][0]["properties"]["new_label"] = LABEL_MAP[label]
        print(f"setting label at {geojson_fname} to {LABEL_MAP[label]}")
        with geojson_fname.open("w") as f:
            json.dump(label_fc, f)
