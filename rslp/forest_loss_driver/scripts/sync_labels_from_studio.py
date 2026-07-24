"""Take labels from ES Studio and put them back into a local rslearn dataset."""

import argparse
import json
import os
import shutil
from typing import Any

import requests
from upath import UPath

BASE_URL = "https://olmoearth.allenai.org/api/v1/"
LABEL_MAP = {
    "Airstrips": "airstrip",
    "Agriculture-Large": "agriculture",
    "Agriculture-Medium": "agriculture",
    "Agriculture-Small": "agriculture",
    "Burned_areas_Fire": "burned",
    "Dry_areas_seasonal": "none",
    "False_positives_not_forest_loss_an_alert_error": "none",
    "Flooded_rivers": "river",
    "Landslides": "landslide",
    "Logging_clear-cut": "logging",
    "Logging_roads": "road",
    "Mining": "mining",
    "Pasture_praderas_ganaderia": "agriculture",
    "Roads": "road",
    "Selective_Logging": "logging",
    "Windthrowsblowdowns_Hurricane_winds": "hurricane",
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
        response = requests.post(
            BASE_URL + "tasks/search",
            json={
                "project_id": {"eq": project_id},
                "offset": cur_offset,
            },
            headers=get_headers(),
            timeout=10,
        )
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"got bad API response {response.status_code}")

        json_data = response.json()
        if len(json_data["records"]) == 0:
            break

        tasks.extend(json_data["records"])
        cur_offset += len(json_data["records"])

    return tasks


def get_annotations(project_id: str) -> list[dict[str, Any]]:
    """Get all annotations for the specified project as GeoJSON."""
    cur_offset = 0
    annotations: list[dict[str, Any]] = []
    while True:
        response = requests.post(
            BASE_URL + "annotations/search",
            json={
                "project_id": {"eq": project_id},
                "offset": cur_offset,
            },
            headers=get_headers(),
            timeout=10,
        )
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"got bad API response {response.status_code}")

        json_data = response.json()
        if len(json_data["records"]) == 0:
            break

        annotations.extend(json_data["records"])
        cur_offset += len(json_data["records"])

    return annotations


def get_label_from_annotation(annotation: dict[str, Any]) -> str | None:
    """Get the labeled category for this annotation (if any)."""
    for metadata_value in annotation["metadata_values"]:
        if metadata_value["name"] != "tag_name":
            continue
        return metadata_value["label_name"]
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
    parser.add_argument(
        "--remap_labels",
        action="store_true",
        help="Whether to remap labels to the Peru label hierarchy",
    )
    args = parser.parse_args()
    ds_path = UPath(args.ds_path)

    tasks = get_tasks(args.project_id)
    annotations = get_annotations(args.project_id)

    task_by_id = {task["id"]: task for task in tasks}

    for annotation in annotations:
        task = task_by_id[annotation["task_id"]]
        group = task["attributes"]["group"]
        window_name = task["attributes"]["window"]

        label = get_label_from_annotation(annotation)

        if args.remap_labels:
            if label not in LABEL_MAP:
                print(f"skipping unknown label {label}")
                continue
            label = LABEL_MAP[label]

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
        label_fc["features"][0]["properties"]["new_label"] = label
        print(f"setting label at {geojson_fname} to {label}")
        with geojson_fname.open("w") as f:
            json.dump(label_fc, f)
