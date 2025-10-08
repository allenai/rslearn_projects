"""Script to add area of polygon to the tasks in ES Studio.

First make the Metadata Field under Annotation Tags for the project:
- Name: Area
- Display Name: Area (hectares)
- Data Type: number
- Allowed range: 0-9999

This script takes two arguments, Studio project ID and API key.
"""

import json
import sys

import requests
import shapely
import shapely.geometry
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

BASE_URL = "https://earth-system-studio.allen.ai/api/v1"

# Arbitrary user ID to save the annotation under.
# This one is ES Studio User.
DEFAULT_ANNOTATOR_ID = "dc379e38-d2ac-41fd-a844-85f43d3e5765"

if __name__ == "__main__":
    project_id = sys.argv[1]
    api_token = sys.argv[2]

    headers = {
        "Authorization": f"Bearer {api_token}",
    }

    # Get the annotation metadata field ID for the Area field.
    url = f"{BASE_URL}/projects/{project_id}"
    response = requests.get(url, headers=headers, timeout=10)
    assert response.status_code == 200
    project_data = response.json()
    metadata_field_id = None
    for metadata_field in project_data["template"]["annotation_metadata_fields"]:
        if metadata_field["name"] != "Area":
            continue
        metadata_field_id = metadata_field["id"]
        break
    assert metadata_field_id is not None

    # Now iterate through tasks.
    url = f"{BASE_URL}/projects/{project_id}/tasks?limit=1000"
    response = requests.get(url, headers=headers, timeout=10)
    assert response.status_code == 200
    item_list = response.json()["items"]
    for task in tqdm.tqdm(item_list):
        task_id = task["id"]
        url = f"{BASE_URL}/tasks/{task_id}/annotations"
        response = requests.get(url, headers=headers, timeout=10)
        assert response.status_code == 200
        fc = response.json()
        if len(fc["features"]) != 1:
            continue
        feat = fc["features"][0]
        shp = shapely.geometry.shape(feat["geometry"])
        properties = feat["properties"]

        # Compute area.
        # One hectare is 100 m x 100 m so we set the pixel size to that.
        wgs84_geom = STGeometry(WGS84_PROJECTION, shp, None)
        dst_proj = get_utm_ups_projection(shp.centroid.x, shp.centroid.y, 100, -100)
        dst_geom = wgs84_geom.to_projection(dst_proj)
        area = round(dst_geom.shp.area, 2)

        # Add the metadata field.
        if "metadata_values" in properties:
            metadata_values = properties["metadata_values"]
        else:
            metadata_values = []
        already_set = False
        for d in metadata_values:
            if d["metadata_field_id"] == metadata_field_id:
                already_set = True
        if already_set:
            print("already set")
            continue
        metadata_values.append(
            {
                "metadata_field_id": metadata_field_id,
                "value": area,
            }
        )

        annotation_id = properties["id"]
        post_data = {
            "id": annotation_id,
            "status": properties["status"],
            "start_time": properties["start_time"],
            "end_time": properties["end_time"],
            "geom": shp.wkt,
            "metadata_values": metadata_values,
            "task_id": properties["task_id"],
            "annotator_id": properties["annotator_id"]
            if "annotator_id" in properties
            else DEFAULT_ANNOTATOR_ID,
        }

        url = f"{BASE_URL}/annotations/{annotation_id}"
        response = requests.put(url, json.dumps(post_data), headers=headers, timeout=10)
        assert response.status_code == 200
