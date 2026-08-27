"""Upload land cover change events to OlmoEarth Studio.

Creates one task per GeoJSON feature from create_land_cover_change_geojson.py.
The task geometry is a 256x256 box (at 10m/pixel = 2560m) centered at the
polygon centroid. The original polygon is uploaded as an annotation on the task.

For each feature, looks up the source window in the dataset, reads the change
GeoTIFF, colorizes the early/late dominant-class maps with WorldCover colors,
and uploads them as images timestamped 2016-01-01 and 2025-01-01.

Task names encode the index, centroid, class name, and direction.
Safe to re-run: existing tasks with matching names are deleted before
re-creation.

Requires STUDIO_API_KEY environment variable.
"""

import argparse
import io
import json
import math
import os
import time
from typing import Any

import numpy as np
import rasterio
import requests
from rslearn.utils.fsspec import open_rasterio_upath_reader
from shapely.geometry import box as shapely_box
from shapely.geometry import shape
from upath import UPath

DEFAULT_API_URL = "https://olmoearth.allenai.org/api/v1"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
BOX_SIZE_M = 2560  # 256 pixels * 10m/pixel

# Model class index -> WorldCover RGB color.
# Classes without a direct WorldCover equivalent get a reasonable stand-in.
WORLDCOVER_COLORS = np.array(
    [
        (0, 0, 0),  # 0  nodata
        (180, 180, 180),  # 1  bare              (WC 60)
        (139, 69, 19),  # 2  burnt             (no WC equivalent)
        (240, 150, 255),  # 3  crops             (WC 40)
        (200, 120, 200),  # 4  fallow/shifting   (no WC equivalent)
        (255, 255, 76),  # 5  grassland         (WC 30)
        (250, 230, 160),  # 6  Lichen and moss   (WC 100)
        (255, 187, 34),  # 7  shrub             (WC 20)
        (240, 240, 240),  # 8  snow and ice      (WC 70)
        (0, 100, 0),  # 9  tree              (WC 10)
        (250, 0, 0),  # 10 urban/built-up    (WC 50)
        (0, 100, 200),  # 11 water             (WC 80)
        (0, 150, 160),  # 12 wetland           (WC 90)
    ],
    dtype=np.uint8,
)

EARLY_CLASS_BAND = 1
LATE_CLASS_BAND = 2


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
    attributes: dict,
) -> str:
    resp = _api_request(
        "POST",
        f"{api_url}/tasks",
        json={
            "name": name,
            "project_id": project_id,
            "geom": geom_wkt,
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


def _upload_image(
    api_url: str,
    task_id: str,
    tiff_bytes: bytes,
    start_time: str,
    end_time: str,
    source: str,
) -> None:
    resp = _api_request(
        "POST",
        f"{api_url}/images/upload",
        data={
            "start_time": start_time,
            "end_time": end_time,
            "source": source,
            "attributes": json.dumps({}),
            "task_id": str(task_id),
        },
        files={"image_file": ("image.tif", io.BytesIO(tiff_bytes), "image/tiff")},
        timeout=120,
    )
    resp.raise_for_status()


def _centroid_box_wkt(lon: float, lat: float) -> str:
    """Create a BOX_SIZE_M x BOX_SIZE_M WKT box centered at (lon, lat)."""
    half_m = BOX_SIZE_M / 2
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    half_lat = half_m / meters_per_deg_lat
    half_lon = half_m / meters_per_deg_lon
    return shapely_box(
        lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat
    ).wkt


def build_task_name(
    index: int, lon: float, lat: float, class_name: str, direction: str
) -> str:
    """Return a deterministic task name combining index, centroid, and class."""
    return f"#{str(index + 1).zfill(4)} ({lon:.4f}, {lat:.4f}) {direction}:{class_name}"


def main() -> None:
    """CLI entrypoint: upload land cover change features as Studio tasks."""
    parser = argparse.ArgumentParser(
        description="Upload land cover change events to Studio"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input GeoJSON from create_land_cover_change_geojson.py",
    )
    parser.add_argument("--ds-path", required=True, help="Path to rslearn dataset")
    parser.add_argument("--project-id", required=True, help="Studio project UUID")
    parser.add_argument(
        "--change-filename",
        default="land_cover_change.tif",
        help="Change GeoTIFF filename per window",
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL, help="Studio API base URL"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        fc = json.load(f)

    features = fc["features"]
    print(f"Loaded {len(features)} land cover change features")

    ds_root = UPath(args.ds_path)

    print("Fetching existing tasks for idempotent re-run...")
    existing_tasks = get_existing_tasks(args.api_url, args.project_id)
    print(f"Found {len(existing_tasks)} existing tasks in project")

    for i, feat in enumerate(features):
        props = feat["properties"]
        geom = shape(feat["geometry"])
        centroid = geom.centroid
        class_name = props["class_name"]
        direction = props["direction"]
        window_group = props["window_group"]
        window_name = props["window_name"]

        name = build_task_name(i, centroid.x, centroid.y, class_name, direction)
        box_wkt = _centroid_box_wkt(centroid.x, centroid.y)

        if name in existing_tasks:
            _delete_task(args.api_url, existing_tasks[name])

        task_id = _create_task(
            api_url=args.api_url,
            project_id=args.project_id,
            name=name,
            geom_wkt=box_wkt,
            attributes={
                "class_id": props["class_id"],
                "class_name": class_name,
                "direction": direction,
                "window_group": window_group,
                "window_name": window_name,
                "num_pixels": props["num_pixels"],
            },
        )

        _create_annotation(args.api_url, task_id, geom.wkt)

        # Upload colorized early/late land cover images.
        tiff_path = (
            ds_root / "windows" / window_group / window_name / args.change_filename
        )
        with open_rasterio_upath_reader(tiff_path) as src:
            bands = src.read()
            crs = src.crs
            transform = src.transform

        for band_idx, label, ts_start, ts_end in [
            (
                EARLY_CLASS_BAND,
                "land_cover_early",
                "2016-01-01T00:00:00Z",
                "2016-12-31T23:59:59Z",
            ),
            (
                LATE_CLASS_BAND,
                "land_cover_late",
                "2025-01-01T00:00:00Z",
                "2025-12-31T23:59:59Z",
            ),
        ]:
            rgb = WORLDCOVER_COLORS[bands[band_idx]].transpose(2, 0, 1)
            buf = io.BytesIO()
            with rasterio.open(
                buf,
                "w",
                driver="GTiff",
                compress="deflate",
                width=rgb.shape[2],
                height=rgb.shape[1],
                count=3,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(rgb)

            _upload_image(
                args.api_url, task_id, buf.getvalue(), ts_start, ts_end, label
            )

        if (i + 1) % 50 == 0:
            print(f"  Uploaded {i + 1}/{len(features)} tasks")

    print(f"Done. Uploaded {len(features)} tasks.")


if __name__ == "__main__":
    main()
