r"""Create an rslearn dataset for NISAR vessel detection from an OlmoEarth Studio project.

Each Studio task with status "reviewed" or "to_be_reviewed" becomes one rslearn
window, with bounds matching the task geometry (in an appropriate UTM projection at
10 m/pixel, matching NISAR L2 GCOV frequency A resolution) and time range matching the
task start/end times. The task's annotations become point features in the "label"
vector layer with the property "category" set to "vessel". Tasks with no annotations
still get an (empty) label layer so they serve as negative examples.

Windows are assigned to a "train" or "val" group (~90/10) based on a deterministic
hash of the window name.

Requires the STUDIO_API_KEY environment variable to be set.

Example:
    python -m rslp.nisar_vessels.scripts.create_dataset \
        --project_id c927e5cb-734b-4323-8cad-7f224b3e850d \
        --ds_path /path/to/nisar_vessels_dataset/
"""

import argparse
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from typing import Any

import requests
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://olmoearth.allenai.org/api/v1"
REQUEST_TIMEOUT = 30
SEARCH_PAGE_SIZE = 1000

# Only tasks with these statuses are converted to windows.
WANTED_TASK_STATUSES = ["reviewed", "to_be_reviewed"]

# Meters per pixel for the windows. This matches the resolution of NISAR L2 GCOV
# frequency A grids (10 or 20 m depending on bandwidth mode).
PIXEL_SIZE = 10

LABEL_LAYER = "label"
CATEGORY = "vessel"

# Buffer to add around the task time range. The Studio tasks have
# start_time == end_time set to the NISAR granule acquisition start time, but the
# data source matches granules with task_start <= collected_at < task_end, so a
# zero-duration window can never match. One minute is plenty (granules span ~35
# seconds) while remaining uniquely selective (other passes over the same location
# are hours to days away).
TIME_BUFFER = timedelta(minutes=1)

# One in VAL_MODULUS windows (by name hash) goes to the val group.
VAL_MODULUS = 10

DATASET_CONFIG_FNAME = "data/nisar_vessels/config.json"


def get_headers() -> dict[str, str]:
    """Get the headers to use for Studio API requests."""
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def search_all(endpoint: str, search_body: dict[str, Any]) -> list[dict[str, Any]]:
    """Get all records from a paginated Studio search endpoint.

    Args:
        endpoint: the search endpoint, e.g. "tasks/search".
        search_body: the search filters (offset/limit are added automatically).

    Returns:
        all matching records.
    """
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.post(
            f"{BASE_URL}/{endpoint}",
            headers=get_headers(),
            json=dict(search_body, offset=offset, limit=SEARCH_PAGE_SIZE),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            logger.error(response.text)
            raise ValueError(
                f"got bad API response {response.status_code} from {endpoint}"
            )
        cur_records = response.json()["records"]
        if len(cur_records) == 0:
            break
        records.extend(cur_records)
        offset += len(cur_records)
    return records


def get_tasks(project_id: str) -> list[dict[str, Any]]:
    """Get the tasks in the project that have a wanted status."""
    return search_all(
        "tasks/search",
        {
            "project_id": {"eq": project_id},
            "status": {"inc": WANTED_TASK_STATUSES},
        },
    )


def get_annotations_by_task(project_id: str) -> dict[str, list[dict[str, Any]]]:
    """Get non-rejected annotations in the project, grouped by task ID."""
    annotations = search_all(
        "annotations/search",
        {
            "project_id": {"eq": project_id},
        },
    )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        if annotation["status"] == "rejected":
            continue
        if annotation["task_id"] is None:
            continue
        by_task.setdefault(annotation["task_id"], []).append(annotation)
    return by_task


def get_split(window_name: str) -> str:
    """Deterministically assign the window to the train or val split."""
    h = hashlib.sha256(window_name.encode()).hexdigest()
    if int(h, 16) % VAL_MODULUS == 0:
        return "val"
    return "train"


def parse_time(s: str) -> datetime:
    """Parse an ISO-format timestamp from the Studio API."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def create_window(
    dataset: Dataset,
    task: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> str | None:
    """Create one window (with label layer) from a Studio task.

    Args:
        dataset: the output rslearn dataset.
        task: the Studio task.
        annotations: the non-rejected annotations belonging to this task.

    Returns:
        the split the window was assigned to, or None if the task was skipped.
    """
    if task["geom_wkt"] is None:
        logger.warning("skipping task %s that has no geometry", task["id"])
        return None
    if task["start_time"] is None or task["end_time"] is None:
        logger.warning("skipping task %s that has no time range", task["id"])
        return None

    task_shp = shapely.from_wkt(task["geom_wkt"])
    src_geometry = STGeometry(WGS84_PROJECTION, task_shp, None)
    centroid = task_shp.centroid
    dst_projection = get_utm_ups_projection(
        centroid.x, centroid.y, PIXEL_SIZE, -PIXEL_SIZE
    )
    dst_shp = src_geometry.to_projection(dst_projection).shp
    bounds = (
        int(dst_shp.bounds[0]),
        int(dst_shp.bounds[1]),
        int(dst_shp.bounds[2]) + 1,
        int(dst_shp.bounds[3]) + 1,
    )
    time_range = (
        parse_time(task["start_time"]) - TIME_BUFFER,
        parse_time(task["end_time"]) + TIME_BUFFER,
    )

    # Include a portion of the task ID in the window name in case task names are not
    # unique within the project.
    window_name = f"{task['name']}_{task['id'][:8]}".replace(" ", "_")
    split = get_split(window_name)

    window = Window(
        storage=dataset.storage,
        group=split,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(
            split=split,
            task_id=task["id"],
            task_name=task["name"],
            task_status=task["status"],
        ),
    )
    window.save()
    window._data = dataset.window_data_storage_factory.create(window)

    # Write the label layer with one point feature per annotation.
    features = []
    for annotation in annotations:
        annotation_shp = shapely.from_wkt(annotation["geom_wkt"])
        features.append(
            Feature(
                STGeometry(WGS84_PROJECTION, annotation_shp.centroid, None),
                {
                    "category": CATEGORY,
                    "annotation_id": annotation["id"],
                },
            )
        )
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_vector(GeojsonVectorFormat(), features)
    window.mark_layer_completed(LABEL_LAYER)

    return split


def create_dataset(project_id: str, ds_path: str) -> None:
    """Create the NISAR vessel detection dataset from the Studio project.

    Args:
        project_id: the Studio project ID.
        ds_path: the path to write the rslearn dataset to.
    """
    ds_upath = UPath(ds_path)
    ds_upath.mkdir(parents=True, exist_ok=True)

    # Copy the dataset configuration (expects to run from the rslearn_projects root).
    with open(DATASET_CONFIG_FNAME, "rb") as src:
        with (ds_upath / "config.json").open("wb") as dst:
            shutil.copyfileobj(src, dst)

    tasks = get_tasks(project_id)
    logger.info(
        "got %d tasks with status in %s", len(tasks), ",".join(WANTED_TASK_STATUSES)
    )
    annotations_by_task = get_annotations_by_task(project_id)

    dataset = Dataset(ds_upath)
    split_counts: dict[str, int] = {}
    num_labels = 0
    for task in tqdm.tqdm(tasks, desc="Creating windows"):
        annotations = annotations_by_task.get(task["id"], [])
        split = create_window(dataset, task, annotations)
        if split is None:
            continue
        split_counts[split] = split_counts.get(split, 0) + 1
        num_labels += len(annotations)

    logger.info(
        "created %d windows (%s) with %d total vessel labels",
        sum(split_counts.values()),
        ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items())),
        num_labels,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create NISAR vessel detection dataset from a Studio project",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="OlmoEarth Studio project ID",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to write the rslearn dataset",
    )
    args = parser.parse_args()
    create_dataset(args.project_id, args.ds_path)
