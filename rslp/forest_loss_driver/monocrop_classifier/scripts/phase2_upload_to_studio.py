"""Sample monocrop phase-2 predictions per class and upload them to ES Studio.

Situation: for phase 2 of monocrop annotation, the monocrop classifier (trained on
the phase-1 annotations, see ``../create_dataset.py`` and the model configs under
``data/forest_loss_driver/monocrop_classifier/``) was applied to a prediction
dataset of ~5K forest loss event polygons built by ``../create_prediction_dataset.py``
from a GeoJSON of candidate events. This script uses those predictions for
stratified sampling of the next annotation round:

1. Optionally, event polygons that intersect any feature in the exclusion
   GeoJSONs given via ``--exclude-geojson`` (files or directories of
   ``*.geojson``, e.g. the per-AOI outputs under
   ``.../20260622_monocrop_setup/new_per_aoi_outputs/``) are dropped, so
   already-covered areas are not annotated again.
2. Each remaining prediction window is assigned a single predicted class: the
   majority class of the ``output`` segmentation raster within the original event
   polygon (rasterized the same way as training labels), excluding nodata (class 0).
3. Up to ``--per-class`` (default 100) windows are sampled per predicted class.
4. The combined selection is shuffled (seeded) and uploaded as tasks to a *new*
   Studio project, named following the scheme of
   ``../../scripts/monocrop_initial_setup_20260624/rename_studio_tasks.py`` but with
   the predicted class included:

       [#001] soybean (-5.7671, -77.1391) at 2022-01-08

   where ``#001`` is the 1-based counter over the shuffle, ``(lat, lon)`` is the
   event polygon centroid, and the date is the event time (``oe_start_time``).

Tasks mirror the structure of the phase-1 monocrop projects: the task geometry is
a fixed 2560 m (256 px at 10 m) box centered at the polygon centroid, with
``oe_start_time``/``oe_end_time`` as its time range and attributes recording the
source window and predicted class. The event polygon itself is added as a pending
annotation on the task with *no* label set: the ``monoculture_tag`` labelset
metadata field (``--label-field``) is left null so a null label means the task
has not been human-labeled yet (the predicted class is only recorded in the task
name and attributes). If the project template is missing that field, the
``confidence`` field (used by annotators and required by ``create_dataset.py``),
or any of their labels, they are created via
the API first. Safe to re-run: names are deterministic given the seed, and tasks
whose name already exists in the project are skipped (annotation included).

Run in an environment with rslearn, rslp, requests, and tqdm, with access to the
prediction dataset (e.g. on a weka-mounted node):

    STUDIO_API_KEY=... python -m \
        rslp.forest_loss_driver.monocrop_classifier.scripts.phase2_upload_to_studio \
        --ds-path /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier/predict_20260721/ \
        --geojson /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20260721_monocrop_phase2/phase2.geojson \
        --exclude-geojson /weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/20260622_monocrop_setup/new_per_aoi_outputs/ \
        --project-id <PROJECT_ID> \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import requests
import shapely
import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from upath import UPath

from rslp.forest_loss_driver.monocrop_classifier.create_dataset import (
    CLASS_NAMES,
    parse_datetime,
    rasterize_label,
)
from rslp.forest_loss_driver.monocrop_classifier.create_prediction_dataset import (
    PREDICT_GROUP,
    feature_window_name,
    parse_feature_polygon,
)

OUTPUT_LAYER = "output"
BASE_URL = "https://olmoearth.allenai.org/api/v1"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
# Feature properties copied into task attributes when present.
ATTRIBUTE_PROPERTY_KEYS = ("country", "new_label", "area_ha")
# Task geometry: 256 pixels * 10 m/pixel box centered at the polygon centroid,
# matching the phase-1 monocrop project tasks.
BOX_SIZE_M = 2560
DEFAULT_LABEL_FIELD = "monoculture_tag"
# Colors for labels created by this script (annotatable classes only).
LABEL_COLORS = {
    "mennonites_nonsoybean": "#8c564b",
    "mennonites_soybean": "#e377c2",
    "oil_palm": "#ff7f0e",
    "other_agriculture": "#bcbd22",
    "pastures": "#2ca02c",
    "rice": "#17becf",
    "soybean": "#1f77b4",
}
# The confidence field used by annotators and required by create_dataset.py.
CONFIDENCE_FIELD = "confidence"
CONFIDENCE_COLORS = {
    "high": "#2ca02c",
    "medium": "#ff7f0e",
    "low": "#d62728",
}


def load_features_by_window_name(
    geojson_path: str,
) -> dict[str, tuple[dict[str, Any], shapely.Geometry]]:
    """Index GeoJSON features by their deterministic prediction window name.

    Uses the same parsing and naming as create_prediction_dataset.py so each
    entry maps to the window that was created for that feature. Features that
    would have been skipped at dataset creation time are skipped here too.
    """
    with UPath(geojson_path).open() as f:
        collection = json.load(f)

    index: dict[str, tuple[dict[str, Any], shapely.Geometry]] = {}
    for feature in collection["features"]:
        properties = feature.get("properties") or {}
        geometry = parse_feature_polygon(feature.get("geometry"))
        if geometry is None or not properties.get("oe_start_time"):
            continue
        name = feature_window_name(properties, geometry)
        index[name] = (properties, geometry)
    return index


def load_exclusion_tree(paths: list[str]) -> shapely.STRtree:
    """Load exclusion feature geometries into a spatial index.

    Each path may be a GeoJSON file or a directory, in which case all
    ``*.geojson`` files directly inside it are loaded.
    """
    geojson_paths: list[UPath] = []
    for path in paths:
        upath = UPath(path)
        if upath.is_dir():
            children = sorted(p for p in upath.iterdir() if p.name.endswith(".geojson"))
            if not children:
                raise ValueError(f"no .geojson files found in directory {path}")
            geojson_paths.extend(children)
        else:
            geojson_paths.append(upath)

    geometries: list[shapely.Geometry] = []
    for upath in geojson_paths:
        with upath.open() as f:
            collection = json.load(f)
        num_loaded = 0
        for feature in collection["features"]:
            geometry = parse_feature_polygon(feature.get("geometry"))
            if geometry is None:
                continue
            geometries.append(geometry)
            num_loaded += 1
        print(f"Loaded {num_loaded} exclusion features from {upath}")
    return shapely.STRtree(geometries)


# Per-worker globals, populated by _init_worker.
_DATASET: Dataset | None = None


def _init_worker(ds_path: str) -> None:
    """Load the dataset once per worker process."""
    global _DATASET
    _DATASET = Dataset(UPath(ds_path))


def _get_predicted_class(args: tuple[str, str]) -> tuple[str, int | None, str]:
    """Compute the majority predicted class within the event polygon.

    Args:
        args: (window_name, polygon_wkt) where the polygon is in WGS84.

    Returns:
        (window_name, class_id or None, outcome) where outcome is "ok" or the
        reason the window was skipped.
    """
    window_name, polygon_wkt = args
    assert _DATASET is not None
    windows = _DATASET.load_windows(groups=[PREDICT_GROUP], names=[window_name])
    if not windows:
        return window_name, None, "missing_window"
    window = windows[0]
    if not window.is_layer_completed(OUTPUT_LAYER):
        return window_name, None, "no_output_layer"

    band_set = _DATASET.layers[OUTPUT_LAYER].band_sets[0]
    raster = window.data.read_raster(
        OUTPUT_LAYER,
        band_set.bands,
        band_set.instantiate_raster_format(),
    )
    array = raster.array
    while array.ndim > 2:
        array = array[0]

    polygon = shapely.from_wkt(polygon_wkt)
    projected = (
        STGeometry(WGS84_PROJECTION, polygon, None).to_projection(window.projection).shp
    )
    try:
        mask = rasterize_label(projected, window.bounds, 1)
    except ValueError:
        return window_name, None, "polygon_outside_window"

    values = array[mask == 1]
    values = values[values != 0]
    if values.size == 0:
        return window_name, None, "all_nodata"
    counts = Counter(int(v) for v in values)
    return window_name, counts.most_common(1)[0][0], "ok"


def compute_predicted_classes(
    ds_path: str,
    features: dict[str, tuple[dict[str, Any], shapely.Geometry]],
    window_names: list[str],
    workers: int,
) -> tuple[dict[str, int], Counter[str]]:
    """Compute the predicted class for each window, in parallel."""
    tasks = [(name, features[name][1].wkt) for name in window_names]
    predicted: dict[str, int] = {}
    outcomes: Counter[str] = Counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(ds_path,)
    ) as pool:
        for window_name, class_id, outcome in tqdm.tqdm(
            pool.map(_get_predicted_class, tasks, chunksize=8),
            total=len(tasks),
            desc="Computing predicted classes",
        ):
            outcomes[outcome] += 1
            if class_id is not None:
                predicted[window_name] = class_id
    return predicted, outcomes


def sample_and_shuffle(
    predicted: dict[str, int], per_class: int, seed: int
) -> list[str]:
    """Sample up to per_class windows per predicted class, then shuffle."""
    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for window_name in sorted(predicted):
        by_class.setdefault(predicted[window_name], []).append(window_name)

    selected: list[str] = []
    for class_id in sorted(by_class):
        names = by_class[class_id]
        if len(names) > per_class:
            names = rng.sample(names, per_class)
        selected.extend(names)
    rng.shuffle(selected)
    return selected


def make_task_name(
    counter: int, class_name: str, lat: float, lon: float, date: str
) -> str:
    """Build the Studio task name."""
    return f"[#{counter:03d}] {class_name} ({lat:.4f}, {lon:.4f}) at {date}"


def centroid_box_wkt(lon: float, lat: float) -> str:
    """Create a BOX_SIZE_M x BOX_SIZE_M WKT box centered at (lon, lat)."""
    half_m = BOX_SIZE_M / 2
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    half_lat = half_m / meters_per_deg_lat
    half_lon = half_m / meters_per_deg_lon
    return shapely.box(
        lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat
    ).wkt


def _get_headers() -> dict[str, str]:
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _api_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("headers", _get_headers())
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    for attempt in range(MAX_RETRIES):
        resp = requests.request(method, url, **kwargs)
        if resp.status_code < 500:
            return resp
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt))
    return resp


def get_existing_task_names(project_id: str) -> set[str]:
    """Fetch the names of all tasks already in the project."""
    names: set[str] = set()
    offset = 0
    while True:
        resp = _api_request(
            "POST",
            f"{BASE_URL}/tasks/search",
            json={"project_id": {"eq": project_id}, "offset": offset, "limit": 1000},
        )
        resp.raise_for_status()
        records = resp.json()["records"]
        if not records:
            break
        names.update(task["name"] for task in records)
        offset += len(records)
    return names


def create_task(
    project_id: str,
    name: str,
    geom_wkt: str,
    start_time: str,
    end_time: str | None,
    attributes: dict[str, Any],
) -> str:
    """Create one Studio task and return its ID."""
    body: dict[str, Any] = {
        "name": name,
        "project_id": project_id,
        "geom": geom_wkt,
        "start_time": start_time,
        "attributes": attributes,
    }
    if end_time is not None:
        body["end_time"] = end_time
    resp = _api_request("POST", f"{BASE_URL}/tasks", json=body)
    if resp.status_code != 200:
        print(resp.text)
    resp.raise_for_status()
    return resp.json()["records"][0]["id"]


def create_annotation(
    task_id: str,
    geom_wkt: str,
) -> None:
    """Create one pending annotation with no label (null = not human-labeled)."""
    resp = _api_request(
        "POST",
        f"{BASE_URL}/annotations",
        json={
            "status": "pending",
            "geom": geom_wkt,
            "task_id": task_id,
        },
    )
    if resp.status_code != 200:
        print(resp.text)
    resp.raise_for_status()


def get_project(project_id: str) -> dict[str, Any]:
    """Fetch the project definition, including its annotation template."""
    resp = _api_request("GET", f"{BASE_URL}/projects/{project_id}")
    resp.raise_for_status()
    records = resp.json()["records"]
    if len(records) != 1:
        raise ValueError(f"expected one project for {project_id}, got {len(records)}")
    return records[0]


def ensure_labelset_field(
    project_id: str,
    template: dict[str, Any],
    field_name: str,
    labels: dict[str, str],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Ensure a labelset metadata field with the given labels exists.

    Creates the labelset and field if the template lacks them, and creates any
    individual missing labels.

    Args:
        project_id: the Studio project ID.
        template: the project's annotation template record.
        field_name: the labelset metadata field name.
        labels: {label_name: color} for the labels the field must have.

    Returns:
        (metadata_field_id, {label_name: label_id}, template) where the template
        is re-fetched if the field had to be created.
    """
    field = next(
        (
            f
            for f in template.get("annotation_metadata_fields") or []
            if f["name"] == field_name and f["data_type"] == "labelset"
        ),
        None,
    )
    if field is None:
        # The API requires the labelset to exist before the field can reference
        # it (inline label creation is not supported by the deployed version).
        labelset = next(
            (ls for ls in template.get("labelsets") or [] if ls["name"] == field_name),
            None,
        )
        if labelset is None:
            print(f"Creating labelset {field_name!r}")
            resp = _api_request(
                "POST",
                f"{BASE_URL}/labelsets",
                json={"name": field_name, "template_id": template["id"]},
            )
            if resp.status_code != 200:
                print(resp.text)
            resp.raise_for_status()
            labelset = resp.json()["records"][0]

        print(f"Creating labelset metadata field {field_name!r}")
        resp = _api_request(
            "POST",
            f"{BASE_URL}/annotation_metadata_fields",
            json={
                "name": field_name,
                "data_type": "labelset",
                "template_id": template["id"],
                "required": True,
                "labelset_id": labelset["id"],
            },
        )
        if resp.status_code != 200:
            print(resp.text)
        resp.raise_for_status()
        # Re-fetch the template; missing labels are created below.
        template = get_project(project_id)["template"]
        field = next(
            f
            for f in template["annotation_metadata_fields"]
            if f["name"] == field_name and f["data_type"] == "labelset"
        )

    labelset_id = field["labelset_id"]
    label_ids = {
        label["name"]: label["id"]
        for label in template.get("labels") or []
        if label["labelset_id"] == labelset_id
    }
    missing = {name: color for name, color in labels.items() if name not in label_ids}
    if missing:
        # POST /labels takes a list body and creates the labels in batch.
        print(f"Creating missing labels: {sorted(missing)}")
        resp = _api_request(
            "POST",
            f"{BASE_URL}/labels",
            json=[
                {"name": name, "color": color, "labelset_id": labelset_id}
                for name, color in missing.items()
            ],
        )
        if resp.status_code != 200:
            print(resp.text)
        resp.raise_for_status()
        for record in resp.json()["records"]:
            label_ids[record["name"]] = record["id"]
    return field["id"], label_ids, template


def resolve_labels(project_id: str, field_name: str) -> None:
    """Ensure the labelset fields annotators need exist, creating them if needed.

    Ensures both the predicted-class field and the confidence field (and their
    labels) exist in the project template. No labels are assigned to the
    uploaded annotations; the fields are only prepared for human annotators.
    """
    project = get_project(project_id)
    template = project.get("template")
    if not template:
        raise ValueError(
            f"project {project_id} has no annotation template; create the project "
            "template in Studio first"
        )

    _, _, template = ensure_labelset_field(
        project_id, template, field_name, LABEL_COLORS
    )
    ensure_labelset_field(project_id, template, CONFIDENCE_FIELD, CONFIDENCE_COLORS)


def main() -> None:
    """Sample predictions per class and upload them as Studio tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-path", required=True, help="Prediction dataset path.")
    parser.add_argument(
        "--geojson",
        required=True,
        help="Event polygon GeoJSON passed to create_prediction_dataset.py.",
    )
    parser.add_argument(
        "--exclude-geojson",
        nargs="+",
        default=[],
        help="GeoJSON files (or directories of *.geojson) whose features mark "
        "areas to skip: event polygons intersecting any of them are ignored.",
    )
    parser.add_argument("--project-id", required=True, help="ES Studio project ID.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=100,
        help="Maximum number of windows to sample per predicted class (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling and the shuffle that assigns counters (default: 42).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Worker processes for reading predictions (default: 32).",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        help="Labelset metadata field for the predicted class label "
        f"(default: {DEFAULT_LABEL_FIELD}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks without calling the Studio API.",
    )
    args = parser.parse_args()

    features = load_features_by_window_name(args.geojson)
    print(f"Loaded {len(features)} usable features from {args.geojson}")

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(groups=[PREDICT_GROUP])
    print(f"Found {len(windows)} windows in group {PREDICT_GROUP}")
    matched = [w.name for w in windows if w.name in features]
    if len(matched) < len(windows):
        print(
            f"Skipping {len(windows) - len(matched)} windows with no matching feature"
        )

    if args.exclude_geojson:
        tree = load_exclusion_tree(args.exclude_geojson)
        num_before = len(matched)
        matched = [
            name
            for name in matched
            if len(tree.query(features[name][1], predicate="intersects")) == 0
        ]
        print(
            f"Excluded {num_before - len(matched)} windows intersecting exclusion "
            f"features; {len(matched)} remain"
        )

    predicted, outcomes = compute_predicted_classes(
        args.ds_path, features, matched, args.workers
    )
    print(f"Prediction outcomes: {dict(sorted(outcomes.items()))}")
    class_counts = Counter(predicted.values())
    print("Predicted class counts:")
    for class_id in sorted(class_counts):
        print(f"  {CLASS_NAMES[class_id]}: {class_counts[class_id]}")

    selected = sample_and_shuffle(predicted, args.per_class, args.seed)
    sampled_counts = Counter(predicted[name] for name in selected)
    print(f"Sampled {len(selected)} windows:")
    for class_id in sorted(sampled_counts):
        print(f"  {CLASS_NAMES[class_id]}: {sampled_counts[class_id]}")

    planned = []
    for counter, window_name in enumerate(selected, start=1):
        properties, geometry = features[window_name]
        class_id = predicted[window_name]
        centroid = geometry.centroid
        date = parse_datetime(properties["oe_start_time"]).date().isoformat()
        name = make_task_name(
            counter, CLASS_NAMES[class_id], centroid.y, centroid.x, date
        )
        attributes = {
            "window_name": window_name,
            "predicted_class_id": class_id,
            "predicted_class_name": CLASS_NAMES[class_id],
            **{
                key: properties[key]
                for key in ATTRIBUTE_PROPERTY_KEYS
                if key in properties
            },
        }
        planned.append((name, window_name, properties, geometry, attributes))

    if args.dry_run:
        for name, window_name, _, _, _ in planned:
            print(f"{name}  <-  {window_name}")
        print(f"Dry run: would create up to {len(planned)} tasks")
        return

    resolve_labels(args.project_id, args.label_field)

    existing_names = get_existing_task_names(args.project_id)
    print(f"Found {len(existing_names)} existing tasks in project")

    num_created = 0
    num_skipped = 0
    for name, window_name, properties, geometry, attributes in tqdm.tqdm(
        planned, desc="Creating tasks"
    ):
        if name in existing_names:
            num_skipped += 1
            continue
        centroid = geometry.centroid
        task_id = create_task(
            project_id=args.project_id,
            name=name,
            geom_wkt=centroid_box_wkt(centroid.x, centroid.y),
            start_time=properties["oe_start_time"],
            end_time=properties.get("oe_end_time"),
            attributes=attributes,
        )
        create_annotation(
            task_id=task_id,
            geom_wkt=geometry.wkt,
        )
        num_created += 1
    print(f"Created {num_created} tasks, skipped {num_skipped} existing")


if __name__ == "__main__":
    main()
