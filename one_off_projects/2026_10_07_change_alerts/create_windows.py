"""Create change alert windows from the annotations in an OlmoEarth Studio project.

Each annotation (a point, or for Forest Loss Driver a polygon) becomes one window
centered on it, with time range (change date, change date). Two label layers are
written:

- label_category: uint8 category ID (1 + index in SourceDataset.categories, 0 =
  nodata) at the annotated pixel(s).
- label_change_day: uint16 days since 1970-01-01 of the change date at the annotated
  pixel(s) for positive annotations, 0 elsewhere and for negatives.

Annotations are skipped if their category is not used or their change date is outside
[MIN_CHANGE_DATE, MAX_CHANGE_DATE] (see select_annotations).

Windows are split into train/val/test by hashing a ~10 km grid cell. Run this once
for the training dataset with --splits train,val and once for each test dataset with
--splits test; the dataset config.json (from make_config.py) must already exist.

Requires STUDIO_API_KEY.
"""

import argparse
import functools
import hashlib
import json
import math
import multiprocessing
from datetime import UTC, datetime
from typing import Any

import numpy as np
import rasterio.features
import shapely
import shapely.affinity
import shapely.wkt
import tqdm
from common import (
    MAX_CHANGE_DATE,
    MIN_CHANGE_DATE,
    RESOLUTION,
    SOURCE_DATASETS,
    SPLIT_CELL_SIZE,
    SPLIT_FRACTIONS,
    WINDOW_SIZE,
    dataset_path,
)
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from rslp.utils.studio import StudioClient

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@functools.cache
def get_dataset(ds_path: str) -> Dataset:
    """Load the dataset once per worker process."""
    return Dataset(UPath(ds_path))


def get_split(projection: Projection, col: int, row: int) -> str:
    """Assign a split based on the grid cell containing the pixel."""
    cell = f"{projection.crs}_{col // SPLIT_CELL_SIZE}_{row // SPLIT_CELL_SIZE}"
    value = int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16) / 16**8
    cumulative = 0.0
    for split, fraction in SPLIT_FRACTIONS.items():
        cumulative += fraction
        if value < cumulative:
            return split
    return split


def fetch_records(source: str) -> dict[str, Any]:
    """Fetch the tasks and annotations of the source's Studio project.

    Returns:
        dict with "tasks" (map from task ID to task) and "annotations" (list).
    """
    cfg = SOURCE_DATASETS[source]
    client = StudioClient()
    tasks = {task["id"]: task for task in client.get_tasks(cfg.project_id)}
    annotations = client.get_annotations(cfg.project_id)
    print(f"fetched {len(tasks)} tasks and {len(annotations)} annotations")
    return {"tasks": tasks, "annotations": annotations}


def select_annotations(source: str, records: dict[str, Any]) -> list[dict[str, Any]]:
    """Select the annotations to create windows for.

    Annotations are skipped if their task's source_file is in skip_source_files, their
    category is not one of the categories, or their change date is outside
    [MIN_CHANGE_DATE, MAX_CHANGE_DATE].

    Returns:
        list of dicts with id, category, change_date, geom_wkt, and task attributes.
    """
    cfg = SOURCE_DATASETS[source]
    tasks = records["tasks"]
    annotations = []
    num_skipped: dict[str, int] = {"source_file": 0, "category": 0, "date": 0}
    for ann in records["annotations"]:
        attributes = tasks[ann["task_id"]]["attributes"]
        if attributes.get("source_file") in cfg.skip_source_files:
            num_skipped["source_file"] += 1
            continue
        categories = [
            mv["label_name"]
            for mv in ann["metadata_values"]
            if mv["name"] == "category"
        ]
        if not categories or categories[0] not in cfg.categories:
            num_skipped["category"] += 1
            continue
        if ann["start_time"] != ann["end_time"]:
            raise ValueError(
                f"annotation {ann['id']} has a time range, expected a date"
            )
        start_time = datetime.fromisoformat(ann["start_time"].replace("Z", "+00:00"))
        if not MIN_CHANGE_DATE <= start_time.date() <= MAX_CHANGE_DATE:
            num_skipped["date"] += 1
            continue
        annotations.append(
            {
                "id": ann["id"],
                "category": categories[0],
                "change_date": start_time.date().isoformat(),
                "geom_wkt": ann["geom_wkt"],
                "attributes": attributes,
            }
        )
    counts: dict[str, int] = {}
    for ann in annotations:
        counts[ann["category"]] = counts.get(ann["category"], 0) + 1
    print(
        f"selected {len(annotations)} annotations from {source} (skipped {num_skipped}): "
        f"{counts}"
    )
    return annotations


def rasterize(
    geom: shapely.Geometry, projection: Projection, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    """Get a boolean HxW mask of the window pixels covered by the geometry."""
    shp = STGeometry(WGS84_PROJECTION, geom, None).to_projection(projection).shp
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    mask = np.zeros((height, width), dtype=bool)
    if isinstance(shp, shapely.Point):
        col = math.floor(shp.x) - bounds[0]
        row = math.floor(shp.y) - bounds[1]
        mask[row, col] = True
        return mask
    shp = shapely.affinity.translate(shp, xoff=-bounds[0], yoff=-bounds[1])
    mask = rasterio.features.rasterize(
        [(shp, 1)], out_shape=(height, width), all_touched=True, dtype=np.uint8
    ).astype(bool)
    if not mask.any():
        # Polygons smaller than a pixel: label the pixel at the representative point.
        point = shp.representative_point()
        mask[math.floor(point.y), math.floor(point.x)] = True
    return mask


def write_label(
    window: Window, dataset: Dataset, layer_name: str, array: np.ndarray
) -> None:
    """Write a single-band label raster and mark the layer completed."""
    band_set = dataset.layers[layer_name].band_sets[0]
    with window.data.open_layer_writer(layer_name) as writer:
        writer.write_raster(
            band_set.bands,
            band_set.instantiate_raster_format(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=array[None, :, :]),
        )
    window.mark_layer_completed(layer_name)


def create_window(
    ds_path: str, source: str, splits: list[str], ann: dict[str, Any]
) -> str | None:
    """Create the window for one annotation.

    Returns:
        the split, or None if the window was not created since its split is not in
        splits.
    """
    cfg = SOURCE_DATASETS[source]
    geom = shapely.wkt.loads(ann["geom_wkt"])
    center = geom.representative_point()
    projection = get_utm_ups_projection(center.x, center.y, RESOLUTION, -RESOLUTION)
    projected = STGeometry(WGS84_PROJECTION, center, None).to_projection(projection)
    col = math.floor(projected.shp.x)
    row = math.floor(projected.shp.y)
    split = get_split(projection, col, row)
    if split not in splits:
        return None

    half = WINDOW_SIZE // 2
    bounds = (col - half, row - half, col + half, row + half)
    change_time = datetime.fromisoformat(ann["change_date"]).replace(tzinfo=UTC)
    category_id = 1 + cfg.categories.index(ann["category"])
    is_positive = category_id != 1

    dataset = get_dataset(ds_path)
    window = Window(
        storage=dataset.storage,
        group=source,
        name=ann["id"],
        projection=projection,
        bounds=bounds,
        time_range=(change_time, change_time),
        options={
            "split": split,
            "category": ann["category"],
            "change_date": ann["change_date"],
            "source_file": ann["attributes"].get("source_file"),
        },
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    mask = rasterize(geom, projection, bounds)
    category = np.zeros(mask.shape, dtype=np.uint8)
    category[mask] = category_id
    write_label(window, dataset, "label_category", category)
    change_day = np.zeros(mask.shape, dtype=np.uint16)
    if is_positive:
        change_day[mask] = (change_time - UNIX_EPOCH).days
    write_label(window, dataset, "label_change_day", change_day)
    return split


def main() -> None:
    """Create the windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCE_DATASETS), required=True)
    parser.add_argument(
        "--ds_path",
        default=None,
        help="Dataset path, defaults to the standard path for --kind",
    )
    parser.add_argument(
        "--kind",
        choices=["train", "test_history", "test_recent"],
        default="train",
        help="Which dataset to create windows in; sets the default --ds_path and "
        "--splits",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits to create windows for (default train,val for "
        "the train dataset, test otherwise)",
    )
    parser.add_argument(
        "--annotations_cache",
        default=None,
        help="Optional JSON file to cache the Studio tasks and annotations in "
        "(before filtering)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use this many annotations (a deterministic subset), for testing",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    ds_path = args.ds_path or dataset_path(args.source, args.kind)
    if args.splits is not None:
        splits = args.splits.split(",")
    elif args.kind == "train":
        splits = ["train", "val"]
    else:
        splits = ["test"]

    if args.annotations_cache and UPath(args.annotations_cache).exists():
        with UPath(args.annotations_cache).open() as f:
            records = json.load(f)
    else:
        records = fetch_records(args.source)
        if args.annotations_cache:
            with UPath(args.annotations_cache).open("w") as f:
                json.dump(records, f)
    annotations = select_annotations(args.source, records)
    if args.limit is not None:
        annotations = sorted(annotations, key=lambda ann: ann["id"])[: args.limit]

    jobs = [
        {"ds_path": ds_path, "source": args.source, "splits": splits, "ann": ann}
        for ann in annotations
    ]
    counts: dict[str, int] = {}
    with multiprocessing.Pool(args.workers) as pool:
        for split in tqdm.tqdm(
            pool.imap_unordered(_create_window_star, jobs), total=len(jobs)
        ):
            if split is not None:
                counts[split] = counts.get(split, 0) + 1
    print(f"created windows in {ds_path}: {counts}")


def _create_window_star(kwargs: dict[str, Any]) -> str | None:
    return create_window(**kwargs)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
