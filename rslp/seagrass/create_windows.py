"""Create rslearn windows for the Jamaica seagrass dataset.

The local seagrass bundle has two useful geospatial views:

1. training sample points with class labels in the CSVs under ``training_samples``.
2. full reference raster tiles described by ``reference_manifests/reference_labels.json``.

This script can create either view as rslearn windows. Sample windows include a
single-pixel ``label_raster`` at the center of each patch. Reference-tile windows
record the tile geometry and label-raster manifest metadata. Per-window metadata
(sample/tile attributes, split assignment, label-raster manifest details) is stored
in the window ``options`` so it persists in the rslearn ``metadata.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS
from rslearn.config.dataset import StorageConfig
from rslearn.dataset import Window
from rslearn.utils import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

DEFAULT_SOURCE_DIR = Path("/weka/dfive-default/piperw/scripts/seagrass")
DEFAULT_DATASET_PATH = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
LABEL_BAND = "label"
LABEL_NAMES = {
    0: "background",
    1: "sparse_seagrass",
    2: "dense_seagrass",
}
SAMPLE_COLUMNS = {
    "sample_id",
    "mode",
    "tile_id",
    "crs",
    "preferred_crs",
    "used_preferred_crs",
    "utm_zone",
    "pixel_row",
    "pixel_col",
    "x_utm",
    "y_utm",
    "lon",
    "lat",
    "valid_band_count",
    "candidate_count",
    "valid_candidate_count",
    "label_conflict",
    "candidate_mode",
    "source_shard_zone",
    "source_shard_class",
    "source_gcs_path",
}


def calculate_bounds(
    center_x: int,
    center_y: int,
    window_size: int,
) -> tuple[int, int, int, int]:
    """Calculate pixel bounds for a centered rslearn window."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if window_size % 2 == 0:
        return (
            center_x - window_size // 2,
            center_y - window_size // 2,
            center_x + window_size // 2,
            center_y + window_size // 2,
        )

    return (
        center_x - window_size // 2,
        center_y - window_size // 2 - 1,
        center_x + window_size // 2 + 1,
        center_y + window_size // 2,
    )


def assign_split(
    key: str,
    val_fraction: float,
    test_fraction: float,
    seed: str,
) -> str:
    """Assign a stable train/val/test split from a hash key."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction and test_fraction must be non-negative and sum to < 1"
        )

    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def to_jsonable(value: Any) -> Any:
    """Convert CSV strings into JSON-serializable values."""
    if value == "":
        return None
    return value


def create_sample_window(
    row: dict[str, str],
    ds_path: Any,
    group: str,
    window_size: int,
    year: int,
    val_fraction: float,
    test_fraction: float,
    split_seed: str,
    split_key_column: str,
) -> str:
    """Create one point-centered sample window with a center-pixel label."""
    sample_id = str(row["sample_id"])
    label_value = int(row["mode"])
    if label_value not in LABEL_NAMES:
        raise ValueError(
            f"Unexpected seagrass label {label_value} for sample {sample_id}"
        )

    lon = float(row["lon"])
    lat = float(row["lat"])
    crs = str(row["crs"])
    projection = Projection(CRS.from_string(crs), WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    center_x = int(float(row["x_utm"]) / WINDOW_RESOLUTION)
    center_y = int(float(row["y_utm"]) / -WINDOW_RESOLUTION)
    bounds = calculate_bounds(center_x, center_y, window_size)

    split_key = str(row[split_key_column])
    split = assign_split(split_key, val_fraction, test_fraction, split_seed)
    # The points dataset uses only train/val: hold out `val` for a clean validation
    # metric and fold the `test` bucket into `train`. The polygon windows in the
    # separate test group serve as the held-out test set. Keeping test_fraction
    # leaves the `val` bucket unchanged from prior runs.
    if split == "test":
        split = "train"
    window_name = f"sample_{sample_id}"
    time_range = (
        datetime(year, 1, 1, tzinfo=timezone.utc),
        datetime(year, 12, 31, tzinfo=timezone.utc),
    )

    # Persist all sample metadata on the window itself (rslearn metadata.json)
    # rather than a separate auxiliary info.json file.
    options = {column: to_jsonable(value) for column, value in row.items()}
    options.update(
        {
            "split": split,
            "label": label_value,
            "label_name": LABEL_NAMES[label_value],
            "latitude": lat,
            "longitude": lon,
            "split_key": split_key,
            "split_seed": split_seed,
        }
    )

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(ds_path),
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options=options,
    )
    window.save()

    raster = np.full((1, window_size, window_size), 255, dtype=np.uint8)
    raster[0, window_size // 2, window_size // 2] = label_value
    raster_dir = window.get_raster_dir(LABEL_LAYER, [LABEL_BAND])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=raster),
    )
    window.mark_layer_completed(LABEL_LAYER)
    return split


def read_sample_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read only the metadata columns needed for rslearn windows."""
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is missing a header row")

        required = {
            "sample_id",
            "mode",
            "tile_id",
            "crs",
            "x_utm",
            "y_utm",
            "lon",
            "lat",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

        return [
            {
                column: value
                for column, value in row.items()
                if column in SAMPLE_COLUMNS and value is not None
            }
            for row in reader
        ]


def iter_reference_tiles(manifest_path: Path) -> Iterable[dict[str, Any]]:
    """Yield reference-tile entries from the label manifest."""
    with manifest_path.open() as f:
        manifest = json.load(f)
    yield from manifest["tiles"]


def tile_bounds_from_grid(grid: dict[str, Any]) -> tuple[int, int, int, int]:
    """Convert a rasterio-style affine grid into rslearn pixel bounds."""
    height, width = grid["shape"]
    transform = grid["transform"]
    x_resolution = float(transform[0])
    y_resolution = float(transform[4])
    if x_resolution != WINDOW_RESOLUTION or y_resolution != -WINDOW_RESOLUTION:
        raise ValueError(
            "Expected 10 m reference tiles; got "
            f"x_resolution={x_resolution}, y_resolution={y_resolution}"
        )

    x_min = int(round(float(transform[2]) / WINDOW_RESOLUTION))
    y_min = int(round(float(transform[5]) / -WINDOW_RESOLUTION))
    return (x_min, y_min, x_min + int(width), y_min + int(height))


def create_reference_tile_window(
    tile: dict[str, Any],
    ds_path: Any,
    group: str,
    year: int,
    val_fraction: float,
    test_fraction: float,
    split_seed: str,
) -> str:
    """Create one full-tile segmentation window from reference_labels.json."""
    tile_id = str(tile["tile_id"])
    grid = tile["grid"]
    projection = Projection(
        CRS.from_string(str(grid["crs"])), WINDOW_RESOLUTION, -WINDOW_RESOLUTION
    )
    bounds = tile_bounds_from_grid(grid)
    split = assign_split(tile_id, val_fraction, test_fraction, split_seed)
    time_range = (
        datetime(year, 1, 1, tzinfo=timezone.utc),
        datetime(year, 12, 31, tzinfo=timezone.utc),
    )

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(ds_path),
        group=group,
        name=f"tile_{tile_id}",
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options={
            "split": split,
            "tile_id": tile_id,
            "split_seed": split_seed,
            "source_tile": tile.get("source_tile"),
            "label_gcs_prefix": tile.get("gcs_prefix"),
            "label_audit": tile.get("label_audit"),
            "mask_pixel_count": tile.get("mask_pixel_count"),
            "depth_audit": tile.get("depth_audit"),
            "grid": grid,
        },
    )
    window.save()
    return split


def summarize_splits(splits: Iterable[str]) -> dict[str, int]:
    """Count split assignments for a completed run."""
    summary = {"train": 0, "val": 0, "test": 0}
    for split in splits:
        summary[split] += 1
    return summary


def run_window_jobs(
    create_fn: Callable[..., str],
    jobs: list[dict[str, Any]],
    workers: int,
) -> list[str]:
    """Run window creation jobs either serially or with rslearn multiprocessing."""
    if workers == 1:
        return [create_fn(**job) for job in tqdm.tqdm(jobs)]

    pool = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(pool, create_fn, jobs)
    splits = [split for split in tqdm.tqdm(outputs, total=len(jobs))]
    pool.close()
    return splits


def create_windows(args: argparse.Namespace) -> dict[str, int]:
    """Create seagrass windows according to CLI args."""
    ds_path = UPath(args.ds_path)
    ds_path.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir)

    if args.mode == "samples":
        csv_path = (
            Path(args.csv_path)
            if args.csv_path
            else (
                source_dir
                / "training_samples"
                / "samples_olmo_strict_grid_same_locations.csv"
            )
        )
        rows = read_sample_rows(csv_path)
        if args.split_key_column not in rows[0]:
            raise ValueError(
                f"split key column {args.split_key_column!r} is not present in {csv_path}"
            )
        jobs: list[dict[str, Any]] = [
            dict(
                row=row,
                ds_path=ds_path,
                group=args.group,
                window_size=args.window_size,
                year=args.year,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                split_seed=args.split_seed,
                split_key_column=args.split_key_column,
            )
            for row in rows
        ]
        splits = run_window_jobs(create_sample_window, jobs, args.workers)
    else:
        reference_labels = (
            Path(args.reference_labels)
            if args.reference_labels
            else (source_dir / "reference_manifests" / "reference_labels.json")
        )
        tiles = list(iter_reference_tiles(reference_labels))
        jobs = [
            dict(
                tile=tile,
                ds_path=ds_path,
                group=args.group,
                year=args.year,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                split_seed=args.split_seed,
            )
            for tile in tiles
        ]
        splits = run_window_jobs(create_reference_tile_window, jobs, args.workers)

    summary = summarize_splits(splits)
    with (ds_path / f"{args.mode}_split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["samples", "reference_tiles"],
        default="samples",
        help="Create point sample windows or full reference-tile windows.",
    )
    parser.add_argument(
        "--source_dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing the seagrass bundle from scripts/seagrass.",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="Seagrass training sample CSV used when --mode=samples. Defaults under --source_dir.",
    )
    parser.add_argument(
        "--reference_labels",
        default=None,
        help="reference_labels.json used when --mode=reference_tiles. Defaults under --source_dir.",
    )
    parser.add_argument(
        "--ds_path",
        default=str(DEFAULT_DATASET_PATH),
        help="Output rslearn dataset path.",
    )
    parser.add_argument(
        "--group",
        default="jamaica_2025",
        help="rslearn window group name.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Sample window size in pixels. Ignored for reference-tile mode.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year used for window time ranges.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of split keys assigned to validation.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.1,
        help="Fraction of split keys assigned to test.",
    )
    parser.add_argument(
        "--split_seed",
        default="seagrass_jamaica_2025",
        help="Seed string mixed into deterministic split hashes.",
    )
    parser.add_argument(
        "--split_key_column",
        default="tile_id",
        help="Sample CSV column used for spatial split hashing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    split_summary = create_windows(parse_args())
    print(json.dumps(split_summary, indent=2, sort_keys=True))
