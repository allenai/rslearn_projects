"""Create rslearn windows from MONITRS total_train.json / total_test.json.

Each JSON object (one QA sample with a multi-date image list) becomes one window.
Spatial extent matches MONITRS patches (default 512 px at 10 m ≈ 5.12 km). Image paths
in JSON use ``all_events/``; on disk they are typically under ``all_events_submit/`` —
see ``--images_subdir``.

Example:

.. code-block:: bash

    python -m rslp.monitrs.create_monitrs_windows \\
        --monitrs_root /weka/dfive-default/piperw/data/MONITRS \\
        --ds_path /weka/dfive-default/piperw/rslearn_projects/data/monitrs/dataset_v1 \\
        --max_samples 10
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from datetime import datetime, timedelta, timezone
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from upath import UPath

from rslp.utils.windows import calculate_bounds

# MONITRS README: RGB Sentinel-2, 10 m/px, ~5.12 km patches → 512 px.
DEFAULT_PIXEL_SIZE_M = 10
DEFAULT_WINDOW_SIZE_PX = 512

DEFAULT_DATASET_CONFIG: dict[str, Any] = {
    "layers": {
        "label": {
            "type": "vector",
        },
    },
}


def _parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def _resolve_image_path(
    monitrs_root: UPath, json_rel_path: str, images_subdir: str
) -> UPath:
    """Map JSON ``all_events/...`` paths to on-disk layout."""
    parts = json_rel_path.strip("/").split("/")
    if not parts:
        raise ValueError(f"empty path: {json_rel_path!r}")
    if parts[0] == "all_events":
        parts[0] = images_subdir
    return monitrs_root.joinpath(*parts)


def _ensure_config(ds_path: UPath) -> None:
    cfg_path = ds_path / "config.json"
    if cfg_path.exists():
        return
    ds_path.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(DEFAULT_DATASET_CONFIG, indent=2))


def _load_json(path: UPath) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return data


def process_datapoint(kwargs: dict[str, Any]) -> str:
    """Create one window; designed for multiprocessing."""
    dataset: Dataset = kwargs["dataset"]
    group: str = kwargs["group"]
    split: str = kwargs["split"]
    row: dict[str, Any] = kwargs["row"]
    monitrs_root: UPath = kwargs["monitrs_root"]
    images_subdir: str = kwargs["images_subdir"]
    window_size: int = kwargs["window_size"]
    pixel_size: float = kwargs["pixel_size"]
    time_buffer: timedelta = kwargs["time_buffer"]
    verify_images: bool = kwargs["verify_images"]
    skip_existing: bool = kwargs["skip_existing"]

    row_id = row["id"]
    window_name = f"{split}_{row_id}"
    window_root = Window.get_window_root(dataset.path, group, window_name)
    if skip_existing and (window_root / "metadata.json").exists():
        return window_name

    lat_lon = row.get("lat_lon") or []
    if not lat_lon:
        raise ValueError(f"{window_name}: missing lat_lon")
    lat0, lon0 = float(lat_lon[0][0]), float(lat_lon[0][1])

    timestamps = row.get("timestamp") or []
    if not timestamps:
        raise ValueError(f"{window_name}: missing timestamp")
    times = [_parse_time(str(t)) for t in timestamps]
    time_range = (min(times) - time_buffer, max(times) + time_buffer)

    src_point = shapely.Point(lon0, lat0)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(lon0, lat0)
    dst_projection = Projection(dst_crs, pixel_size, -pixel_size)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    window = Window(
        storage=dataset.storage,
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options={
            "split": split,
            "monitrs_id": row_id,
            "folder_id": row.get("folder_id"),
            "dataset": row.get("dataset"),
            "task": row.get("task"),
        },
    )
    window.save()

    video = list(row.get("video") or [])
    resolved_paths = [
        str(_resolve_image_path(monitrs_root, rel, images_subdir)) for rel in video
    ]
    if verify_images:
        for p in resolved_paths:
            if not UPath(p).exists():
                raise FileNotFoundError(f"{window_name}: missing image {p}")

    payload = {
        "monitrs_root": str(monitrs_root),
        "images_subdir": images_subdir,
        "video_json": video,
        "video_paths": resolved_paths,
        "lat_lon": lat_lon,
        "timestamp": timestamps,
        "sensor": row.get("sensor"),
        "conversations": row.get("conversations"),
        "folder_id": row.get("folder_id"),
        "dataset": row.get("dataset"),
        "task": row.get("task"),
        "id": row_id,
        "split": split,
    }
    (window_root / "info.json").write_text(json.dumps(payload, indent=2))

    return window_name


def _build_jobs(
    rows: list[dict[str, Any]],
    split: str,
    dataset: Dataset,
    group: str,
    monitrs_root: UPath,
    images_subdir: str,
    window_size: int,
    pixel_size: float,
    time_buffer_days: float,
    verify_images: bool,
    skip_existing: bool,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if max_samples is not None:
        rows = rows[:max_samples]
    tb = timedelta(days=time_buffer_days)
    return [
        dict(
            dataset=dataset,
            group=group,
            split=split,
            row=row,
            monitrs_root=monitrs_root,
            images_subdir=images_subdir,
            window_size=window_size,
            pixel_size=pixel_size,
            time_buffer=tb,
            verify_images=verify_images,
            skip_existing=skip_existing,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monitrs_root",
        type=str,
        default="/weka/dfive-default/piperw/data/MONITRS",
        help="Root directory containing JSON splits and image folders",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="",
        help="Path to total_train.json (default: MONITRS_ROOT/total_train.json)",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="",
        help="Path to total_test.json (default: MONITRS_ROOT/total_test.json)",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Output rslearn dataset directory",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="monitrs",
        help="Window group name under windows/",
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="all_events_submit",
        help="Folder under MONITRS_ROOT that replaces the ``all_events`` prefix in JSON paths",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PX,
        help="Window size in pixels (MONITRS patches are 512 at 10 m)",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_M,
        help="Pixel size in meters (positive x; y uses negative of this)",
    )
    parser.add_argument(
        "--time_buffer_days",
        type=float,
        default=0.0,
        help="Half-width extension applied to min/max scene times (± days)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Process pool size",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, only process this many rows per split (for debugging)",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only read total_train.json",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only read total_test.json",
    )
    parser.add_argument(
        "--verify_images",
        action="store_true",
        help="Require every resolved image path to exist on disk",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recreate windows even if metadata.json already exists",
    )
    args = parser.parse_args()

    if args.train_only and args.test_only:
        parser.error("cannot combine --train_only and --test_only")

    monitrs_root = UPath(args.monitrs_root)
    train_path = UPath(args.train_json) if args.train_json else monitrs_root / "total_train.json"
    test_path = UPath(args.test_json) if args.test_json else monitrs_root / "total_test.json"
    ds_path = UPath(args.ds_path)

    _ensure_config(ds_path)
    dataset = Dataset(ds_path)

    splits_rows: list[tuple[str, list[dict[str, Any]]]] = []
    if not args.test_only:
        splits_rows.append(("train", _load_json(train_path)))
    if not args.train_only:
        splits_rows.append(("test", _load_json(test_path)))

    skip_existing = not args.no_skip_existing
    jobs: list[dict[str, Any]] = []
    for split, rows in splits_rows:
        jobs.extend(
            _build_jobs(
                rows,
                split,
                dataset,
                args.group,
                monitrs_root,
                args.images_subdir,
                args.window_size,
                args.pixel_size,
                args.time_buffer_days,
                args.verify_images,
                skip_existing,
                args.max_samples,
            )
        )

    if not jobs:
        return

    if args.workers <= 1:
        for j in tqdm.tqdm(jobs):
            process_datapoint(j)
    else:
        with multiprocessing.Pool(args.workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(process_datapoint, jobs),
                total=len(jobs),
            ):
                pass


if __name__ == "__main__":
    main()
