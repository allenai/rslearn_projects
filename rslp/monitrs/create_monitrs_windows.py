r"""Create rslearn windows from MONITRS total_train.json / total_test.json.

MONITRS is a **vision–language** benchmark: each JSON row is a multiple-choice or
free-form question about a **short Sentinel-2 image sequence** (JPEG paths in
``video``), plus ``lat_lon``, ``timestamp``, ``sensor``, ``task``, ``dataset``, and
``conversations`` (human prompt + model answer text). There are **no** segmentation
polygons or raster class masks in the source JSON.

Each row becomes one rslearn ``Window``. The full row (including ``conversations``) is
copied to ``windows/<group>/<split>_<id>/info.json`` for training or inspection.

Optional: ``--qa_vector_layer`` writes one **GeoJSON point** at the window center with
truncated QA strings in feature properties, so tooling that expects a vector layer can
read labels without parsing ``info.json``. Add a matching **vector** layer name to
``config.json`` (no ``class_names`` required).

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
from rslearn.config.dataset import LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

# MONITRS README: RGB Sentinel-2, 10 m/px, ~5.12 km patches → 512 px.
DEFAULT_PIXEL_SIZE_M = 10
DEFAULT_WINDOW_SIZE_PX = 512

# No built-in layers: MONITRS has no mask labels in JSON. Add raster/vector layers in
# config.json (e.g. Sentinel-2) before or after window creation.
DEFAULT_DATASET_CONFIG: dict[str, Any] = {"layers": {}}

# GeoJSON property strings longer than this are truncated (very long MCQ prompts).
_MAX_QA_GEOJSON_PROP_LEN = 24_000


def _first_conversation_turn(
    conversations: list[dict[str, Any]] | None, role: str
) -> str:
    """Return the first ``value`` for ``from == role``, or empty string."""
    if not conversations:
        return ""
    for turn in conversations:
        if turn.get("from") == role:
            v = turn.get("value")
            return str(v) if v is not None else ""
    return ""


def _truncate_prop(s: str, max_len: int = _MAX_QA_GEOJSON_PROP_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _qa_vector_properties(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Feature properties for optional QA GeoJSON (text, not segmentation classes)."""
    conv = row.get("conversations")
    if not isinstance(conv, list):
        conv = []
    human = _truncate_prop(_first_conversation_turn(conv, "human"))
    gpt = _truncate_prop(_first_conversation_turn(conv, "gpt"))
    return {
        "monitrs_id": row.get("id"),
        "monitrs_split": split,
        "monitrs_task": row.get("task"),
        "monitrs_dataset": row.get("dataset"),
        "monitrs_folder_id": row.get("folder_id"),
        "monitrs_question": human,
        "monitrs_answer": gpt.strip() if gpt else "",
    }


def _write_qa_vector_layer(
    window: Window,
    center_geometry: STGeometry,
    row: dict[str, Any],
    split: str,
    layer: str,
) -> None:
    """One point feature at window center; full QA also remains in ``info.json``."""
    feat = Feature(center_geometry, _qa_vector_properties(row, split))
    out_dir = window.get_layer_dir(layer)
    GeojsonVectorFormat().encode_vector(out_dir, [feat])
    window.mark_layer_completed(layer)


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


def _first_lat_lon_pair(row: dict[str, Any]) -> tuple[float, float] | None:
    """First (lat, lon) from ``lat_lon``, or None if missing or unusable."""
    lat_lon = row.get("lat_lon") or []
    if not lat_lon:
        return None
    try:
        pair = lat_lon[0]
        if pair is None or len(pair) < 2:
            return None
        lat0, lon0 = float(pair[0]), float(pair[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (lat0, lon0)


def _time_range_for_row(
    row: dict[str, Any], time_buffer: timedelta
) -> tuple[datetime, datetime] | None:
    """Window (start, end) time range, or None if timestamps missing or invalid."""
    timestamps = row.get("timestamp") or []
    if not timestamps:
        return None
    try:
        times = [_parse_time(str(t)) for t in timestamps]
    except (TypeError, ValueError):
        return None
    return (min(times) - time_buffer, max(times) + time_buffer)


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
    qa_vector_layer: str | None = kwargs.get("qa_vector_layer")

    row_id = row["id"]
    window_name = f"{split}_{row_id}"
    window_root = Window.get_window_root(dataset.path, group, window_name)
    if skip_existing and (window_root / "metadata.json").exists():
        return window_name

    latlon = _first_lat_lon_pair(row)
    if latlon is None:
        print(f"[skip {window_name}] missing or invalid lat_lon", flush=True)
        return window_name
    lat0, lon0 = latlon

    time_range = _time_range_for_row(row, time_buffer)
    if time_range is None:
        print(f"[skip {window_name}] missing or invalid timestamp", flush=True)
        return window_name

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
        "lat_lon": row.get("lat_lon") or [],
        "timestamp": row.get("timestamp") or [],
        "sensor": row.get("sensor"),
        "conversations": row.get("conversations"),
        "folder_id": row.get("folder_id"),
        "dataset": row.get("dataset"),
        "task": row.get("task"),
        "id": row_id,
        "split": split,
    }
    (window_root / "info.json").write_text(json.dumps(payload, indent=2))

    if qa_vector_layer:
        _write_qa_vector_layer(window, dst_geometry, row, split, qa_vector_layer)

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
    qa_vector_layer: str | None,
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
            qa_vector_layer=qa_vector_layer,
        )
        for row in rows
    ]


def main() -> None:
    """Create MONITRS windows for rslearn dataset."""
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
        help=(
            "If set, only process this many rows from each JSON split after ordering "
            "(e.g. 100 with train+test → up to 200 windows). Use --train_only for "
            "exactly N train rows."
        ),
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
    parser.add_argument(
        "--qa_vector_layer",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "If set, write one GeoJSON point per window under layers/<NAME>/ with "
            "properties monitrs_question / monitrs_answer (truncated). "
            "The same NAME must exist as a vector layer in config.json. "
            "Use --no_skip_existing to write QA vectors for windows that already exist."
        ),
    )
    args = parser.parse_args()

    if args.train_only and args.test_only:
        parser.error("cannot combine --train_only and --test_only")

    monitrs_root = UPath(args.monitrs_root)
    train_path = (
        UPath(args.train_json) if args.train_json else monitrs_root / "total_train.json"
    )
    test_path = (
        UPath(args.test_json) if args.test_json else monitrs_root / "total_test.json"
    )
    ds_path = UPath(args.ds_path)

    _ensure_config(ds_path)
    dataset = Dataset(ds_path)

    if args.qa_vector_layer:
        if args.qa_vector_layer not in dataset.layers:
            parser.error(
                f"--qa_vector_layer {args.qa_vector_layer!r} is not in dataset config "
                f"(layers: {list(dataset.layers.keys())})"
            )
        if dataset.layers[args.qa_vector_layer].type != LayerType.VECTOR:
            parser.error(
                f"--qa_vector_layer {args.qa_vector_layer!r} must be type vector, not "
                f"{dataset.layers[args.qa_vector_layer].type}"
            )

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
                args.qa_vector_layer,
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
