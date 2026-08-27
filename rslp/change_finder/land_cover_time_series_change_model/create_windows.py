"""Create rslearn windows for the time-series land-cover change detector.

Inputs:

- A polygons GeoJSON produced by ``rslp.change_finder.scripts.create_land_cover_change_geojson``
  (e.g. ``land_cover_change_src_dst_sel100.geojson``). It has two kinds of
  features per source window: ``"change"`` features (one per (src, dst) class
  pair, with a ``MultiPolygon`` of changed pixels) and ``"no_change"`` features
  (one per source window, with a ``MultiPolygon`` of confidently-unchanged
  pixels). Both carry ``window_group``/``window_name`` referencing the source
  10-year rslearn dataset.

- An annotations sidecar JSON written by the land-cover-change viewer
  (e.g. ``land_cover_change_src_dst_sel100.annotations.json``), keyed by
  ``feature_idx`` (index into the polygons GeoJSON), with ``YYYY-MM`` values
  for ``pre_change``, ``change_start``, ``change_end``, ``post_change``. Only
  ``change`` features get annotated.

- The source rslearn dataset, used only to recover the projection and pixel
  bounds for each source window.

For each annotated change feature this script:

1. Creates a new ``Window`` in the output rslearn dataset with the same
   projection and bounds as the source window. The window's ``time_range``
   spans 5 years ending 1 year after the latest annotated month, so any valid
   3-year sub-window picked at training time can include both pre-change and
   post-change context.
2. Pre-writes three label GeoTIFFs (``label_binary``, ``label_src``,
   ``label_dst``) rasterized from the WGS-84 polygons. These layers are marked
   completed so rslearn skips them at materialize time.

It also writes a sidecar ``ts_change_annotations.json`` at the dataset root
mapping ``"{group}/{name}"`` -> ``{pre_change, change_start, change_end,
post_change}`` so the training-time transform can look up the annotation
without re-reading window metadata files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

import affine
import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
import tqdm
from rasterio.features import rasterize
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

# 13 WorldCover classes (0 = nodata). Mirrors
# rslp/change_finder/scripts/create_land_cover_change_geojson.py CLASS_NAMES.
NUM_CLASSES = 13

# 5 years of data, ending 1 year after the latest annotated month.
WINDOW_DURATION = timedelta(days=365 * 5)
WINDOW_END_PAD = timedelta(days=365)

ANNOTATIONS_SIDECAR_FNAME = "ts_change_annotations.json"

BINARY_LAYER = "label_binary"
SRC_LAYER = "label_src"
DST_LAYER = "label_dst"
LABEL_BAND = "label"

# Binary class IDs (must match config.json class_names).
BIN_NODATA = 0
BIN_NO_CHANGE = 1
BIN_CHANGE = 2

RASTER_FORMAT = GeotiffRasterFormat()


def _parse_month(s: str) -> datetime:
    """Parse a 'YYYY-MM' string into a UTC datetime at the first of the month."""
    return datetime.strptime(s, "%Y-%m").replace(tzinfo=timezone.utc)


def _wgs84_to_pixel(
    geom: shapely.geometry.base.BaseGeometry,
    src_window: Window,
) -> shapely.geometry.base.BaseGeometry | None:
    """Reproject a WGS-84 geometry into the source window's pixel coordinate frame."""
    if geom is None or geom.is_empty:
        return None
    st = STGeometry(WGS84_PROJECTION, geom, time_range=None).to_projection(
        src_window.projection
    )
    if st.shp.is_empty:
        return None
    return st.shp


def _rasterize_polygon(
    poly_pixel: shapely.geometry.base.BaseGeometry | None,
    bounds: tuple[int, int, int, int],
    fill_value: int,
) -> np.ndarray:
    """Rasterize one polygon (in pixel coords) at the given fill value.

    Returns a uint8 HW array of zeros + ``fill_value`` inside the polygon.
    """
    min_x, min_y, max_x, max_y = bounds
    h = max_y - min_y
    w = max_x - min_x
    if poly_pixel is None:
        return np.zeros((h, w), dtype=np.uint8)

    clip = shapely.box(min_x, min_y, max_x, max_y)
    local = poly_pixel.intersection(clip)
    if local.is_empty:
        return np.zeros((h, w), dtype=np.uint8)
    if not local.is_valid:
        local = shapely.make_valid(local)
    if local.is_empty:
        return np.zeros((h, w), dtype=np.uint8)

    # Shift to window-local pixel coords so the rasterize transform is identity.
    local = shapely.affinity.translate(local, xoff=-min_x, yoff=-min_y)
    transform = affine.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return rasterize(
        [(local, fill_value)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )


def _write_label_layer(
    window: Window,
    layer_name: str,
    array_hw: np.ndarray,
) -> None:
    """Write a single-band uint8 HW label raster into ``layer_name`` and mark complete."""
    raster_dir = window.get_raster_dir(layer_name, [LABEL_BAND])
    chw = array_hw[np.newaxis, :, :].astype(np.uint8, copy=False)
    RASTER_FORMAT.encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=chw),
    )
    window.mark_layer_completed(layer_name)


def _create_one_window(
    out_dataset: Dataset,
    src_window: Window,
    change_geom_wgs84: shapely.geometry.base.BaseGeometry,
    no_change_geom_wgs84: shapely.geometry.base.BaseGeometry | None,
    src_class_id: int,
    dst_class_id: int,
    annotation: dict[str, str],
    feature_idx: int,
    group: str,
) -> str:
    """Create one output Window + its 3 label rasters; return the new window name."""
    pre = _parse_month(annotation["pre_change"])
    cs = _parse_month(annotation["change_start"])
    ce = _parse_month(annotation["change_end"])
    post = _parse_month(annotation["post_change"])
    t_end = max(pre, cs, ce, post) + WINDOW_END_PAD
    t_start = t_end - WINDOW_DURATION

    name = f"{src_window.group}__{src_window.name}__f{feature_idx}"
    split = "val" if hashlib.sha256(name.encode()).hexdigest()[0] in "01" else "train"

    window = Window(
        storage=out_dataset.storage,
        group=group,
        name=name,
        projection=src_window.projection,
        bounds=src_window.bounds,
        time_range=(t_start, t_end),
        options=dict(
            split=split,
            src_group=src_window.group,
            src_name=src_window.name,
            feature_idx=feature_idx,
            src_class_id=src_class_id,
            dst_class_id=dst_class_id,
            pre_change=annotation["pre_change"],
            change_start=annotation["change_start"],
            change_end=annotation["change_end"],
            post_change=annotation["post_change"],
        ),
    )
    window.save()

    change_pixel = _wgs84_to_pixel(change_geom_wgs84, src_window)
    no_change_pixel = (
        _wgs84_to_pixel(no_change_geom_wgs84, src_window)
        if no_change_geom_wgs84 is not None
        else None
    )

    change_mask = _rasterize_polygon(change_pixel, src_window.bounds, 1)
    no_change_mask = _rasterize_polygon(no_change_pixel, src_window.bounds, 1)

    # Binary label: change wins over no_change on overlap; everything else nodata.
    binary = np.full(change_mask.shape, BIN_NODATA, dtype=np.uint8)
    binary[no_change_mask == 1] = BIN_NO_CHANGE
    binary[change_mask == 1] = BIN_CHANGE

    src_label = np.zeros_like(binary)
    src_label[change_mask == 1] = src_class_id
    dst_label = np.zeros_like(binary)
    dst_label[change_mask == 1] = dst_class_id

    _write_label_layer(window, BINARY_LAYER, binary)
    _write_label_layer(window, SRC_LAYER, src_label)
    _write_label_layer(window, DST_LAYER, dst_label)

    return name


def _process_job(job: dict[str, Any]) -> str | None:
    try:
        return _create_one_window(**job)
    except Exception as e:
        traceback.print_exc()
        print(f"Error on feature_idx={job.get('feature_idx')}: {e}")
        return None


def create_windows(
    *,
    polygons_geojson: str,
    annotations_json: str,
    src_ds_path: str,
    out_ds_path: str,
    group: str = "default",
    workers: int = 32,
) -> None:
    """Create new rslearn windows + label rasters from annotated change polygons.

    Args:
        polygons_geojson: GeoJSON with ``change`` and ``no_change`` features.
        annotations_json: sidecar JSON with one annotation dict per change feature
            keyed by ``feature_idx``.
        src_ds_path: source rslearn dataset path (only window metadata is read).
        out_ds_path: output rslearn dataset path. ``config.json`` must already exist.
        group: window group name for the new windows.
        workers: number of worker processes.
    """
    with open(polygons_geojson) as f:
        fc = json.load(f)
    features = fc["features"]
    print(f"Loaded {len(features)} features from {polygons_geojson}")

    with open(annotations_json) as f:
        annotations_list = json.load(f)
    annotations_by_idx = {int(a["feature_idx"]): a for a in annotations_list}
    print(f"Loaded {len(annotations_by_idx)} annotations from {annotations_json}")

    # Index no_change features by source window.
    no_change_by_window: dict[tuple[str, str], dict] = {}
    src_windows_needed: set[tuple[str, str]] = set()
    for feat in features:
        p = feat["properties"]
        key = (p["window_group"], p["window_name"])
        if p["feature_type"] == "no_change":
            no_change_by_window[key] = feat

    # Collect change features that have annotations.
    annotated_features: list[tuple[int, dict]] = []
    for idx, feat in enumerate(features):
        p = feat["properties"]
        if p["feature_type"] != "change":
            continue
        if idx not in annotations_by_idx:
            continue
        annotated_features.append((idx, feat))
        src_windows_needed.add((p["window_group"], p["window_name"]))
    print(f"{len(annotated_features)} annotated change features to process")

    # Load only the source windows we actually reference.
    src_dataset = Dataset(UPath(src_ds_path))
    print(f"Loading {len(src_windows_needed)} source windows...")
    groups = sorted({g for g, _ in src_windows_needed})
    names = sorted({n for _, n in src_windows_needed})
    all_src_windows = src_dataset.load_windows(
        groups=groups, names=names, workers=workers, show_progress=True
    )
    src_window_by_key = {(w.group, w.name): w for w in all_src_windows}
    missing = src_windows_needed - set(src_window_by_key)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} source windows, e.g. {next(iter(missing))}"
        )

    out_dataset = Dataset(UPath(out_ds_path))

    jobs: list[dict[str, Any]] = []
    annotations_sidecar: dict[str, dict[str, str | int]] = {}
    for feature_idx, feat in annotated_features:
        p = feat["properties"]
        key = (p["window_group"], p["window_name"])
        src_window = src_window_by_key[key]
        change_geom = shapely.geometry.shape(feat["geometry"])
        no_change_feat = no_change_by_window.get(key)
        no_change_geom = (
            shapely.geometry.shape(no_change_feat["geometry"])
            if no_change_feat is not None
            else None
        )
        annotation = annotations_by_idx[feature_idx]
        jobs.append(
            dict(
                out_dataset=out_dataset,
                src_window=src_window,
                change_geom_wgs84=change_geom,
                no_change_geom_wgs84=no_change_geom,
                src_class_id=int(p["src_class_id"]),
                dst_class_id=int(p["dst_class_id"]),
                annotation=annotation,
                feature_idx=feature_idx,
                group=group,
            )
        )
        out_name = f"{src_window.group}__{src_window.name}__f{feature_idx}"
        annotations_sidecar[f"{group}/{out_name}"] = {
            "pre_change": annotation["pre_change"],
            "change_start": annotation["change_start"],
            "change_end": annotation["change_end"],
            "post_change": annotation["post_change"],
            "src_class_id": int(p["src_class_id"]),
            "dst_class_id": int(p["dst_class_id"]),
        }

    print(f"Processing {len(jobs)} windows with {workers} workers")
    if workers <= 1:
        results = [_process_job(j) for j in tqdm.tqdm(jobs)]
    else:
        with multiprocessing.Pool(workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(_process_job, jobs),
                    total=len(jobs),
                )
            )
    num_ok = sum(1 for r in results if r is not None)
    print(f"Created {num_ok}/{len(jobs)} windows")

    sidecar_path = UPath(out_ds_path) / ANNOTATIONS_SIDECAR_FNAME
    with sidecar_path.open("w") as f:
        json.dump(annotations_sidecar, f)
    print(f"Wrote annotation sidecar to {sidecar_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create rslearn windows for the time-series change detector."
    )
    parser.add_argument("--polygons_geojson", required=True)
    parser.add_argument("--annotations_json", required=True)
    parser.add_argument(
        "--src_ds_path",
        required=True,
        help="Source rslearn dataset (used only for window projection/bounds).",
    )
    parser.add_argument(
        "--out_ds_path",
        required=True,
        help="Output rslearn dataset path (config.json must already exist).",
    )
    parser.add_argument("--group", default="default")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    multiprocessing.set_start_method("forkserver")
    create_windows(
        polygons_geojson=args.polygons_geojson,
        annotations_json=args.annotations_json,
        src_ds_path=args.src_ds_path,
        out_ds_path=args.out_ds_path,
        group=args.group,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
