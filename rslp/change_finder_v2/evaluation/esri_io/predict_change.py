"""Derive change scores from the ESRI/Impact-Observatory land-cover map at eval points.

This is a baseline that does NOT run a model: it reads the pre-downloaded ESRI 10 m
Annual Land Cover maps (by Impact Observatory) directly. The maps live as one
GeoTIFF per MGRS grid zone (e.g. ``01C_lc_stack.tif``), each in its own UTM CRS, with
one ``uint8`` band per year (2017..2024). For each eval point we read the land-cover
class at the point pixel for ``src_year`` and ``dst_year`` and compare them.

Per point:
- ``pred_src_category`` / ``pred_dst_category``: the ESRI class (mapped to the
  annotation category vocabulary) for src / dst year.
- ``predicted_changed``: ``class_src != class_dst``.
- ``change_score`` (binary, in {0.0, 1.0}): ``1.0`` if the class changed else ``0.0``.
  ESRI gives a hard class label (no probability), so this is a hard-decision baseline.

Tiles are located via a spatial index built once over all GeoTIFFs (each tile's WGS84
bounding box). For a point we consider every tile whose bbox contains it (grid zones
overlap) and use the first where both the src-year and dst-year classes are valid
(not nodata / clouds). This sidesteps the MGRS UTM-zone naming exceptions.

The output CSV schema is shared across change methods (e.g. the WorldCover model), so a
single metric script can consume any method's CSV.

    python -m rslp.change_finder_v2.evaluation.esri_io.predict_change \
        --csv eval.csv \
        --data-dir /weka/dfive-default/rslearn-eai/datasets/esri_lc_stacks \
        --index-path esri_tile_index.json \
        --output eval_esri_io.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Any

import rasterio
import rasterio.warp
import tqdm
from rasterio.windows import Window
from upath import UPath

# Default number of worker processes for the tile index build and pixel reads.
DEFAULT_WORKERS = 32

# ESRI 10 m Annual Land Cover (Impact Observatory) class ids -> annotation category
# names. The annotation vocabulary is the WorldCover class list (see the annotation
# app's CATEGORIES); ESRI has no burnt / fallow / Lichen and moss / shrub classes, so
# those category comparisons simply won't match (acceptable for a baseline). Class 0
# (nodata) and class 10 (clouds) are treated as invalid (no prediction).
ESRI_TO_CATEGORY = {
    1: "water",
    2: "tree",
    4: "wetland (herbaceous)",
    5: "crops",
    7: "urban/built-up",
    8: "bare",
    9: "snow and ice",
    11: "grassland",
}

# ESRI class ids that mean "no valid land cover" at the pixel.
INVALID_CLASSES = {0, 10}

# The ESRI maps cover these years, one raster band each (band index = year offset + 1).
FIRST_YEAR = 2017
LAST_YEAR = 2024

WGS84_CRS = "EPSG:4326"

MERGED_FIELDS = [
    "row_index",
    "lon",
    "lat",
    "src_year",
    "dst_year",
    "has_changed",
    "src_category",
    "dst_category",
    "has_prediction",
    "predicted_changed",
    "change_score",
    "pred_src_category",
    "pred_dst_category",
]


def _year_to_band(year: int) -> int | None:
    """Return the 1-based raster band index for a year, or None if out of range."""
    if year < FIRST_YEAR or year > LAST_YEAR:
        return None
    return year - FIRST_YEAR + 1


def _index_one_tile(path_str: str) -> dict[str, Any]:
    """Open one tile (header only) and return its index entry (WGS84 bbox)."""
    path = UPath(path_str)
    with path.open("rb") as f:
        with rasterio.open(f) as ds:
            bbox = rasterio.warp.transform_bounds(
                ds.crs, WGS84_CRS, *ds.bounds, densify_pts=21
            )
            return {
                "name": path.name,
                "path": path_str,
                "crs": ds.crs.to_string(),
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            }


def _read_tile(
    task: tuple[int, dict[str, Any], list[tuple[int, float, float, int, int]]],
) -> tuple[int, list[tuple[int, int, int]]]:
    """Read all candidate points from one tile.

    ``task`` is ``(tile_index, tile, points)`` where each point is
    ``(row_index, lon, lat, src_band, dst_band)``. Returns ``(tile_index, results)``
    with ``results`` a list of ``(row_index, src_class, dst_class)`` for points that
    fall inside the raster and have a valid (non-nodata/cloud) class in both years.
    """
    tile_index, tile, points = task
    results: list[tuple[int, int, int]] = []
    with UPath(tile["path"]).open("rb") as f:
        with rasterio.open(f) as ds:
            for idx, lon, lat, src_band, dst_band in points:
                xs, ys = rasterio.warp.transform(WGS84_CRS, ds.crs, [lon], [lat])
                row, col = ds.index(xs[0], ys[0])
                if not (0 <= row < ds.height and 0 <= col < ds.width):
                    continue
                arr = ds.read(
                    indexes=[src_band, dst_band], window=Window(col, row, 1, 1)
                )
                src_class, dst_class = int(arr[0, 0, 0]), int(arr[1, 0, 0])
                if src_class in INVALID_CLASSES or dst_class in INVALID_CLASSES:
                    continue
                results.append((idx, src_class, dst_class))
    return tile_index, results


def build_tile_index(data_dir: UPath, workers: int = DEFAULT_WORKERS) -> list[dict[str, Any]]:
    """Build a spatial index over the ESRI tile GeoTIFFs.

    Returns a list of ``{"name", "path", "crs", "bbox"}`` entries, where ``bbox`` is the
    tile's ``[west, south, east, north]`` bounding box in WGS84 degrees. Tiles are
    indexed in parallel across ``workers`` processes.
    """
    path_strs = [str(p) for p in sorted(data_dir.glob("*_lc_stack.tif"))]
    with multiprocessing.Pool(workers) as pool:
        tiles = list(
            tqdm.tqdm(
                pool.imap(_index_one_tile, path_strs),
                total=len(path_strs),
                desc="index tiles",
            )
        )
    # Sort for deterministic order (imap preserves order, but be explicit).
    tiles.sort(key=lambda t: t["name"])
    return tiles


def load_or_build_index(
    data_dir: UPath, index_path: Path | None, workers: int = DEFAULT_WORKERS
) -> list[dict[str, Any]]:
    """Load the tile index from ``index_path`` if present, else build (and cache) it."""
    if index_path is not None and index_path.exists():
        with index_path.open() as f:
            tiles = json.load(f)
        print(f"Loaded tile index ({len(tiles)} tiles) from {index_path}")
        return tiles

    print(f"Building tile index from {data_dir} ...")
    tiles = build_tile_index(data_dir, workers=workers)
    print(f"Indexed {len(tiles)} tiles")
    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w") as f:
            json.dump(tiles, f)
        print(f"Cached tile index to {index_path}")
    return tiles


def _bbox_contains(bbox: list[float], lon: float, lat: float) -> bool:
    """Whether a WGS84 ``[west, south, east, north]`` bbox contains a point."""
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def predict_change(
    csv_path: Path,
    data_dir: str,
    index_path: Path | None,
    output: Path,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[str, Any]]:
    """Score change from the ESRI land-cover map at each eval point."""
    data_upath = UPath(data_dir)
    tiles = load_or_build_index(data_upath, index_path, workers=workers)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # For each point, find candidate tiles (those whose bbox contains it) and the
    # src/dst band indices. Points with an out-of-range year have no candidates.
    candidates: list[list[int]] = []
    bands: list[tuple[int | None, int | None]] = []
    for row in rows:
        lon = float(row["lon"])
        lat = float(row["lat"])
        src_band = _year_to_band(int(row["src_year"]))
        dst_band = _year_to_band(int(row["dst_year"]))
        bands.append((src_band, dst_band))
        if src_band is None or dst_band is None:
            candidates.append([])
            continue
        candidates.append(
            [t for t, tile in enumerate(tiles) if _bbox_contains(tile["bbox"], lon, lat)]
        )

    # Invert to tile -> point indices so each (large) GeoTIFF is opened at most once.
    tile_to_points: dict[int, list[int]] = defaultdict(list)
    for idx, tile_indices in enumerate(candidates):
        for t in tile_indices:
            tile_to_points[t].append(idx)

    # One task per candidate tile (read in parallel); each reads all its points.
    tasks: list[tuple[int, dict[str, Any], list[tuple[int, float, float, int, int]]]] = []
    for t in sorted(tile_to_points):
        points: list[tuple[int, float, float, int, int]] = []
        for i in tile_to_points[t]:
            src_band, dst_band = bands[i]
            assert src_band is not None and dst_band is not None
            points.append(
                (i, float(rows[i]["lon"]), float(rows[i]["lat"]), src_band, dst_band)
            )
        tasks.append((t, tiles[t], points))

    # Resolved class pairs per point; the smallest-index valid (non-nodata) tile wins
    # (deterministic regardless of the order tasks complete).
    resolved: list[tuple[int, int] | None] = [None] * len(rows)
    winning_tile: list[int | None] = [None] * len(rows)
    with multiprocessing.Pool(workers) as pool:
        for tile_index, results in tqdm.tqdm(
            pool.imap_unordered(_read_tile, tasks),
            total=len(tasks),
            desc="read tiles",
        ):
            for idx, src_class, dst_class in results:
                if winning_tile[idx] is None or tile_index < winning_tile[idx]:
                    winning_tile[idx] = tile_index
                    resolved[idx] = (src_class, dst_class)

    merged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        lon = float(row["lon"])
        lat = float(row["lat"])
        gt_changed = row["has_changed"].strip().lower() == "true"

        out: dict[str, Any] = {
            "row_index": idx,
            "lon": lon,
            "lat": lat,
            "src_year": row["src_year"],
            "dst_year": row["dst_year"],
            "has_changed": gt_changed,
            "src_category": row.get("src_category", ""),
            "dst_category": row.get("dst_category", ""),
            "has_prediction": False,
            "predicted_changed": "",
            "change_score": "",
            "pred_src_category": "",
            "pred_dst_category": "",
        }

        if resolved[idx] is not None:
            src_class, dst_class = resolved[idx]
            changed = src_class != dst_class
            out["predicted_changed"] = bool(changed)
            out["change_score"] = 1.0 if changed else 0.0
            out["pred_src_category"] = ESRI_TO_CATEGORY.get(src_class, "")
            out["pred_dst_category"] = ESRI_TO_CATEGORY.get(dst_class, "")
            out["has_prediction"] = True

        merged.append(out)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    scored = sum(1 for r in merged if r["has_prediction"])
    missing = len(merged) - scored
    print(
        f"Wrote {len(merged)} rows to {output}; {scored} with predictions, "
        f"{missing} missing (no tile or nodata at point)"
    )
    return merged


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive ESRI/IO land-cover change scores at eval points into a "
            "standardized CSV (baseline; reads pre-downloaded maps, no model)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV (from export_annotations_to_csv.py).",
    )
    parser.add_argument(
        "--data-dir",
        default="/weka/dfive-default/rslearn-eai/datasets/esri_lc_stacks",
        help="Directory of ESRI '*_lc_stack.tif' tiles (one per MGRS grid zone).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON path to cache/load the tile bbox index. Built from --data-dir "
            "if absent."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output method CSV path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of worker processes for the tile index build and pixel reads. "
            f"Default: {DEFAULT_WORKERS}."
        ),
    )
    args = parser.parse_args()

    predict_change(
        csv_path=args.csv,
        data_dir=args.data_dir,
        index_path=args.index_path,
        output=args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
