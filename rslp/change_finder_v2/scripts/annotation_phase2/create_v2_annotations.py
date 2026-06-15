"""Create a v2 annotation JSON from change_finder_v2 write_raster prediction outputs.

The random-2048 prediction runs (``write_jobs_random_2048``) produce, for each
2048x2048 tile, a 49-band ``output_change`` GeoTIFF (when ``write_raster`` is set)
plus a sibling GeoJSON with the same basename. This script:

1. Scans every ``.tif`` in the input directory.
2. Thresholds the binary change band (>=0.5) and randomly selects ONE change pixel
   per tile.
3. Reads the per-pixel argmax source/destination land cover category and the
   timestep (argmax over the 20 timestamp bands) at that pixel.
4. Maps the timestep index to an actual date using the sibling GeoJSON, whose
   features carry ``timestamp_idx`` / ``timestamp_start`` / ``timestamp_end`` that
   were resolved from window metadata at prediction time (this metadata is not
   present in the standalone raster).
5. Emits a v2 annotation entry (a 128x128 window centered on the pixel, with one
   positive point) suitable for
   ``rslp.change_finder_v2.annotation_app.create_windows``.

Usage::

    python -m rslp.change_finder_v2.scripts.annotation_phase2.create_v2_annotations \
        --input_dir /path/to/write_raster_outputs/ \
        --output v2_annotations.json \
        --group phase2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from datetime import datetime

import numpy as np
import rasterio
import shapely
import shapely.geometry
import tqdm
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from upath import UPath

from rslp.change_finder_v2.lcc_model.postprocess import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    NUM_LC_CLASSES,
    NUM_TIMESTAMPS,
    SRC_BAND_OFFSET,
    TS_BAND_OFFSET,
)

# Binary change probability threshold on the 0-255 uint8 scale (>= 0.5).
DEFAULT_THRESHOLD = 128
# Side length (pixels) of the annotation window centered on the chosen pixel.
DEFAULT_WINDOW_SIZE = 128
# Number of years on either side of the change date for the entry time_range.
TIME_RANGE_YEARS = 3


def _shift_years(dt: datetime, years: int) -> datetime:
    """Shift a datetime by a whole number of years, handling Feb 29."""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Feb 29 -> Feb 28 in a non-leap target year.
        return dt.replace(year=dt.year + years, day=28)


def _pixel_to_lonlat(px: int, py: int, projection: object) -> tuple[float, float]:
    """Convert absolute pixel coordinates (column, row) to lon/lat."""
    pt = shapely.geometry.Point(px + 0.5, py + 0.5)
    wgs84_pt = STGeometry(projection, pt, None).to_projection(WGS84_PROJECTION).shp
    return float(wgs84_pt.x), float(wgs84_pt.y)


def _load_timestamp_map(
    geojson_path: UPath,
) -> dict[int, tuple[str, str]]:
    """Load the timestep-index -> (start, end) date map from the sibling GeoJSON.

    Returns:
        a dict mapping a timestamp index to (timestamp_start, timestamp_end) ISO
        strings, built from the GeoJSON features.
    """
    idx_to_dates: dict[int, tuple[str, str]] = {}
    if not geojson_path.exists():
        return idx_to_dates

    with geojson_path.open("r") as f:
        fc = json.load(f)
    features = fc.get("features", [])

    for feat in features:
        props = feat.get("properties", {})
        ts_idx = props.get("timestamp_idx")
        ts_start = props.get("timestamp_start")
        ts_end = props.get("timestamp_end")
        if ts_idx is None or ts_start is None or ts_end is None:
            continue
        idx_to_dates.setdefault(int(ts_idx), (ts_start, ts_end))

    return idx_to_dates


def _resolve_dates(
    ts_idx: int,
    idx_to_dates: dict[int, tuple[str, str]],
) -> tuple[str, str] | None:
    """Resolve (start, end) ISO dates for the pixel's timestep.

    Tries an exact timestamp-index match in the GeoJSON, then falls back to the
    nearest available index.
    """
    if ts_idx in idx_to_dates:
        return idx_to_dates[ts_idx]

    # Fallback: the nearest available timestamp index.
    if idx_to_dates:
        nearest = min(idx_to_dates.keys(), key=lambda k: abs(k - ts_idx))
        return idx_to_dates[nearest]

    return None


def process_tile(
    tif_path_str: str,
    group: str,
    threshold: int,
    window_size: int,
    seed: int,
) -> dict | None:
    """Process one prediction raster and return a v2 annotation entry, or None."""
    tif_path = UPath(tif_path_str)

    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            arr = src.read()
            projection, bounds = get_raster_projection_and_bounds(src)

    change_score = arr[BINARY_CHANGE_BAND]
    ys, xs = np.where(change_score >= threshold)
    if len(ys) == 0:
        return None

    # Deterministically pick one change pixel for this tile.
    rng = random.Random(f"{tif_path.name}:{seed}")
    i = rng.randrange(len(ys))
    row, col = int(ys[i]), int(xs[i])

    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    ts_probs = arr[TS_BAND_OFFSET : TS_BAND_OFFSET + NUM_TIMESTAMPS, row, col]

    # Skip class 0 (nodata) by taking argmax over classes 1..12 and adding 1.
    src_idx = int(src_probs[1:].argmax()) + 1
    dst_idx = int(dst_probs[1:].argmax()) + 1
    ts_idx = int(ts_probs.argmax())

    px = bounds[0] + col
    py = bounds[1] + row
    lon, lat = _pixel_to_lonlat(px, py, projection)

    geojson_path = tif_path.parent / (tif_path.stem + ".geojson")
    idx_to_dates = _load_timestamp_map(geojson_path)
    dates = _resolve_dates(ts_idx, idx_to_dates)

    point: dict = {
        "lon": lon,
        "lat": lat,
        "pre_category": LC_CLASS_NAMES[src_idx],
        "post_category": LC_CLASS_NAMES[dst_idx],
    }

    time_range: list[str] | None = None
    if dates is not None:
        pre_change = datetime.fromisoformat(dates[0]).date().isoformat()
        point["pre_change"] = pre_change

        change_date = datetime.fromisoformat(dates[0])
        time_range = [
            _shift_years(change_date, -TIME_RANGE_YEARS).date().isoformat(),
            _shift_years(change_date, TIME_RANGE_YEARS).date().isoformat(),
        ]

    half = window_size // 2
    entry_bounds = [
        px - half,
        py - half,
        px - half + window_size,
        py - half + window_size,
    ]

    entry: dict = {
        "projection": projection.serialize(),
        "bounds": entry_bounds,
        "window_name": f"{projection.crs}_{px}_{py}",
        "group": group,
        "positive_points": [point],
        "negative_points": [],
    }
    if time_range is not None:
        entry["time_range"] = time_range

    return entry


def create_annotations(
    input_dir: str,
    output: str,
    group: str,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed: int = 0,
    workers: int = 32,
) -> None:
    """Scan write_raster outputs and write a v2 annotation JSON."""
    in_path = UPath(input_dir)
    tif_paths = sorted(str(p) for p in in_path.glob("*.tif"))
    print(f"Found {len(tif_paths)} prediction rasters in {input_dir}")

    kwargs_list = [
        dict(
            tif_path_str=p,
            group=group,
            threshold=threshold,
            window_size=window_size,
            seed=seed,
        )
        for p in tif_paths
    ]

    entries: list[dict] = []
    with multiprocessing.Pool(workers) as pool:
        results = star_imap_unordered(pool, process_tile, kwargs_list)
        for entry in tqdm.tqdm(results, total=len(kwargs_list), desc="Processing"):
            if entry is not None:
                entries.append(entry)

    out_path = UPath(output)
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} annotation entries to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a v2 annotation JSON from change_finder_v2 write_raster outputs."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the per-tile .tif and sibling .geojson outputs.",
    )
    parser.add_argument(
        "--output", required=True, help="Output v2 annotation JSON path."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Binary change probability threshold (0-255). Default 128 (>=0.5).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Side length (pixels) of each annotation window. Default 128.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Window group name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for per-tile pixel selection. Default 0.",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    create_annotations(
        input_dir=args.input_dir,
        output=args.output,
        group=args.group,
        threshold=args.threshold,
        window_size=args.window_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
