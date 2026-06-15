"""Export annotated v2 JSON points to an evaluation CSV.

One CSV row per annotated point (positive or negative) across all input JSONs.
Columns: lon, lat, src_year, dst_year, has_changed, src_category, dst_category.

- Positive (change) points contribute has_changed=True. src_year is the year before
  the year containing pre_change; dst_year is the year after the year containing
  post_change. src_category/dst_category come from pre_category/post_category.
- Negative (no-change) points contribute has_changed=False with blank categories and
  fixed src_year/dst_year (CLI defaults 2019/2021).

Positive points missing any of pre_change/post_change/pre_category/post_category are
skipped (and counted), mirroring prepare._entry_has_complete_annotations.

Points are also dropped (and counted) when:
- they fall outside the supported imagery range: src_year < --min-src-year (the WorldCover
  imagery starts in 2017, so a src_year of 2016 is out of range) or dst_year >
  --max-dst-year, or
- (positive points only) src_category == dst_category, i.e. no real land-cover change.

Points within --min-distance-m meters of an already-kept point are dropped (greedy),
so the exported set has no two points closer than that threshold. Positive points are
processed first, so a positive is kept over a nearby negative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import tqdm
from rslearn.utils.grid_index import GridIndex

# Approximate meters per degree of latitude (used to size the dedup grid and to
# convert the min-distance threshold into a lon/lat query box).
_METERS_PER_DEG_LAT = 111_320.0

# Minimum allowed distance (meters) between any two exported points. Points
# closer than this to an already-kept point are dropped.
DEFAULT_MIN_DISTANCE_M = 50.0

# Supported imagery range (inclusive): points with src_year before this or dst_year
# after DEFAULT_MAX_DST_YEAR are dropped.
DEFAULT_MIN_SRC_YEAR = 2017
DEFAULT_MAX_DST_YEAR = 2024

CSV_FIELDS = [
    "lon",
    "lat",
    "src_year",
    "dst_year",
    "has_changed",
    "src_category",
    "dst_category",
]


def _year_of(date_str: str) -> int:
    """Parse an ISO date string and return its year."""
    return datetime.fromisoformat(date_str).year


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two lon/lat points."""
    earth_radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * earth_radius_m * math.asin(math.sqrt(a))


def _query_box(
    lon: float, lat: float, min_distance_m: float
) -> tuple[float, float, float, float]:
    """Lon/lat bounding box that contains everything within min_distance_m of a point."""
    dlat = min_distance_m / _METERS_PER_DEG_LAT
    coslat = math.cos(math.radians(lat))
    if coslat > 1e-6:
        dlon = min(min_distance_m / (_METERS_PER_DEG_LAT * coslat), 180.0)
    else:
        dlon = 180.0
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _dedupe_by_distance(
    rows: list[dict[str, Any]],
    min_distance_m: float,
    kept: list[dict[str, Any]] | None = None,
    index: GridIndex | None = None,
    desc: str = "dedup",
) -> tuple[list[dict[str, Any]], int]:
    """Greedily drop rows within min_distance_m of an already-kept row.

    Returns (kept_rows, num_dropped). Rows are processed in input order, so the
    first point in any close cluster is the one kept. An existing `kept` list (and its
    matching `index`) can be passed to dedup against (and extend); both are mutated in
    place.

    A GridIndex over lon/lat keeps this near-linear: each candidate only haversine-checks
    points in nearby grid cells (the cell size matches the min-distance threshold) rather
    than every kept point. The exact haversine check is still applied to those candidates,
    so the result is identical to the brute-force version.
    """
    if kept is None:
        kept = []
    if index is None:
        index = GridIndex(size=min_distance_m / _METERS_PER_DEG_LAT)
    dropped = 0
    for row in tqdm.tqdm(rows, desc=desc):
        lon, lat = row["lon"], row["lat"]
        too_close = any(
            _haversine_m(lon, lat, k["lon"], k["lat"]) < min_distance_m
            for k in index.query(_query_box(lon, lat, min_distance_m))
        )
        if too_close:
            dropped += 1
            continue
        kept.append(row)
        index.insert((lon, lat, lon, lat), row)
    return kept, dropped


def _positive_row(pt: dict[str, Any]) -> dict[str, Any] | None:
    """Build a CSV row for a positive (change) point, or None if incomplete."""
    pre_change = pt.get("pre_change")
    post_change = pt.get("post_change")
    pre_category = pt.get("pre_category")
    post_category = pt.get("post_category")
    if not (pre_change and post_change and pre_category and post_category):
        return None

    return {
        "lon": pt["lon"],
        "lat": pt["lat"],
        "src_year": _year_of(pre_change) - 1,
        "dst_year": _year_of(post_change) + 1,
        "has_changed": True,
        "src_category": pre_category,
        "dst_category": post_category,
    }


def _negative_row(
    pt: dict[str, Any], negative_src_year: int, negative_dst_year: int
) -> dict[str, Any]:
    """Build a CSV row for a negative (no-change) point."""
    return {
        "lon": pt["lon"],
        "lat": pt["lat"],
        "src_year": negative_src_year,
        "dst_year": negative_dst_year,
        "has_changed": False,
        "src_category": "",
        "dst_category": "",
    }


def _filter_reason(
    row: dict[str, Any], min_src_year: int, max_dst_year: int
) -> str | None:
    """Return a drop reason for a row, or None if it should be kept.

    Drops points outside the supported imagery range, and positive points whose
    source and destination categories are identical (no real land-cover change).
    """
    if row["src_year"] < min_src_year or row["dst_year"] > max_dst_year:
        return "out_of_range"
    if row["has_changed"] and row["src_category"] == row["dst_category"]:
        return "same_category"
    return None


def export_rows(
    v2_json_paths: list[Path],
    negative_src_year: int = 2019,
    negative_dst_year: int = 2021,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    min_src_year: int = DEFAULT_MIN_SRC_YEAR,
    max_dst_year: int = DEFAULT_MAX_DST_YEAR,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Load v2 JSONs and return kept rows plus drop counts.

    Returns (rows, num_skipped_incomplete, num_out_of_range, num_same_category,
    num_dropped_close).

    Points outside [min_src_year, max_dst_year] and positives with identical
    src/dst categories are dropped. Points within min_distance_m of an already-kept
    point are dropped. Positives are processed first so they win over nearby negatives.
    """
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    skipped_incomplete = 0
    out_of_range = 0
    same_category = 0

    def _keep(row: dict[str, Any], bucket: list[dict[str, Any]]) -> None:
        nonlocal out_of_range, same_category
        reason = _filter_reason(row, min_src_year, max_dst_year)
        if reason == "out_of_range":
            out_of_range += 1
        elif reason == "same_category":
            same_category += 1
        else:
            bucket.append(row)

    for json_path in tqdm.tqdm(v2_json_paths, desc="loading JSONs"):
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            for pt in entry.get("positive_points", []):
                row = _positive_row(pt)
                if row is None:
                    skipped_incomplete += 1
                    continue
                _keep(row, positive_rows)
            for pt in entry.get("negative_points", []):
                _keep(
                    _negative_row(pt, negative_src_year, negative_dst_year),
                    negative_rows,
                )

    # Share one grid index (and kept list) across both passes so positives are inserted
    # first and win over nearby negatives, matching the original greedy semantics.
    index = GridIndex(size=min_distance_m / _METERS_PER_DEG_LAT)
    kept: list[dict[str, Any]] = []
    kept, dropped_close = _dedupe_by_distance(
        positive_rows, min_distance_m, kept, index, desc="dedup positives"
    )
    kept, dropped_neg = _dedupe_by_distance(
        negative_rows, min_distance_m, kept, index, desc="dedup negatives"
    )
    return (
        kept,
        skipped_incomplete,
        out_of_range,
        same_category,
        dropped_close + dropped_neg,
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Export annotated change_finder_v2 JSON points to an evaluation CSV."
        )
    )
    parser.add_argument(
        "--v2-json-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to annotated v2 annotation JSONs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--negative-src-year",
        type=int,
        default=2019,
        help="src_year assigned to negative (no-change) points. Default: 2019.",
    )
    parser.add_argument(
        "--negative-dst-year",
        type=int,
        default=2021,
        help="dst_year assigned to negative (no-change) points. Default: 2021.",
    )
    parser.add_argument(
        "--min-distance-m",
        type=float,
        default=DEFAULT_MIN_DISTANCE_M,
        help=(
            "Drop points within this many meters of an already-kept point. "
            f"Default: {DEFAULT_MIN_DISTANCE_M:g}."
        ),
    )
    parser.add_argument(
        "--min-src-year",
        type=int,
        default=DEFAULT_MIN_SRC_YEAR,
        help=(
            "Drop points whose src_year is before this. "
            f"Default: {DEFAULT_MIN_SRC_YEAR}."
        ),
    )
    parser.add_argument(
        "--max-dst-year",
        type=int,
        default=DEFAULT_MAX_DST_YEAR,
        help=(
            "Drop points whose dst_year is after this. "
            f"Default: {DEFAULT_MAX_DST_YEAR}."
        ),
    )
    args = parser.parse_args()

    rows, skipped_incomplete, out_of_range, same_category, dropped_close = export_rows(
        v2_json_paths=args.v2_json_paths,
        negative_src_year=args.negative_src_year,
        negative_dst_year=args.negative_dst_year,
        min_distance_m=args.min_distance_m,
        min_src_year=args.min_src_year,
        max_dst_year=args.max_dst_year,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    num_changed = sum(1 for r in rows if r["has_changed"])
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"({num_changed} changed, {len(rows) - num_changed} no-change); "
        f"skipped {skipped_incomplete} incomplete positive points; "
        f"dropped {out_of_range} outside years "
        f"[{args.min_src_year}, {args.max_dst_year}], "
        f"{same_category} with same src/dst category, "
        f"{dropped_close} within {args.min_distance_m:g} m of another"
    )


if __name__ == "__main__":
    main()
