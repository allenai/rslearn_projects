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

# Minimum allowed distance (meters) between any two exported points. Points
# closer than this to an already-kept point are dropped.
DEFAULT_MIN_DISTANCE_M = 50.0

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


def _dedupe_by_distance(
    rows: list[dict[str, Any]],
    min_distance_m: float,
    kept: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Greedily drop rows within min_distance_m of an already-kept row.

    Returns (kept_rows, num_dropped). Rows are processed in input order, so the
    first point in any close cluster is the one kept. An existing `kept` list can
    be passed to dedup against (and extend); it is mutated in place.
    """
    if kept is None:
        kept = []
    dropped = 0
    for row in rows:
        too_close = any(
            _haversine_m(row["lon"], row["lat"], k["lon"], k["lat"]) < min_distance_m
            for k in kept
        )
        if too_close:
            dropped += 1
            continue
        kept.append(row)
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


def export_rows(
    v2_json_paths: list[Path],
    negative_src_year: int = 2019,
    negative_dst_year: int = 2021,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
) -> tuple[list[dict[str, Any]], int, int]:
    """Load v2 JSONs and return (rows, num_skipped_incomplete, num_dropped_close).

    Points within min_distance_m of an already-kept point are dropped. Positives
    are processed first so they win over nearby negatives.
    """
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    skipped_incomplete = 0

    for json_path in v2_json_paths:
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            for pt in entry.get("positive_points", []):
                row = _positive_row(pt)
                if row is None:
                    skipped_incomplete += 1
                    continue
                positive_rows.append(row)
            for pt in entry.get("negative_points", []):
                negative_rows.append(
                    _negative_row(pt, negative_src_year, negative_dst_year)
                )

    kept, dropped_close = _dedupe_by_distance(positive_rows, min_distance_m)
    kept, dropped_neg = _dedupe_by_distance(negative_rows, min_distance_m, kept)
    return kept, skipped_incomplete, dropped_close + dropped_neg


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
    args = parser.parse_args()

    rows, skipped_incomplete, dropped_close = export_rows(
        v2_json_paths=args.v2_json_paths,
        negative_src_year=args.negative_src_year,
        negative_dst_year=args.negative_dst_year,
        min_distance_m=args.min_distance_m,
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
        f"dropped {dropped_close} points within {args.min_distance_m:g} m of another"
    )


if __name__ == "__main__":
    main()
