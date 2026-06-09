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
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
) -> tuple[list[dict[str, Any]], int]:
    """Load v2 JSONs and return (rows, num_skipped_incomplete_positives)."""
    rows: list[dict[str, Any]] = []
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
                rows.append(row)
            for pt in entry.get("negative_points", []):
                rows.append(_negative_row(pt, negative_src_year, negative_dst_year))

    return rows, skipped_incomplete


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
    args = parser.parse_args()

    rows, skipped_incomplete = export_rows(
        v2_json_paths=args.v2_json_paths,
        negative_src_year=args.negative_src_year,
        negative_dst_year=args.negative_dst_year,
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
        f"skipped {skipped_incomplete} incomplete positive points"
    )


if __name__ == "__main__":
    main()
