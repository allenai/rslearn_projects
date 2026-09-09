"""Filter v2 annotation JSONs to the entries that prepare.py would accept.

Applies the same window-level rules as ``rslp.change_finder_v2.lcc_model.prepare``:
- Entries with invalid positive-point date ordering (expected
  pre_change < first_date_change_noticeable <= post_change) are omitted
  (prepare.py raises on these).
- "Mixed" entries where some positive points have all three date fields and
  some don't are omitted.
- Entries without complete annotations (no fully-annotated positive point, and
  not a negative-only entry with a time_range) are omitted.
- Duplicate group/window_name keys across all inputs are omitted (first
  occurrence wins, matching prepare.py).

No rule omits individual points; mixed entries drop the whole window.

Each input JSON produces an output JSON with the same basename in the output
directory, which may be a GCS path (gs://...).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from upath import UPath

from rslp.change_finder_v2.lcc_model.prepare import (
    _entry_has_complete_annotations,
    _get_positive_points_missing_dates,
    _parse_date,
)


def _positive_point_dates_valid(entry: dict[str, Any]) -> bool:
    """Check date ordering of fully-annotated positive points.

    Same check as prepare._validate_positive_point_dates, but returns False
    instead of raising so the entry can be omitted.
    """
    for pt in entry.get("positive_points", []):
        if not (
            pt.get("pre_change")
            and pt.get("post_change")
            and pt.get("first_date_change_noticeable")
        ):
            continue
        pre_change = _parse_date(pt["pre_change"])
        post_change = _parse_date(pt["post_change"])
        first_observable = _parse_date(pt["first_date_change_noticeable"])
        if pre_change >= first_observable or first_observable > post_change:
            return False
    return True


def filter_v2_jsons(v2_json_paths: list[str], out_dir: str) -> None:
    """Filter v2 annotation JSONs, writing filtered copies to out_dir.

    Args:
        v2_json_paths: Paths to the input v2 annotation JSONs.
        out_dir: Output directory (local or GCS) for the filtered JSONs, one
            per input file with the same basename.
    """
    out_upath = UPath(out_dir)
    out_upath.mkdir(parents=True, exist_ok=True)

    seen_window_keys: set[tuple[str, str]] = set()
    totals = {
        "kept": 0,
        "invalid_dates": 0,
        "mixed_points": 0,
        "incomplete": 0,
        "duplicate": 0,
    }

    for v2_json_path in v2_json_paths:
        in_upath = UPath(v2_json_path)
        with in_upath.open("r") as f:
            entries = json.load(f)

        kept: list[dict[str, Any]] = []
        counts = {key: 0 for key in totals}

        for entry in entries:
            window_key = (entry.get("group"), entry.get("window_name"))
            if not _positive_point_dates_valid(entry):
                counts["invalid_dates"] += 1
                print(
                    f"  omitting {window_key[0]}/{window_key[1]}: "
                    "invalid positive point date ordering"
                )
            elif _get_positive_points_missing_dates(entry):
                counts["mixed_points"] += 1
                print(
                    f"  omitting {window_key[0]}/{window_key[1]}: "
                    "mixed positive points (some with dates, some without)"
                )
            elif not _entry_has_complete_annotations(entry):
                counts["incomplete"] += 1
            elif window_key in seen_window_keys:
                counts["duplicate"] += 1
                print(f"  omitting {window_key[0]}/{window_key[1]}: duplicate")
            else:
                seen_window_keys.add(window_key)
                counts["kept"] += 1
                kept.append(entry)

        out_path = out_upath / in_upath.name
        with out_path.open("w") as f:
            json.dump(kept, f)

        print(
            f"{v2_json_path} -> {out_path}: kept {counts['kept']}/{len(entries)} "
            f"({counts['invalid_dates']} invalid dates, "
            f"{counts['mixed_points']} mixed points, "
            f"{counts['incomplete']} incomplete, "
            f"{counts['duplicate']} duplicates)"
        )
        for key in totals:
            totals[key] += counts[key]

    total_entries = sum(totals.values())
    print(
        f"Total: kept {totals['kept']}/{total_entries} "
        f"({totals['invalid_dates']} invalid dates, "
        f"{totals['mixed_points']} mixed points, "
        f"{totals['incomplete']} incomplete, "
        f"{totals['duplicate']} duplicates)"
    )


def main() -> None:
    """Filter v2 annotation JSONs to the entries prepare.py would accept."""
    parser = argparse.ArgumentParser(
        description="Filter v2 annotation JSONs to entries prepare.py would accept."
    )
    parser.add_argument(
        "--v2-json-paths",
        nargs="+",
        required=True,
        help="Path(s) to v2 annotation JSONs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (local or gs://) for the filtered JSONs.",
    )
    args = parser.parse_args()

    filter_v2_jsons(v2_json_paths=args.v2_json_paths, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
