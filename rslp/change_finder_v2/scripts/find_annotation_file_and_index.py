"""Find which v2 annotation JSON(s) contain a valid entry for a given window name.

Given a list of v2 annotation JSONs and a window name, this reports which JSON(s)
have an entry for that window, and at which index, that the LCC model export would
actually turn into a training window.

The "valid entry" rule mirrors ``_entry_has_complete_annotations`` from the LCC model
export (``rslp.change_finder_v2.lcc_model.prepare``): an entry is usable if it has
either at least one fully-annotated positive point (pre/post change dates, first
noticeable date, and pre/post categories), or no positive points but at least one
negative point together with a ``time_range``.

Usage::

    python -m rslp.change_finder_v2.scripts.find_annotation_file_and_index \
        --window_name EPSG:32633_123_456 \
        annotations_a.json annotations_b.json ...
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from upath import UPath


def entry_has_complete_annotations(entry: dict[str, Any]) -> bool:
    """Check that an entry has enough annotation info to create a training window.

    Mirrors ``_entry_has_complete_annotations`` in the LCC model export. Accepts
    entries with either:
    - At least one fully-annotated positive point (with dates and categories), OR
    - No positive points but at least one negative point and a time_range field.
    """
    for pt in entry.get("positive_points", []):
        if (
            pt.get("pre_change")
            and pt.get("post_change")
            and pt.get("first_date_change_noticeable")
            and pt.get("pre_category")
            and pt.get("post_category")
        ):
            return True
    if (
        not entry.get("positive_points")
        and entry.get("negative_points")
        and entry.get("time_range")
    ):
        return True
    return False


def find_entries(json_path: str, window_name: str) -> tuple[list[int], list[int]]:
    """Find entries matching window_name in one annotation JSON.

    Returns:
        a tuple (valid_indices, invalid_indices) of entry indices whose
        ``window_name`` matches, split by whether the entry is a valid (usable) entry
        per the LCC model export rules.
    """
    with UPath(json_path).open("r") as f:
        entries = json.load(f)

    valid: list[int] = []
    invalid: list[int] = []
    for idx, entry in enumerate(entries):
        if entry.get("window_name") != window_name:
            continue
        if entry_has_complete_annotations(entry):
            valid.append(idx)
        else:
            invalid.append(idx)
    return valid, invalid


def find_annotation_file_and_index(
    json_paths: list[str], window_name: str
) -> dict[str, list[int]]:
    """Report which JSON(s) have a valid entry for window_name.

    Returns a mapping from JSON path to the list of valid entry indices for that
    path (only paths with at least one valid entry are included).
    """
    result: dict[str, list[int]] = {}
    for json_path in json_paths:
        valid, invalid = find_entries(json_path, window_name)
        if valid:
            result[json_path] = valid
            print(f"{json_path}: valid entry at index/indices {valid}")
        elif invalid:
            print(
                f"{json_path}: found {len(invalid)} matching entry/entries "
                f"at {invalid} but none are valid (incomplete annotations)"
            )

    if not result:
        print(f"\nNo valid entry for window '{window_name}' in any JSON.")
    return result


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Find which v2 annotation JSON(s) have a valid entry for a window name, "
            "per the LCC model export rules."
        )
    )
    parser.add_argument("--window_name", required=True, help="Window name to look for.")
    parser.add_argument(
        "--json_paths",
        required=True,
        nargs="+",
        help="Paths to v2 annotation JSON files.",
    )
    args = parser.parse_args()

    find_annotation_file_and_index(args.json_paths, args.window_name)


if __name__ == "__main__":
    main()
