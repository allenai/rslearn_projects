"""Add LCC-flagged points from a v2 annotation file to the VLM image database.

Two modes:

- ``evaluation``: include labeled positive points (those with pre_change, post_change,
  first_date_change_noticeable, and both pre/post categories) plus negative points,
  recording the ground-truth label. Use this to quantitatively evaluate Gemini's
  accuracy. The key year is the midpoint of the pre-change and post-change dates (for
  negatives, the midpoint of the entry's time range).
- ``deployment``: include every point as something to validate. Requires each point to
  have pre_change and both categories (predicted by the LCC model). Errors out if any
  negative points are present or if any point has post_change or
  first_date_change_noticeable set. The key year is the pre-change year.

This both creates the windows in the image database (so the user can run rslearn
prepare/materialize) and writes a JSON :class:`PointSet` summarizing the points.

Example:
    python -m rslp.change_finder_v2.vlm.validate_points.add_points \
        --annotations annotations.json \
        --image-db-path /weka/.../image_db \
        --output points.json \
        --mode evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from rslp.change_finder_v2.vlm.image_db import image_database

from .schema import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    MODE_DEPLOYMENT,
    MODE_EVALUATION,
    PointRecord,
    PointSet,
)

logger = logging.getLogger(__name__)

# A positive point needs these to have a usable change date and transition.
_POSITIVE_FIELDS = ("pre_change", "post_change", "first_date_change_noticeable")
_CATEGORY_FIELDS = ("pre_category", "post_category")


def _midpoint(d1: date, d2: date) -> date:
    """Return the calendar midpoint of two dates."""
    return date.fromordinal((d1.toordinal() + d2.toordinal()) // 2)


def _make_record(
    lon: float,
    lat: float,
    year: int,
    predicted_change_date: str | None,
    pre_category: str | None,
    post_category: str | None,
    label: str | None,
    entry: dict,
    point_type: str,
    point_index: int,
) -> PointRecord:
    return PointRecord(
        lon=lon,
        lat=lat,
        year=year,
        window_name=image_database.window_name(lon, lat, year),
        predicted_change_date=predicted_change_date,
        pre_category=pre_category,
        post_category=post_category,
        label=label,
        annotation_group=entry.get("group", ""),
        annotation_window_name=entry.get("window_name", ""),
        point_type=point_type,
        point_index=point_index,
    )


def _evaluation_records(entries: list[dict]) -> list[PointRecord]:
    """Build records for evaluation mode (labeled positives + negatives)."""
    records: list[PointRecord] = []
    for entry in entries:
        for idx, point in enumerate(entry.get("positive_points", [])):
            if not all(point.get(field) for field in _POSITIVE_FIELDS):
                continue
            if not all(point.get(field) for field in _CATEGORY_FIELDS):
                continue
            pre = date.fromisoformat(point["pre_change"])
            post = date.fromisoformat(point["post_change"])
            mid = _midpoint(pre, post)
            records.append(
                _make_record(
                    lon=float(point["lon"]),
                    lat=float(point["lat"]),
                    year=mid.year,
                    predicted_change_date=mid.isoformat(),
                    pre_category=point.get("pre_category"),
                    post_category=point.get("post_category"),
                    label=LABEL_POSITIVE,
                    entry=entry,
                    point_type="positive",
                    point_index=idx,
                )
            )

        for idx, point in enumerate(entry.get("negative_points", [])):
            time_range = entry.get("time_range")
            if not time_range:
                logger.warning(
                    "skipping negative point in %s: entry has no time_range",
                    entry.get("window_name"),
                )
                continue
            start = date.fromisoformat(time_range[0])
            end = date.fromisoformat(time_range[1])
            mid = _midpoint(start, end)
            records.append(
                _make_record(
                    lon=float(point["lon"]),
                    lat=float(point["lat"]),
                    year=mid.year,
                    predicted_change_date=None,
                    pre_category=None,
                    post_category=None,
                    label=LABEL_NEGATIVE,
                    entry=entry,
                    point_type="negative",
                    point_index=idx,
                )
            )
    return records


def _deployment_records(entries: list[dict]) -> list[PointRecord]:
    """Build records for deployment mode, validating the input is prediction-like."""
    records: list[PointRecord] = []
    for entry in entries:
        if entry.get("negative_points"):
            raise ValueError(
                f"entry {entry.get('window_name')} has negative points; deployment "
                "mode expects only positive (flagged) points"
            )
        for idx, point in enumerate(entry.get("positive_points", [])):
            for field in ("post_change", "first_date_change_noticeable"):
                if point.get(field):
                    raise ValueError(
                        f"entry {entry.get('window_name')} point {idx} has {field}; "
                        "deployment mode expects only predicted points (pre_change "
                        "and categories)"
                    )
            if not point.get("pre_change"):
                raise ValueError(
                    f"entry {entry.get('window_name')} point {idx} is missing "
                    "pre_change (the predicted change date)"
                )
            if not all(point.get(field) for field in _CATEGORY_FIELDS):
                raise ValueError(
                    f"entry {entry.get('window_name')} point {idx} is missing "
                    "pre_category/post_category (predicted by the LCC model)"
                )
            pre = date.fromisoformat(point["pre_change"])
            records.append(
                _make_record(
                    lon=float(point["lon"]),
                    lat=float(point["lat"]),
                    year=pre.year,
                    predicted_change_date=pre.isoformat(),
                    pre_category=point.get("pre_category"),
                    post_category=point.get("post_category"),
                    label=None,
                    entry=entry,
                    point_type="positive",
                    point_index=idx,
                )
            )
    return records


def build_records(entries: list[dict], mode: str) -> list[PointRecord]:
    """Build point records from annotation entries according to the mode."""
    if mode == MODE_EVALUATION:
        return _evaluation_records(entries)
    if mode == MODE_DEPLOYMENT:
        return _deployment_records(entries)
    raise ValueError(f"unknown mode: {mode}")


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Add points from a v2 annotation file to the VLM image database."
    )
    parser.add_argument(
        "--annotations", required=True, help="Path to the v2 annotation JSON file."
    )
    parser.add_argument(
        "--image-db-path", required=True, help="Root path of the rslearn image database."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the point set JSON."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[MODE_EVALUATION, MODE_DEPLOYMENT],
        help="evaluation (labeled points) or deployment (predicted points).",
    )
    parser.add_argument(
        "--group",
        default=image_database.DEFAULT_GROUP,
        help="Window group in the image database.",
    )
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    entries = json.loads(open(parsed.annotations).read())
    records = build_records(entries, parsed.mode)
    logger.info("Built %d point records (mode=%s)", len(records), parsed.mode)

    point_keys = list({(r.lon, r.lat, r.year) for r in records})
    created = image_database.add_points(
        parsed.image_db_path, point_keys, group=parsed.group
    )
    logger.info(
        "Added %d new windows (%d total unique points) to %s",
        len(created),
        len(point_keys),
        parsed.image_db_path,
    )

    point_set = PointSet(
        mode=parsed.mode,
        image_db_path=str(parsed.image_db_path),
        group=parsed.group,
        points=records,
    )
    point_set.save(parsed.output)
    logger.info("Wrote %d points to %s", len(records), parsed.output)


if __name__ == "__main__":
    main()
