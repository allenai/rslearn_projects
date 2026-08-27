"""Add change points from a v2 annotation file to the VLM image database.

Two modes, both of which only consider fully-labeled positive points (those with
``pre_change``, ``post_change``, ``first_date_change_noticeable``, ``pre_category``, and
``post_category`` all set); negative points and partially-labeled positives are skipped:

- ``all`` (default): add every fully-labeled positive point. Use this to categorize
  unlabeled changes.
- ``evaluation``: only add fully-labeled positive points that also have at least one
  hand-labeled ground-truth change category (``pre_change_category``,
  ``post_change_category``, or ``same_change_category``). Use this to quantitatively
  evaluate Gemini's accuracy.

The ground-truth change categories are always recorded on the point record when present,
regardless of mode; the mode only changes which points are included.

The key year used for each point is the midpoint year of its pre-change and post-change
dates.

This both creates the windows in the image database (so the user can run rslearn
prepare/materialize) and writes a JSON :class:`PointSet` summarizing the points.

Example:
    python -m rslp.change_finder_v2.vlm.category_tagger.add_points \
        --annotations annotations1.json annotations2.json \
        --image-db-path /weka/.../image_db \
        --output points.json \
        --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from rslp.change_finder_v2.vlm.image_db import image_database

from .schema import PointRecord, PointSet

logger = logging.getLogger(__name__)

# Modes for adding points.
MODE_ALL = "all"
MODE_EVALUATION = "evaluation"

# A positive point must have all of these set to be eligible.
REQUIRED_FIELDS = (
    "pre_change",
    "post_change",
    "first_date_change_noticeable",
    "pre_category",
    "post_category",
)

# Hand-labeled ground-truth fine-grained change categories on a positive point.
GT_CATEGORY_FIELDS = (
    "pre_change_category",
    "post_change_category",
    "same_change_category",
)


def _midpoint(d1: date, d2: date) -> date:
    """Return the calendar midpoint of two dates."""
    return date.fromordinal((d1.toordinal() + d2.toordinal()) // 2)


def build_records(entries: list[dict], mode: str = MODE_ALL) -> list[PointRecord]:
    """Build a record for every fully-labeled positive point in the entries.

    In ``evaluation`` mode, only points with at least one ground-truth change category
    set are included. Ground-truth categories are recorded whenever present.
    """
    records: list[PointRecord] = []
    skipped_unlabeled = 0
    for entry in entries:
        for idx, point in enumerate(entry.get("positive_points", [])):
            if not all(point.get(field) for field in REQUIRED_FIELDS):
                continue
            gt_pre = point.get("pre_change_category")
            gt_post = point.get("post_change_category")
            gt_same = point.get("same_change_category")
            if mode == MODE_EVALUATION and not (gt_pre or gt_post or gt_same):
                skipped_unlabeled += 1
                continue
            lon = float(point["lon"])
            lat = float(point["lat"])
            pre_change = date.fromisoformat(point["pre_change"])
            post_change = date.fromisoformat(point["post_change"])
            mid = _midpoint(pre_change, post_change)
            records.append(
                PointRecord(
                    lon=lon,
                    lat=lat,
                    year=mid.year,
                    window_name=image_database.window_name(lon, lat, mid.year),
                    predicted_change_date=mid.isoformat(),
                    pre_category=point.get("pre_category"),
                    post_category=point.get("post_category"),
                    annotation_group=entry.get("group", ""),
                    annotation_window_name=entry.get("window_name", ""),
                    point_index=idx,
                    pre_change=point["pre_change"],
                    post_change=point["post_change"],
                    first_observable=point["first_date_change_noticeable"],
                    gt_pre_change_category=gt_pre,
                    gt_post_change_category=gt_post,
                    gt_same_change_category=gt_same,
                )
            )
    if mode == MODE_EVALUATION:
        logger.info(
            "Skipped %d valid positive points with no ground-truth change category",
            skipped_unlabeled,
        )
    return records


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Add points from a v2 annotation file to the VLM image database."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        nargs="+",
        help="Path(s) to the v2 annotation JSON file(s).",
    )
    parser.add_argument(
        "--image-db-path",
        required=True,
        help="Root path of the rslearn image database.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the point set JSON."
    )
    parser.add_argument(
        "--mode",
        default=MODE_ALL,
        choices=[MODE_ALL, MODE_EVALUATION],
        help="all (every valid positive) or evaluation (only ground-truth-labeled).",
    )
    parser.add_argument(
        "--group",
        default=image_database.DEFAULT_GROUP,
        help="Window group in the image database.",
    )
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    entries: list[dict] = []
    for path in parsed.annotations:
        entries.extend(json.loads(open(path).read()))
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
        image_db_path=str(parsed.image_db_path),
        group=parsed.group,
        points=records,
    )
    point_set.save(parsed.output)
    logger.info("Wrote %d points to %s", len(records), parsed.output)


if __name__ == "__main__":
    main()
