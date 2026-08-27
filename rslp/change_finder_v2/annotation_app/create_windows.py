"""Create rslearn windows from a v2 annotation JSON file.

Reads the v2 JSON (list of dicts with projection, bounds, time_range, points,
window_name, group) and creates one rslearn Window per entry so that imagery
can be materialized via ``rslearn dataset prepare`` / ``rslearn dataset ingest``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection
from upath import UPath

DEFAULT_TIME_RANGE = (
    datetime(2016, 1, 1, tzinfo=timezone.utc),
    datetime(2026, 1, 1, tzinfo=timezone.utc),
)


def create_windows(
    v2_json_path: str,
    ds_path: str,
    time_range: tuple[datetime, datetime] = DEFAULT_TIME_RANGE,
) -> None:
    """Create rslearn windows from a v2 annotation JSON.

    Args:
        v2_json_path: Path to the v2 JSON file.
        ds_path: Path to the rslearn dataset (config.json must already exist).
        time_range: Time range for all windows (controls what imagery gets
            fetched). The per-entry ``time_range`` in the JSON is metadata about
            when negative points are valid, not the imagery time range.
    """
    with open(v2_json_path) as f:
        entries = json.load(f)

    dataset = Dataset(UPath(ds_path))

    created = 0
    for entry in entries:
        projection = Projection.deserialize(entry["projection"])
        bounds = tuple(entry["bounds"])

        window = Window(
            storage=dataset.storage,
            group=entry["group"],
            name=entry["window_name"],
            projection=projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()
        created += 1

    print(f"Created {created} windows in {ds_path}")


def main() -> None:
    """Create rslearn windows from a v2 annotation JSON."""
    parser = argparse.ArgumentParser(
        description="Create rslearn windows from a v2 annotation JSON."
    )
    parser.add_argument("v2_json_path", help="Path to the v2 JSON file.")
    parser.add_argument("ds_path", help="Path to the rslearn dataset.")
    parser.add_argument(
        "--time-start",
        default="2016-01-01",
        help="Default time range start (ISO date). Default: 2016-01-01",
    )
    parser.add_argument(
        "--time-end",
        default="2026-01-01",
        help="Default time range end (ISO date). Default: 2026-01-01",
    )
    args = parser.parse_args()

    time_range = (
        datetime.fromisoformat(args.time_start).replace(tzinfo=timezone.utc),
        datetime.fromisoformat(args.time_end).replace(tzinfo=timezone.utc),
    )
    create_windows(
        v2_json_path=args.v2_json_path,
        ds_path=args.ds_path,
        time_range=time_range,
    )


if __name__ == "__main__":
    main()
