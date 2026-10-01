"""Rename Studio tasks to a counter + lat/lon + date naming scheme.

This is step 5 of the monocrop initial setup. After the annotation sets have been
uploaded to an ES Studio project, this script renames every task from the upload-time
name (e.g. ``Oilpalmperu_agriculture_large_-5.767112_-77.139067``) to a compact name:

    [#001] (-5.7671, -77.1391) at 2022-01-08

where:
  - ``#001`` is a 1-based counter over a *random shuffle* of the project's tasks
    (zero-padded to 3 digits; no project in this round has more than 999 tasks),
  - ``(-5.7671, -77.1391)`` is ``(lat, lon)`` parsed from the trailing two
    underscore-separated floats of the original name, rounded to 4 decimals,
  - ``2022-01-08`` is the task's ``start_time`` (date only).

Tasks that are already renamed (name starts with ``[#``) are skipped, and the counter
continues after the highest existing ``[#NNN]`` so numbers are not reused. This makes
the script safe to re-run after uploading additional tasks to the same project.

The STUDIO_API_KEY environment variable must be set. Run from the rslearn_projects
root in an environment with rslp installed:

    STUDIO_API_KEY=... python -m \
        rslp.forest_loss_driver.scripts.monocrop_initial_setup_20260624.rename_studio_tasks \
        --project-id <PROJECT_ID> \
        --dry-run
"""

import argparse
import random
import re
from datetime import datetime

import tqdm

from rslp.utils.studio import StudioClient


def parse_lat_lon(name: str) -> tuple[float, float]:
    """Parse (lat, lon) from the trailing two underscore-separated floats of a name."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse lat/lon from task name: {name!r}")
    try:
        lat = float(parts[1])
        lon = float(parts[2])
    except ValueError as e:
        raise ValueError(f"cannot parse lat/lon from task name: {name!r}") from e
    return lat, lon


def parse_date(start_time: str | None, name: str) -> str:
    """Parse a YYYY-MM-DD date from a task's ISO-8601 start_time."""
    if not start_time:
        raise ValueError(f"task {name!r} has no start_time")
    return datetime.fromisoformat(start_time).date().isoformat()


def make_new_name(counter: int, lat: float, lon: float, date: str) -> str:
    """Build the new task name."""
    return f"[#{counter:03d}] ({lat:.4f}, {lon:.4f}) at {date}"


def main() -> None:
    """Parse arguments and rename all tasks in the project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="ES Studio project ID.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random shuffle that assigns counters (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without calling the API.",
    )
    args = parser.parse_args()

    client = StudioClient()
    tasks = client.get_tasks(args.project_id)
    print(f"Found {len(tasks)} tasks")

    # Skip tasks that already have the new naming scheme (name starts with "[#").
    already = [t for t in tasks if t["name"].startswith("[#")]
    todo = [t for t in tasks if not t["name"].startswith("[#")]
    if already:
        print(f"Skipping {len(already)} already-renamed task(s)")

    # Continue the counter after the highest existing "[#NNN]" so we don't reuse numbers.
    start = 1
    for t in already:
        m = re.match(r"\[#(\d+)\]", t["name"])
        if m is not None:
            start = max(start, int(m.group(1)) + 1)

    # Assign counters over a random shuffle of the remaining tasks.
    rng = random.Random(args.seed)
    rng.shuffle(todo)

    renames = []
    for i, task in enumerate(todo, start=start):
        lat, lon = parse_lat_lon(task["name"])
        date = parse_date(task.get("start_time"), task["name"])
        new_name = make_new_name(i, lat, lon, date)
        renames.append((task, new_name))

    if args.dry_run:
        for task, new_name in renames:
            print(f"{task['name']}  ->  {new_name}")
        print(f"Dry run: would rename {len(renames)} tasks")
        return

    for task, new_name in tqdm.tqdm(renames, desc="Renaming"):
        # PUT preserves the task's geometry and time range.
        client.update_task(
            task["id"], {"name": new_name, "project_id": args.project_id}
        )
    print(f"Renamed {len(renames)} tasks")


if __name__ == "__main__":
    main()
