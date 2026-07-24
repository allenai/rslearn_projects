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

The STUDIO_API_KEY environment variable must be set. Run in any environment with
requests and tqdm (e.g. the rslearn venv):

    STUDIO_API_KEY=... python \
        rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/rename_studio_tasks.py \
        --project-id <PROJECT_ID> \
        --dry-run
"""

import argparse
import os
import random
import re
from datetime import datetime
from typing import Any

import requests
import tqdm

BASE_URL = "https://olmoearth.allenai.org/api/v1/"


def get_headers() -> dict[str, str]:
    """Get the headers to use for HTTP requests."""
    api_key = os.environ["STUDIO_API_KEY"]
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def get_tasks(project_id: str) -> list[dict[str, Any]]:
    """Get all tasks in a project, handling pagination."""
    cur_offset = 0
    tasks: list[dict[str, Any]] = []
    while True:
        response = requests.post(
            BASE_URL + "tasks/search",
            json={
                "project_id": {"eq": project_id},
                "offset": cur_offset,
            },
            headers=get_headers(),
            timeout=10,
        )
        if response.status_code != 200:
            print(response.text)
            raise Exception(f"got bad API response {response.status_code}")

        json_data = response.json()
        if len(json_data["records"]) == 0:
            break

        tasks.extend(json_data["records"])
        cur_offset += len(json_data["records"])

    return tasks


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


def rename_task(task_id: str, new_name: str, project_id: str) -> None:
    """Rename a single task via the Studio API (preserves geometry/time)."""
    response = requests.put(
        BASE_URL + f"tasks/{task_id}",
        json={"name": new_name, "project_id": project_id},
        headers=get_headers(),
        timeout=10,
    )
    if response.status_code != 200:
        print(response.text)
        raise Exception(f"got bad API response {response.status_code}")


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

    tasks = get_tasks(args.project_id)
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
        rename_task(task["id"], new_name, args.project_id)
    print(f"Renamed {len(renames)} tasks")


if __name__ == "__main__":
    main()
