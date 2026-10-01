"""Upload a v2 LCC annotation JSON into an OlmoEarth Studio project.

Each entry becomes a task named "<5-digit JSON index> <window_name>", and each
point an annotation on that task. Missing fields and labels are added to the
project first (see schema.py). Tasks whose name already exists are skipped, so
the script is safe to re-run after an interruption; an existing task whose
annotation count differs from the JSON is reported rather than touched, since
it may have been edited in the lab since.

    STUDIO_API_KEY=... python -m rslp.olmoearth_lcc.studio.upload_to_studio \
        --json annotations.json --project-id <PROJECT_ID> [--dry-run]

Set STUDIO_API_URL (or --base-url) to target a non-production Studio, e.g.
http://localhost:8000/api/v1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tqdm
from upath import UPath

from rslp.olmoearth_lcc.studio.client import StudioClient
from rslp.olmoearth_lcc.studio.mapping import (
    KNOWN_ENTRY_KEYS,
    entry_to_task_payload,
    point_to_annotation_payload,
    task_name,
    unmapped_point_keys,
)
from rslp.olmoearth_lcc.studio.schema import (
    ALL_FIELDS,
    NEGATIVE,
    POSITIVE,
    ProjectSchema,
    ensure_template,
    required_labels,
    resolve_schema,
)


def iter_points(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(kind, point) pairs in upload order: positives, then negatives."""
    return [(POSITIVE, p) for p in entry.get("positive_points", [])] + [
        (NEGATIVE, p) for p in entry.get("negative_points", [])
    ]


def report_entries(entries: list[dict[str, Any]], schema: ProjectSchema) -> None:
    """Print what the upload will add to the template and lose from the JSON."""
    extra_keys: Counter[str] = Counter()
    dropped_point_keys: Counter[str] = Counter()
    for entry in entries:
        extra_keys.update(key for key in entry if key not in KNOWN_ENTRY_KEYS)
        for kind, point in iter_points(entry):
            dropped_point_keys.update(unmapped_point_keys(point, kind))
    if extra_keys:
        print(f"Extra entry keys kept in task attributes: {dict(extra_keys)}")
    if dropped_point_keys:
        print(
            "WARNING: point keys with no Studio field will be dropped: "
            f"{dict(dropped_point_keys)}"
        )
    missing_fields = [name for name in ALL_FIELDS if name not in schema.field_ids]
    if missing_fields:
        print(f"Fields to create: {missing_fields}")
    for name, labels in required_labels(entries).items():
        if name not in schema.field_ids:
            continue
        missing = [label for label in labels if label not in schema.label_ids[name]]
        if missing:
            print(f"Labels to add to {name!r}: {missing}")


def upload_entry(
    client: StudioClient,
    project_id: str,
    schema: ProjectSchema,
    index: int,
    entry: dict[str, Any],
) -> None:
    """Create one entry's task and then its annotations, in order."""
    task = client.create(
        "/tasks", {**entry_to_task_payload(index, entry), "project_id": project_id}
    )
    for kind, point in iter_points(entry):
        client.create(
            "/annotations",
            point_to_annotation_payload(
                point,
                kind,
                schema.field_ids,
                schema.label_ids,
                task["id"],
                entry.get("time_range"),
            ),
        )


def check_existing(
    client: StudioClient,
    project_id: str,
    entries: list[dict[str, Any]],
    existing_tasks: dict[str, str],
) -> None:
    """Warn about existing tasks whose annotation count differs from the JSON."""
    counts: Counter[str] = Counter(
        annotation["task_id"]
        for annotation in client.search_all(
            "/annotations/search", {"project_id": {"eq": project_id}}
        )
    )
    mismatched = []
    for index, entry in enumerate(entries):
        task_id = existing_tasks.get(task_name(index, entry["window_name"]))
        if task_id is None:
            continue
        expected = len(iter_points(entry))
        if counts[task_id] != expected:
            mismatched.append((index, counts[task_id], expected))
    if mismatched:
        print(
            f"WARNING: {len(mismatched)} existing tasks have a different number of "
            "points than the JSON (left unchanged); first few (index, studio, json): "
            f"{mismatched[:10]}"
        )


def main() -> None:
    """Upload the JSON's entries as Studio tasks and annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="v2 annotation JSON path.")
    parser.add_argument("--project-id", required=True, help="Studio project ID.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Studio API root (default: $STUDIO_API_URL, else production).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Entries uploaded in parallel (each entry's points stay sequential).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created without writing to Studio.",
    )
    args = parser.parse_args()

    with UPath(args.json).open() as f:
        entries: list[dict[str, Any]] = json.load(f)
    num_points = sum(len(iter_points(entry)) for entry in entries)
    print(f"Loaded {len(entries)} entries with {num_points} points from {args.json}")
    for index, entry in enumerate(entries):
        # Fail on a malformed entry before anything is written.
        entry_to_task_payload(index, entry)

    client = StudioClient(args.base_url)
    project = client.get_project(args.project_id)
    print(f"Project: {project['name']} ({client.base_url})")

    existing_tasks = {
        task["name"]: task["id"]
        for task in client.search_all(
            "/tasks/search", {"project_id": {"eq": args.project_id}}
        )
    }
    todo = [
        (index, entry)
        for index, entry in enumerate(entries)
        if task_name(index, entry["window_name"]) not in existing_tasks
    ]
    print(
        f"{len(existing_tasks)} tasks already in the project; "
        f"{len(todo)} entries to upload"
    )

    report_entries(entries, resolve_schema(project))
    if args.dry_run:
        print("Dry run: nothing written")
        return

    schema = ensure_template(client, args.project_id, entries)
    if existing_tasks:
        check_existing(client, args.project_id, entries, existing_tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(upload_entry, client, args.project_id, schema, index, entry)
            for index, entry in todo
        ]
        for future in tqdm.tqdm(futures, desc="Uploading entries"):
            future.result()
    print(f"Uploaded {len(todo)} entries")


if __name__ == "__main__":
    main()
