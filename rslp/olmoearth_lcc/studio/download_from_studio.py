"""Download an OlmoEarth Studio LCC project back to a v2 annotation JSON.

Inverse of upload_to_studio.py: entries come out in task-name (JSON index)
order, in the format lcc_model/prepare.py reads, with empty fields omitted.

    STUDIO_API_KEY=... python -m rslp.olmoearth_lcc.studio.download_from_studio \
        --project-id <PROJECT_ID> --out annotations.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from rslearn.utils.fsspec import open_atomic
from upath import UPath

from rslp.olmoearth_lcc.studio.client import StudioClient
from rslp.olmoearth_lcc.studio.mapping import (
    task_and_annotations_to_entry,
    task_index,
)
from rslp.olmoearth_lcc.studio.schema import resolve_schema


def download_entries(client: StudioClient, project_id: str) -> list[dict[str, Any]]:
    """All of the project's tasks as v2 entries, sorted by JSON index."""
    schema = resolve_schema(client.get_project(project_id))

    tasks = list(
        client.search_all(
            "/tasks/search",
            {
                "project_id": {"eq": project_id},
                "include_images": False,
                "include_annotation_metadata": False,
            },
        )
    )
    print(f"Fetched {len(tasks)} tasks")

    annotations_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    num_annotations = 0
    for annotation in client.search_all(
        "/annotations/search", {"project_id": {"eq": project_id}}
    ):
        if annotation.get("task_id"):
            annotations_by_task[annotation["task_id"]].append(annotation)
            num_annotations += 1
    print(f"Fetched {num_annotations} task annotations")

    indexed = []
    for task in tasks:
        try:
            index = task_index(task["name"])
        except ValueError:
            print(f"WARNING: skipping task {task['name']!r} (no JSON index in name)")
            continue
        indexed.append((index, task))
    indexed.sort(key=lambda pair: pair[0])

    indices = [index for index, _ in indexed]
    if len(set(indices)) != len(indices):
        print("WARNING: some JSON indices appear on more than one task")

    return [
        task_and_annotations_to_entry(
            task,
            annotations_by_task[task["id"]],
            schema.field_names,
            schema.label_names,
        )
        for _, task in indexed
    ]


def main() -> None:
    """Write the project's annotations as a v2 JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="Studio project ID.")
    parser.add_argument("--out", required=True, help="Output v2 JSON path.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Studio API root (default: $STUDIO_API_URL, else production).",
    )
    args = parser.parse_args()

    client = StudioClient(args.base_url)
    entries = download_entries(client, args.project_id)
    with open_atomic(UPath(args.out), "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
