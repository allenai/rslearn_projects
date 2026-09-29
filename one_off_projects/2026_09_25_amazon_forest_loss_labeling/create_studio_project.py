"""Create the Studio validation project and add one task per event for annotation.

The events come from {out_dir}/events_for_annotation.geojson (select_for_annotation.py).
To avoid biasing annotators, nothing about the model prediction or the sampling
stratum goes into Studio: the project only has the country and validate labelset
fields, tasks are shuffled before numbering, and task attributes only hold event_id.
The predicted category is added after annotation by sync_predictions_to_studio.py.

Each task:
- name "[#001] 2025-03-08 at -3.9997, -46.6966" (date, then lat, lon of the event
  polygon centroid),
- geometry: a 1280 m (128 px at 10 m) box centered at the centroid,
- time range: the event date,
- one pending annotation with the event polygon and country set.

The mapping from task to event is written to {out_dir}/studio_tasks.json. Re-running
skips tasks whose name already exists in the project.

Example:

    STUDIO_API_KEY=... python create_studio_project.py --out_dir /path/to/out \
        --project_id 542e4955-ae9d-4c34-9e22-ad0ddaea17f2
"""

import argparse
import json
import math
import random

import shapely
import shapely.geometry
import tqdm
from upath import UPath

from studio_api import CLASS_COLORS, COUNTRY_COLORS, Studio

ORGANIZATION_ID = "73c31b7f-af83-4da8-8b61-7c7accd03864"
PROJECT_NAME = "Forest Loss Driver Validation 2026-09"
PROJECT_DESCRIPTION = (
    "Validation annotation of 538 GLAD forest loss events (Jan 2025 - Jul 2026) "
    "selected with the 20260924 utm forest loss driver model. Predicted category is "
    "added only after annotation."
)
BOX_SIZE_M = 1280


def centroid_box(lon: float, lat: float) -> shapely.Polygon:
    """Make a BOX_SIZE_M box centered at (lon, lat)."""
    half_lat = BOX_SIZE_M / 2 / 111_320.0
    half_lon = BOX_SIZE_M / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return shapely.box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def main(cli_args: argparse.Namespace) -> None:
    """Create the project (if needed), fields, tasks and annotations."""
    out_dir = UPath(cli_args.out_dir)
    with (out_dir / "events_for_annotation.geojson").open() as f:
        events = json.load(f)["features"]

    # Shuffle so the task number does not reveal the sampling stratum.
    events.sort(key=lambda feat: feat["properties"]["event_id"])
    random.Random(cli_args.seed).shuffle(events)
    planned = []
    for counter, feat in enumerate(events, start=1):
        shp = shapely.geometry.shape(feat["geometry"])
        centroid = shp.centroid
        date = feat["properties"]["oe_start_time"][:10]
        name = f"[#{counter:03d}] {date} at {centroid.y:.4f}, {centroid.x:.4f}"
        planned.append((name, feat, shp, centroid))

    if cli_args.dry_run:
        for name, feat, _, _ in planned[:10]:
            print(name, feat["properties"]["event_id"], feat["properties"]["country"])
        print(f"dry run: would create {len(planned)} tasks")
        return

    studio = Studio()
    project_id = cli_args.project_id
    if project_id is None:
        project_id = studio.request(
            "POST",
            "/projects",
            json={
                "name": PROJECT_NAME,
                "description": PROJECT_DESCRIPTION,
                "organization_id": ORGANIZATION_ID,
            },
        )["records"][0]["id"]
        print(f"created project {project_id}")

    country_field_id, country_label_ids = studio.ensure_labelset_field(
        project_id, "country", COUNTRY_COLORS
    )
    studio.ensure_labelset_field(project_id, "validate", CLASS_COLORS)

    mapping_fname = out_dir / "studio_tasks.json"
    mapping = {"project_id": project_id, "tasks": {}}
    if mapping_fname.exists():
        with mapping_fname.open() as f:
            mapping = json.load(f)
        assert mapping["project_id"] == project_id
    existing = {
        task["name"]: task
        for task in studio.search_all("tasks", {"project_id": {"eq": project_id}})
    }
    print(f"{len(existing)} tasks already in project")

    todo = planned[: cli_args.limit] if cli_args.limit else planned
    for name, feat, shp, centroid in tqdm.tqdm(todo):
        props = feat["properties"]
        if name in existing:
            task = existing[name]
            # Skip tasks that already have their annotation (e.g. from a previous
            # run); otherwise add the missing annotation below.
            if studio.search_all("annotations", {"task_id": {"eq": task["id"]}}):
                continue
        else:
            task = studio.request(
                "POST",
                "/tasks",
                json={
                    "name": name,
                    "project_id": project_id,
                    "geom": centroid_box(centroid.x, centroid.y).wkt,
                    "start_time": props["oe_start_time"],
                    "end_time": props["oe_end_time"],
                    "attributes": {"event_id": props["event_id"]},
                },
            )["records"][0]
        annotation = studio.request(
            "POST",
            "/annotations",
            json={
                "status": "pending",
                "geom": shp.wkt,
                "task_id": task["id"],
                "start_time": props["oe_start_time"],
                "end_time": props["oe_end_time"],
                "metadata_values": [
                    {
                        "metadata_field_id": country_field_id,
                        "value": None,
                        "label_id": country_label_ids[props["country"].lower()],
                    }
                ],
            },
        )["records"][0]
        mapping["tasks"][name] = {
            "task_id": task["id"],
            "annotation_id": annotation["id"],
            "event_id": props["event_id"],
        }
        # Save after every task so an interrupted run keeps its mapping.
        with mapping_fname.open("w") as f:
            json.dump(mapping, f, indent=1)
    print(f"{len(mapping['tasks'])} tasks in {mapping_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--project_id", default=None, help="existing project (else create a new one)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only create N tasks")
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
