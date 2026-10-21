"""Create the Q4 monocrop Studio project and add one task per selected event.

The events come from {out_dir}/events_for_annotation.geojson (select_for_annotation.py).
As in ../2026_09_25_amazon_forest_loss_labeling/create_studio_project.py, nothing about
the model prediction or the sampling stratum goes into Studio: tasks are shuffled
before numbering and task attributes only hold event_uid and country. The predictions
stay in events_for_annotation.geojson, matched through {out_dir}/studio_tasks.json.

The project has the same annotation fields as the earlier monocrop projects:
monoculture_tag (the seven raw classes, required) and confidence (high/medium/low),
and the same basemaps as Monocrop - Phase 2 (OSM, ESRI and Planet monthly mosaics,
Planet 2026-05 by default).

Each task:
- name "[#001] 2025-03-08 at -3.9997, -46.6966" (date, then lat, lon of the event
  polygon centroid),
- geometry: a 2560 m box centered at the centroid (as in the earlier monocrop projects),
- time range: the event date,
- one pending annotation with the (pixel-exact) event polygon and no labels.

Re-running skips tasks whose name already exists in the project.

Example:
    STUDIO_API_KEY=... python create_studio_project.py --out_dir /path/to/out --dry_run
"""

import argparse
import json
import math
import os
import random
import sys

import shapely
import shapely.geometry
import tqdm
from upath import UPath

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "..", "2026_09_25_amazon_forest_loss_labeling"
    )
)
from studio_api import Studio  # noqa: E402

ORGANIZATION_ID = "73c31b7f-af83-4da8-8b61-7c7accd03864"
PROJECT_NAME = "Monocrop Validation 2026-10"
PROJECT_DESCRIPTION = (
    "Q4 monocrop annotation of {num_events} GLAD forest loss events of at least 5 ha "
    "(Jan 2023 - Oct 2025, BO/BR/CO/EC/PE) predicted as agriculture, selected with the "
    "20260914 monocrop classifier. Predicted class is not shown."
)
BOX_SIZE_M = 2560
# Monocrop - Phase 2, whose basemap settings are copied.
BASEMAP_SOURCE_PROJECT_ID = "ffe01218-87d6-4ebe-85c8-70b5515b951a"
BASEMAP_KEYS = [
    "name",
    "url",
    "type",
    "tile_size",
    "max_zoom",
    "min_zoom",
    "display_order",
    "attribution",
    "default",
]
# (name, display name, color), as in the Monocrop - Peru project.
FIELDS = {
    "monoculture_tag": (
        "Monoculture Tag",
        True,
        [
            ("soybean", "Soybean", "#d900ff"),
            ("oil_palm", "Oil Palm", "#32d100"),
            ("mennonites_nonsoybean", "Mennonites Non-Soybean", "#ff0d00"),
            ("mennonites_soybean", "Mennonites Soybean", "#ff7f00"),
            ("other_agriculture", "Other Agriculture", "#000000"),
            ("pastures", "Pastures", "#79f391"),
            ("rice", "Rice", "#fffa00"),
        ],
    ),
    "confidence": (
        "Confidence",
        False,
        [
            ("high", "High", "#ff002b"),
            ("medium", "Medium", "#ffc400"),
            ("low", "Low", "#00ff14"),
        ],
    ),
}


def centroid_box(lon: float, lat: float) -> shapely.Polygon:
    """Make a BOX_SIZE_M box centered at (lon, lat)."""
    half_lat = BOX_SIZE_M / 2 / 111_320.0
    half_lon = BOX_SIZE_M / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return shapely.box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def ensure_fields(studio: Studio, project_id: str) -> None:
    """Create the labelset fields (with display names) if they do not exist yet."""
    settings = studio.get_project(project_id)["settings"]
    existing = {f["name"] for f in settings["annotation_metadata_fields"]}
    for field_name, (display_name, required, labels) in FIELDS.items():
        if field_name in existing:
            continue
        print(f"creating field {field_name}")
        labelset = studio.request(
            "POST",
            "/labelsets",
            json={
                "name": field_name,
                "display_name": display_name,
                "project_settings_id": settings["id"],
            },
        )["records"][0]
        studio.request(
            "POST",
            "/labels",
            json=[
                {
                    "name": name,
                    "display_name": label_display_name,
                    "color": color,
                    "labelset_id": labelset["id"],
                }
                for name, label_display_name, color in labels
            ],
        )
        studio.request(
            "POST",
            "/annotation_metadata_fields",
            json={
                "name": field_name,
                "display_name": display_name,
                "data_type": "labelset",
                "project_settings_id": settings["id"],
                "labelset_id": labelset["id"],
                "required": required,
            },
        )


def copy_basemaps(studio: Studio, project_id: str) -> None:
    """Make the project's basemaps match BASEMAP_SOURCE_PROJECT_ID (matched by name).

    Basemaps of the source project are created if missing, and existing ones with the
    same name are updated so default and display_order match. URLs are not printed
    since the Planet ones contain an API key.
    """
    source = studio.get_project(BASEMAP_SOURCE_PROJECT_ID)["settings"]["basemaps"]
    settings = studio.get_project(project_id)["settings"]
    existing = {basemap["name"]: basemap for basemap in settings["basemaps"]}
    for basemap in sorted(source, key=lambda b: b["display_order"]):
        data = {key: basemap[key] for key in BASEMAP_KEYS}
        current = existing.get(basemap["name"])
        if current is None:
            print(f"creating basemap {basemap['name']}")
            studio.request(
                "POST",
                "/basemaps",
                json={**data, "project_settings_id": settings["id"]},
            )
        elif any(current.get(key) != data[key] for key in BASEMAP_KEYS):
            print(f"updating basemap {basemap['name']}")
            studio.request("PUT", f"/basemaps/{current['id']}", json=data)


def main(cli_args: argparse.Namespace) -> None:
    """Create the project (if needed), fields, tasks and annotations."""
    out_dir = UPath(cli_args.out_dir)
    with (out_dir / "events_for_annotation.geojson").open() as f:
        events = json.load(f)["features"]

    # Shuffle so the task number does not reveal the sampling stratum.
    events.sort(key=lambda feat: feat["properties"]["event_uid"])
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
            print(name, feat["properties"]["event_uid"], feat["properties"]["country"])
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
                "description": PROJECT_DESCRIPTION.format(num_events=len(planned)),
                "organization_id": ORGANIZATION_ID,
            },
        )["records"][0]["id"]
        print(f"created project {project_id}")
    ensure_fields(studio, project_id)
    copy_basemaps(studio, project_id)

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
                    "attributes": {
                        "event_uid": props["event_uid"],
                        "country": props["country"],
                    },
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
                "metadata_values": [],
            },
        )["records"][0]
        mapping["tasks"][name] = {
            "task_id": task["id"],
            "annotation_id": annotation["id"],
            "event_uid": props["event_uid"],
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
