"""Add the new model's predicted category to the annotations in the validation project.

Run this only after the validation annotation is finished: until then the predicted
category is deliberately absent from Studio to avoid biasing annotators (see
create_studio_project.py).

It creates the category labelset field (same classes as validate) if needed, then for
each task in {out_dir}/studio_tasks.json fetches the current annotation and writes it
back with the category label added. All existing metadata values (in particular
validate), the status, annotator, geometry and times are preserved. Annotations that
already have a category are skipped.

Example:

    STUDIO_API_KEY=... python sync_predictions_to_studio.py --out_dir /path/to/out --dry_run
"""

import argparse
import json

import tqdm
from upath import UPath

from studio_api import CLASS_COLORS, Studio


def main(cli_args: argparse.Namespace) -> None:
    """Add category to every annotation in the project."""
    out_dir = UPath(cli_args.out_dir)
    with (out_dir / "studio_tasks.json").open() as f:
        mapping = json.load(f)
    with (out_dir / "events_for_annotation.geojson").open() as f:
        category_by_event = {
            feat["properties"]["event_id"]: feat["properties"]["category"]
            for feat in json.load(f)["features"]
        }
    project_id = mapping["project_id"]
    studio = Studio()

    if cli_args.dry_run:
        field_id, label_ids = None, {}
    else:
        field_id, label_ids = studio.ensure_labelset_field(
            project_id, "category", CLASS_COLORS
        )

    num_updated = num_skipped = 0
    for name, task in tqdm.tqdm(sorted(mapping["tasks"].items())):
        annotations = studio.search_all(
            "annotations", {"task_id": {"eq": task["task_id"]}}
        )
        if len(annotations) != 1:
            print(f"{name}: expected one annotation but got {len(annotations)}, skipping")
            num_skipped += 1
            continue
        annotation = annotations[0]
        category = category_by_event[task["event_id"]]
        if any(m["name"] == "category" for m in annotation["metadata_values"]):
            num_skipped += 1
            continue
        if cli_args.dry_run:
            print(f"[dry run] {name}: would set category={category}")
            num_updated += 1
            continue

        metadata_values = [
            {
                "metadata_field_id": m["metadata_field_id"],
                "value": m["value"],
                "label_id": m["label_id"],
            }
            for m in annotation["metadata_values"]
        ]
        metadata_values.append(
            {"metadata_field_id": field_id, "value": None, "label_id": label_ids[category]}
        )
        body = {
            "id": annotation["id"],
            "status": annotation["status"],
            "start_time": annotation["start_time"],
            "end_time": annotation["end_time"],
            "geom": annotation["geom_wkt"],
            "task_id": annotation["task_id"],
            "metadata_values": metadata_values,
        }
        if annotation.get("annotator_id"):
            body["annotator_id"] = annotation["annotator_id"]
        studio.request("PUT", f"/annotations/{annotation['id']}", json=body)
        num_updated += 1
    print(f"updated {num_updated}, skipped {num_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
