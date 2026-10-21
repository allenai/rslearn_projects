"""Export every labeled monocrop annotation in Studio, for the 1 km exclusion rule.

New Q4 monocrop tasks must be at least 1 km from any existing monocrop label, so this
writes the polygon of every annotation with a monoculture_tag in the monocrop Studio
projects (any status or confidence), regardless of whether it is used for training.
Unlabeled Phase 2 tasks are not exported, so new tasks may duplicate them.

Output: {out_dir}/labeled_monocrop_annotations.geojson.

Example:

    STUDIO_API_KEY=... python export_labeled_annotations.py --out_dir /path/to/out
"""

import argparse
import json
from collections import Counter

import shapely
import shapely.geometry
from upath import UPath

from rslp.forest_loss_driver.monocrop_classifier.studio import StudioClient

# Monocrop - Peru, Bolivia, Ecuador, Phase 2 (org 73c31b7f).
PROJECT_IDS = [
    "30b9bbb2-1ac9-4cf6-baa5-1b77ce79881c",
    "dd3e9ecb-4060-4a9b-af71-05b4bf8ad747",
    "8188a029-fd50-4670-a6f2-243afc3e1b83",
    "ffe01218-87d6-4ebe-85c8-70b5515b951a",
]
LABEL_FIELD = "monoculture_tag"


def main(out_dir: UPath) -> None:
    """Write labeled_monocrop_annotations.geojson."""
    client = StudioClient()
    features = []
    counts: Counter[tuple[str, str]] = Counter()
    for project_id in PROJECT_IDS:
        project_name = client.get_project(project_id)["name"]
        for annotation in client.get_annotations(project_id):
            values = {
                value["name"]: value.get("label_name")
                for value in annotation.get("metadata_values") or []
            }
            label = values.get(LABEL_FIELD)
            if not label or not annotation.get("geom_wkt"):
                continue
            geometry = shapely.from_wkt(annotation["geom_wkt"])
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "project": project_name,
                        "annotation_id": annotation["id"],
                        "task_id": annotation["task_id"],
                        "status": annotation.get("status"),
                        LABEL_FIELD: label,
                        "confidence": values.get("confidence"),
                    },
                    "geometry": shapely.geometry.mapping(geometry),
                }
            )
            counts[(project_name, label)] += 1

    for (project_name, label), count in sorted(counts.items()):
        print(f"{project_name}\t{label}\t{count}")
    print(f"total {len(features)} labeled annotations")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labeled_monocrop_annotations.geojson").open("w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    main(UPath(parser.parse_args().out_dir))
