"""Match Studio prediction outputs back to the selected events.

Studio outputs only keep the centroid Point, the time range, the predicted label
(new_label) and the class probabilities (probs). Here we match each output to the
selected event whose centroid it was submitted as (the coordinates are identical), and
write one GeoJSON per model with the original event polygon and properties plus:

- event_id: index of the event in selected_events.geojson.
- category: predicted class (renamed from new_label as in the deploy pipeline).
- category_prob: probability of the predicted class.
- probs: probabilities over CLASSES, in that order.

Outputs:
    {out_dir}/events_with_new_model_predictions.geojson (utm model)
    {out_dir}/events_with_old_model_predictions.geojson (peru_phase2 model)

Example:

    python merge_predictions.py --out_dir /path/to/out
"""

import argparse
import json

import numpy as np
import shapely.geometry
from scipy.spatial import cKDTree
from upath import UPath

# Class order of probs, from olmoearth_run_data/forest_loss_driver/model.yaml. We
# verify that argmax(probs) matches new_label for every output.
CLASSES = [
    "agriculture",
    "mining",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
    "none",
]

# Output name -> job name prefix in studio_job_ids.json (see studio_jobs.py).
MODELS = {
    "new": "amazon_labeling_20260925_utm_chunk_",
    "old": "amazon_labeling_20260925_peru_phase2_chunk_",
}


def main(out_dir: UPath) -> None:
    """Write the per-model events_with_*_model_predictions.geojson files."""
    with (out_dir / "selected_events.geojson").open() as f:
        events = json.load(f)["features"]
    # The submitted points were the polygon centroids (simplify_features_to_centroids).
    centroids = np.array(
        [shapely.geometry.shape(feat["geometry"]).centroid.coords[0] for feat in events]
    )
    tree = cKDTree(centroids)

    with (out_dir / "studio_job_ids.json").open() as f:
        job_names = list(json.load(f).keys())

    for model_name, prefix in MODELS.items():
        outputs = []
        for job_name in job_names:
            if not job_name.startswith(prefix):
                continue
            with (out_dir / "studio_outputs" / f"{job_name}.geojson").open() as f:
                outputs.extend(json.load(f)["features"])

        dists, idxs = tree.query(
            np.array([feat["geometry"]["coordinates"] for feat in outputs])
        )
        if dists.max() > 1e-9:
            raise ValueError(f"{model_name}: some outputs do not match an input point")
        if len(set(idxs.tolist())) != len(idxs):
            raise ValueError(f"{model_name}: multiple outputs matched the same event")

        merged = []
        for output, event_id in zip(outputs, idxs.tolist()):
            event = events[event_id]
            probs = output["properties"]["probs"]
            category = output["properties"]["new_label"]
            if CLASSES[int(np.argmax(probs))] != category:
                raise ValueError(f"{model_name}: argmax(probs) != new_label {output}")
            if (
                output["properties"]["oe_start_time"][:10]
                != event["properties"]["oe_start_time"][:10]
            ):
                raise ValueError(f"{model_name}: date mismatch for event {event_id}")
            properties = dict(event["properties"])
            properties["event_id"] = event_id
            properties["category"] = category
            properties["category_prob"] = max(probs)
            properties["probs"] = probs
            merged.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": event["geometry"],
                }
            )
        merged.sort(key=lambda feat: feat["properties"]["event_id"])

        missing = sorted(set(range(len(events))) - set(idxs.tolist()))
        print(
            f"{model_name}: {len(merged)} of {len(events)} events have predictions, "
            f"missing event_ids {missing}"
        )
        out_fname = out_dir / f"events_with_{model_name}_model_predictions.geojson"
        with out_fname.open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {"classes": CLASSES},
                    "features": merged,
                },
                f,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    main(UPath(parser.parse_args().out_dir))
