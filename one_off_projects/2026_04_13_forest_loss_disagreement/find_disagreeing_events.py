"""Find forest loss events where old-label and new-label models disagree."""

import argparse
import hashlib
import json


def geom_hash(geom: dict) -> str:
    return hashlib.sha256(json.dumps(geom, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old", default="/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/2026_04_13_forest_loss_disagreement/result_old.geojson", help="Old-label predictions"
    )
    parser.add_argument(
        "--new", default="/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/2026_04_13_forest_loss_disagreement/result_new.geojson", help="New-label predictions"
    )
    parser.add_argument(
        "--output",
        default="/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/2026_04_13_forest_loss_disagreement/disagreements.geojson",
        help="Output GeoJSON with disagreements",
    )
    args = parser.parse_args()

    with open(args.old) as f:
        old_fc = json.load(f)
    with open(args.new) as f:
        new_fc = json.load(f)

    old_by_hash = {geom_hash(feat["geometry"]): feat for feat in old_fc["features"]}
    new_by_hash = {geom_hash(feat["geometry"]): feat for feat in new_fc["features"]}

    old_only = set(old_by_hash) - set(new_by_hash)
    new_only = set(new_by_hash) - set(old_by_hash)
    if old_only or new_only:
        raise ValueError(
            f"Geometry mismatch: {len(old_only)} old-only, {len(new_only)} new-only"
        )

    disagreements = []
    for h, old_feat in old_by_hash.items():
        new_feat = new_by_hash[h]
        old_label = old_feat["properties"]["new_label"]
        new_label = new_feat["properties"]["new_label"]
        if old_label == new_label:
            continue
        disagreements.append(
            {
                "type": "Feature",
                "geometry": old_feat["geometry"],
                "properties": {
                    "old_model_prediction": old_label,
                    "new_model_prediction": new_label,
                    "old_probs": old_feat["properties"].get("probs"),
                    "new_probs": new_feat["properties"].get("probs"),
                    "oe_start_time": old_feat["properties"].get("oe_start_time"),
                    "oe_end_time": old_feat["properties"].get("oe_end_time"),
                },
            }
        )

    output_fc = {"type": "FeatureCollection", "features": disagreements}
    with open(args.output, "w") as f:
        json.dump(output_fc, f)

    print(f"Total events compared: {len(old_by_hash)}")
    print(f"Disagreements written: {len(disagreements)} -> {args.output}")


if __name__ == "__main__":
    main()
