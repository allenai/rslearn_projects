"""Build the human-annotation manifest from per-scene detection JSONs.

Joins the detection JSONs written by the predict pipeline (run with
include_rejected=True so classifier-rejected candidates are present) against the
scene sample CSV from sample_scenes.py, caps detections per scene (keeping
confident/ambiguous candidates before sampling likely negatives), and writes one
manifest row per detection with everything an annotator needs to acquire the
imagery themselves and label it:

  * detection_id, lat/lon (WGS84), acquisition timestamp
  * scene_id, WRS-2 path/row, S3 prefix of the T1/T2 product (usgs-landsat bucket)
  * pixel col/row + projection CRS, crop image path if crops were written
  * detector score, classifier label + prob(correct), stratum/region/cloud metadata
  * empty human_label / annotator / notes columns to fill in

Rows are ordered for review efficiency: classifier-confident detections first
(each one an annotator rejects is a gold hard negative), then ambiguous ones,
then likely negatives (batch-confirmable).

Usage:
    python -m rslp.landsat_vessels.scripts.make_annotation_manifest \
        --scene_csv /weka/dfive-default/yawenz/landsat/scene_sample_v1.csv \
        --json_dir /weka/dfive-default/yawenz/landsat/round1_detections/json/ \
        --out_prefix /weka/dfive-default/yawenz/landsat/round1_manifest
"""

import argparse
import csv
import json
import os
import random

PRIORITY_BUCKETS = [
    # (name, min prob). Reviewed in this order.
    ("confident_vessel", 0.85),
    ("ambiguous", 0.40),
    ("likely_negative", 0.0),
]

MANIFEST_FIELDS = [
    "detection_id",
    "latitude",
    "longitude",
    "ts",
    "scene_id",
    "wrs_path",
    "wrs_row",
    "s3_prefix",
    "pixel_col",
    "pixel_row",
    "crs",
    "detector_score",
    "classifier_label",
    "classifier_prob_correct",
    "review_bucket",
    "stratum",
    "region",
    "scene_cloud_cover",
    "crop_fname",
    "human_label",
    "annotator",
    "notes",
]


def bucket_for(prob: float | None) -> str:
    """Review bucket for a detection's classifier probability."""
    if prob is None:
        return "ambiguous"
    for name, min_prob in PRIORITY_BUCKETS:
        if prob >= min_prob:
            return name
    return "likely_negative"


def main() -> None:
    """Build the annotation manifest CSV + GeoJSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument(
        "--json_dir",
        required=True,
        help="directory of per-scene detection JSONs named <scene_id>.json",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="writes <out_prefix>.csv and <out_prefix>.geojson",
    )
    parser.add_argument("--cap_per_scene", type=int, default=75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.scene_csv) as f:
        scenes = {row["scene_id"]: row for row in csv.DictReader(f)}

    rows = []
    missing, over_cap_dropped = [], 0
    for scene_id, scene in scenes.items():
        json_path = os.path.join(args.json_dir, f"{scene_id}.json")
        if not os.path.exists(json_path):
            missing.append(scene_id)
            continue
        with open(json_path) as f:
            detections = json.load(f)

        if len(detections) > args.cap_per_scene:
            # Bucket-stratified cap: a stormy/icy scene can emit hundreds of
            # candidates, and a uniform sample would be dominated by easy
            # low-prob rejects. Keep the valuable ones first (confident, then
            # ambiguous) and fill whatever budget remains with a random sample
            # of likely negatives.
            by_bucket: dict[str, list] = {name: [] for name, _ in PRIORITY_BUCKETS}
            for det in detections:
                by_bucket[bucket_for(det.get("classifier_prob_correct"))].append(det)
            kept: list[dict] = []
            for name, _ in PRIORITY_BUCKETS:
                budget = args.cap_per_scene - len(kept)
                if budget <= 0:
                    break
                group = by_bucket[name]
                if len(group) > budget:
                    group = rng.sample(group, budget)
                kept.extend(group)
            over_cap_dropped += len(detections) - len(kept)
            detections = kept

        for det in detections:
            prob = det.get("classifier_prob_correct")
            rows.append(
                {
                    "detection_id": f"{scene_id}_{det['col']}_{det['row']}",
                    "latitude": round(det["latitude"], 6),
                    "longitude": round(det["longitude"], 6),
                    "ts": det.get("ts"),
                    "scene_id": scene_id,
                    "wrs_path": scene["path"],
                    "wrs_row": scene["row"],
                    "s3_prefix": f"s3://usgs-landsat/{scene['blob_path']}",
                    "pixel_col": det["col"],
                    "pixel_row": det["row"],
                    "crs": det["projection"].get("crs"),
                    "detector_score": det["score"],
                    "classifier_label": det.get("classifier_label"),
                    "classifier_prob_correct": prob,
                    "review_bucket": bucket_for(prob),
                    "stratum": scene["stratum"],
                    "region": scene["region"],
                    "scene_cloud_cover": scene["cloud_cover"],
                    # The Landsat pipeline stores per-visualization crops in
                    # crop_fnames ({"rgb": ..., "b8": ...}); older outputs used
                    # a single crop_fname.
                    "crop_fname": (det.get("crop_fnames") or {}).get("rgb")
                    or det.get("crop_fname"),
                    "human_label": "",
                    "annotator": "",
                    "notes": "",
                }
            )

    # Review order: confident first, then ambiguous, then likely negatives;
    # within a bucket, highest classifier prob first.
    bucket_order = {name: i for i, (name, _) in enumerate(PRIORITY_BUCKETS)}
    rows.sort(
        key=lambda r: (
            bucket_order[r["review_bucket"]],
            -(r["classifier_prob_correct"] or 0.0),
        )
    )

    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    geojson_path = f"{args.out_prefix}.geojson"
    with open(geojson_path, "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": row,
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row["longitude"], row["latitude"]],
                        },
                    }
                    for row in rows
                ],
            },
            f,
        )

    print(f"Wrote {len(rows)} detections to {csv_path} and {geojson_path}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["review_bucket"]] = counts.get(row["review_bucket"], 0) + 1
    print(f"by review bucket: {counts}")
    if over_cap_dropped:
        print(f"dropped {over_cap_dropped} detections over the per-scene cap")
    if missing:
        print(
            f"WARNING: no detection JSON for {len(missing)} scenes, e.g. {missing[:5]}"
        )


if __name__ == "__main__":
    main()
