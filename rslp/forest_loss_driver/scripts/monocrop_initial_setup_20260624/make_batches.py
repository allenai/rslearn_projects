"""Split the prediction request geometry into batches and simplified centroid copies.

This is step 2 of the monocrop initial setup. Studio inference jobs are capped at
``EVENTS_PER_STUDIO_JOB`` (10000) events, so we split
``prediction_request_geometry.geojson`` into consecutive batches of that size. For each
batch we write two files:

- ``..._batch{i}_orig.geojson``: the original polygon geometries (kept so we can
  restore polygons after inference, since Studio only returns/needs the centroid).
- ``..._batch{i}_simple.geojson``: the same features with each geometry replaced by its
  centroid Point. These are what get uploaded to Studio; submitting Points avoids job
  failures caused by complex polygon geometries.

Run in any environment with ``shapely`` installed, e.g. the rslearn venv:

    python rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/make_batches.py \
        --input /weka/.../predictions_2022_to_2025/prediction_request_geometry.geojson \
        --output-dir /weka/.../predictions_2022_to_2025/
"""

import argparse
import copy
import json
import os

import shapely.geometry

# Matches EVENTS_PER_STUDIO_JOB in
# olmoearth_projects.projects.forest_loss_driver.deploy.
DEFAULT_BATCH_SIZE = 10000


def simplify_features_to_centroids(features: list[dict]) -> list[dict]:
    """Replace each feature's geometry with its centroid Point.

    Mirrors ``simplify_features_to_centroids`` in the deploy pipeline.
    """
    simplified = []
    for feat in features:
        feat = copy.deepcopy(feat)
        shp = shapely.geometry.shape(feat["geometry"])
        feat["geometry"] = shapely.geometry.mapping(shp.centroid)
        simplified.append(feat)
    return simplified


def main() -> None:
    """Parse arguments and write per-batch orig/simple GeoJSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to prediction_request_geometry.geojson.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the per-batch orig/simple GeoJSON files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Events per batch (default {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        fc = json.load(f)
    features = fc["features"]
    # Preserve top-level keys (e.g. "crs") other than "features".
    base = {k: v for k, v in fc.items() if k != "features"}
    base.setdefault("type", "FeatureCollection")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    num_batches = (len(features) + args.batch_size - 1) // args.batch_size
    print(f"Splitting {len(features)} features into {num_batches} batches")
    for i in range(num_batches):
        chunk = features[i * args.batch_size : (i + 1) * args.batch_size]
        orig_path = os.path.join(args.output_dir, f"{stem}_batch{i}_orig.geojson")
        simple_path = os.path.join(args.output_dir, f"{stem}_batch{i}_simple.geojson")
        with open(orig_path, "w") as f:
            json.dump({**base, "features": chunk}, f)
        with open(simple_path, "w") as f:
            json.dump({**base, "features": simplify_features_to_centroids(chunk)}, f)
        print(f"batch{i}: wrote {len(chunk)} features -> {orig_path}, {simple_path}")


if __name__ == "__main__":
    main()
