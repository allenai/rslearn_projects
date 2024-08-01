import csv
import json
import os
import sys

from flask import Flask, jsonify, request, send_file

# e.g. /multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/data.csv
csv_fname = sys.argv[1]
# e.g. /multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/
ds_dir = sys.argv[2]
port = int(sys.argv[3])

# Load examples.
examples = []
with open(csv_fname) as f:
    reader = csv.DictReader(f)
    for csv_row in reader:
        window_id, col, row = csv_row["window_id"], csv_row["col"], csv_row["row"]
        csv_row["crop_window_id"] = f"{window_id}_{col}_{row}"
        examples.append(csv_row)

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("phase1_index.html")


@app.route("/examples")
def get_examples():
    return jsonify(examples)


@app.route("/metadata/<idx>")
def get_example(idx):
    example = examples[int(idx)]
    metadata = {}
    metadata["example_id"] = example["crop_window_id"]
    metadata["url"] = example["url"]

    with open(
        os.path.join(
            ds_dir,
            "windows",
            "selected",
            example["crop_window_id"],
            "layers",
            "label",
            "data.geojson",
        )
    ) as f:
        feat = json.load(f)["features"][0]
        metadata["label"] = feat["properties"]["label"]

    return jsonify(metadata)


@app.route("/image/<example_idx>")
def get_image(example_idx):
    example = examples[int(example_idx)]
    image_fname = os.path.join(
        ds_dir,
        "windows",
        "selected",
        example["crop_window_id"],
        "layers",
        "landsat",
        "R_G_B",
        "image.png",
    )
    return send_file(image_fname)


@app.route("/update/<idx>", methods=["POST"])
def update(idx):
    example = examples[int(idx)]
    label_fname = os.path.join(
        ds_dir,
        "windows",
        "selected",
        example["crop_window_id"],
        "layers",
        "label",
        "data.geojson",
    )
    with open(label_fname) as f:
        fc = json.load(f)
    for feat in fc["features"]:
        feat["properties"]["label"] = request.json
    with open(label_fname + ".tmp", "w") as f:
        json.dump(fc, f)
    os.rename(label_fname + ".tmp", label_fname)
    return jsonify(True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
