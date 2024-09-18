"""Webserver for showing the predictions."""

import hashlib
import json
import math
import sys

from flask import Flask, jsonify, send_file
from upath import UPath

# e.g. gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_20240828/
ds_root = UPath(sys.argv[1])
port = int(sys.argv[2])
window_names_fname = sys.argv[3]

group = "default"

# Load example IDs.
window_dir = ds_root / "windows" / group
with open(window_names_fname) as f:
    window_names = json.load(f)
window_names.sort(
    key=lambda window_name: hashlib.sha256(window_name.encode()).hexdigest()
)

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/examples")
def get_examples():
    return jsonify(window_names)


def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)


@app.route("/metadata/<idx>")
def get_example(idx):
    metadata = {}

    window_name = window_names[int(idx)]
    metadata["example_id"] = window_name

    parts = window_name.split("_")
    point = (int(parts[2]), int(parts[3]))
    point = mercator_to_geo(point, zoom=13, pixels=512)
    metadata["point"] = point

    with (window_dir / window_name / "metadata.json").open() as f:
        window_properties = json.load(f)
        metadata["date"] = window_properties["time_range"][0][0:7]

    with (window_dir / window_name / "best_times.json").open() as f:
        metadata["best_times"] = json.load(f)

    with (window_dir / window_name / "layers" / "output" / "data.geojson").open() as f:
        fc = json.load(f)
        props = fc["features"][0]["properties"]
        metadata["label"] = props["new_label"]
        metadata["prob_str"] = str(props["probs"])

    return jsonify(metadata)


def get_image_fname(example_id, band, image_idx):
    if band == "mask":
        return "layers/mask/mask/image.png"
    if "planet" in band:
        return f"layers/{band}_{image_idx+1}/R_G_B/image.png"
    return f"layers/best_{band}_{image_idx}/R_G_B/image.png"


@app.route("/image/<example_idx>/<band>/<image_idx>")
def get_image(example_idx, band, image_idx):
    assert band in ["pre", "post", "mask", "planet_pre", "planet_post"]
    window_name = window_names[int(example_idx)]
    image_idx = int(image_idx)

    image_fname = get_image_fname(window_name, band, image_idx)
    f = (window_dir / window_name / image_fname).open("rb")
    return send_file(f, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
