import hashlib
import json
import os
import sys

from flask import Flask, jsonify, request, send_file

group_dir = sys.argv[1]
port = int(sys.argv[2])

window_ids = os.listdir(group_dir)
window_ids.sort(key=lambda window_id: hashlib.sha256(window_id.encode()).hexdigest())
app = Flask(__name__)


@app.route("/")
def index():
    return send_file("phase1_index.html")


@app.route("/examples")
def get_examples():
    return jsonify(window_ids)


@app.route("/metadata/<idx>")
def get_example(idx):
    window_id = window_ids[int(idx)]
    metadata = {}
    metadata["example_id"] = window_id

    with open(
        os.path.join(group_dir, window_id, "layers", "label", "data.geojson")
    ) as f:
        feat = json.load(f)["features"][0]
        metadata["label"] = feat["properties"]["label"]
        metadata["url"] = feat["properties"]["url"]

    return jsonify(metadata)


@app.route("/image/<example_idx>")
def get_image(example_idx):
    window_id = window_ids[int(example_idx)]
    image_fname = os.path.join(
        group_dir, window_id, "layers", "landsat", "R_G_B", "image.png"
    )
    return send_file(image_fname)


@app.route("/update/<idx>", methods=["POST"])
def update(idx):
    window_id = window_ids[int(idx)]
    label_fname = os.path.join(group_dir, window_id, "layers", "label", "data.geojson")
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
