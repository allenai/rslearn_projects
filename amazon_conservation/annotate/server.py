import glob
import hashlib
import json
import math
import os
import sys

from flask import Flask, jsonify, request, send_file

# e.g. /multisat/datasets/rslearn_amazon_conservation/
ds_root = sys.argv[1]
# e.g. peru2
group = sys.argv[2]
port = int(sys.argv[3])

# Load example IDs.
window_dir = os.path.join(ds_root, "windows", group)
example_ids = []
for fname in glob.glob(os.path.join(window_dir, "*/good_images.json")):
    example_ids.append(fname.split("/")[-2])
example_ids.sort(key=lambda example_id: hashlib.md5(example_id.encode()).hexdigest())

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/examples")
def get_examples():
    return jsonify(example_ids)


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

    example_id = example_ids[int(idx)]
    metadata["example_id"] = example_id

    parts = example_id.split("_")
    point = (int(parts[2]), int(parts[3]))
    point = mercator_to_geo(point, zoom=13, pixels=512)
    metadata["point"] = point

    with open(os.path.join(window_dir, example_id, "metadata.json")) as f:
        window_properties = json.load(f)
        metadata["date"] = window_properties["time_range"][0][0:7]

    with open(os.path.join(window_dir, example_id, "best_times.json")) as f:
        metadata["best_times"] = json.load(f)

    with open(
        os.path.join(window_dir, example_id, "layers", "label", "data.geojson")
    ) as f:
        wanted = ["old_label", "new_label"]
        for feat in json.load(f)["features"]:
            props = feat["properties"]
            for k in wanted:
                if k in props:
                    metadata[k] = props[k]

    return jsonify(metadata)


def get_image_fname(example_id, band, image_idx):
    if band == "mask":
        return "mask.png"
    if "planet" in band:
        return f"layers/{band}_{image_idx+1}/R_G_B/image.png"
    return f"layers/best_{band}_{image_idx}/R_G_B/image.png"


@app.route("/image/<example_idx>/<band>/<image_idx>")
def get_image(example_idx, band, image_idx):
    assert band in ["pre", "post", "mask", "planet_pre", "planet_post"]
    example_id = example_ids[int(example_idx)]
    image_idx = int(image_idx)

    image_fname = get_image_fname(example_id, band, image_idx)
    return send_file(os.path.join(window_dir, example_id, image_fname))


@app.route("/update/<idx>", methods=["POST"])
def update(idx):
    example_id = example_ids[int(idx)]
    label_fname = os.path.join(
        window_dir, example_id, "layers", "label", "data.geojson"
    )
    with open(label_fname) as f:
        fc = json.load(f)
    for feat in fc["features"]:
        feat["properties"]["new_label"] = request.json
    with open(label_fname + ".tmp", "w") as f:
        json.dump(fc, f)
    os.rename(label_fname + ".tmp", label_fname)
    return jsonify(True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
