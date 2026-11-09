"""Small Flask app for reviewing Sentinel-1 vessel labels.

Serves items.json and the chips made by build_chips.py from --data_dir, and stores the
reviewer's decisions in <data_dir>/labels.json (one entry per item id). Optional files in
the same directory are shown when present:

    groups.json   {group key: [display name, one-line description]}
    scores.json   {item id: {name: score or {image: score}}}, e.g. model scores
    gemini.json   {item id: {field: value}}, e.g. a Gemini answer

Run: python app.py --data_dir /path/to/review_data [--port 8080]
"""

import argparse
import json
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
lock = threading.Lock()
DATA_DIR = "."


def load(name: str, default):  # type: ignore[no-untyped-def]
    """Load a JSON file from the data directory, or return default."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


@app.route("/")
def index():  # type: ignore[no-untyped-def]
    """The app page."""
    return send_from_directory(HERE, "index.html")


@app.route("/chips/<path:fn>")
def chips(fn: str):  # type: ignore[no-untyped-def]
    """Chip images."""
    resp = send_from_directory(os.path.join(DATA_DIR, "chips"), fn)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/api/items")
def items():  # type: ignore[no-untyped-def]
    """Items, labels and the optional side-panel data."""
    return jsonify(
        {
            "items": load("items.json", []),
            "labels": load("labels.json", {}),
            "groups": load("groups.json", {}),
            "scores": load("scores.json", {}),
            "gemini": load("gemini.json", {}),
        }
    )


@app.route("/api/label", methods=["POST"])
def label():  # type: ignore[no-untyped-def]
    """Set or clear the label of one item."""
    body = request.get_json(force=True)
    path = os.path.join(DATA_DIR, "labels.json")
    with lock:
        labels = load("labels.json", {})
        if body.get("label") is None:
            labels.pop(body["id"], None)
        else:
            labels[body["id"]] = {
                "label": body["label"],
                "note": body.get("note", ""),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        with open(path + ".tmp", "w") as f:
            json.dump(labels, f, indent=0)
        os.replace(path + ".tmp", path)
    return jsonify({"ok": True, "n": len(labels)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    app.run(host="0.0.0.0", port=args.port, threaded=True)
