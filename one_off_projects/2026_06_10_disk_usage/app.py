"""Flask app to visualize the collapsed disk-usage tree.

Serves the prebuilt JSON produced by collapse.py (which already aggregated and
pruned the full disk_usage.py scan). The file is loaded once at startup and
handed back verbatim, so page loads are just a small in-memory blob.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template


def create_app(input_path: str) -> Flask:
    """Load the prebuilt tree once and return an app that serves it from memory.

    The response is captured in this closure (no module globals), so each
    request just hands back the in-memory blob.
    """
    input_path = os.path.abspath(input_path)
    print(f"loading {input_path}...")
    with open(input_path) as f:
        response = json.load(f)
    print(f"loaded tree (total {response['tree']['size']} bytes), serving")

    here = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(here / "templates"),
        static_folder=str(here / "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/tree")
    def api_tree():
        return jsonify(response)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="collapsed.json", help="JSON produced by collapse.py."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.input)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
