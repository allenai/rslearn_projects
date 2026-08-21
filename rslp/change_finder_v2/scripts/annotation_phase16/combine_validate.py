"""Combine partial phase16 JSONs, validate, shuffle, and write the final file.

Reads the partial_<dataset>.json files produced by gen_phase16.py, checks the
entries against the v2 annotation conventions (128x128 windows at 10 m in a UTM
zone, points inside their window, unique window names, no post_change /
first_date_change_noticeable set), shuffles them with a fixed seed so the
datasets are interleaved, and writes the combined annotation JSON.

Usage::

    python -m rslp.change_finder_v2.scripts.annotation_phase16.combine_validate \
        --data-dir /path/to/downloaded_datasets/ \
        --output annotations_phase16_mining_from_datasets.json
"""

import argparse
import json
import random
from collections import Counter

import pyproj

ORDER = [
    "ipis_drc",
    "ipis_car",
    "ipis_zwe",
    "usgs_copperbelt",
    "lames",
    "amw",
    "smallminesds",
    "pasanisi",
    "dethier",
    "ivc",
]
SHUFFLE_SEED = 16016


def main() -> None:
    """Combine, validate, and shuffle partial phase16 JSONs into one annotation file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the partial_<dataset>.json files.",
    )
    parser.add_argument("--output", required=True, help="Output annotation JSON.")
    args = parser.parse_args()
    sp = args.data_dir.rstrip("/")

    entries = []
    for name in ORDER:
        with open(f"{sp}/partial_{name}.json") as f:
            part = json.load(f)
        assert len(part) == 100, f"{name}: {len(part)} entries"
        entries.extend(part)

    names = [e["window_name"] for e in entries]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    assert not dupes, f"duplicate window names: {dupes[:5]}"

    n_points = 0
    n_dated = 0
    for e in entries:
        assert e["group"] == "phase16"
        b = e["bounds"]
        assert b[2] - b[0] == 128 and b[3] - b[1] == 128, e["window_name"]
        crs = e["projection"]["crs"]
        assert crs.startswith("EPSG:326") or crs.startswith("EPSG:327"), crs
        assert e["projection"]["x_resolution"] == 10.0
        assert e["projection"]["y_resolution"] == -10.0
        t0, t1 = e["time_range"]
        assert t0 < t1
        tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        for pt in e["positive_points"]:
            n_points += 1
            if pt.get("pre_change"):
                n_dated += 1
                assert "2019-01-01" <= pt["pre_change"] <= "2025-12-31", (
                    e["window_name"],
                    pt["pre_change"],
                )
            assert not pt.get("post_change") and not pt.get(
                "first_date_change_noticeable"
            )
            # The point must land inside the window (in pixel coords).
            x, y = tf.transform(pt["lon"], pt["lat"])
            col, row = x / 10, y / -10
            assert b[0] <= col <= b[2] and b[1] <= row <= b[3], (
                e["window_name"],
                col,
                row,
                b,
            )
            assert -90 < pt["lat"] < 90 and -180 < pt["lon"] < 180

    random.Random(SHUFFLE_SEED).shuffle(entries)

    print(f"total entries: {len(entries)}")
    print(f"entries with a positive point: {n_points}")
    print(f"points with pre_change date: {n_dated}")
    no_pt = sum(1 for e in entries if not e["positive_points"])
    print(f"entries with no points (rough location): {no_pt}")

    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
