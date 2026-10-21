"""Pick monocrop events for the Q4 annotation round from the monocrop predictions.

Inputs: {out_dir}/monocrop_candidates.geojson (build_monocrop_candidates.py) and the
prediction dataset at --ds_path (create_prediction_dataset.py + rslearn model predict),
whose windows are matched to candidates by feature_window_name. Candidates without a
prediction (e.g. fewer than 12 post-event months of imagery) are skipped.

Two samples, drawn in this order:

1. country: for each country and predicted (top-1) class, up to PER_COUNTRY_CLASS
   events uniformly at random. The shortfall of each class (strata with fewer
   events) is then filled from the same class in other countries, round-robin over
   countries.
2. recall: for each class in RECALL_CLASSES, events uniformly at random among those
   with P(class) >= RECALL_MIN_PROB that are not predicted as that class, so that
   positives the model misses can be found. The recall classes share what is left
   of --target_total equally.

Every accepted event must be at least --min_distance_m from all previously accepted
events (polygon distance), so tasks do not overlap. Each stratum has its own random
generator (seeded from --seed and the stratum name).

Output: {out_dir}/events_for_annotation.geojson (with predicted_class, probs and
sample_stratum properties) and {out_dir}/events_for_annotation_counts.csv.

Example:

    python select_for_annotation.py --out_dir /path/to/out --ds_path /path/to/predict_ds
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict

import shapely
import shapely.geometry
from build_monocrop_candidates import ExclusionIndex
from upath import UPath

from rslp.forest_loss_driver.monocrop_classifier.create_prediction_dataset import (
    PREDICT_GROUP,
    feature_window_name,
    parse_feature_polygon,
)

# Order of probs in the output_vector layer (model_classify_pool.yaml class names).
CLASSES = [
    "nodata",
    "mennonites_nonsoybean",
    "soybean",
    "oil_palm",
    "other_agriculture",
    "pastures",
    "rice",
]
PREDICTED_CLASSES = CLASSES[1:]
COUNTRIES = ["BO", "BR", "CO", "EC", "PE"]
PER_COUNTRY_CLASS = 30
RECALL_CLASSES = ["rice", "oil_palm", "mennonites_nonsoybean"]
RECALL_MIN_PROB = 0.05


class Selector:
    """Accept events while keeping them at least min_distance_m apart.

    Accepted geometries are bucketed in a grid of CELL_DEG cells (larger than
    min_distance_m in degrees), so only nearby cells need an exact distance check.
    """

    CELL_DEG = 0.02

    def __init__(self, min_distance_m: float) -> None:
        """Create an empty selection."""
        self.min_distance_m = min_distance_m
        self.selected: list[dict] = []
        self.selected_uids: set[str] = set()
        self.cells: dict[tuple[int, int], list[shapely.Geometry]] = defaultdict(list)

    def _cell(self, lon: float, lat: float) -> tuple[int, int]:
        return (math.floor(lon / self.CELL_DEG), math.floor(lat / self.CELL_DEG))

    def try_add(self, event: dict, stratum: str) -> bool:
        """Accept the event unless it was already accepted or is too close."""
        uid = event["properties"]["event_uid"]
        if uid in self.selected_uids:
            return False
        geometry = event["geometry_shp"]
        minx, miny, maxx, maxy = geometry.bounds
        x0, y0 = self._cell(minx, miny)
        x1, y1 = self._cell(maxx, maxy)
        nearby = [
            other
            for cx in range(x0 - 1, x1 + 2)
            for cy in range(y0 - 1, y1 + 2)
            for other in self.cells.get((cx, cy), [])
        ]
        if nearby and ExclusionIndex(nearby).is_near(geometry, self.min_distance_m):
            return False
        self.selected.append({**event, "stratum": stratum})
        self.selected_uids.add(uid)
        centroid = geometry.centroid
        self.cells[self._cell(centroid.x, centroid.y)].append(geometry)
        return True


def load_predictions(ds_path: UPath, candidates: list[dict]) -> list[dict]:
    """Attach predicted_class and probs to each candidate that has a prediction."""
    events = []
    missing = 0
    for feat in candidates:
        geometry = parse_feature_polygon(feat["geometry"])
        name = feature_window_name(feat["properties"], geometry)
        fname = (
            ds_path
            / "windows"
            / PREDICT_GROUP
            / name
            / "layers"
            / "output_vector"
            / "data.geojson"
        )
        if not fname.exists():
            missing += 1
            continue
        with fname.open() as f:
            output = json.load(f)["features"][0]["properties"]
        probs = output["probs"]
        if len(probs) != len(CLASSES):
            raise ValueError(f"unexpected probs {probs} in {fname}")
        if CLASSES[max(range(len(probs)), key=probs.__getitem__)] != output["class_name"]:
            raise ValueError(f"argmax(probs) != class_name in {fname}")
        props = dict(feat["properties"])
        props["window_name"] = name
        props["predicted_class"] = output["class_name"]
        props["predicted_prob"] = max(probs)
        props["monocrop_probs"] = probs
        events.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": feat["geometry"],
                "geometry_shp": shapely.geometry.shape(feat["geometry"]),
            }
        )
    print(f"{len(events)} candidates with predictions, {missing} without")
    return events


def shuffled(events: list[dict], seed: int, stratum: str) -> list[dict]:
    """Return the events in a random order that depends only on seed and stratum."""
    events = sorted(events, key=lambda e: e["properties"]["event_uid"])
    random.Random(f"{seed}:{stratum}").shuffle(events)
    return events


def main(cli_args: argparse.Namespace) -> None:
    """Write events_for_annotation.geojson."""
    out_dir = UPath(cli_args.out_dir)
    with (out_dir / "monocrop_candidates.geojson").open() as f:
        candidates = json.load(f)["features"]
    events = load_predictions(UPath(cli_args.ds_path), candidates)

    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        props = event["properties"]
        by_stratum[(props["country"], props["predicted_class"])].append(event)
    available = Counter({key: len(value) for key, value in by_stratum.items()})

    selector = Selector(cli_args.min_distance_m)
    # Per (country, class): remaining shuffled events, consumed from the front.
    queues = {
        (country, cls): shuffled(by_stratum[(country, cls)], cli_args.seed, f"{country}:{cls}")
        for country in COUNTRIES
        for cls in PREDICTED_CLASSES
    }

    def take(country: str, cls: str, stratum: str) -> bool:
        queue = queues[(country, cls)]
        while queue:
            if selector.try_add(queue.pop(0), stratum):
                return True
        return False

    # 1. Country sample, then fill each class's shortfall from other countries.
    for cls in PREDICTED_CLASSES:
        shortfall = 0
        for country in COUNTRIES:
            got = 0
            while got < PER_COUNTRY_CLASS and take(country, cls, f"country:{country}:{cls}"):
                got += 1
            shortfall += PER_COUNTRY_CLASS - got
        while shortfall > 0:
            progressed = False
            for country in COUNTRIES:
                if shortfall > 0 and take(country, cls, f"fill:{country}:{cls}"):
                    shortfall -= 1
                    progressed = True
            if not progressed:
                break
    num_country = len(selector.selected)

    # 2. Recall sample for the rare classes.
    remaining = max(cli_args.target_total - num_country, 0)
    per_recall = remaining // len(RECALL_CLASSES)
    for i, cls in enumerate(RECALL_CLASSES):
        quota = per_recall + (1 if i < remaining % len(RECALL_CLASSES) else 0)
        cls_idx = CLASSES.index(cls)
        pool = [
            e
            for e in events
            if e["properties"]["predicted_class"] != cls
            and e["properties"]["monocrop_probs"][cls_idx] >= RECALL_MIN_PROB
        ]
        got = 0
        for event in shuffled(pool, cli_args.seed, f"recall:{cls}"):
            if got >= quota:
                break
            if selector.try_add(event, f"recall:{cls}"):
                got += 1
        print(f"recall {cls}: {got} of quota {quota} (pool {len(pool)})")

    features = []
    for event in selector.selected:
        props = dict(event["properties"])
        props["sample_stratum"] = event["stratum"]
        features.append({"type": "Feature", "properties": props, "geometry": event["geometry"]})
    with (out_dir / "events_for_annotation.geojson").open("w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    counts = Counter(
        (f["properties"]["country"], f["properties"]["predicted_class"], f["properties"]["sample_stratum"].split(":")[0])
        for f in features
    )
    with (out_dir / "events_for_annotation_counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "predicted_class", "available", "country", "fill", "recall"])
        for country in COUNTRIES:
            for cls in PREDICTED_CLASSES:
                writer.writerow(
                    [country, cls, available[(country, cls)]]
                    + [counts[(country, cls, kind)] for kind in ("country", "fill", "recall")]
                )
    print(f"selected {len(features)} events ({num_country} country/fill)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--target_total", type=int, default=1000)
    parser.add_argument("--min_distance_m", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
