"""Pick forest loss events for the next phase of annotation using the new (utm) model.

Two samples are drawn from events_with_new_model_predictions.geojson, after removing
events whose polygon intersects any window of the latest training set
(training_set_20260924_utm_windows.geojson):

1. country: for each country in COUNTRY_SAMPLE_COUNTRIES and each category, sample
   PER_COUNTRY_CATEGORY events uniformly among events whose predicted (top-1)
   category is that category.
2. recall: for each category, sample PER_CATEGORY events uniformly (all countries)
   among events with P(category) >= max(MIN_THRESHOLD, the threshold reaching 0.95
   recall on the val split).

The two samples are drawn independently so each stays a uniform sample of its
stratum. Events drawn more than once are written once, with every stratum listed in
the sample_strata property.

Output: {out_dir}/events_for_annotation.geojson and
{out_dir}/events_for_annotation_counts.csv.

Example:

    python select_for_annotation.py --out_dir /path/to/out
"""

import argparse
import csv
import json
import random
from collections import Counter

import shapely
import shapely.geometry
from upath import UPath

CLASSES = [
    "agriculture",
    "mining",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
    "none",
]

# Probability threshold at which the new model reaches 0.95 recall on the val split of
# dataset_v1/20260924_utm (296 windows, computed with best.ckpt of
# 20260924_forest_loss_driver_utm/config.yaml_02). airstrip has a single val
# window, which the model missed (P=1.66e-5), so its threshold is not meaningful and
# MIN_THRESHOLD applies.
RECALL_95_THRESHOLDS = {
    "agriculture": 0.0003191,
    "mining": 0.03313,
    "airstrip": 1.656e-05,
    "road": 0.00716,
    "logging": 0.001577,
    "burned": 1.997e-06,
    "landslide": 0.05482,
    "hurricane": 0.001672,
    "river": 0.002652,
    "none": 3.694e-05,
}
MIN_THRESHOLD = 0.01

COUNTRY_SAMPLE_COUNTRIES = ["BR", "BO", "EC"]
PER_COUNTRY_CATEGORY = 10
PER_CATEGORY = 25


def main(out_dir: UPath, seed: int) -> None:
    """Select the events and write the outputs."""
    with (out_dir / "events_with_new_model_predictions.geojson").open() as f:
        events = json.load(f)["features"]
    with (out_dir / "events_with_old_model_predictions.geojson").open() as f:
        old_by_id = {
            feat["properties"]["event_id"]: feat["properties"]
            for feat in json.load(f)["features"]
        }

    # Exclude events intersecting the training set.
    tree_shps = {}
    for kind in ["windows", "labels"]:
        with (out_dir / f"training_set_20260924_utm_{kind}.geojson").open() as f:
            tree_shps[kind] = [
                shapely.geometry.shape(feat["geometry"])
                for feat in json.load(f)["features"]
            ]
    event_shps = [shapely.geometry.shape(feat["geometry"]) for feat in events]
    hits = {}
    for kind, shps in tree_shps.items():
        tree = shapely.STRtree(shps)
        event_idx, _ = tree.query(event_shps, predicate="intersects")
        hits[kind] = set(event_idx.tolist())
    print(
        f"events intersecting training windows: {len(hits['windows'])}, "
        f"training label polygons: {len(hits['labels'])}"
    )
    eligible = [feat for i, feat in enumerate(events) if i not in hits["windows"]]
    print(f"{len(eligible)} of {len(events)} events eligible")

    rng = random.Random(seed)
    strata_by_event: dict[int, list[str]] = {}
    count_rows = []

    def sample(stratum: str, pool: list[dict], k: int) -> None:
        # Sort by event_id so the sample only depends on the seed.
        pool = sorted(pool, key=lambda feat: feat["properties"]["event_id"])
        chosen = rng.sample(pool, min(k, len(pool)))
        for feat in chosen:
            strata_by_event.setdefault(feat["properties"]["event_id"], []).append(
                stratum
            )
        count_rows.append([stratum, len(pool), len(chosen)])
        if len(chosen) < k:
            print(f"shortfall {stratum}: only {len(pool)} eligible events")

    for country in COUNTRY_SAMPLE_COUNTRIES:
        for cls in CLASSES:
            pool = [
                feat
                for feat in eligible
                if feat["properties"]["country"] == country
                and feat["properties"]["category"] == cls
            ]
            sample(f"country:{country}:{cls}", pool, PER_COUNTRY_CATEGORY)

    for ci, cls in enumerate(CLASSES):
        threshold = max(MIN_THRESHOLD, RECALL_95_THRESHOLDS[cls])
        pool = [feat for feat in eligible if feat["properties"]["probs"][ci] >= threshold]
        sample(f"recall:{cls}", pool, PER_CATEGORY)

    out_feats = []
    for feat in events:
        event_id = feat["properties"]["event_id"]
        if event_id not in strata_by_event:
            continue
        props = dict(feat["properties"])
        props["sample_strata"] = strata_by_event[event_id]
        old = old_by_id.get(event_id)
        props["old_model_category"] = old["category"] if old else None
        props["old_model_probs"] = old["probs"] if old else None
        out_feats.append(dict(type="Feature", properties=props, geometry=feat["geometry"]))

    num_draws = sum(len(v) for v in strata_by_event.values())
    print(
        f"{num_draws} draws, {len(out_feats)} unique events "
        f"({num_draws - len(out_feats)} drawn in more than one stratum)"
    )
    with (out_dir / "events_for_annotation.geojson").open("w") as f:
        json.dump(
            dict(
                type="FeatureCollection",
                properties=dict(
                    classes=CLASSES,
                    recall_thresholds={
                        cls: max(MIN_THRESHOLD, t)
                        for cls, t in RECALL_95_THRESHOLDS.items()
                    },
                    seed=seed,
                ),
                features=out_feats,
            ),
            f,
        )
    with (out_dir / "events_for_annotation_counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["stratum", "eligible_events", "sampled"])
        writer.writerows(count_rows)
    print(Counter(len(v) for v in strata_by_event.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    cli_args = parser.parse_args()
    main(UPath(cli_args.out_dir), cli_args.seed)
