"""Select the round-1 human-annotation pool from the full manifest.

Takes the manifest CSV from make_annotation_manifest.py and picks ~budget
detections for human annotation, prioritized for the round-1 goal (diverse hard
negatives) under a no-AIS-triage constraint (these scenes never went through
Skylight correlation, so humans absorb everything — the slices spend their time
where labels buy the most).

Slices, in priority order:

  1. suspect_confident — classifier-passed (prob >= .85) detections in contexts
     where real traffic is unlikely: ice stratum, Southern Ocean (lat <= -45),
     high Arctic (lat >= 66), or scenes where the classifier rejected almost
     everything else (pass rate < 10% among >= 30 candidates — a clutter field).
     Every one an annotator rejects is a production-grade hard negative. Take all.
  2. ambiguous — classifier prob in [.40, .85): decision-boundary cases, highest
     training signal per label. Take all.
  3. plausible_confident — passed detections in trafficked-looking water. These
     are mostly real vessels (positive coverage is already adequate), so sample
     only a few per scene as an FP-contamination audit + dark-vessel spot check.
  4. easy_negative — rejected (prob < .40) candidates for negative volume and
     class balance, stratified by (stratum, tier); within each cell, half taken
     highest-prob-first (near-boundary informative), half uniformly at random
     (diversity). Fills whatever budget remains.

The context rules are deliberately coarse (no shipping-density data); slices set
review priority, not labels — annotators label every row on its pixels.

Usage:
    python -m rslp.landsat_vessels.scripts.select_annotation_pool \
        --manifest_csv /weka/dfive-default/yawenz/landsat/round1_manifest.csv \
        --out_prefix /weka/dfive-default/yawenz/landsat/round1_annotation_pool \
        --budget 2500
"""

import argparse
import csv
import json
import random
from collections import defaultdict

CONFIDENT = "confident_vessel"
AMBIGUOUS = "ambiguous"
NEGATIVE = "likely_negative"

# Context rules for "traffic unlikely here".
SOUTHERN_OCEAN_LAT = -45.0
HIGH_ARCTIC_LAT = 66.0
CLUTTER_SCENE_MIN_CANDS = 30
CLUTTER_SCENE_MAX_PASS_RATE = 0.10


def is_implausible(row: dict, scene_stats: dict) -> tuple[bool, str]:
    """Whether a detection sits in a context where real traffic is unlikely."""
    lat = float(row["latitude"])
    # Distance-to-coast columns exist when the manifest was enriched via
    # scripts/coast_distance.py (Skylight's own dataset). A classifier-passed
    # detection that is not over water is a certain hard negative.
    lc = row.get("land_cover_class")
    if lc and lc != "PermanentWaterBody":
        return True, f"on land ({lc})"
    if row["stratum"] == "ice":
        return True, "ice stratum"
    if lat <= SOUTHERN_OCEAN_LAT:
        return True, f"Southern Ocean (lat {lat:.1f})"
    if lat >= HIGH_ARCTIC_LAT:
        return True, f"high Arctic (lat {lat:.1f})"
    st = scene_stats[row["scene_id"]]
    if (
        st["cands"] >= CLUTTER_SCENE_MIN_CANDS
        and st["passed"] / st["cands"] <= CLUTTER_SCENE_MAX_PASS_RATE
    ):
        return True, f"clutter-field scene ({st['passed']}/{st['cands']} passed)"
    return False, ""


def main() -> None:
    """Assign slices and sample the pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--budget", type=int, default=2500)
    # Positives are already well covered (~1k good positives from selected_copy/
    # phase2a), so the plausible-confident audit slice is kept small; freed budget
    # flows to negatives via the easy_negative fill.
    parser.add_argument("--plausible_confident_per_scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_rt_rows",
        type=int,
        default=None,
        help="cap on RT-tier rows in the pool (RT is all from one month, so this "
        "bounds the temporal/tier skew); RT rows are kept in review-priority "
        "order until the cap",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.manifest_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["classifier_prob_correct"] = float(r["classifier_prob_correct"] or 0)

    # Scene-level pass stats (over the capped manifest — close enough to raw).
    scene_stats: dict[str, dict] = defaultdict(lambda: {"cands": 0, "passed": 0})
    for r in rows:
        st = scene_stats[r["scene_id"]]
        st["cands"] += 1
        if r["review_bucket"] == CONFIDENT:
            st["passed"] += 1

    slices: dict[str, list[dict]] = {
        s: []
        for s in [
            "suspect_confident",
            "ambiguous",
            "plausible_confident",
            "easy_negative",
        ]
    }
    plausible_by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket = r["review_bucket"]
        if bucket == CONFIDENT:
            implausible, reason = is_implausible(r, scene_stats)
            if implausible:
                r["slice"], r["selection_reason"] = "suspect_confident", reason
                slices["suspect_confident"].append(r)
            else:
                plausible_by_scene[r["scene_id"]].append(r)
        elif bucket == AMBIGUOUS:
            r["slice"] = "ambiguous"
            r["selection_reason"] = "decision boundary (.40-.85)"
            slices["ambiguous"].append(r)
        else:
            slices["easy_negative"].append(r)  # sampled below

    # Slice 3: cap per scene.
    for scene_id, dets in plausible_by_scene.items():
        take = args.plausible_confident_per_scene
        picked = dets if len(dets) <= take else rng.sample(dets, take)
        for r in picked:
            r["slice"] = "plausible_confident"
            r["selection_reason"] = (
                f"FP audit / dark-vessel check ({len(dets)} passed in scene)"
            )
        slices["plausible_confident"].extend(picked)

    def row_tier(r: dict) -> str:
        return "RT" if r["scene_id"].endswith("_RT") else "T1T2"

    # Slices 1-3 in review-priority order, enforcing the optional RT cap: RT
    # rows are admitted highest-priority-first until the cap, then dropped.
    rt_cap = args.max_rt_rows if args.max_rt_rows is not None else float("inf")
    selected: list[dict] = []
    rt_used = rt_dropped = 0
    for name in ["suspect_confident", "ambiguous", "plausible_confident"]:
        for r in sorted(slices[name], key=lambda r: -r["classifier_prob_correct"]):
            if row_tier(r) == "RT":
                if rt_used >= rt_cap:
                    rt_dropped += 1
                    continue
                rt_used += 1
            selected.append(r)
    remaining = args.budget - len(selected)

    def pick_negatives(cell_rows: list[dict], want: int) -> list[dict]:
        """Half near-boundary (highest prob first), half uniform random."""
        cell_rows.sort(key=lambda r: -r["classifier_prob_correct"])
        top = cell_rows[: (want + 1) // 2]
        rand = rng.sample(
            cell_rows[(want + 1) // 2 :],
            min(want - len(top), len(cell_rows) - len(top)),
        )
        for r in top:
            r["slice"] = "easy_negative"
            r["selection_reason"] = "near-boundary negative"
        for r in rand:
            r["slice"] = "easy_negative"
            r["selection_reason"] = "random negative (diversity)"
        return top + rand

    # Slice 4: stratified fill of easy negatives (RT cells bounded by the cap;
    # budget the RT cells can't take flows to the T1/T2 cells).
    picked_negatives: list[dict] = []
    if remaining > 0:
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for r in slices["easy_negative"]:
            cells[(r["stratum"], row_tier(r))].append(r)
        quota, extra = divmod(remaining, len(cells))
        shortfall = 0
        t1t2_leftover: list[list[dict]] = []
        for i, (cell, cell_rows) in enumerate(sorted(cells.items())):
            want = quota + (1 if i < extra else 0)
            if cell[1] == "RT":
                allowed = int(min(want, max(0, rt_cap - rt_used)))
                shortfall += want - allowed
                want = allowed
            want = min(want, len(cell_rows))
            picked = pick_negatives(cell_rows, want)
            if cell[1] == "RT":
                rt_used += len(picked)
            picked_negatives.extend(picked)
            if cell[1] == "T1T2":
                t1t2_leftover.append([r for r in cell_rows if "slice" not in r])
        # Redistribute RT shortfall across T1/T2 cells round-robin.
        while shortfall > 0 and any(t1t2_leftover):
            for cell_rows in t1t2_leftover:
                if shortfall <= 0 or not cell_rows:
                    continue
                picked_negatives.extend(pick_negatives(cell_rows, 1))
                cell_rows[:] = [r for r in cell_rows if "slice" not in r]
                shortfall -= 1
    selected += picked_negatives
    if rt_dropped:
        print(
            f"RT cap {args.max_rt_rows}: dropped {rt_dropped} RT rows from slices 1-3"
        )

    # Tier column so RT rows (whose s3_prefix may be gone by annotation time —
    # use crop_fname instead) are easy to spot.
    for r in selected:
        r["tier"] = "RT" if r["scene_id"].endswith("_RT") else "T1T2"

    # Review order: slice priority, then descending prob within slice.
    order = {
        "suspect_confident": 0,
        "ambiguous": 1,
        "plausible_confident": 2,
        "easy_negative": 3,
    }
    selected.sort(key=lambda r: (order[r["slice"]], -r["classifier_prob_correct"]))

    fieldnames = list(rows[0].keys()) + ["tier", "slice", "selection_reason"]
    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    with open(f"{args.out_prefix}.geojson", "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": r,
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                float(r["longitude"]),
                                float(r["latitude"]),
                            ],
                        },
                    }
                    for r in selected
                ],
            },
            f,
        )

    print(
        f"Selected {len(selected)} / budget {args.budget} "
        f"-> {args.out_prefix}.csv/.geojson"
    )
    for name in order:
        n = sum(1 for r in selected if r["slice"] == name)
        print(f"  {name:<20} {n}")
    by: dict[tuple[str, str], int] = defaultdict(int)
    for r in selected:
        tier = "RT" if r["scene_id"].endswith("_RT") else "T1T2"
        by[(r["stratum"], tier)] += 1
    print("  by stratum/tier:", dict(sorted(by.items())))


if __name__ == "__main__":
    main()
