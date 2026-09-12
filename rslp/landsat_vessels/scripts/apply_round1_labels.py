"""Reconcile the annotation labels JSONL against the rslearn windows.

The annotation app already writes each label through into its window's ``label`` layer as
it is made, so in normal use this script confirms rather than repairs. Run it to:

- **reconcile**: re-apply the whole log (last record per window wins, so a relabel
  supersedes the original), catching anything labelled while the dataset was unwritable
  or edited in the JSONL by hand;
- **report**: the label mix by split and slice, the reason breakdown, and how many
  classifier-passed detections a human called incorrect — the hard negatives this round
  exists to collect;
- **export**: ``*_pool_labeled.{csv,geojson}``, the annotation pool with ``human_label`` /
  ``annotator`` / ``notes`` / ``reason`` filled in, leaving the original pool untouched.

What becomes a label layer is decided in ``annotations/writeback.py``, shared with the
app: ``correct`` / ``incorrect`` / ``unsure`` are written, ``skip`` is not.

Idempotent: only windows whose annotation differs from what is on disk are rewritten.

Usage:
    python apply_round1_labels.py [--dry_run] [--group round1_20260803]
"""

import argparse
import csv
import json
from collections import defaultdict

import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.landsat_vessels.annotations.writeback import (
    TRAINABLE_LABELS,
    WRITTEN_LABELS,
    build_properties,
    clear_label,
    is_current,
    read_existing_label,
    write_label,
)

DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
ROUND_DIR = UPath("/weka/dfive-default/yawenz/landsat/annotation_round1")
POOL_GEOJSON = UPath(
    "/weka/dfive-default/yawenz/landsat/round1_annotation_pool_v2.geojson"
)
POOL_CSV = UPath("/weka/dfive-default/yawenz/landsat/round1_annotation_pool_v2.csv")

GROUP = "round1_20260803"


def read_labels(path: UPath) -> tuple[dict[str, dict], int, set[str]]:
    """Reduce the JSONL to the final label per window.

    Returns (final label per window, event count, every window the log mentions). The
    third value is what makes retraction safe: a window that appears in the log but has no
    final label was deliberately cleared, and its label layer should go — whereas a window
    the log never mentions is simply unannotated and is left alone.
    """
    labels: dict[str, dict] = {}
    seen: set[str] = set()
    events = 0
    if not path.exists():
        return labels, 0, seen
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping unparseable line: {line[:80]}")
                continue
            if "window" not in record:
                continue
            events += 1
            seen.add(record["window"])
            if record.get("label") is None:
                labels.pop(record["window"], None)
            else:
                labels[record["window"]] = record
    return labels, events, seen


def main() -> None:
    """Reconcile the annotation JSONL log into the dataset label layers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument(
        "--dry_run", action="store_true", help="report what would change, write nothing"
    )
    args = parser.parse_args()

    labels_path = ROUND_DIR / "labels" / f"{args.group}_labels.jsonl"
    labels, events, seen = read_labels(labels_path)
    print(f"{labels_path}: {events} events -> {len(labels)} windows with a final label")
    if not labels:
        print("nothing to apply")
        return

    with (ROUND_DIR / f"{args.group}_index.json").open() as f:
        index = {record["window"]: record for record in json.load(f)}

    to_write = {w: r for w, r in labels.items() if r["label"] in WRITTEN_LABELS}
    skipped = {w: r for w, r in labels.items() if r["label"] == "skip"}
    unknown = sorted(set(to_write) - set(index))
    if unknown:
        print(
            f"  {len(unknown)} labelled windows are not in the index, ignoring: {unknown[:3]}"
        )
        to_write = {w: r for w, r in to_write.items() if w in index}

    # Windows the annotator touched and then took back: cleared outright, or moved to
    # "skip". Any label layer they still carry is stale and must go.
    to_retract = sorted((seen - set(labels)) | set(skipped))

    dataset = Dataset(DATASET_ROOT)
    windows = {
        window.name: window
        for window in dataset.load_windows(
            groups=[args.group], names=sorted(set(to_write) | set(to_retract))
        )
    }
    missing = sorted(set(to_write) - set(windows))
    if missing:
        print(
            f"  {len(missing)} windows missing from the dataset, ignoring: {missing[:3]}"
        )

    vector_format = GeojsonVectorFormat()
    written = 0
    unchanged = 0
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_slice: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flipped: list[tuple[str, str, str]] = []

    for name, record in tqdm.tqdm(sorted(to_write.items()), desc="writing labels"):
        window = windows.get(name)
        if window is None:
            continue
        pool = index[name]
        human_label = record["label"]
        counts[human_label] += 1
        if record.get("reason"):
            reasons[record["reason"]] += 1
        by_split[pool["split"]][human_label] += 1
        by_slice[pool["slice"]][human_label] += 1

        # A classifier-passed detection the human called incorrect is a hard negative:
        # exactly what this round set out to collect.
        if pool["classifier_label"] == "correct" and human_label == "incorrect":
            flipped.append((name, pool["slice"], str(pool["classifier_prob_correct"])))

        properties = build_properties(record, pool, args.group)

        if args.dry_run:
            if not is_current(read_existing_label(window, vector_format), properties):
                written += 1
            else:
                unchanged += 1
            continue
        if write_label(window, properties, vector_format):
            written += 1
        else:
            unchanged += 1

    retracted = 0
    for name in to_retract:
        window = windows.get(name)
        if window is None:
            continue
        if args.dry_run:
            retracted += int(window.is_layer_completed("label"))
        else:
            retracted += int(clear_label(window))

    trainable = sum(counts[label] for label in TRAINABLE_LABELS)
    print(
        f"\n{'would write' if args.dry_run else 'wrote'} {written} label layers, "
        f"{unchanged} already current"
    )
    print(
        f"{'would retract' if args.dry_run else 'retracted'} {retracted} stale label "
        f"layers ({len(skipped)} skipped, {len(seen - set(labels))} cleared)"
    )
    print(f"\nlabels: {dict(sorted(counts.items()))}")
    print(f"trainable (correct + incorrect): {trainable}")
    if reasons:
        print(f"reasons: {dict(sorted(reasons.items()))}")
    print("\nby split:")
    for split, split_counts in sorted(by_split.items()):
        print(f"  {split:5s} {dict(sorted(split_counts.items()))}")
    print("by slice:")
    for slice_name, slice_counts in sorted(by_slice.items()):
        print(f"  {slice_name:20s} {dict(sorted(slice_counts.items()))}")
    print(
        f"\nhard negatives found (classifier said correct, human said incorrect): {len(flipped)}"
    )

    if args.dry_run:
        return

    # A labelled copy of the pool, for the record and for anything that reads the pool
    # rather than the dataset. The originals are left alone.
    with POOL_GEOJSON.open() as f:
        pool_geojson = json.load(f)
    for feature in pool_geojson["features"]:
        record = labels.get(feature["properties"]["detection_id"])
        feature["properties"]["human_label"] = record["label"] if record else ""
        feature["properties"]["annotator"] = (
            record.get("annotator", "") if record else ""
        )
        feature["properties"]["notes"] = record.get("note", "") if record else ""
        feature["properties"]["reason"] = record.get("reason") or "" if record else ""
    out_geojson = ROUND_DIR / f"{args.group}_pool_labeled.geojson"
    with out_geojson.open("w") as f:
        json.dump(pool_geojson, f)

    with POOL_CSV.open() as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    for field in ("human_label", "annotator", "notes", "reason"):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        record = labels.get(row["detection_id"])
        row["human_label"] = record["label"] if record else ""
        row["annotator"] = record.get("annotator", "") if record else ""
        row["notes"] = record.get("note", "") if record else ""
        row["reason"] = (record.get("reason") or "") if record else ""
    out_csv = ROUND_DIR / f"{args.group}_pool_labeled.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_geojson}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
