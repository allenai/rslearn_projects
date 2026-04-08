"""Show confusion matrices for speed, heading, and vessel type within each length bucket."""

import argparse
import csv
from collections import defaultdict

import numpy as np

SHIP_TYPES = [
    "cargo",
    "tanker",
    "passenger",
    "service",
    "tug",
    "pleasure",
    "fishing",
    "enforcement",
    "sar",
]


def bucketize(value: float, thresholds: list[float]) -> int:
    """Get bucket index for the given value and bucket thresholds."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def bucket_label(thresholds: list[float], idx: int) -> str:
    """Get the name to use for the given bucket index.

    The name is based on the range of values falling into that bucket.
    """
    if idx == 0:
        return f"0-{thresholds[0]}"
    if idx == len(thresholds):
        return f"{thresholds[-1]}+"
    return f"{thresholds[idx-1]}-{thresholds[idx]}"


def heading_bucket(degrees: float, thresholds: list[float]) -> int:
    """Bucketize heading, normalizing to [0, 360) first."""
    degrees = degrees % 360
    return bucketize(degrees, thresholds)


def print_confusion_matrix(
    gt_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    title: str,
) -> None:
    """Print the confusion matrix for this subset of labels."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gt_labels, pred_labels):
        cm[g, p] += 1

    print(f"\n{'=' * 60}")
    print(f"  {title}  (n={len(gt_labels)})")
    print(f"{'=' * 60}")

    col_w = max(len(c) for c in class_names) + 2
    print(f"{'GT / Pred':>{col_w}}" + "".join(c.rjust(col_w) for c in class_names))
    for i, name in enumerate(class_names):
        row_str = "".join(str(cm[i, j]).rjust(col_w) for j in range(n))
        print(f"{name:>{col_w}}{row_str}")

    correct = np.trace(cm)
    total = cm.sum()
    if total > 0:
        print(f"  Accuracy: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confusion matrices per length bucket from vessel attribute CSV",
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Input CSV from predictions_to_csv.py"
    )
    parser.add_argument(
        "--length_buckets",
        type=float,
        nargs="+",
        default=[50, 100, 200],
        help="Length bucket thresholds (meters)",
    )
    parser.add_argument(
        "--speed_buckets",
        type=float,
        nargs="+",
        default=[2, 4, 8],
        help="Speed bucket thresholds (knots)",
    )
    parser.add_argument(
        "--heading_buckets",
        type=float,
        nargs="+",
        default=[45, 90, 135, 180, 225, 270, 315],
        help="Heading bucket thresholds (degrees, 0-360)",
    )
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    # Group rows by GT length bucket.
    length_names = [
        bucket_label(args.length_buckets, i)
        for i in range(len(args.length_buckets) + 1)
    ]
    groups: dict[int, list[dict]] = defaultdict(list)
    skipped_no_length = 0
    for row in rows:
        if row["gt_length"] == "":
            skipped_no_length += 1
            continue
        lb = bucketize(float(row["gt_length"]), args.length_buckets)
        groups[lb].append(row)

    print(f"Total rows: {len(rows)}, skipped (no gt_length): {skipped_no_length}")
    for lb_idx in sorted(groups):
        group = groups[lb_idx]
        length_label = length_names[lb_idx]
        print(f"\n{'#' * 70}")
        print(f"  LENGTH BUCKET: {length_label}  ({len(group)} vessels)")
        print(f"{'#' * 70}")

        # --- Speed confusion matrix ---
        speed_names = [
            bucket_label(args.speed_buckets, i)
            for i in range(len(args.speed_buckets) + 1)
        ]
        gt_speed, pred_speed = [], []
        for row in group:
            if row["gt_sog"] == "" or row["pred_sog"] == "":
                continue
            gt_speed.append(bucketize(float(row["gt_sog"]), args.speed_buckets))
            pred_speed.append(bucketize(float(row["pred_sog"]), args.speed_buckets))
        if gt_speed:
            print_confusion_matrix(
                gt_speed, pred_speed, speed_names, f"Speed | length={length_label}"
            )

        # --- Heading confusion matrix ---
        heading_names = [
            bucket_label(args.heading_buckets, i)
            for i in range(len(args.heading_buckets) + 1)
        ]
        gt_heading, pred_heading = [], []
        for row in group:
            if row["gt_cog"] == "" or row["pred_heading"] == "":
                continue
            gt_heading.append(
                heading_bucket(float(row["gt_cog"]), args.heading_buckets)
            )
            pred_heading.append(
                heading_bucket(float(row["pred_heading"]), args.heading_buckets)
            )
        if gt_heading:
            print_confusion_matrix(
                gt_heading,
                pred_heading,
                heading_names,
                f"Heading | length={length_label}",
            )

        # --- Ship type confusion matrix ---
        gt_type, pred_type = [], []
        for row in group:
            gt_t = row["gt_type"]
            pred_t = row["pred_type"]
            if (
                gt_t == ""
                or pred_t == ""
                or gt_t not in SHIP_TYPES
                or pred_t not in SHIP_TYPES
            ):
                continue
            gt_type.append(SHIP_TYPES.index(gt_t))
            pred_type.append(SHIP_TYPES.index(pred_t))
        if gt_type:
            print_confusion_matrix(
                gt_type, pred_type, SHIP_TYPES, f"Ship type | length={length_label}"
            )
