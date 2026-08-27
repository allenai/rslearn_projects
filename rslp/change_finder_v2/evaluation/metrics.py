"""Compute change-detection metrics from a method's standardized CSV.

Works on any standardized CSV produced by a change method (e.g.
``predict_change_lcc.py`` or ``worldcover/predict_change.py``), which has columns:
``row_index, lon, lat, src_year, dst_year, has_changed, src_category, dst_category,
has_prediction, predicted_changed, change_score, pred_src_category,
pred_dst_category``.

Only rows with ``has_prediction == True`` are scored. Prints AUROC, binary change
accuracy, and src/dst category accuracy to stdout, and writes a precision-recall curve
(swept at every ``--threshold-step``, default 0.05) to ``--output``.

    python -m rslp.change_finder_v2.evaluation.metrics \
        --csv eval_lcc.csv --output eval_lcc_pr_curve.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

PR_FIELDS = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]


def _as_bool(value: Any) -> bool:
    """Parse a CSV cell into a bool (handles 'True'/'False'/'1'/'0')."""
    return str(value).strip().lower() in ("true", "1")


def _confusion(
    labels: list[bool], predictions: list[bool]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for positive class = changed."""
    tp = sum(1 for y, p in zip(labels, predictions, strict=True) if y and p)
    fp = sum(1 for y, p in zip(labels, predictions, strict=True) if not y and p)
    fn = sum(1 for y, p in zip(labels, predictions, strict=True) if y and not p)
    tn = sum(1 for y, p in zip(labels, predictions, strict=True) if not y and not p)
    return tp, fp, fn, tn


def _pr_row(
    threshold: float, labels: list[bool], scores: list[float]
) -> dict[str, Any]:
    """Precision/recall/F1 and confusion counts at one score threshold."""
    predictions = [s >= threshold for s in scores]
    tp, fp, fn, tn = _confusion(labels, predictions)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _thresholds(step: float) -> list[float]:
    """Thresholds from 0.0 to 1.0 inclusive, spaced by step."""
    if step <= 0:
        raise ValueError("threshold-step must be positive")
    n = int(round(1.0 / step))
    return [min(i * step, 1.0) for i in range(n + 1)]


def compute_metrics(
    csv_path: Path,
    output: Path,
    threshold_step: float = 0.05,
    score_col: str = "change_score",
    label_col: str = "has_changed",
) -> list[dict[str, Any]]:
    """Compute AUROC/accuracy/category accuracy and write the PR-curve CSV."""
    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if _as_bool(r.get("has_prediction"))]

    if not rows:
        print("No rows with predictions to score.")
        return []

    labels = [_as_bool(r[label_col]) for r in rows]
    scores = [float(r[score_col]) for r in rows]
    predicted_changed = [_as_bool(r["predicted_changed"]) for r in rows]

    print(f"Scoring {len(rows)} rows with predictions from {csv_path}")

    # AUROC (needs both classes present).
    if len(set(labels)) < 2:
        print("AUROC: n/a (only one class present in has_changed)")
    else:
        print(f"AUROC: {roc_auc_score(labels, scores):.4f}")

    # Binary change accuracy from the argmax-based predicted_changed.
    tp, fp, fn, tn = _confusion(labels, predicted_changed)
    accuracy = (tp + tn) / len(rows)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print("\nBinary change (positive = changed, from predicted_changed):")
    print(f"  accuracy : {accuracy:.4f}  ({tp + tn}/{len(rows)})")
    print(f"  precision: {precision:.4f}")
    print(f"  recall   : {recall:.4f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    # src/dst category accuracy where both GT and prediction are "changed".
    both_changed = [
        r for r, y, p in zip(rows, labels, predicted_changed, strict=True) if y and p
    ]
    if both_changed:
        n = len(both_changed)
        src_correct = sum(
            1 for r in both_changed if r["src_category"] == r["pred_src_category"]
        )
        dst_correct = sum(
            1 for r in both_changed if r["dst_category"] == r["pred_dst_category"]
        )
        print(f"\nCategory accuracy (over {n} points changed in both GT & pred):")
        print(f"  src: {src_correct / n:.4f}  ({src_correct}/{n})")
        print(f"  dst: {dst_correct / n:.4f}  ({dst_correct}/{n})")
    else:
        print("\nNo points where both GT and prediction are 'changed'.")

    # Precision-recall curve over thresholds.
    pr_rows = [_pr_row(t, labels, scores) for t in _thresholds(threshold_step)]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PR_FIELDS)
        writer.writeheader()
        writer.writerows(pr_rows)

    print(f"\nWrote PR curve ({len(pr_rows)} thresholds) to {output}")
    return pr_rows


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute change-detection metrics from a standardized method CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Standardized method CSV (from predict_change_lcc.py or worldcover).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output precision-recall curve CSV path.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Spacing of PR-curve thresholds from 0 to 1. Default: 0.05.",
    )
    parser.add_argument(
        "--score-col",
        default="change_score",
        help="Score column for AUROC / PR curve. Default: change_score.",
    )
    parser.add_argument(
        "--label-col",
        default="has_changed",
        help="Ground-truth label column. Default: has_changed.",
    )
    args = parser.parse_args()

    compute_metrics(
        csv_path=args.csv,
        output=args.output,
        threshold_step=args.threshold_step,
        score_col=args.score_col,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
