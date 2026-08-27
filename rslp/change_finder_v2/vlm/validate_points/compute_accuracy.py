"""Compute accuracy metrics for a prediction set (evaluation mode only).

Reads the prediction set produced by ``run_gemini`` and, for points that have a
ground-truth label, reports overall accuracy and a confusion matrix (treating
"positive" = a real change). Points without a label or without a prediction are skipped.

Example:
    python -m rslp.change_finder_v2.vlm.validate_points.compute_accuracy \
        --predictions predictions.json
"""

from __future__ import annotations

import argparse

from .schema import LABEL_NEGATIVE, LABEL_POSITIVE, PredictionSet


def compute_metrics(prediction_set: PredictionSet) -> dict[str, float | int]:
    """Compute accuracy and confusion-matrix counts over labeled, predicted points."""
    tp = fp = tn = fn = 0
    skipped = 0
    for item in prediction_set.predictions:
        label = item.record.label
        pred = item.prediction
        if label is None or pred is None:
            skipped += 1
            continue
        if label == LABEL_POSITIVE and pred == LABEL_POSITIVE:
            tp += 1
        elif label == LABEL_NEGATIVE and pred == LABEL_POSITIVE:
            fp += 1
        elif label == LABEL_NEGATIVE and pred == LABEL_NEGATIVE:
            tn += 1
        elif label == LABEL_POSITIVE and pred == LABEL_NEGATIVE:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "n": total,
        "skipped": skipped,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics for a prediction set."
    )
    parser.add_argument(
        "--predictions", required=True, help="Prediction set JSON from run_gemini."
    )
    parsed = parser.parse_args(args=args)

    prediction_set = PredictionSet.load(parsed.predictions)
    metrics = compute_metrics(prediction_set)

    print(f"Evaluated points:   {metrics['n']} (skipped {metrics['skipped']})")
    print(f"Accuracy:           {metrics['accuracy']:.3f}")
    print(f"Precision (change): {metrics['precision']:.3f}")
    print(f"Recall (change):    {metrics['recall']:.3f}")
    print(f"F1 (change):        {metrics['f1']:.3f}")
    print("Confusion matrix (rows=truth, cols=prediction):")
    print("            pred+   pred-")
    print(f"  truth+    {metrics['tp']:>5}   {metrics['fn']:>5}")
    print(f"  truth-    {metrics['fp']:>5}   {metrics['tn']:>5}")


if __name__ == "__main__":
    main()
