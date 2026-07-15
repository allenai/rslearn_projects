"""Diagnostic: dump classifier probabilities for one scene.

Runs the full pipeline on a single scene, keeps the scratch dataset, then reads the
per-detection classifier outputs from the ``classify_predict`` group and reports the
distribution of ``prob`` values vs. the assigned ``label``.

This is used to check whether ``positive_class_threshold`` (0.85) is actually being
applied: if detections with prob(correct) in [0.5, 0.85) are labeled "correct", the
threshold is being ignored by the installed rslearn version.
"""

import argparse
import json
from pathlib import Path

from rslp.landsat_vessels.predict_pipeline import predict_pipeline
from rslp.utils.mp import init_mp

DEFAULT_SCENE = "LC09_L1TP_001090_20241103_20241103_02_T1"  # whitecaps, expected [0,10]


def _extract_prob(props: dict) -> float | None:
    """Pull the probability of the positive class ("correct") from feature props."""
    prob = props.get("prob")
    if prob is None:
        return None
    if isinstance(prob, int | float):
        return float(prob)
    if isinstance(prob, dict):
        if "correct" in prob:
            return float(prob["correct"])
        return float(max(prob.values()))
    if isinstance(prob, list | tuple) and prob:
        # Assume [correct, incorrect] ordering per config classes.
        return float(prob[0])
    return None


def main() -> None:
    """Run one scene and print the classifier probability breakdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default=DEFAULT_SCENE)
    parser.add_argument("--scratch", default="/tmp/inspect_probs_scratch")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    init_mp()

    result = predict_pipeline(scene_id=args.scene_id, scratch_path=args.scratch)
    print(
        f"\nScene {args.scene_id}: detector={result.detector_count} "
        f"classifier={result.classifier_count}\n"
    )

    # Find every geojson written under the classify_predict group.
    geojson_paths = [
        p for p in Path(args.scratch).rglob("*.geojson") if "classify_predict" in str(p)
    ]
    if not geojson_paths:
        print("No classify_predict geojson outputs found under scratch.")
        print("Looked under:", args.scratch)
        return

    rows: list[tuple[str, float | None]] = []
    for gp in geojson_paths:
        try:
            with open(gp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            if "label" not in props:
                continue
            rows.append((props["label"], _extract_prob(props)))

    if not rows:
        print("Found geojson files but no features with a 'label' property.")
        return

    correct = [prob for label, prob in rows if label == "correct"]
    have_prob = [p for p in correct if p is not None]

    print(f"Total classifier windows read: {len(rows)}")
    print(f"Labeled 'correct': {len(correct)}")

    if not have_prob:
        print(
            "\nNo 'prob' property present on the outputs -- the writer isn't emitting "
            "probabilities, so the threshold cannot be verified this way."
        )
        return

    below = [p for p in have_prob if p < args.threshold]
    print(
        f"Of those, {len(below)} have prob(correct) < {args.threshold} "
        f"(these should have been filtered out if the threshold were applied)."
    )

    buckets = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    print("\nprob(correct) histogram for label=='correct':")
    for lo, hi in zip(buckets, buckets[1:]):
        n = sum(1 for p in have_prob if lo <= p < hi)
        bar = "#" * n
        print(f"  [{lo:.2f}, {hi:.2f}): {n:3d} {bar}")

    verdict = (
        "THRESHOLD IGNORED (regression confirmed)"
        if below
        else "threshold appears to be applied"
    )
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
