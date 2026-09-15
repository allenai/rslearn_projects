"""Smoke test scored at every candidate operating threshold, from one pipeline run.

``smoke_test.py`` scores each scene at whatever ``positive_class_threshold`` the
classifier config happens to set, so comparing operating points means re-downloading and
re-running every scene per threshold. The threshold is not part of the model though: the
classifier writes ``prob(correct)`` for each candidate and the verdict is a comparison
against it, and ``smoke_test.py``'s metric (``classifier_count``) is exactly the number of
candidates that clear it — the infra filter and attribute model run afterwards and do not
feed that count.

So this runs the pipeline once per scene with ``include_rejected=True``, which keeps every
detector candidate along with its probability, and then scores the same pass/fail table at
each threshold offline. One pass over the scenes answers both "does it pass" and "at which
threshold".

Per-scene probabilities are cached under ``--out_dir``, so a re-run skips scenes already
done and a crash mid-sweep costs only the scene in flight.

Usage:
    python -m rslp.landsat_vessels.evaluation.smoke_test_sweep \
        --classify_config data/landsat_vessels/config_classifier_round1_plus_20260803.yaml
"""

import argparse
import json
import os
import shutil
import sys
import traceback

from rslp.landsat_vessels.evaluation.smoke_test import SCENES
from rslp.utils.mp import init_mp

# predict_pipeline is re-exported as a function by the package __init__, shadowing the
# module, so reach the module directly to override its config constant.
_landsat_mod = sys.modules["rslp.landsat_vessels.predict_pipeline"]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def run_scene(scene_id: str, out_dir: str, keep_scratch: bool) -> dict:
    """Run the pipeline on one scene and return its candidate probabilities."""
    from rslp.landsat_vessels.predict_pipeline import predict_pipeline

    scratch = f"/tmp/smoke_{scene_id}"
    result = predict_pipeline(
        scene_id=scene_id,
        scratch_path=scratch,
        include_rejected=True,
    )
    try:
        probs = []
        for detection in result.detections:
            prob = detection.metadata.get("classifier_prob_correct")
            if prob is not None:
                probs.append(float(prob))
        return {
            "scene_id": scene_id,
            "detector_count": result.detector_count,
            "classifier_count_config_threshold": result.classifier_count,
            "n_candidates_with_prob": len(probs),
            "probs": probs,
        }
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    """Run each scene once, then score the smoke test at every threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classify_config",
        default=None,
        help="classifier config to use (defaults to the pipeline's configured one)",
    )
    parser.add_argument("--out_dir", default="/tmp/smoke_sweep")
    parser.add_argument(
        "--keep_scratch",
        action="store_true",
        help="keep each scene's rslearn scratch dataset (large)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.classify_config:
        _landsat_mod.CLASSIFY_MODEL_CONFIG = args.classify_config  # type: ignore[attr-defined]
    print(f"classifier config: {_landsat_mod.CLASSIFY_MODEL_CONFIG}")

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for scene_id, (low, high), desc in SCENES:
        cache_path = os.path.join(args.out_dir, f"{scene_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                results[scene_id] = json.load(f)
            print(f"[CACHED] {scene_id} ({desc})")
            continue
        print(f"[RUNNING] {scene_id} ({desc}) expected [{low}, {high}]...", flush=True)
        try:
            record = run_scene(scene_id, args.out_dir, args.keep_scratch)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failures[scene_id] = str(e)
            continue
        results[scene_id] = record
        with open(cache_path, "w") as f:
            json.dump(record, f)
        print(
            f"  detector={record['detector_count']} "
            f"classifier@config={record['classifier_count_config_threshold']} "
            f"candidates_with_prob={record['n_candidates_with_prob']}",
            flush=True,
        )

    if not results:
        print("\nno scenes completed; nothing to score")
        raise SystemExit(1)

    # ---- Score the smoke test at each threshold ----
    print("\n" + "=" * 110)
    print("Classifier count per scene at each threshold (expected range in brackets)")
    print("=" * 110)
    header = f"{'scene':<48}{'exp':>10}{'det':>7}"
    for thr in THRESHOLDS:
        header += f"{thr:>7.2f}"
    print(header)
    print("-" * 110)

    passes: dict[float, int] = {thr: 0 for thr in THRESHOLDS}
    for scene_id, (low, high), _desc in SCENES:
        rec = results.get(scene_id)
        if rec is None:
            print(f"{scene_id:<48}{f'[{low},{high}]':>10}{'ERROR':>7}")
            continue
        row = f"{scene_id:<48}{f'[{low},{high}]':>10}{rec['detector_count']:>7}"
        for thr in THRESHOLDS:
            count = sum(1 for p in rec["probs"] if p >= thr)
            ok = low <= count <= high
            if ok:
                passes[thr] += 1
            row += f"{count:>6}{'*' if ok else ' '}"
        print(row)

    print("-" * 110)
    total = len([s for s in SCENES if s[0] in results])
    summary = f"{'PASS / scored (* = in range)':<48}{'':>10}{'':>7}"
    for thr in THRESHOLDS:
        summary += f"{passes[thr]:>5}/{total}"[-7:]
    print(summary)
    print("-" * 110)

    best = max(THRESHOLDS, key=lambda t: (passes[t], t))
    print(f"\nBest threshold on these scenes: {best:.2f} ({passes[best]}/{total} pass)")
    if failures:
        print(f"\nScenes that errored ({len(failures)}):")
        for scene_id, err in failures.items():
            print(f"  {scene_id}: {err}")

    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as f:
        json.dump(
            {
                "classify_config": _landsat_mod.CLASSIFY_MODEL_CONFIG,
                "passes_by_threshold": {str(k): v for k, v in passes.items()},
                "scored": total,
                "failures": failures,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    init_mp()
    main()
