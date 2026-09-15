"""Smoke test scored over a 2D grid of detector x classifier thresholds.

``smoke_test_sweep.py`` sweeps only the classifier threshold: it runs the pipeline once
per scene (with ``include_rejected=True``) and then counts, at each classifier
``prob(correct)`` threshold, how many candidates clear it. The detector operating point
is fixed at whatever ``score_threshold`` the detector config sets (0.7 in
``config_detector.yaml``).

This variant also sweeps the *detector* threshold. Two things make that possible from a
single pass per scene:

  1. Each detector candidate carries its own detector confidence (``VesselDetection.score``)
     and its classifier probability (``metadata["classifier_prob_correct"]``). Recording
     both per candidate lets us re-apply any (detector_thr, classifier_thr) pair offline.
  2. The detector's own ``score_threshold`` is lowered to ``--det_floor`` (default 0.5) for
     the run, so candidates with detector scores down to that floor are emitted. Any
     detector threshold >= det_floor in the grid is then reproduced by filtering offline;
     det_floor=0.7 reproduces the production detector.

A candidate "passes" a cell (det_thr, cls_thr) iff ``score >= det_thr`` AND
``prob(correct) >= cls_thr``. The smoke-test metric is the per-scene candidate count vs
its expected ``[low, high]`` range; the cell's score is how many scenes land in range.

Per-scene candidates are cached under ``--out_dir`` so re-runs skip completed scenes.

Usage:
    python -m rslp.landsat_vessels.evaluation.smoke_test_sweep_2d \
        --classify_config data/landsat_vessels/config_classifier_20260908d.yaml \
        --det_floor 0.5
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any

import yaml

from rslp.landsat_vessels.evaluation.smoke_test import SCENES
from rslp.utils.mp import init_mp

# predict_pipeline is re-exported as a function by the package __init__, shadowing the
# module, so reach the module directly to override its config constants (the pipeline
# reads DETECT_MODEL_CONFIG / CLASSIFY_MODEL_CONFIG as module globals).
_landsat_mod = sys.modules["rslp.landsat_vessels.predict_pipeline"]

# Detector thresholds must all be >= --det_floor (the floor the detector actually ran at).
DET_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CLS_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _set_score_threshold(obj: object, value: float) -> int:
    """Recursively set every ``score_threshold`` key in a nested dict/list. Returns count."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "score_threshold":
                obj[k] = value
                n += 1
            else:
                n += _set_score_threshold(v, value)
    elif isinstance(obj, list):
        for v in obj:
            n += _set_score_threshold(v, value)
    return n


def _detector_config_with_floor(det_floor: float) -> str:
    """Write a temp detector config whose score_threshold is lowered to det_floor."""
    with open(_landsat_mod.DETECT_MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    n = _set_score_threshold(cfg, det_floor)
    if n == 0:
        raise ValueError("no score_threshold key found in detector config")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="detector_floor_", delete=False
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    print(
        f"patched detector config: score_threshold -> {det_floor} ({n} site(s)) at {tmp.name}"
    )
    return tmp.name


def run_scene(scene_id: str, keep_scratch: bool) -> dict:
    """Run the pipeline on one scene, returning per-candidate (det score, cls prob)."""
    from rslp.landsat_vessels.predict_pipeline import predict_pipeline

    scratch = f"/tmp/smoke2d_{scene_id}"
    result = predict_pipeline(
        scene_id=scene_id,
        scratch_path=scratch,
        include_rejected=True,
    )
    try:
        candidates = []
        for det in result.detections:
            prob = det.metadata.get("classifier_prob_correct")
            candidates.append(
                {
                    "score": float(det.score) if det.score is not None else None,
                    "prob": float(prob) if prob is not None else None,
                }
            )
        return {
            "scene_id": scene_id,
            "detector_count": result.detector_count,
            "classifier_count_config_threshold": result.classifier_count,
            "candidates": candidates,
        }
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _count(candidates: list[dict], det_thr: float, cls_thr: float) -> int:
    return sum(
        1
        for c in candidates
        if c["score"] is not None
        and c["prob"] is not None
        and c["score"] >= det_thr
        and c["prob"] >= cls_thr
    )


def main() -> None:
    """Run each scene once at the detector floor, then score the det x cls grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classify_config",
        default=None,
        help="classifier config to use (defaults to the pipeline's configured one)",
    )
    parser.add_argument(
        "--det_floor",
        type=float,
        default=0.5,
        help="detector score_threshold to run at; grid det thresholds must be >= this",
    )
    parser.add_argument("--out_dir", default="/tmp/smoke_sweep_2d")
    parser.add_argument("--keep_scratch", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.classify_config:
        _landsat_mod.CLASSIFY_MODEL_CONFIG = os.path.abspath(args.classify_config)  # type: ignore[attr-defined]
    det_cfg = _detector_config_with_floor(args.det_floor)
    _landsat_mod.DETECT_MODEL_CONFIG = det_cfg  # type: ignore[attr-defined]
    print(f"classifier config: {_landsat_mod.CLASSIFY_MODEL_CONFIG}")
    print(f"detector config:   {_landsat_mod.DETECT_MODEL_CONFIG}")

    det_thresholds = [t for t in DET_THRESHOLDS if t >= args.det_floor - 1e-9]

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
            record = run_scene(scene_id, args.keep_scratch)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failures[scene_id] = str(e)
            continue
        results[scene_id] = record
        with open(cache_path, "w") as f:
            json.dump(record, f)
        print(
            f"  detector(floor)={record['detector_count']} "
            f"candidates={len(record['candidates'])}",
            flush=True,
        )

    if not results:
        print("\nno scenes completed; nothing to score")
        raise SystemExit(1)

    total = len([s for s in SCENES if s[0] in results])

    # ---- Pass matrix: rows = detector threshold, cols = classifier threshold ----
    print("\n" + "=" * 100)
    print(
        f"Scenes in range [low,high] per (detector_thr row x classifier_thr col). Max = {total}"
    )
    print("=" * 100)
    corner = "det\\cls"
    header = f"{corner:>8}" + "".join(f"{c:>6.2f}" for c in CLS_THRESHOLDS)
    print(header)
    print("-" * len(header))
    passes_grid: dict[tuple, int] = {}
    for dt in det_thresholds:
        row = f"{dt:>8.2f}"
        for ct in CLS_THRESHOLDS:
            p = 0
            for scene_id, (low, high), _d in SCENES:
                rec = results.get(scene_id)
                if rec is None:
                    continue
                if low <= _count(rec["candidates"], dt, ct) <= high:
                    p += 1
            passes_grid[(dt, ct)] = p
            row += f"{p:>6}"
        print(row)

    best = max(passes_grid.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
    (best_dt, best_ct), best_p = best
    print("-" * len(header))
    print(
        f"\nBest cell: detector_thr={best_dt:.2f}, classifier_thr={best_ct:.2f} -> {best_p}/{total} scenes in range"
    )

    # List all cells that achieve the max.
    top = sorted([k for k, v in passes_grid.items() if v == best_p])
    print(
        f"All cells achieving {best_p}/{total}: "
        + ", ".join(f"(det={d:.2f},cls={c:.2f})" for d, c in top)
    )

    # ---- Per-scene counts at the best cell ----
    print("\n" + "=" * 100)
    print(
        f"Per-scene candidate counts at best cell (det>={best_dt:.2f}, cls>={best_ct:.2f})"
    )
    print("=" * 100)
    print(f"{'scene':<48}{'expected':>12}{'count':>7}{'':>3}{'desc'}")
    print("-" * 100)
    for scene_id, (low, high), desc in SCENES:
        rec = results.get(scene_id)
        if rec is None:
            print(f"{scene_id:<48}{f'[{low},{high}]':>12}{'ERR':>7}")
            continue
        cnt = _count(rec["candidates"], best_dt, best_ct)
        ok = "*" if low <= cnt <= high else " "
        print(f"{scene_id:<48}{f'[{low},{high}]':>12}{cnt:>7}{ok:>3}  {desc}")

    # ---- Detection-count matrices: how counts move as thresholds change ----
    # These are what a user tunes against: for each (det_thr row x cls_thr col) cell,
    # how many detections survive. One matrix per scene, plus the total across scenes.
    def print_count_matrix(title: str, count_at: Any) -> None:
        print("\n" + "=" * len(header))
        print(title)
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for dt in det_thresholds:
            row = f"{dt:>8.2f}"
            for ct in CLS_THRESHOLDS:
                row += f"{count_at(dt, ct):>6}"
            print(row)

    counts_grid_by_scene: dict[str, dict[str, int]] = {}
    for scene_id, (low, high), desc in SCENES:
        rec = results.get(scene_id)
        if rec is None:
            continue
        counts_grid_by_scene[scene_id] = {
            f"{dt}|{ct}": _count(rec["candidates"], dt, ct)
            for dt in det_thresholds
            for ct in CLS_THRESHOLDS
        }
        print_count_matrix(
            f"Detections for {scene_id} ({desc}, expected [{low},{high}])"
            f"  [det row x cls col]",
            lambda dt, ct, r=rec: _count(r["candidates"], dt, ct),
        )

    print_count_matrix(
        "TOTAL detections across all scenes  [det row x cls col]",
        lambda dt, ct: sum(
            _count(rec["candidates"], dt, ct) for rec in results.values()
        ),
    )

    with open(os.path.join(args.out_dir, "sweep2d_summary.json"), "w") as f:
        json.dump(
            {
                "classify_config": _landsat_mod.CLASSIFY_MODEL_CONFIG,
                "det_floor": args.det_floor,
                "det_thresholds": det_thresholds,
                "cls_thresholds": CLS_THRESHOLDS,
                "scored": total,
                "passes_grid": {f"{d}|{c}": v for (d, c), v in passes_grid.items()},
                "counts_grid_by_scene": counts_grid_by_scene,
                "best": {"det": best_dt, "cls": best_ct, "passes": best_p},
                "failures": failures,
            },
            f,
            indent=2,
        )
    if failures:
        print(f"\nScenes that errored ({len(failures)}):")
        for scene_id, err in failures.items():
            print(f"  {scene_id}: {err}")


if __name__ == "__main__":
    init_mp()
    main()
