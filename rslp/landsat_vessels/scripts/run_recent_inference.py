"""Run the Landsat vessel pipeline over a sampled scene list and render candidate grids.

Uses rslp.landsat_vessels.predict_pipeline.predict_pipeline (the API we deploy) with
include_rejected=True, so EVERY detector candidate above the detector's score_threshold is
carried through the classifier (each with classifier_prob_correct). The detector is patched
to a lower --det_thr (default 0.7) so the 0.7-0.9 candidate band is visible for inspection.

For each scene an annotated ``candidates_grid.png`` is rendered with a 2D detector x
classifier colour scheme (relative to --cls_thr, default 0.9):
  * green  : det>=0.9 & cls>=cls_thr   (previously kept, at the old 0.9 detector cutoff)
  * red    : det>=0.9 & cls< cls_thr
  * orange : det 0.7-0.9 & cls>=cls_thr (confident vessels below the old detector cutoff)
  * yellow : det 0.7-0.9 & cls< cls_thr
Per-candidate crops are rendered into the grid then discarded (they live only in scratch).
CSV/row stats (num_detections, min/max/median prob) are over the green detections.

The deployed operating point is det0.7 / cls0.99 (see rslp.landsat_vessels.config); this
script keeps --det_thr / --cls_thr configurable so candidate bands can be inspected.

Parallelism uses independent shard subprocesses (predict_pipeline spawns its own worker
processes for materialize, so a daemonic multiprocessing.Pool can't host it). The
orchestrator (--workers N, no --shard) writes the patched configs, launches N shard
subprocesses (round-robin over the scene list), waits, then assembles the CSV.
Resumable: a scene whose rows/<scene_id>.json already exists is skipped.

Outputs under --out_dir:
  * detections.csv        -- scene_id, num_detections, min/max/median classifier prob
  * figures/<scene_id>/   -- candidates_grid.png only, and ONLY for scenes with >=1
                             detector candidate (no empty folders); per-candidate crops
                             are rendered into it then discarded (live only in scratch)
  * rows/<scene_id>.json  -- per-scene record (for resume + final CSV assembly)
  * patched_detector.yaml, patched_classifier.yaml -- the exact configs used

Usage:
    python -m rslp.landsat_vessels.scripts.run_recent_inference \
        --scene_csv /weka/dfive-default/yawenz/landsat/20260908_results/sample_scenes.csv \
        --out_dir  /weka/dfive-default/yawenz/landsat/20260908_results \
        --classify_config data/landsat_vessels/config_classifier_20260908d.yaml \
        --workers 10
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess  # nosec B404 - orchestrates our own shard subprocesses
import sys
import time
import traceback
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

# Import the predict_pipeline module explicitly (rather than fetching it from
# sys.modules) so this script works regardless of whether the landsat_vessels package
# __init__ imports predict_pipeline eagerly. We patch DETECT_MODEL_CONFIG /
# CLASSIFY_MODEL_CONFIG on this module object in shard mode.
import rslp.landsat_vessels.predict_pipeline as _landsat_mod
from rslp.utils.mp import init_mp

CSV_FIELDS = [
    "scene_id",
    "num_detections",
    "detector_candidates",
    "n_green",
    "n_red",
    "n_orange",
    "n_yellow",
    "min_prob",
    "max_prob",
    "median_prob",
    "datetime",
    "cloud_cover",
    "lat",
    "lon",
    "seconds",
    "status",
]

# Grid appearance for the per-scene candidate visualization.
GRID_CELL = 288
GRID_COLS = 6
# Detector "high" threshold that splits green/red (deploy point) from orange/yellow.
GRID_DET_HI = 0.9

# 2D det x cls color scheme (borders):
#   green  : det>=det_hi & cls>=cls_thr  (deployable kept)
#   red    : det>=det_hi & cls< cls_thr
#   orange : det< det_hi & cls>=cls_thr
#   yellow : det< det_hi & cls< cls_thr
COLOR_GREEN = (40, 200, 60)
COLOR_RED = (220, 50, 50)
COLOR_ORANGE = (240, 140, 30)
COLOR_YELLOW = (230, 215, 40)


def _candidate_color(
    det: float, prob: float, cls_thr: float, det_hi: float
) -> tuple[int, int, int]:
    if det >= det_hi:
        return COLOR_GREEN if prob >= cls_thr else COLOR_RED
    return COLOR_ORANGE if prob >= cls_thr else COLOR_YELLOW


def _font(size: int) -> Any:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def collect_candidates(result: Any, cls_thr: float) -> list[dict]:
    """Flatten a PipelineResult (run with include_rejected=True) into candidate dicts.

    Each candidate carries its detector score, classifier prob(correct), verdict, and
    the path to its pan-sharpened RGB crop, sorted by detector score then prob desc.
    """
    cands = []
    for d in result.detections:
        prob = d.metadata.get("classifier_prob_correct")
        rgb = d.crop_fnames.get("rgb") if d.crop_fnames else None
        cands.append(
            {
                "col": d.col,
                "row": d.row,
                "det_score": float(d.score) if d.score is not None else None,
                "prob": float(prob) if prob is not None else None,
                "label": d.metadata.get("classifier_label"),
                "rgb": str(rgb) if rgb is not None else None,
            }
        )
    # Sort by detector score desc, then classifier prob desc, so the high-detector
    # (green/red) candidates come first and the 0.7-0.9 band (orange/yellow) after.
    cands.sort(key=lambda c: (-(c["det_score"] or 0.0), -(c["prob"] or 0.0)))
    return cands


def render_grid(
    cands: list[dict],
    out_path: str,
    cls_thr: float,
    det_hi: float = GRID_DET_HI,
    cell: int = GRID_CELL,
    cols: int = GRID_COLS,
) -> None:
    """Render an annotated grid with the 2D det x cls color scheme + a legend header."""
    pad, label_h = 6, 34
    legend_h = 40
    n = max(len(cands), 1)
    rows = (n + cols - 1) // cols
    cw = cell + 2 * pad
    ch = cell + label_h + 2 * pad
    width = cols * cw
    grid = Image.new("RGB", (width, legend_h + rows * ch), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    font = _font(15)
    legend_font = _font(16)

    # Legend header.
    legend = [
        (COLOR_GREEN, "det>0.9 cls>0.9 (kept)"),
        (COLOR_RED, "det>0.9 cls<0.9"),
        (COLOR_ORANGE, "det0.7-0.9 cls>0.9"),
        (COLOR_YELLOW, "det0.7-0.9 cls<0.9"),
    ]
    lx = 8
    for color, text in legend:
        draw.rectangle([lx, 12, lx + 18, 30], fill=color)
        draw.text((lx + 24, 12), text, fill=(230, 230, 230), font=legend_font)
        lx += 30 + int(draw.textlength(text, font=legend_font)) + 22

    for i, c in enumerate(cands):
        gx = (i % cols) * cw
        gy = legend_h + (i // cols) * ch
        det = c["det_score"] or 0.0
        prob = c["prob"] or 0.0
        border = _candidate_color(det, prob, cls_thr, det_hi)
        if c["rgb"] and os.path.exists(c["rgb"]):
            im = Image.open(c["rgb"]).convert("RGB").resize((cell, cell), Image.NEAREST)
        else:
            im = Image.new("RGB", (cell, cell), (60, 60, 60))
        grid.paste(im, (gx + pad, gy + pad + label_h))
        draw.rectangle(
            [
                gx + pad - 2,
                gy + pad + label_h - 2,
                gx + pad + cell + 1,
                gy + pad + label_h + cell + 1,
            ],
            outline=border,
            width=3,
        )
        pp = f"{c['prob']:.3f}" if c["prob"] is not None else "n/a"
        ds = f"{c['det_score']:.2f}" if c["det_score"] is not None else "n/a"
        draw.text((gx + pad, gy + 2), f"#{i} cls={pp}", fill=border, font=font)
        draw.text((gx + pad, gy + 17), f"det={ds}", fill=(200, 200, 200), font=font)
    grid.save(out_path)


def _set_key(obj: object, key: str, value: float) -> int:
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                obj[k] = value
                n += 1
            else:
                n += _set_key(v, key, value)
    elif isinstance(obj, list):
        for v in obj:
            n += _set_key(v, key, value)
    return n


def make_detector_config(det_thr: float, out_path: str) -> None:
    """Copy the detector config with score_threshold set to det_thr."""
    with open(_landsat_mod.DETECT_MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if _set_key(cfg, "score_threshold", det_thr) == 0:
        raise ValueError("no score_threshold in detector config")
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)


def make_classifier_config(src: str, cls_thr: float, out_path: str) -> None:
    """Copy the Run-d classifier config with positive_class_threshold set to cls_thr."""
    with open(src) as f:
        cfg = yaml.safe_load(f)
    task_args = cfg["data"]["init_args"]["task"]["init_args"]
    if task_args.get("positive_class") is None:
        raise ValueError("classifier task has no positive_class; cannot threshold")
    task_args["positive_class_threshold"] = cls_thr
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)


def process_scene(scene: dict, out_dir: str, scratch_root: str, cls_thr: float) -> dict:
    """Run the pipeline on one scene; render the candidate grid + per-scene row JSON.

    The pipeline is run with include_rejected=True so every detector candidate is carried
    through the classifier (each with classifier_prob_correct), crops are written for all
    of them (temporarily, only the grid is persisted), and an annotated candidates_grid.png
    is rendered with the 2D detector x classifier colour scheme. Detection/prob stats in the
    row are over the green detections (det>=0.9 & prob>=cls_thr).
    """
    from rslp.landsat_vessels.predict_pipeline import predict_pipeline

    scene_id = scene["scene_id"]
    rows_dir = os.path.join(out_dir, "rows")
    fig_dir = os.path.join(out_dir, "figures", scene_id)
    row_path = os.path.join(rows_dir, f"{scene_id}.json")

    base = {
        "scene_id": scene_id,
        "datetime": scene.get("datetime", ""),
        "cloud_cover": scene.get("cloud_cover", ""),
        "lat": scene.get("lat", ""),
        "lon": scene.get("lon", ""),
    }
    # Transient S3 read failures under high concurrency can leave a materialized band
    # missing (retry_max_attempts=0 in the pipeline turns that into a scene failure), so
    # retry the whole scene a few times before giving up. Each attempt uses its OWN fresh
    # scratch dir: predict_pipeline only recreates the scratch root, not the data source's
    # cache/ subtree, so reusing a wiped path would fail setup -- a unique path per attempt
    # makes every retry behave like a pristine first run. Crops go under scratch (only the
    # candidates_grid.png is persisted, and only when there is >=1 candidate).
    max_attempts = 3
    row = None
    for attempt in range(1, max_attempts + 1):
        scratch = os.path.join(scratch_root, f"{scene_id}__a{attempt}")
        crop_dir = os.path.join(scratch, "crops")
        start = time.time()
        try:
            result = predict_pipeline(
                scene_id=scene_id,
                scratch_path=scratch,
                crop_path=crop_dir,
                include_rejected=True,
            )
            cands = collect_candidates(result, cls_thr)
            # Only persist a figure folder + grid when there is something to show.
            if cands:
                os.makedirs(fig_dir, exist_ok=True)
                render_grid(
                    cands, os.path.join(fig_dir, "candidates_grid.png"), cls_thr
                )
            # Per-color tallies for the 2D det x cls scheme.
            n_green = n_red = n_orange = n_yellow = 0
            for c in cands:
                col = _candidate_color(
                    c["det_score"] or 0.0, c["prob"] or 0.0, cls_thr, GRID_DET_HI
                )
                n_green += col == COLOR_GREEN
                n_red += col == COLOR_RED
                n_orange += col == COLOR_ORANGE
                n_yellow += col == COLOR_YELLOW
            # "kept" = deployable operating point (det>=det_hi & cls>=cls_thr) = green.
            kept = [
                c
                for c in cands
                if (c["det_score"] or 0.0) >= GRID_DET_HI
                and (c["prob"] or 0.0) >= cls_thr
            ]
            probs = [c["prob"] for c in kept if c["prob"] is not None]
            row = {
                **base,
                "num_detections": len(kept),
                "detector_candidates": len(cands),
                "n_green": n_green,
                "n_red": n_red,
                "n_orange": n_orange,
                "n_yellow": n_yellow,
                "min_prob": round(min(probs), 4) if probs else "",
                "max_prob": round(max(probs), 4) if probs else "",
                "median_prob": round(statistics.median(probs), 4) if probs else "",
                "seconds": round(time.time() - start, 1),
                "status": "ok",
            }
            shutil.rmtree(scratch, ignore_errors=True)
            break
        except Exception:
            shutil.rmtree(scratch, ignore_errors=True)
            with open(os.path.join(out_dir, "failures.log"), "a") as f:
                f.write(
                    f"=== {scene_id} (attempt {attempt}/{max_attempts}) ===\n"
                    f"{traceback.format_exc()}\n"
                )
            if attempt < max_attempts:
                time.sleep(5)
                continue
            row = {
                **base,
                "num_detections": "",
                "detector_candidates": "",
                "n_green": "",
                "n_red": "",
                "n_orange": "",
                "n_yellow": "",
                "min_prob": "",
                "max_prob": "",
                "median_prob": "",
                "seconds": round(time.time() - start, 1),
                "status": "error",
            }

    assert row is not None
    with open(row_path, "w") as f:
        json.dump(row, f)
    print(
        f"{scene_id}: {row['status']} dets={row['num_detections']} "
        f"cand={row['detector_candidates']} {row['seconds']}s",
        flush=True,
    )
    return row


def assemble_csv(scenes: list[dict], out_dir: str) -> None:
    """Build detections.csv from all existing per-scene row JSONs."""
    rows = []
    for s in scenes:
        row_path = os.path.join(out_dir, "rows", f"{s['scene_id']}.json")
        if os.path.exists(row_path):
            with open(row_path) as f:
                rows.append(json.load(f))

    def sort_key(r: dict) -> tuple:
        nd = r.get("num_detections")
        return (r.get("status") != "ok", -(nd if isinstance(nd, int) else 0))

    rows.sort(key=sort_key)
    csv_path = os.path.join(out_dir, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    err = [r for r in rows if r.get("status") != "ok"]
    tot = sum(r["num_detections"] for r in ok if isinstance(r["num_detections"], int))
    print(f"wrote {csv_path}: {len(ok)} ok, {len(err)} errored, {tot} total detections")


def load_scenes(scene_csv: str, limit: int | None) -> list[dict]:
    """Load the sampled scene rows from the scene CSV (optionally limited)."""
    with open(scene_csv) as f:
        scenes = list(csv.DictReader(f))
    return scenes[:limit] if limit else scenes


def main() -> None:
    """Orchestrate shard subprocesses (default) or run one shard (--shard)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--classify_config", required=True)
    parser.add_argument("--det_thr", type=float, default=0.7)
    parser.add_argument("--cls_thr", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--scratch_root", default="/tmp/recent_scratch")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    # Internal: when set, this process is a single shard worker.
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "rows"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(args.scratch_root, exist_ok=True)
    detector_cfg = os.path.join(out_dir, "patched_detector.yaml")
    classifier_cfg = os.path.join(out_dir, "patched_classifier.yaml")
    scenes = load_scenes(args.scene_csv, args.limit)

    # --- Shard worker mode ---
    if args.shard is not None:
        _landsat_mod.DETECT_MODEL_CONFIG = detector_cfg  # type: ignore[attr-defined]
        _landsat_mod.CLASSIFY_MODEL_CONFIG = classifier_cfg  # type: ignore[attr-defined]
        mine = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard]
        for s in mine:
            row_path = os.path.join(out_dir, "rows", f"{s['scene_id']}.json")
            if os.path.exists(row_path) and not args.overwrite:
                continue
            process_scene(s, out_dir, args.scratch_root, args.cls_thr)
        return

    # --- Orchestrator mode ---
    make_detector_config(args.det_thr, detector_cfg)
    make_classifier_config(
        os.path.abspath(args.classify_config), args.cls_thr, classifier_cfg
    )
    print(f"detector cfg   : {detector_cfg} (score_threshold={args.det_thr})")
    print(
        f"classifier cfg : {classifier_cfg} (positive_class_threshold={args.cls_thr})"
    )

    remaining = [
        s
        for s in scenes
        if args.overwrite
        or not os.path.exists(os.path.join(out_dir, "rows", f"{s['scene_id']}.json"))
    ]
    print(
        f"{len(scenes)} scenes total, {len(remaining)} to run, launching {args.workers} shards"
    )

    t0 = time.time()
    procs = []
    for shard in range(args.workers):
        cmd = [
            sys.executable,
            "-m",
            "rslp.landsat_vessels.scripts.run_recent_inference",
            "--scene_csv",
            args.scene_csv,
            "--out_dir",
            out_dir,
            "--classify_config",
            args.classify_config,
            "--det_thr",
            str(args.det_thr),
            "--cls_thr",
            str(args.cls_thr),
            "--scratch_root",
            args.scratch_root,
            "--shard",
            str(shard),
            "--num_shards",
            str(args.workers),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.overwrite:
            cmd += ["--overwrite"]
        log = open(os.path.join(out_dir, f"shard_{shard}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))  # nosec B603 - cmd is built internally, no shell
    for p in procs:
        p.wait()
    print(f"all shards finished in {time.time() - t0:.0f}s")

    assemble_csv(scenes, out_dir)


if __name__ == "__main__":
    init_mp()
    main()
