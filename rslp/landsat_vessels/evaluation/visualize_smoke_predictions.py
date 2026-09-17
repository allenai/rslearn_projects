"""Render smoke-test predictions (bboxes on imagery) at a fixed det/cls operating point.

For each smoke-test scene, run the full pipeline with the detector's ``score_threshold``
lowered to ``--det_thr`` (so the detector only emits candidates at or above that
confidence) and the Run-d classifier. Keep every candidate (``include_rejected=True``),
then keep only those whose classifier ``prob(correct) >= --cls_thr``. For the surviving
detections we draw two things per scene:

  1. ``<scene>_overview.png`` -- the full-scene panchromatic (B8) image, downsampled,
     with a red box drawn on every kept detection. Shows where the vessels landed.
  2. ``<scene>_crops.png`` -- a grid of the per-detection 64 px pan-sharpened RGB crops
     (written by the pipeline's ``crop_path``), each with a centered box and its
     detector score / classifier prob(correct) annotated. Shows what each hit looks like.

Usage:
    python -m rslp.landsat_vessels.evaluation.visualize_smoke_predictions \
        --classify_config data/landsat_vessels/config_classifier_20260908d.yaml \
        --det_thr 0.9 --cls_thr 0.9 --out_dir /tmp/smoke_vis_0909
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from matplotlib import patches
from PIL import Image

from rslp.landsat_vessels.evaluation.smoke_test import SCENES
from rslp.utils.mp import init_mp

# predict_pipeline is re-exported as a function by the package __init__, shadowing the
# module, so reach the module directly to override its config constants.
_landsat_mod = sys.modules["rslp.landsat_vessels.predict_pipeline"]


def _set_score_threshold(obj: object, value: float) -> int:
    """Recursively set every ``score_threshold`` key in a nested dict/list."""
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
    if _set_score_threshold(cfg, det_floor) == 0:
        raise ValueError("no score_threshold key found in detector config")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="detector_floor_", delete=False
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _open_scene_bands(scratch: str) -> dict:
    """Open the detector-window B2/B3/B4/B8 geotiffs and return them + B8 reader.

    The detector window stores the 7-band ``landsat`` layer with each band in its own
    sub-directory. B8 is full resolution (15 m, our reference grid); B2/B3/B4 are stored
    at half resolution (30 m). Everything is uint8 (already remapped from raw DN). We use
    these to Brovey pan-sharpen larger crops around each detection.
    """
    base = os.path.join(scratch, "windows", "default", "default", "layers", "landsat")
    bands: dict = {}
    for band in ["B2", "B3", "B4", "B8"]:
        p = os.path.join(base, band, "geotiff.tif")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing band geotiff {p}")
        bands[band] = rasterio.open(p)
    return bands


def _pan_sharp_crop(bands: dict, det: Any, half: int) -> np.ndarray | None:
    """Brovey pan-sharpened natural-color crop (HxWx3 uint8) centered on a detection.

    Reads a ``2*half`` px window (B8 15 m grid) around the detection from the detector
    window, upsamples the 30 m color bands to the B8 grid, and applies the same joint
    2-98 percentile normalization + Brovey ratio as ``visualize_split.rgb_thumb``.
    """
    b8_src = bands["B8"]
    b8_full = b8_src.read(1)
    H, W = b8_full.shape
    wx = det.col * det.projection.x_resolution
    wy = det.row * det.projection.y_resolution
    py, px = b8_src.index(wx, wy)
    y0, y1 = max(0, py - half), min(H, py + half)
    x0, x1 = max(0, px - half), min(W, px + half)
    if y1 <= y0 or x1 <= x0:
        return None
    tw, th = x1 - x0, y1 - y0

    b8 = b8_full[y0:y1, x0:x1].astype(np.float32)
    colors: dict[str, np.ndarray] = {}
    for band in ["B2", "B3", "B4"]:
        arr = bands[band].read(1)
        sh, sw = arr.shape
        sy, sx = H / sh, W / sw  # ~2.0 for the half-res color bands
        by0, by1 = int(y0 / sy), max(int(y0 / sy) + 1, int(math.ceil(y1 / sy)))
        bx0, bx1 = int(x0 / sx), max(int(x0 / sx) + 1, int(math.ceil(x1 / sx)))
        sub = arr[by0:by1, bx0:bx1].astype(np.float32)
        colors[band] = np.asarray(
            Image.fromarray(sub, mode="F").resize((tw, th), Image.BILINEAR)
        )

    allv = np.concatenate([a.ravel() for a in [b8, *colors.values()]])
    lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    span = hi - lo if hi > lo else 1.0

    def n8(a: np.ndarray) -> np.ndarray:
        return np.clip((a - lo) / span * 255, 0, 255)

    b8n = n8(b8)
    total = np.clip(
        (n8(colors["B2"]) + n8(colors["B3"]) + n8(colors["B4"])) / 3, 1, 255
    )
    sharp = {
        band: np.clip(n8(colors[band]) * b8n / total, 0, 255).astype(np.uint8)
        for band in ["B2", "B3", "B4"]
    }
    return np.stack([sharp["B4"], sharp["B3"], sharp["B2"]], axis=-1)


def _draw_overview(
    scene_id: str,
    desc: str,
    bands: dict,
    kept: list,
    out_path: str,
    max_dim: int = 1800,
) -> None:
    """Draw the full-scene B8 with a red box on each kept detection."""
    b8_src = bands["B8"]
    arr = b8_src.read(1)
    h, w = arr.shape
    scale = min(1.0, max_dim / max(h, w))
    disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = Image.fromarray(arr).resize((disp_w, disp_h), Image.BILINEAR)

    fig, ax = plt.subplots(figsize=(disp_w / 100, disp_h / 100), dpi=130)
    ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
    box = max(8, int(0.012 * max(disp_w, disp_h)))
    for det in kept:
        wx = det.col * det.projection.x_resolution
        wy = det.row * det.projection.y_resolution
        py, px = b8_src.index(wx, wy)  # full-res pixel
        cx, cy = px * scale, py * scale
        ax.add_patch(
            patches.Rectangle(
                (cx - box / 2, cy - box / 2),
                box,
                box,
                linewidth=1.2,
                edgecolor="red",
                facecolor="none",
            )
        )
    ax.set_title(
        f"{scene_id}\n{desc} — {len(kept)} detections (det>=thr & cls>=thr)",
        fontsize=8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _draw_crop_grid(
    scene_id: str,
    desc: str,
    bands: dict,
    kept: list,
    out_path: str,
    cols: int = 5,
    half: int = 32,
) -> None:
    """Grid of large Brovey pan-sharpened crops, each with a centered box + scores."""
    if not kept:
        return
    n = len(kept)
    rows = math.ceil(n / cols)
    # Big tiles (~2.8 in each) so the ~64 px crops render large and clear.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.1))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        det = kept[i]
        crop = _pan_sharp_crop(bands, det, half)
        if crop is not None:
            ax.imshow(crop, interpolation="nearest")
            ch, cw = crop.shape[:2]
            b = max(8, int(0.20 * min(ch, cw)))
            ax.add_patch(
                patches.Rectangle(
                    (cw / 2 - b / 2, ch / 2 - b / 2),
                    b,
                    b,
                    linewidth=1.6,
                    edgecolor="red",
                    facecolor="none",
                )
            )
        else:
            ax.text(0.5, 0.5, "no crop", ha="center", va="center", fontsize=7)
        prob = det.metadata.get("classifier_prob_correct")
        ax.set_title(
            f"#{i}  det={det.score:.2f}  cls={prob:.2f}"
            if prob is not None
            else f"#{i}  det={det.score:.2f}",
            fontsize=8,
            pad=2,
        )
    fig.suptitle(
        f"{scene_id} — {desc} — {n} detections (det>=0.9, cls>=0.9)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_scene(
    scene_id: str, desc: str, det_thr: float, cls_thr: float, out_dir: str
) -> dict:
    """Run the pipeline for one scene and render overview + crop grid."""
    from rslp.landsat_vessels.predict_pipeline import predict_pipeline

    scratch = f"/tmp/smoke_vis_{scene_id}"
    result = predict_pipeline(
        scene_id=scene_id,
        scratch_path=scratch,
        include_rejected=True,
    )
    bands: dict = {}
    try:
        kept = [
            d
            for d in result.detections
            if d.score is not None
            and d.score >= det_thr
            and d.metadata.get("classifier_prob_correct") is not None
            and d.metadata["classifier_prob_correct"] >= cls_thr
        ]
        kept.sort(key=lambda d: d.metadata["classifier_prob_correct"], reverse=True)

        bands = _open_scene_bands(scratch)
        _draw_overview(
            scene_id,
            desc,
            bands,
            kept,
            os.path.join(out_dir, f"{scene_id}_overview.png"),
        )
        _draw_crop_grid(
            scene_id,
            desc,
            bands,
            kept,
            os.path.join(out_dir, f"{scene_id}_crops.png"),
        )
        return {
            "scene_id": scene_id,
            "kept": len(kept),
            "detector_floor": result.detector_count,
        }
    finally:
        for src in bands.values():
            src.close()
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    """Render smoke-test predictions at the given det/cls thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify_config", default=None)
    parser.add_argument("--det_thr", type=float, default=0.9)
    parser.add_argument("--cls_thr", type=float, default=0.9)
    parser.add_argument("--out_dir", default="/tmp/smoke_vis")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="skip the first N scenes (0-based) in the SCENES list",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.classify_config:
        _landsat_mod.CLASSIFY_MODEL_CONFIG = os.path.abspath(args.classify_config)  # type: ignore[attr-defined]
    det_cfg = _detector_config_with_floor(args.det_thr)
    _landsat_mod.DETECT_MODEL_CONFIG = det_cfg  # type: ignore[attr-defined]
    print(f"classifier config: {_landsat_mod.CLASSIFY_MODEL_CONFIG}")
    print(f"detector floor:    {args.det_thr}  (config {det_cfg})")
    print(f"keep if det>={args.det_thr} and cls>={args.cls_thr}")

    summary = []
    for scene_id, (low, high), desc in SCENES[args.start_index :]:
        print(f"\n[RUNNING] {scene_id} ({desc}) expected [{low}, {high}]", flush=True)
        try:
            rec = run_scene(scene_id, desc, args.det_thr, args.cls_thr, args.out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            summary.append((scene_id, desc, low, high, -1))
            continue
        summary.append((scene_id, desc, low, high, rec["kept"]))
        print(
            f"  kept {rec['kept']} detections (det>={args.det_thr}, cls>={args.cls_thr})"
        )

    print("\n" + "=" * 90)
    print(f"{'scene':<48}{'expected':>12}{'kept':>7}   desc")
    print("-" * 90)
    for scene_id, desc, low, high, kept in summary:
        k = "ERR" if kept < 0 else str(kept)
        print(f"{scene_id:<48}{f'[{low},{high}]':>12}{k:>7}   {desc}")
    print(f"\nImages written to {args.out_dir}")


if __name__ == "__main__":
    init_mp()
    main()
