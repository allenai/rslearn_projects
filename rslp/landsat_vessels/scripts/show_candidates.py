"""Re-run one scene with include_rejected=True and render ALL detector candidates.

Uses the exact patched det=0.9 / cls=0.9 configs from a run's out_dir so the verdict
matches what the overnight run produced. Every detector candidate is carried through the
classifier (include_rejected) with classifier_prob_correct recorded; we then build an
annotated grid: GREEN border = kept (prob>=cls_thr), RED = removed, sorted by prob desc,
each labeled with detector score and classifier prob(correct).
"""

import argparse
import os
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from rslp.utils.mp import init_mp

_landsat_mod = sys.modules["rslp.landsat_vessels.predict_pipeline"]


def _font(size: int) -> Any:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> None:
    """Render a detector-candidate grid for a single scene."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", required=True)
    ap.add_argument(
        "--run_dir", default="/weka/dfive-default/yawenz/landsat/20260908_results"
    )
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--cls_thr", type=float, default=0.9)
    ap.add_argument("--scratch", default="/tmp/cand_scratch")
    ap.add_argument("--cell", type=int, default=200, help="grid cell px")
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    scene_id = args.scene_id
    out_dir = args.out_dir or os.path.join(args.run_dir, "candidates", scene_id)
    os.makedirs(out_dir, exist_ok=True)

    # Point the pipeline at the exact patched configs the run used.
    det_cfg = os.path.join(args.run_dir, "patched_detector.yaml")
    cls_cfg = os.path.join(args.run_dir, "patched_classifier.yaml")
    assert os.path.exists(det_cfg) and os.path.exists(
        cls_cfg
    ), "patched configs missing"
    _landsat_mod.DETECT_MODEL_CONFIG = det_cfg  # type: ignore[attr-defined]
    _landsat_mod.CLASSIFY_MODEL_CONFIG = cls_cfg  # type: ignore[attr-defined]

    from rslp.landsat_vessels.predict_pipeline import predict_pipeline

    result = predict_pipeline(
        scene_id=scene_id,
        scratch_path=os.path.join(args.scratch, scene_id),
        crop_path=out_dir,
        include_rejected=True,
    )

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

    cands.sort(key=lambda c: (c["prob"] is None, -(c["prob"] or 0.0)))  # type: ignore[operator]
    kept = [c for c in cands if (c["prob"] or 0.0) >= args.cls_thr]
    removed = [c for c in cands if (c["prob"] or 0.0) < args.cls_thr]

    print(
        f"\n{scene_id}: {len(cands)} detector candidates -> "
        f"{len(kept)} kept (prob>={args.cls_thr}), {len(removed)} removed"
    )
    print(
        f"{'idx':>3} {'det_score':>9} {'cls_prob':>9} {'verdict':>8}  ({'col':>5},{'row':>5})"
    )
    for i, c in enumerate(cands):
        verdict = "KEEP" if (c["prob"] or 0.0) >= args.cls_thr else "remove"
        ds = f"{c['det_score']:.3f}" if c["det_score"] is not None else "  -  "
        pp = f"{c['prob']:.4f}" if c["prob"] is not None else "  -  "
        print(f"{i:>3} {ds:>9} {pp:>9} {verdict:>8}  ({c['col']:>5},{c['row']:>5})")

    # Build annotated grid.
    cell, cols = args.cell, args.cols
    pad, label_h = 6, 34
    n = len(cands)
    rows = (n + cols - 1) // cols
    cw = cell + 2 * pad
    ch = cell + label_h + 2 * pad
    grid = Image.new("RGB", (cols * cw, rows * ch), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    font = _font(15)

    for i, c in enumerate(cands):
        gx = (i % cols) * cw
        gy = (i // cols) * ch
        keep = (c["prob"] or 0.0) >= args.cls_thr
        border = (40, 200, 60) if keep else (220, 50, 50)
        # crop image
        if c["rgb"] and os.path.exists(c["rgb"]):  # type: ignore[arg-type]
            im = Image.open(c["rgb"]).convert("RGB")
            # upsample small crop to cell with nearest for crispness
            im = im.resize((cell, cell), Image.NEAREST)
        else:
            im = Image.new("RGB", (cell, cell), (60, 60, 60))
        grid.paste(im, (gx + pad, gy + pad + label_h))
        # border
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

    grid_path = os.path.join(out_dir, "candidates_grid.png")
    grid.save(grid_path)
    print(f"\ncrops + grid written to {out_dir}")
    print(f"grid: {grid_path}")


if __name__ == "__main__":
    init_mp()
    main()
