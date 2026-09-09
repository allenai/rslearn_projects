"""Render a contact sheet of all windows in a split (val/test) for the classifier.

For the groups the training config actually uses, gather every window whose
``options.split`` tag matches the requested split, render a natural-color
(B4/B3/B2) thumbnail, and lay them out in a grid. Each tile is annotated with its
source group and outlined by its ground-truth label (green=correct, red=incorrect),
so the group composition and class balance of the split are visible at a glance.

Usage:
    python -m rslp.landsat_vessels.evaluation.visualize_split --split test
    python -m rslp.landsat_vessels.evaluation.visualize_split --split val
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

DS = "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
GROUPS = ["selected_copy", "phase2a_completed", "feedback_20260325", "phase3a_selected"]
STACK_DIR = "B1_B2_B3_B4_B5_B6_B7_B9_B10_B11"
# Index of each band within the stacked geotiff (order = dir name).
STACK_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]


def rgb_thumb(window_dir: str) -> np.ndarray | None:
    """Return a Brovey pan-sharpened natural-color (B4/B3/B2) thumbnail as HxWx3 uint8.

    Follows the standard Brovey transform: each color band is scaled by the panchromatic
    band over the RGB intensity, ``band_sharp = band * B8 / mean(B2,B3,B4)``. The 30 m
    color bands (B2/B3/B4, 32x32) are upsampled 2x to the B8 grid (15 m, 64x64) first.
    Because all four bands share one normalization to 0-255, the B8/intensity ratio is
    well-behaved. If B8 is missing, falls back to the upsampled color-only image.
    """
    path = os.path.join(window_dir, "layers", "landsat", STACK_DIR, "geotiff.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (10, H, W)

    pan_path = os.path.join(window_dir, "layers", "landsat", "B8", "geotiff.tif")
    have_pan = os.path.exists(pan_path)
    pan_raw = rasterio.open(pan_path).read(1).astype(np.float32) if have_pan else None

    # Single joint 0-255 normalization across B2/B3/B4 (+B8) so the Brovey ratio
    # B8/intensity stays radiometrically meaningful.
    color_raw = {b: arr[STACK_ORDER.index(b)] for b in ["B2", "B3", "B4"]}
    stack_for_scale = list(color_raw.values()) + ([pan_raw] if have_pan else [])
    allvals = np.concatenate([a.ravel() for a in stack_for_scale])
    lo, hi = np.percentile(allvals, 2), np.percentile(allvals, 98)
    span = hi - lo if hi > lo else 1

    def norm8(a: np.ndarray) -> np.ndarray:
        return np.clip((a - lo) / span * 255, 0, 255)

    if not have_pan:
        rgb = np.stack(
            [norm8(color_raw["B4"]), norm8(color_raw["B3"]), norm8(color_raw["B2"])],
            axis=-1,
        )
        return rgb.astype(np.uint8)

    # Upsample color 2x to the B8 grid via pixel repeat (matches 32->64), Brovey-sharpen.
    b8 = norm8(pan_raw)
    sharp = {}
    for band in ["B2", "B3", "B4"]:
        up = norm8(color_raw[band]).repeat(2, axis=0).repeat(2, axis=1)
        sharp[band] = up
    total = np.clip((sharp["B2"] + sharp["B3"] + sharp["B4"]) / 3, 1, 255)
    for band in ["B2", "B3", "B4"]:
        sharp[band] = np.clip(sharp[band] * b8 / total, 0, 255).astype(np.uint8)
    return np.stack([sharp["B4"], sharp["B3"], sharp["B2"]], axis=-1)


COLOR = {"correct": "#2ca02c", "incorrect": "#d62728"}
# Short group tag for tile labels.
SHORT = {
    "feedback_20260325": "fb",
    "phase2a_completed": "p2a",
    "phase3a_selected": "p3a",
    "selected_copy": "sel",
}


def render_sheet(
    items: list[tuple[str, str, str]],
    cols: int,
    title: str,
    out_path: str,
) -> None:
    """Render a grid of window thumbnails to out_path."""
    n = len(items)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        g, label, wdir = items[i]
        tag = "pos" if label == "correct" else "neg"
        thumb = rgb_thumb(wdir)
        if thumb is None:
            ax.text(
                0.5,
                0.5,
                f"no img\n{SHORT.get(g, g)} · {tag}",
                ha="center",
                va="center",
                fontsize=6,
                color=COLOR[label],
            )
            continue
        ax.imshow(thumb)
        ax.set_title(
            f"{SHORT.get(g, g)} · {tag}", fontsize=6, pad=1, color=COLOR[label]
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR[label])
            spine.set_linewidth(2.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    """Build and save the contact sheet(s) for one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", default="./pr_out")
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument(
        "--parts",
        type=int,
        default=1,
        help="Split into this many separate figures (useful for large splits).",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect (group, label, window_dir) for windows in this split.
    items: list[tuple[str, str, str]] = []
    for g in GROUPS:
        gdir = os.path.join(DS, "windows", g)
        if not os.path.isdir(gdir):
            continue
        for w in os.listdir(gdir):
            wdir = os.path.join(gdir, w)
            mpath = os.path.join(wdir, "metadata.json")
            lpath = os.path.join(wdir, "layers", "label", "data.geojson")
            if not (os.path.exists(mpath) and os.path.exists(lpath)):
                continue
            with open(mpath) as f:
                split = json.load(f).get("options", {}).get("split")
            if split != args.split:
                continue
            with open(lpath) as f:
                feats = json.load(f).get("features", [])
            if not feats:
                continue
            label = feats[0].get("properties", {}).get("label", "UNKNOWN")
            if label not in ("correct", "incorrect"):
                continue
            items.append((g, label, wdir))

    # Sort so the grid groups by source group, then by label.
    items.sort(key=lambda t: (t[0], t[1]))
    n = len(items)
    comp: dict[str, dict[str, int]] = {}
    for g, label, _ in items:
        comp.setdefault(g, {"correct": 0, "incorrect": 0})[label] += 1
    print(f"split={args.split}: {n} windows")
    for g, c in comp.items():
        print(f"  {g}: correct={c['correct']} incorrect={c['incorrect']}")

    comp_str = "  ".join(
        f"{SHORT.get(g, g)}: {c['correct']}+/{c['incorrect']}-" for g, c in comp.items()
    )

    parts = max(1, args.parts)
    if parts == 1:
        render_sheet(
            items,
            args.cols,
            f"{args.split} split — {n} windows (green=correct, red=incorrect)\n{comp_str}",
            os.path.join(args.out_dir, f"contact_sheet_{args.split}.png"),
        )
        return

    # Split into `parts` contiguous chunks (order preserved: grouped by group/label).
    chunk = math.ceil(n / parts)
    for p in range(parts):
        sub = items[p * chunk : (p + 1) * chunk]
        if not sub:
            continue
        title = (
            f"{args.split} split — part {p + 1}/{parts} "
            f"(windows {p * chunk + 1}-{p * chunk + len(sub)} of {n}; "
            f"green=correct, red=incorrect)\n{comp_str}"
        )
        render_sheet(
            sub,
            args.cols,
            title,
            os.path.join(args.out_dir, f"contact_sheet_{args.split}_part{p + 1}.png"),
        )


if __name__ == "__main__":
    main()
