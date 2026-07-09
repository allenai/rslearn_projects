"""Visualize a PASTIS2 eval-tensor patch: 12 monthly S2 RGB panels + the label mask.

Renders one patch (s2_images/{i}.pt = (12,13,64,64), targets.pt row i) to a PNG montage so
you can eyeball that (a) the S2 imagery is real/coherent, (b) it varies seasonally across
the 12 months, and (c) the label classes/shapes look right. Works on both smoke and real
outputs.

RGB is bands B4/B3/B2 -> indices 3/2/1 of the 13-band stack [B1,B2,B3,B4,B5,B6,B7,B8,B8A,
B9,B10,B11,B12], per-panel 2-98 percentile stretched. Empty (zero-filled) months render black.

Run:  python viz_tensors.py --dir data/tensors/pastis2_test --index 0 --out patch0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ALL_MONTHS = [201809, 201810, 201811, 201812, 201901, 201902,
              201903, 201904, 201905, 201906, 201907, 201908]
RGB_IDX = [3, 2, 1]  # B4, B3, B2 in the 13-band stack


def stretch(rgb: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile stretch to [0,1]; all-zero panel stays black."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        band = rgb[c]
        if band.max() == 0:
            continue
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        out[c] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    return out.transpose(1, 2, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="a pastis2_<split> dir")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="patch.png")
    args = ap.parse_args()

    d = Path(args.dir)
    s2 = torch.load(d / "s2_images" / f"{args.index}.pt").numpy()  # (12,13,64,64)
    targets = torch.load(d / "targets.pt")[args.index].numpy()     # (64,64)
    months = torch.load(d / "months.pt")[args.index].tolist()

    fig, axes = plt.subplots(2, 7, figsize=(21, 6))
    for t in range(12):
        ax = axes.flat[t]
        ax.imshow(stretch(s2[t][RGB_IDX]))
        nonempty = s2[t].sum() != 0
        ax.set_title(f"{months[t]}{'' if nonempty else ' (empty)'}", fontsize=9)
        ax.axis("off")
    # 13th panel: the label mask; 14th: colorbar-ish legend via unique values.
    ax = axes.flat[12]
    im = ax.imshow(targets, cmap="tab20", vmin=0, vmax=19)
    ax.set_title(f"label (classes {sorted(np.unique(targets).tolist())})", fontsize=9)
    ax.axis("off")
    axes.flat[13].axis("off")
    fig.colorbar(im, ax=axes.flat[13], fraction=0.8)

    fig.suptitle(f"{args.dir}  patch {args.index}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out}  (s2 {s2.shape}, target classes {sorted(np.unique(targets).tolist())})")


if __name__ == "__main__":
    main()
