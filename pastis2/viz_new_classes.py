"""Find + visualize the windows in the QA sample that actually contain the newly-added
tropical classes (25..32), rasterized under the expanded class map. Answers: did the
(original-map) sampler incidentally capture the new classes, and do they look sane?
Saves viz_qa/newclass_<name>.png per class, and prints per-class window coverage.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import viz_qa_samples as V  # reuse expanded-map rasterize, S2 reader, territory, colors

TARGETS = [25, 26, 27, 28, 29, 30, 31, 32]  # the added classes


def main() -> None:
    wins = sorted(glob.glob(str(V.DS / "windows" / V.GROUP / "*")))
    found: dict[int, list] = {c: [] for c in TARGETS}
    scanned = 0
    for wdir in wins:
        wdir = Path(wdir)
        if not (wdir / "metadata.json").exists():
            continue
        if not glob.glob(str(wdir / "layers" / "sentinel2*" / "*" / "geotiff.tif")):
            continue
        meta = json.load(open(wdir / "metadata.json"))
        _, _, lon, lat = V.window_geo(meta)
        terr = V.territory_of(lon, lat)
        if terr == "unknown":
            continue
        mask = V.rasterize_label(meta, terr)
        scanned += 1
        for c in TARGETS:
            px = int((mask == c).sum())
            if px > 0:
                found[c].append((px, wdir, meta, mask, terr))

    print(f"scanned {scanned} materialized windows")
    for c in TARGETS:
        print(f"  {c:2} {V.NAMES[c]:18}: {len(found[c])} windows contain it")

    for c in TARGETS:
        items = sorted(found[c], key=lambda x: -x[0])[:4]
        if not items:
            continue
        n = len(items)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
        for j, (px, wdir, meta, mask, terr) in enumerate(items):
            rgb = V.read_best_s2(wdir)
            axes[0, j].imshow(rgb if rgb is not None else np.zeros((128, 128, 3)))
            axes[0, j].set_title(f"{terr}\nS2 RGB", fontsize=8)
            axes[1, j].imshow(mask, cmap=V_cmap, norm=V_norm, interpolation="nearest")
            # outline the target class in red for visibility
            axes[1, j].contour((mask == c).astype(float), levels=[0.5], colors="red", linewidths=1.2)
            axes[1, j].set_title(f"mask ({px} px of {V.NAMES[c]})", fontsize=8)
            for r in range(2):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        fig.suptitle(f"PASTIS2 new class {c} — {V.NAMES[c]}  (red outline; {len(found[c])} windows in sample)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = V.HERE / "viz_qa" / f"newclass_{c}_{V.NAMES[c].split('/')[0].replace(' ', '_')}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"saved {out.name}")


# build colormap/norm once (mirror viz_qa_samples)
palette = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])
from matplotlib.colors import ListedColormap, BoundaryNorm
_colors = np.vstack([[0.85, 0.85, 0.85, 1], palette[1:V.NCLS]])[:V.NCLS]
V_cmap = ListedColormap(_colors)
V_norm = BoundaryNorm(np.arange(-0.5, V.NCLS + 0.5, 1), V_cmap.N)


if __name__ == "__main__":
    main()
