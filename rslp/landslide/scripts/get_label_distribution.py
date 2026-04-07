import os
import numpy as np
import rasterio
from collections import defaultdict
from pathlib import Path

ROOT = "/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives/windows/sen12_landslides"
#ROOT = "/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/1k_positives_fix/windows/sen12_landslides"
LABEL_PATH = "layers/label_raster/label/geotiff.tif"
LABELS = [0, 1, 2]

def make_stats():
    return {
        "total_pixels": defaultdict(int),
        "per_sample_pcts": defaultdict(list),
        "n_samples": 0,
        "missing": 0,
    }

stats = {"positive": make_stats(), "negative": make_stats()}

subdirs = sorted(Path(ROOT).iterdir())
for subdir in subdirs:
    if not subdir.is_dir():
        continue
    name = subdir.name
    if "positive" in name:
        key = "positive"
    elif "negative" in name:
        key = "negative"
    else:
        continue

    tif = subdir / LABEL_PATH
    if not tif.exists():
        print(f"  [WARN] missing: {tif}")
        stats[key]["missing"] += 1
        continue

    with rasterio.open(tif) as src:
        data = src.read(1)  # first band

    total = data.size
    stats[key]["n_samples"] += 1
    for lbl in LABELS:
        count = int(np.sum(data == lbl))
        stats[key]["total_pixels"][lbl] += count
        stats[key]["per_sample_pcts"][lbl].append(100.0 * count / total if total > 0 else 0.0)

# ── Report ──────────────────────────────────────────────────────────────────
for key in ("positive", "negative"):
    s = stats[key]
    n = s["n_samples"]
    grand_total = sum(s["total_pixels"].values())
    print(f"\n{'='*50}")
    print(f"  {key.upper()}  ({n} samples, {s['missing']} missing)")
    print(f"{'='*50}")
    print(f"  {'Label':<8} {'Total Pixels':>14} {'% of All':>10} {'Avg % / Sample':>16}")
    print(f"  {'-'*50}")
    for lbl in LABELS:
        tp = s["total_pixels"][lbl]
        pct_all = 100.0 * tp / grand_total if grand_total > 0 else 0.0
        pcts = s["per_sample_pcts"][lbl]
        avg_pct = np.mean(pcts) if pcts else 0.0
        print(f"  {lbl:<8} {tp:>14,} {pct_all:>9.2f}% {avg_pct:>15.2f}%")
    print(f"  {'TOTAL':<8} {grand_total:>14,}")

print()
