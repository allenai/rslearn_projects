"""Plot the global geolocation of classifier windows by split and label.

Each window's metadata carries its UTM projection (crs + pixel resolution) and pixel
bounds. We compute the window center, reproject it to lon/lat (EPSG:4326), and scatter
all windows on world maps -- one panel per split (train/val/test), colored by label
(green=correct/pos, red=incorrect/neg).

Usage:
    python -m rslp.landsat_vessels.evaluation.plot_geolocations --out_dir ./pr_out
"""

import argparse
import json
import os
from collections import defaultdict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

DS = "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
GROUPS = ["selected_copy", "phase2a_completed", "feedback_20260325", "phase3a_selected"]
SPLITS = ["train", "val", "test"]
COLOR = {"correct": "#2ca02c", "incorrect": "#d62728"}


def collect() -> dict:
    """Return {split: {label: [(lon, lat), ...]}} for all windows in the config groups."""
    # Group raw centers by CRS so we can batch-reproject.
    by_crs: dict[str, list] = defaultdict(list)  # crs -> [(px, py, split, label)]
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
                meta = json.load(f)
            split = meta.get("options", {}).get("split")
            if split not in SPLITS:
                continue
            with open(lpath) as f:
                feats = json.load(f).get("features", [])
            if not feats:
                continue
            label = feats[0].get("properties", {}).get("label")
            if label not in ("correct", "incorrect"):
                continue
            proj = meta["projection"]
            b = meta["bounds"]
            # Center pixel -> projected meters via per-axis resolution.
            px = (b[0] + b[2]) / 2 * proj["x_resolution"]
            py = (b[1] + b[3]) / 2 * proj["y_resolution"]
            by_crs[proj["crs"]].append((px, py, split, label))

    out: dict = {s: defaultdict(list) for s in SPLITS}
    for crs, rows in by_crs.items():
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        lons, lats = tf.transform(xs, ys)
        for (_, _, split, label), lon, lat in zip(rows, lons, lats):
            out[split][label].append((lon, lat))
    return out


def main() -> None:
    """Build and save the global geolocation map."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./pr_out")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = collect()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(3, 1, figsize=(16, 20), subplot_kw={"projection": proj})
    for ax, split in zip(axes, SPLITS):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#eeeeee")
        ax.add_feature(cfeature.OCEAN, facecolor="#d6ecff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888888")
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4)
        counts = {}
        for label in ("correct", "incorrect"):
            pts = data[split][label]
            counts[label] = len(pts)
            if not pts:
                continue
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            tag = "pos" if label == "correct" else "neg"
            ax.scatter(
                lons,
                lats,
                s=14,
                c=COLOR[label],
                marker="o" if label == "correct" else "x",
                linewidths=0.8,
                alpha=0.6,
                edgecolors="none" if label == "correct" else COLOR[label],
                transform=ccrs.PlateCarree(),
                label=f"{tag} ({counts[label]})",
                zorder=5,
            )
        ax.set_title(
            f"{split} — pos={counts['correct']}  neg={counts['incorrect']}",
            fontsize=13,
        )
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    fig.suptitle(
        "Landsat vessel classifier — window geolocations by split & label",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(args.out_dir, "geolocations.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")
    for s in SPLITS:
        print(f"  {s}: pos={len(data[s]['correct'])} neg={len(data[s]['incorrect'])}")


if __name__ == "__main__":
    main()
