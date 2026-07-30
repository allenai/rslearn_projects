"""Grouped bar charts: per class, one colored bar per DROM territory.
Two figures (parcels and pixels), log y-scale (huge dynamic range). -> maps/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio

HERE = Path(__file__).parent
OUT = HERE / "maps"; OUT.mkdir(exist_ok=True)

CRS = {"Guadeloupe": 5490, "Martinique": 5490, "Guyane": 2972, "Réunion": 2975, "Mayotte": 4471}
FILE = {"Guadeloupe": "guadeloupe", "Martinique": "martinique", "Guyane": "guyane",
        "Réunion": "reunion", "Mayotte": "mayotte"}
COLORS = {"Guadeloupe": "#e6194B", "Martinique": "#f58231", "Guyane": "#3cb44b",
          "Réunion": "#4363d8", "Mayotte": "#911eb4"}

m = json.load(open(HERE / "pastis_rpg_class_map_expanded.json"))
c2c = {str(k): int(v) for k, v in m["code_to_class"].items()}
names = {int(k): v for k, v in m["class_names"].items()}

info = pyogrio.read_info(str(HERE / "data/rpg/martinique.gpkg"))
col = next(c for c in ("code_cultu", "CODE_CULTU", "code", "CODE") if c in info["fields"])

parcels: dict = {t: {} for t in CRS}
pixels: dict = {t: {} for t in CRS}
totpx: dict = {}
for terr, crs in CRS.items():
    g = pyogrio.read_dataframe(str(HERE / f"data/rpg/{FILE[terr]}.gpkg"), columns=[col]).to_crs(crs)
    g["cls"] = [c2c.get(str(x), 0) for x in g[col].values]
    g["px"] = g.geometry.area / 100.0
    for cid, sub in g.groupby("cls"):
        if cid == 0:
            continue
        parcels[terr][cid] = len(sub)
        pixels[terr][cid] = int(sub["px"].sum())
        totpx[cid] = totpx.get(cid, 0) + int(sub["px"].sum())

classes = sorted(totpx, key=lambda c: -totpx[c])  # by total pixel support
labels = [f"{c} {names.get(c, '?')}" for c in classes]
terrs = list(CRS)


def grouped(data: dict, title: str, ylabel: str, fname: str) -> None:
    x = np.arange(len(classes))
    w = 0.16
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, terr in enumerate(terrs):
        vals = [data[terr].get(c, 0) for c in classes]
        # plot 0 as a tiny sliver so log-scale doesn't drop it entirely
        vals = [v if v > 0 else 0 for v in vals]
        ax.bar(x + (i - 2) * w, vals, w, label=terr, color=COLORS[terr], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=13)
    ax.legend(title="territory", ncol=5, fontsize=9)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


grouped(parcels, "PASTIS2 DROM — parcels per class, by territory", "parcels (log)", "bars_parcels_by_territory.png")
grouped(pixels, "PASTIS2 DROM — pixels per class, by territory", "pixels @10m (log)", "bars_pixels_by_territory.png")
