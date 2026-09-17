"""Map the scene sampler's strata pools (and the selected scenes, if sampled).

Renders one world-map panel per stratum showing every candidate WRS-2 path/row
centroid that qualifies for it, plus a final panel with the actually-selected
scenes once sample_scenes.py has written its CSV. Reuses the sampler's own
candidate-building code so the map shows exactly what the sampler draws from.

Usage:
    python -m rslp.landsat_vessels.scripts.map_scene_sample \
        --marine_regions /weka/dfive-default/yawenz/landsat/World_Marine_Regions.geojson \
        --out /weka/dfive-default/yawenz/landsat/strata_pools_map.png \
        [--sample_csv /weka/dfive-default/yawenz/landsat/scene_sample_v1.csv]
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.io import shapereader
from upath import UPath

from rslp.landsat_vessels.scripts.sample_scenes import (
    STRATUM_FRACTIONS,
    build_pathrow_candidates,
    load_land_geometries,
    load_marine_regions,
)

# Categorical slots 1-5 of the skill-validated default palette (light mode), in
# fixed order; stratum identity keeps the same hue in every panel.
STRATUM_COLORS = {
    "coastal": "#2a78d6",
    "storm": "#eb6834",
    "ice": "#1baf7a",
    "glint": "#eda100",
    "cloud": "#e87ba4",
    "open": "#008300",
}
LAND_COLOR = "#e6e4dd"
COAST_COLOR = "#b5b3aa"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e56"


def main() -> None:
    """Render the strata-pool map."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--marine_regions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cache_dir",
        default="/weka/dfive-default/yawenz/landsat/scene_sampling_cache",
    )
    parser.add_argument(
        "--sample_csv",
        default="/weka/dfive-default/yawenz/landsat/scene_sample_v1.csv",
        help="if this exists, the selected scenes are drawn in the last panel",
    )
    parser.add_argument(
        "--rt_csv",
        default="/weka/dfive-default/yawenz/landsat/scene_sample_rt_v1.csv",
        help="RT sample CSV (sample_rt_scenes.py); drawn as triangles if it exists",
    )
    parser.add_argument(
        "--selected_only",
        action="store_true",
        help="render a single large map of the selected scenes (no pool panels)",
    )
    args = parser.parse_args()

    print("Building candidates (cached WRS-2 shapefile)...")
    regions = load_marine_regions(args.marine_regions, include_lakes=False)
    land_geoms = load_land_geometries()
    candidates = build_pathrow_candidates(
        UPath(args.cache_dir) / "wrs2", regions, land_geoms
    )
    print(f"  {len(candidates)} candidate path/rows")

    selected = []
    if os.path.exists(args.sample_csv):
        with open(args.sample_csv) as f:
            selected = list(csv.DictReader(f))
        print(f"  {len(selected)} selected scenes from {args.sample_csv}")

    rt_selected = []
    if args.rt_csv and os.path.exists(args.rt_csv):
        with open(args.rt_csv) as f:
            rt_selected = list(csv.DictReader(f))
        print(f"  {len(rt_selected)} RT scenes from {args.rt_csv}")

    land_shp = shapereader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    land_recs = list(shapereader.Reader(land_shp).geometries())

    strata = list(STRATUM_FRACTIONS)

    if args.selected_only:
        if not selected:
            raise SystemExit(f"no sample CSV at {args.sample_csv}")
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_geometries(
            land_recs,
            ccrs.PlateCarree(),
            facecolor=LAND_COLOR,
            edgecolor=COAST_COLOR,
            linewidth=0.4,
            zorder=1,
        )
        ax.spines["geo"].set_edgecolor(COAST_COLOR)
        for stratum in strata:
            pts = [s for s in selected if s["stratum"] == stratum]
            if not pts:
                continue
            ax.scatter(
                [float(s["lon"]) for s in pts],
                [float(s["lat"]) for s in pts],
                transform=ccrs.PlateCarree(),
                s=42,
                color=STRATUM_COLORS[stratum],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                label=f"{stratum} ({len(pts)})",
            )
        for stratum in strata:
            pts = [s for s in rt_selected if s["stratum"] == stratum]
            if not pts:
                continue
            ax.scatter(
                [float(s["lon"]) for s in pts],
                [float(s["lat"]) for s in pts],
                transform=ccrs.PlateCarree(),
                s=52,
                marker="^",
                color=STRATUM_COLORS[stratum],
                edgecolors=TEXT_PRIMARY,
                linewidths=0.8,
                zorder=4,
            )
        if rt_selected:
            # One legend entry for the RT marker shape (colors follow strata).
            ax.scatter(
                [],
                [],
                s=52,
                marker="^",
                color="none",
                edgecolors=TEXT_PRIMARY,
                linewidths=0.8,
                label=f"RT tier ({len(rt_selected)})",
            )
        title = (
            f"Round 1 scene sample — {len(selected)} Landsat T1/T2 scenes by stratum"
        )
        if rt_selected:
            title = (
                f"Round 1 scene sample — {len(selected)} T1/T2 (circles) + "
                f"{len(rt_selected)} RT (triangles) scenes by stratum"
            )
        ax.set_title(title, fontsize=14, color=TEXT_PRIMARY, loc="left")
        ax.legend(
            loc="lower left",
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            labelcolor=TEXT_SECONDARY,
            markerscale=1.2,
        )
        fig.tight_layout()
        fig.savefig(args.out, dpi=150, facecolor="white")
        print(f"Saved map: {args.out}")
        return
    n_panels = len(strata) + (1 if selected else 0)
    ncols, nrows = 2, (n_panels + 1) // 2
    fig = plt.figure(figsize=(13, 3.6 * nrows))

    def draw_base(ax: plt.Axes) -> None:
        ax.set_global()
        ax.add_geometries(
            land_recs,
            ccrs.PlateCarree(),
            facecolor=LAND_COLOR,
            edgecolor=COAST_COLOR,
            linewidth=0.3,
            zorder=1,
        )
        ax.spines["geo"].set_edgecolor(COAST_COLOR)
        ax.spines["geo"].set_linewidth(0.5)

    for i, stratum in enumerate(strata):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=ccrs.Robinson())
        draw_base(ax)
        lons = [c["lon"] for c in candidates if c[stratum]]
        lats = [c["lat"] for c in candidates if c[stratum]]
        ax.scatter(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            s=1.5,
            color=STRATUM_COLORS[stratum],
            alpha=0.45,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
        frac = STRATUM_FRACTIONS[stratum]
        ax.set_title(
            f"{stratum} — {len(lons):,} path/rows (quota {frac:.0%})",
            fontsize=11,
            color=TEXT_PRIMARY,
            loc="left",
        )

    if selected:
        ax = fig.add_subplot(nrows, ncols, n_panels, projection=ccrs.Robinson())
        draw_base(ax)
        for stratum in strata:
            pts = [s for s in selected if s["stratum"] == stratum]
            if not pts:
                continue
            ax.scatter(
                [float(s["lon"]) for s in pts],
                [float(s["lat"]) for s in pts],
                transform=ccrs.PlateCarree(),
                s=14,
                color=STRATUM_COLORS[stratum],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label=f"{stratum} ({len(pts)})",
            )
        for stratum in strata:
            pts = [s for s in rt_selected if s["stratum"] == stratum]
            if not pts:
                continue
            ax.scatter(
                [float(s["lon"]) for s in pts],
                [float(s["lat"]) for s in pts],
                transform=ccrs.PlateCarree(),
                s=20,
                marker="^",
                color=STRATUM_COLORS[stratum],
                edgecolors=TEXT_PRIMARY,
                linewidths=0.5,
                zorder=4,
            )
        if rt_selected:
            ax.scatter(
                [],
                [],
                s=20,
                marker="^",
                color="none",
                edgecolors=TEXT_PRIMARY,
                linewidths=0.5,
                label=f"RT tier ({len(rt_selected)})",
            )
        title = f"selected scenes — n={len(selected)}"
        if rt_selected:
            title = (
                f"selected scenes — {len(selected)} T1/T2 (○) + "
                f"{len(rt_selected)} RT (△)"
            )
        ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
        ax.legend(
            loc="lower left",
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            borderpad=0.4,
            handletextpad=0.3,
            labelcolor=TEXT_SECONDARY,
        )

    fig.suptitle(
        "Landsat scene sampler — stratum candidate pools (WRS-2 path/rows in marine ROI)",
        fontsize=13,
        color=TEXT_PRIMARY,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved map: {args.out}")


if __name__ == "__main__":
    main()
