"""Comparison stats for the seagrass splits.

Point windows are counted per-sample by their label_name (each point is one class).
Polygon windows are counted by PIXEL across their label_raster (0=background,
1=sparse_seagrass, 2=dense_seagrass, 255=outside/ignore = no-seagrass).

Reports OLD (jamaica_2025_points + jamaica_2025_test_polygons[_no_sparse]) and
NEW (jamaica_2025_proper_splits) side by side.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio

DS = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")
W = DS / "windows"
PIX_NAME = {
    0: "no_seagrass(0/bg)",
    1: "sparse_seagrass",
    2: "dense_seagrass",
    255: "no_seagrass(255/outside)",
}


def _point_rec(d: Path) -> tuple | None:
    """(split, label_name) for a point window."""
    try:
        o = json.load((d / "metadata.json").open())["options"]
        return (o.get("split"), o.get("label_name"))
    except Exception:
        return None


def point_recs(group: str) -> list[tuple]:
    """Read (split, label_name) for all point windows in a group."""
    dirs = [
        d
        for d in (W / group).iterdir()
        if d.is_dir() and not d.name.startswith("polygon_")
    ]
    with multiprocessing.Pool(32) as p:
        return [r for r in p.map(_point_rec, dirs, chunksize=256) if r]


def _poly_pix(d: Path) -> tuple[str, dict[int, int]] | None:
    """(name, {pixel_value: count}) from a polygon window's label_raster."""
    try:
        with rasterio.open(
            d / "layers" / "label_raster" / "label" / "geotiff.tif"
        ) as src:
            a = src.read(1)
        v, c = np.unique(a, return_counts=True)
        return (d.name, dict(zip(v.tolist(), c.tolist())))
    except Exception:
        return None


def poly_pix_counts(group: str) -> dict[str, dict[int, int]]:
    """{window_name: {pixel_value: count}} for all polygon windows in a group."""
    dirs = [
        d for d in (W / group).iterdir() if d.is_dir() and d.name.startswith("polygon_")
    ]
    with multiprocessing.Pool(32) as p:
        recs = [r for r in p.map(_poly_pix, dirs, chunksize=128) if r]
    return dict(recs)


def proper_test_names() -> set[str]:
    """Names of polygon windows tagged split=test in jamaica_2025_proper_splits."""
    out = set()
    for d in (W / "jamaica_2025_proper_splits").iterdir():
        if d.is_dir() and d.name.startswith("polygon_"):
            with contextlib.suppress(OSError, json.JSONDecodeError, KeyError):
                if (
                    json.load((d / "metadata.json").open())["options"].get("split")
                    == "test"
                ):
                    out.add(d.name)
    return out


def sum_pix(
    per_window: dict[str, dict[int, int]], names: set[str] | None = None
) -> Counter:
    """Sum pixel-value counts over the named windows (or all)."""
    total: Counter = Counter()
    for name, vc in per_window.items():
        if names is not None and name not in names:
            continue
        for v, c in vc.items():
            total[v] += c
    return total


def fmt_pix(total: Counter) -> str:
    """Pretty per-class pixel summary."""
    tot = sum(total.values())
    lines = []
    for v in sorted(total):
        lines.append(
            f"      {PIX_NAME.get(v, f'value {v}'):26} {total[v]:>13,}  ({100*total[v]/tot:.2f}%)"
        )
    lines.append(f"      {'TOTAL':26} {tot:>13,}")
    return "\n".join(lines)


def main() -> None:
    """Compute and print all the comparison stats."""
    print("Reading point windows ...")
    old_pts = point_recs("jamaica_2025_points")
    new_pts = point_recs("jamaica_2025_proper_splits")
    print("Reading polygon label rasters (this reads ~13k small geotiffs) ...")
    poly = poly_pix_counts("jamaica_2025_test_polygons")
    # "test minus sparse-containing windows" == the actual dense-only group on disk
    nosparse_names = {
        d.name
        for d in (W / "jamaica_2025_test_polygons_no_sparse").iterdir()
        if d.is_dir() and d.name.startswith("polygon_")
    }
    zero_dense = sum(1 for n in nosparse_names if 2 not in poly.get(n, {}))
    new_test = proper_test_names()

    print("\n================  OLD  ================")
    c = Counter(ln for _s, ln in old_pts)
    print(
        f"OLD train points  (jamaica_2025_points, all {len(old_pts):,} samples) - per-sample class counts:"
    )
    for k in ["dense_seagrass", "sparse_seagrass", "background"]:
        print(f"      {('no_seagrass' if k=='background' else k):16} {c.get(k,0):>8,}")

    print(
        f"\nOLD test polygons (jamaica_2025_test_polygons, {len(poly):,} windows) - PIXEL counts:"
    )
    print(fmt_pix(sum_pix(poly)))

    print(
        f"\nOLD test minus sparse-containing windows (= no_sparse, {len(nosparse_names):,} windows; "
        f"{zero_dense} of them have 0 dense pixels - polygon too small to rasterize) - PIXEL counts:"
    )
    print(fmt_pix(sum_pix(poly, nosparse_names)))

    print("\n================  NEW  ================")
    bysplit: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for s, ln in new_pts:
        bysplit[s][ln] += 1
    print("NEW points (jamaica_2025_proper_splits) - per-sample class counts by split:")
    for s in ["train", "val"]:
        cc = bysplit[s]
        print(
            f"  {s} ({sum(cc.values()):,}):  dense_seagrass={cc.get('dense_seagrass',0):,}  "
            f"sparse_seagrass={cc.get('sparse_seagrass',0):,}  no_seagrass={cc.get('background',0):,}"
        )

    print(
        f"\nNEW test samples (proper_splits split=test, {len(new_test):,} polygons) - PIXEL counts:"
    )
    print(fmt_pix(sum_pix(poly, new_test)))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
