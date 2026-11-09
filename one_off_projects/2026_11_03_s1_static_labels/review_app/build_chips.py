"""Build review-app items and chips for a list of label positions.

Input: a JSON list of {"id", "group", "win", "row", "col"} (any other fields are kept),
where win is "group/name" of a window in the dataset and (row, col) the pixel in it, e.g.
labels picked at random or model detections to look at. "rank" defaults to the order in
the list.

Per item, chips of the target image (t) and the up to four same-track historicals
(h1..h4): 96 x 96 px around the position (chips/) and 256 x 256 px zoomed out (chips/z/).
They show VV and VH in dB, with one stretch per band shared across dates and both zooms.
For each historical, a simple "is it there too" test is also recorded: VV contrast >= 10 dB
at the position and 15 x 15 px VV+VH NCC >= 0.7 with the target (best over +-2 px).

Output: <data_dir>/items.json, <data_dir>/chips/{,z/}<id>_<t|h1..h4>_<vv|vh>.png

Usage: python build_chips.py --ds_path .../dataset_v1/20261103 --input picks.json --data_dir review_data
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform

LAYERS = {
    "t": "sentinel1",
    "h1": "sentinel1_historical",
    "h2": "sentinel1_historical.1",
    "h3": "sentinel1_historical.2",
    "h4": "sentinel1_historical.3",
}
R, RZ = 48, 128


def crop_db(a: np.ndarray, r0: int, c0: int, rad: int) -> np.ndarray:
    """Crop (2 rad)^2 px around (r0, c0), zero-padded, in dB."""
    pad = np.pad(a, ((0, 0), (rad, rad), (rad, rad)))
    return 20 * np.log10(np.maximum(pad[:, r0 : r0 + 2 * rad, c0 : c0 + 2 * rad], 1))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation."""
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def present(t: np.ndarray, h: np.ndarray, r0: int, c0: int) -> dict | None:
    """Whether the object at (r0, c0) of the target t (dB) is also in historical h (dB)."""
    if h.shape != t.shape or r0 < 10 or c0 < 10 or r0 + 10 > t.shape[1] or c0 + 10 > t.shape[2]:
        return None
    v = h[0]
    peak = v[max(0, r0 - 2) : r0 + 3, max(0, c0 - 2) : c0 + 3].max()
    bg = v[max(0, r0 - 30) : r0 + 31, max(0, c0 - 30) : c0 + 31]
    bg = bg[bg > 0]
    if bg.size < 100 or peak <= 0:
        return None
    contrast = float(peak - np.median(bg))
    a = t[:, r0 - 7 : r0 + 8, c0 - 7 : c0 + 8]
    best = max(
        ncc(a, h[:, r0 - 7 + dy : r0 + 8 + dy, c0 - 7 + dx : c0 + 8 + dx])
        for dy in range(-2, 3)
        for dx in range(-2, 3)
    )
    return {
        "con": round(contrast, 1),
        "ncc": round(best, 3),
        "present": bool(contrast >= 10 and best >= 0.7),
    }


def build(windows_dir: str, chips_dir: str, it: dict) -> dict:
    """Render the chips of one item and fill in its metadata."""
    win, r0, c0 = it["win"], it["row"], it["col"]
    with open(f"{windows_dir}/{win}/metadata.json") as f:
        meta = json.load(f)
    with open(f"{windows_dir}/{win}/items.json") as f:
        layer_items = {x["layer_name"]: x for x in json.load(f)}
    scenes = {"t": layer_items["sentinel1"]["serialized_item_groups"][0][0]}
    hist = layer_items.get("sentinel1_historical", {}).get("serialized_item_groups", [])
    for i, group in enumerate(hist):
        scenes[f"h{i + 1}"] = group[0]
    imgs = {}
    for k, layer in LAYERS.items():
        path = f"{windows_dir}/{win}/layers/{layer}/vv_vh/geotiff.tif"
        completed = os.path.exists(f"{windows_dir}/{win}/layers/{layer}/completed")
        if k in scenes and os.path.exists(path) and (k == "t" or completed):
            with rasterio.open(path) as r:
                imgs[k] = r.read().astype(np.float32)
    target_db = 20 * np.log10(np.maximum(imgs["t"], 1))
    it["present"] = {
        k: present(target_db, 20 * np.log10(np.maximum(a, 1)), r0, c0)
        for k, a in imgs.items()
        if k != "t"
    }
    it["dates"] = {k: s["geometry"]["time_range"][0][:10] for k, s in scenes.items()}
    it["scenes"] = {k: s["name"] for k, s in scenes.items()}
    proj = meta["projection"]
    x = (meta["bounds"][0] + c0 + 0.5) * proj["x_resolution"]
    y = (meta["bounds"][1] + r0 + 0.5) * proj["y_resolution"]
    lon, lat = transform(proj["crs"], "EPSG:4326", [x], [y])
    it["lat"], it["lon"] = lat[0], lon[0]
    it.setdefault("split", "val" if "-val_" in win else "train")

    big = {k: crop_db(a, r0, c0, RZ) for k, a in imgs.items()}
    for b, band in enumerate(["vv", "vh"]):
        centres = [c[b][RZ - R : RZ + R, RZ - R : RZ + R] for c in big.values()]
        vals = np.concatenate([c[c > 0] for c in centres])
        lo = np.percentile(vals, 2) if vals.size else 0
        hi = max(c[R - 2 : R + 3, R - 2 : R + 3].max() for c in centres)
        for k, c in big.items():
            im = np.clip((c[b] - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(im).save(f"{chips_dir}/z/{it['id']}_{k}_{band}.png")
            Image.fromarray(im[RZ - R : RZ + R, RZ - R : RZ + R]).save(
                f"{chips_dir}/{it['id']}_{k}_{band}.png"
            )
    it["have"] = sorted(big)
    return it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    chips_dir = f"{args.data_dir}/chips"
    os.makedirs(f"{chips_dir}/z", exist_ok=True)
    with open(args.input) as f:
        todo = json.load(f)
    for i, it in enumerate(todo):
        it.setdefault("rank", i)
    with Pool(args.workers) as p:
        new = p.map(partial(build, f"{args.ds_path}/windows", chips_dir), todo)
    # Add to (or replace items in) an existing app.
    path = f"{args.data_dir}/items.json"
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    new_ids = {x["id"] for x in new}
    items = [x for x in old if x["id"] not in new_ids] + new
    with open(path + ".tmp", "w") as f:
        json.dump(items, f, indent=0)
    os.replace(path + ".tmp", path)
    print(len(new), "items built,", len(items), "in the app")
