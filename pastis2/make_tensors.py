"""Phase 6: turn materialized rslearn windows into PASTIS-format eval tensors.

Adapts olmoearth_pretrain/.../pastis_processor.py to read rslearn output instead of raw
PASTIS .npy. Per window it stacks the monthly Sentinel-2 composites onto the fixed
12-month grid (Sep<y-1>..Aug<y>), imputes B1/B9/B10 (10 -> 13 bands, identical band map
to the original), reads the rasterized `label`, quadrant-splits 128->4x64, and writes the
same on-disk layout the eval consumes: <out>/pastis2_<split>/{s2_images/{i}.pt, targets.pt,
months.pt}.

Missing months (rslearn may materialize fewer than 12; see README) are zero-filled on the
grid so every sample has T=12; months.pt records the canonical 12-month grid. (TODO:
nearest-month fill or a MISSING sentinel instead of zeros.)

Run:  python make_tensors.py --dataset smoke_ds --group rpg_2019 --out data/tensors
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from upath import UPath

from rslearn.dataset.dataset import Dataset

N_BANDS_IN, N_BANDS_OUT, PATCH = 10, 13, 128
IGNORE_LABEL = -1


def season_slots(year: int) -> list[int]:
    """The 12 canonical YYYYMM slots Sep<year-1>..Aug<year> (e.g. 201809..201908)."""
    idx0 = (year - 1) * 12 + 8  # Sep<year-1> as absolute month index (0-based month)
    return [((idx0 + k) // 12) * 100 + ((idx0 + k) % 12) + 1 for k in range(12)]


def _mindex(yyyymm: int) -> int:
    """YYYYMM -> absolute month index for distance math."""
    return (yyyymm // 100) * 12 + (yyyymm % 100 - 1)


def impute(img: torch.Tensor) -> torch.Tensor:
    """10 PASTIS bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] -> 13 (B1/B9/B10 imputed).

    Identical mapping to pastis_processor.impute so tensors match the existing eval.
    """
    return torch.stack([
        img[0], img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7],
        img[7], img[8], img[8], img[9],
    ])


def _month_of_group(items_group: list) -> int | None:
    """Parse YYYYMM from a group's first item name (e.g. S2A_31TCJ_20190804_..)."""
    for it in items_group:
        s = it.get("name", "") if isinstance(it, dict) else str(it)
        m = re.search(r"_(\d{8})", s)
        if m:
            return int(m.group(1)[:6])
    return None


def _read_geotiff(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()  # (C, H, W)


def window_to_grid(
    wdir: Path, slots: list[int], max_dist: int = 12
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Reduce a window's (possibly over-fetched) monthly composites to 12 season slots.

    Returns (s2 (12,13,128,128), targets (128,128), months) where months[k] is the actual
    YYYYMM used for slot k (0 if empty). Each slot is filled by the composite whose month
    is nearest (in absolute month distance, <= max_dist) to the slot's canonical month --
    so a cloudy target-year month is back-filled from an adjacent year rather than left
    empty, while present target-year months are always preferred (distance 0).
    """
    items = json.loads((wdir / "items.json").read_text())
    groups = next(l["serialized_item_groups"] for l in items if l["layer_name"] == "sentinel2")
    idx_to_month = {i: _month_of_group(g) for i, g in enumerate(groups)}

    # Gather all materialized composites as {month: imputed (13,H,W)}.
    available: dict[int, torch.Tensor] = {}
    for tif in glob.glob(str(wdir / "layers" / "sentinel2*" / "*" / "geotiff.tif")):
        layer = tif.split("/layers/")[1].split("/")[0]
        gidx = 0 if layer == "sentinel2" else int(layer.split(".")[1])
        month = idx_to_month.get(gidx)
        if month is None:
            continue
        available[month] = impute(torch.tensor(_read_geotiff(tif).astype("float32")))

    grid = torch.zeros(len(slots), N_BANDS_OUT, PATCH, PATCH, dtype=torch.float32)
    used_months = [0] * len(slots)
    for k, canon in enumerate(slots):
        if not available:
            break
        best = min(available, key=lambda m: abs(_mindex(m) - _mindex(canon)))
        if abs(_mindex(best) - _mindex(canon)) <= max_dist:
            grid[k] = available[best]
            used_months[k] = best

    label_tifs = glob.glob(str(wdir / "layers" / "label" / "*" / "geotiff.tif"))
    targets = torch.tensor(_read_geotiff(label_tifs[0])[0].astype("int64"))
    return grid, targets, used_months


def quad_split_s2(x: torch.Tensor) -> torch.Tensor:
    """(T,C,128,128) -> (4,T,C,64,64)."""
    return torch.stack([x[..., :64, :64], x[..., 64:, :64],
                        x[..., :64, 64:], x[..., 64:, 64:]])


def quad_split_t(x: torch.Tensor) -> torch.Tensor:
    """(128,128) -> (4,64,64)."""
    return torch.stack([x[:64, :64], x[64:, :64], x[:64, 64:], x[64:, 64:]])


def split_for(name: str) -> str:
    """Deterministic train/val/test by window name hash (60/20/20).

    TODO: use a proper spatial/geographic holdout (PASTIS used 5 folds) so train and
    test are not spatially adjacent.
    """
    h = int.from_bytes(name.encode(), "big") % 5
    return {0: "valid", 1: "test"}.get(h, "train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2019,
                    help="RPG/season year; the 12 slots are Sep<year-1>..Aug<year>")
    args = ap.parse_args()

    ds = Dataset(UPath(args.dataset))
    windows = ds.load_windows(groups=[args.group])
    slots = season_slots(args.year)
    print(f"processing {len(windows)} windows ... season slots {slots[0]}..{slots[-1]}")

    buckets: dict[str, dict[str, list]] = {
        s: {"s2": [], "targets": [], "months": []} for s in ("train", "valid", "test")
    }
    windows_root = Path(args.dataset) / "windows" / args.group
    filled_hist = []
    for w in windows:
        s2, targets, used = window_to_grid(windows_root / w.name, slots)
        filled_hist.append(sum(1 for m in used if m))
        s2_q, tgt_q = quad_split_s2(s2), quad_split_t(targets)  # (4,12,13,64,64),(4,64,64)
        months_t = torch.tensor(used, dtype=torch.long)
        b = buckets[split_for(w.name)]
        for q in range(4):
            b["s2"].append(s2_q[q])
            b["targets"].append(tgt_q[q])
            b["months"].append(months_t)
    if filled_hist:
        print(f"filled slots per window (of 12): min={min(filled_hist)} "
              f"max={max(filled_hist)} mean={sum(filled_hist)/len(filled_hist):.1f}")

    for split, b in buckets.items():
        if not b["s2"]:
            continue
        out = Path(args.out) / f"pastis2_{split}"
        (out / "s2_images").mkdir(parents=True, exist_ok=True)
        for i, s2 in enumerate(b["s2"]):
            torch.save(s2.clone(), out / "s2_images" / f"{i}.pt")
        torch.save(torch.stack(b["targets"]), out / "targets.pt")
        torch.save(torch.stack(b["months"]), out / "months.pt")
        print(f"  {split}: {len(b['s2'])} patches -> {out}")


if __name__ == "__main__":
    main()
