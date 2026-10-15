"""Seed variance on the headline probe cells: mean +/- sd over 3 seeds.

Seed 42 is the published run (its dumps live in p{dim}-{split}-lr*-r5); 43 and 44
are the new sv-* runs. All three are scored by the SAME code path here rather than
reusing the published numbers, so the spread reflects seeds and not two different
scorers.

Per seed the cell takes its best LR -- the published selection procedure -- so the
sd is the variance of the whole procedure, which is what an error bar should mean.
"""
import glob, json, os, re
import numpy as np
import torch

P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
ISL = ["reunion", "guadeloupe", "martinique", "guyane", "mayotte"]
MODTOK = {"S2": "hw16_sentinel2_l2a", "S1": "hw16_sentinel1",
          "S1+S2": "hw16_sentinel1-sentinel2_l2a",
          "S2+LS": "hw16_landsat-sentinel2_l2a",
          "S1+S2+LS": "hw16_landsat-sentinel1-sentinel2_l2a"}


def sc(p):
    x = torch.load(p, map_location="cpu")
    pr, gt = x["preds"].numpy().ravel(), x["labels"].numpy().ravel()
    m = (gt >= 0) & (gt <= 7)
    pr, gt = pr[m], gt[m]
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (gt, pr), 1)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(tp + cm.sum(0) - tp + cm.sum(1) - tp, 1e-9)
    pres = cm.sum(1) > 0
    return round(float(iou[pres].mean() * 100), 2)


def dirs_for(dim, split, seed):
    if seed == 42:
        return sorted(glob.glob(f"{P2}/oe_preddump_p{dim}-{split}-lr*-r5"))
    return sorted(glob.glob(f"{P2}/oe_preddump_sv-p{dim}-{split}-lr*-s{seed}"))


def best(dim, split, seed, match):
    vals = []
    for d in dirs_for(dim, split, seed):
        for f in os.listdir(d):
            if f.endswith("_preds.pt") and match(f):
                vals.append(sc(os.path.join(d, f)))
    return max(vals) if vals else None


out = {}
for dim in (64, 128):
    for mod, tok in MODTOK.items():
        # 443
        seeds = {s: best(dim, "443", s, lambda f, t=tok: f"{t}_lr" in f)
                 for s in (42, 43, 44)}
        got = [v for v in seeds.values() if v is not None]
        if len(got) == 3:
            out[f"443|{dim}|{mod}"] = {"mean": round(float(np.mean(got)), 2),
                                       "sd": round(float(np.std(got, ddof=1)), 2),
                                       "n": 3, "seeds": got}
        # loio
        for isl in ISL:
            seeds = {s: best(dim, "loio", s,
                             lambda f, t=tok, i=isl: f"_loio_{i}_" in f and f"{t}_lr" in f)
                     for s in (42, 43, 44)}
            got = [v for v in seeds.values() if v is not None]
            if len(got) == 3:
                out[f"loio|{dim}|{mod}|{isl}"] = {
                    "mean": round(float(np.mean(got)), 2),
                    "sd": round(float(np.std(got, ddof=1)), 2),
                    "n": 3, "seeds": got}
json.dump(out, open(f"{P2}/seed_variance.json", "w"), indent=1)
sds = [v["sd"] for v in out.values()]
print(f"  {len(out)} cells with n=3")
print(f"  sd: min {min(sds):.2f}  median {np.median(sds):.2f}  max {max(sds):.2f}")
for k in sorted(out, key=lambda k: -out[k]["sd"])[:6]:
    v = out[k]
    print(f"    widest: {k:28s} {v['mean']:.2f} +/- {v['sd']:.2f}  seeds={v['seeds']}")
