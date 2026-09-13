"""Backfill every trop-4 that existing dumps support, LOIO + 443.

Method: pick the dump from the LR that MAXIMISES mIoU -- the same selection the
published tables use -- then require it to reproduce the published mIoU. Where LRs
tie, their trop-4 agrees to within 0.02, so the tie is immaterial (verified).

TSViT is scored from its tiledump cropped to the inner 120px: its reassembly pads
a 5x5 grid of 24px tiles into a 128px array, and the zero border is fabricated
Background it never predicted (costs 2.63 mIoU if scored).
"""
import json, os, pickle, glob
import numpy as np
import torch

P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
D = json.load(open(f"{P2}/v3_data.json"))
ISL = D["meta"]["isl"]
SLUG = dict(zip(ISL, ["reunion", "guadeloupe", "martinique", "guyane", "mayotte"]))


def from_arrays(pr, gt):
    # Void is spelled differently per producer: the probe dumps use -1, the TSViT
    # bg8void tiledumps use 8 (labels 0..8, predictions 0..7). Masking on the valid
    # class range handles both; assuming -1 raised IndexError on the tiledumps.
    m = (gt >= 0) & (gt <= 7)
    pr, gt = pr[m], gt[m]
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (gt, pr), 1)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(tp + cm.sum(0) - tp + cm.sum(1) - tp, 1e-9)
    pres = cm.sum(1) > 0
    tr = [iou[i] for i in (4, 5, 6, 7) if pres[i]]
    return (round(float(iou[pres].mean() * 100), 2),
            round(float(np.mean(tr) * 100), 2) if tr else None)


def sc(p, inner=None):
    x = torch.load(p, map_location="cpu")
    pr, gt = x["preds"].numpy(), x["labels"].numpy()
    if inner:
        pr, gt = pr[:, :inner, :inner], gt[:, :inner, :inner]
    return from_arrays(pr.ravel(), gt.ravel())


def best(cands):
    """(mIoU, trop4) of the max-mIoU candidate."""
    if not cands:
        return None
    return max(cands, key=lambda r: r[0])


out, unresolved = {}, []


def put(key, label, pub, got):
    if got is None:
        unresolved.append((label, pub, "no dump"))
        return
    mi, tr = got
    if abs(mi - pub) > 0.06:
        unresolved.append((label, pub, f"best dump gives {mi}"))
        return
    out[key] = tr
    print(f"  {label:40s} pub {pub:6.2f}  dump {mi:6.2f}  trop-4 {tr}", flush=True)


# ---- LOIO S2: Tessera, AEF (probe dumps), TSViT (tiledump) ----
for model, d, tokfmt in (("Tessera v2", "oe_preddump_v2loio_tessera", None),
                         ("AEF / GSE", "oe_preddump_v2loio_aef", None)):
    row = next(r for r in D["loio"]["S2"] if r["model"] == model)
    for i, isl in enumerate(ISL):
        c = row["cells"][i]
        if not (isinstance(c, dict) and "v" in c) or c.get("t") is not None:
            continue
        cands = [sc(f"{P2}/{d}/{f}") for f in os.listdir(f"{P2}/{d}")
                 if f.endswith("_preds.pt") and f"_loio_{SLUG[isl]}_" in f]
        put(f"loio|S2|{model}|{isl}", f"LOIO {model} {isl}", c["v"], best(cands))

row = next(r for r in D["loio"]["S2"] if r["model"] == "TSViT (scratch)")
for i, isl in enumerate(ISL):
    c = row["cells"][i]
    if not (isinstance(c, dict) and "v" in c) or c.get("t") is not None:
        continue
    p = f"{P2}/oe_preddump_443/tsvit_bg8void_loio_{SLUG[isl]}_tiledump.pt"
    put(f"loio|S2|TSViT (scratch)|{isl}", f"LOIO TSViT {isl}", c["v"],
        sc(p) if os.path.exists(p) else None)

# ---- 443 ----
t443 = {r["model"]: r for r in D["t443"]}
# probes: their own 443 dumps
for dim in (64, 128):
    c = t443[f"OlmoEarth probe · {dim}d"]["cells"][0]
    if c.get("t") is None:
        cands = []
        for d in glob.glob(f"{P2}/oe_preddump_p{dim}-443-lr*-r5"):
            for f in os.listdir(d):
                if f.endswith("_preds.pt") and "hw16_sentinel2_l2a_lr" in f:
                    cands.append(sc(os.path.join(d, f)))
        put(f"t443|OlmoEarth probe · {dim}d", f"443 probe {dim}d", c["v"], best(cands))
# TSViT variants from tiledumps (inner 120)
for model, fn in (("TSViT (scratch)", "tsvit_bg8void_tiledump.pt"),
                  ("TSViT + focal γ=2", "tsvit_bg8void_cw_focal2_tiledump.pt")):
    c = t443[model]["cells"][0]
    p = f"{P2}/oe_preddump_443/{fn}"
    if c.get("t") is None:
        put(f"t443|{model}", f"443 {model}", c["v"],
            sc(p) if os.path.exists(p) else None)
# U-TAE from its confusion matrix
for model, sub in (("U-TAE (scratch)", "utae_paps_run"),
                   ("U-TAE + invsqrt·focal", "utae_paps_diag10")):
    c = t443[model]["cells"][0]
    if c.get("t") is not None:
        continue
    ps = glob.glob(f"/weka/dfive-default/piperw/pastis2_drom_bg8void/{sub}/**/conf_mat.pkl",
                   recursive=True)
    if not ps:
        unresolved.append((f"443 {model}", c["v"], "no conf_mat"))
        continue
    cm = np.asarray(pickle.load(open(ps[0], "rb")), dtype=np.float64)[:8, :8]
    tp = np.diag(cm)
    iou = tp / np.maximum(tp + cm.sum(0) - tp + cm.sum(1) - tp, 1e-9)
    pres = cm.sum(1) > 0
    tr = [iou[i] for i in (4, 5, 6, 7) if pres[i]]
    put(f"t443|{model}", f"443 {model}", c["v"],
        (round(float(iou[pres].mean() * 100), 2),
         round(float(np.mean(tr) * 100), 2) if tr else None))
# trivial predicts Background only -> every tropical IoU is 0 by construction
out["t443|Trivial (all Background)"] = 0.0
print(f"  {'443 Trivial':40s} trop-4 0.00 (by construction)")

json.dump(out, open(f"{P2}/trop4_backfill.json", "w"), indent=1)
print(f"\n  filled {len(out)} cells; unresolved {len(unresolved)}")
for lbl, pub, why in unresolved:
    print(f"    {lbl:40s} pub {pub:6.2f}  {why}")
