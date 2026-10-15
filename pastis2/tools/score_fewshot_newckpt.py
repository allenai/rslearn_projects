"""Best-LR-per-cell for the new-checkpoint few-shot grid.

Selection matches how the 443/LOIO probe tables are built: for each
(dim, arm, X) take the highest mIoU across the 8 learning rates. Scored from the
prediction dumps, so mIoU and trop-4 always come from the same run.

The px arms are UNIFIED 23-class (void = 23), not bg8void: score over the BG8
subset [0,1,12,16,19,20,21,22]. Using the bg8void class list here would produce
plausible nonsense.
"""
import glob, json, os, re
import numpy as np
import torch

P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
XS = [10, 25, 100, 1000]
ARMS = {"unbal": "pxi", "bal": "bal", "plo": "plo"}
BG8 = [0, 1, 12, 16, 19, 20, 21, 22]
VOID = 23


def score(path):
    x = torch.load(path, map_location="cpu")
    pr, gt = x["preds"].numpy().ravel(), x["labels"].numpy().ravel()
    m = (gt != VOID) & (gt >= 0)
    pr, gt = pr[m], gt[m]
    C = int(max(gt.max(), pr.max())) + 1
    C = max(C, VOID + 1)
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (gt, pr), 1)
    tp = np.array([cm[c, c] for c in BG8], float)
    fn = np.array([cm[c, :].sum() - cm[c, c] for c in BG8], float)
    fp = np.array([cm[:, c].sum() - cm[c, c] for c in BG8], float)
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    pres = np.array([cm[c, :].sum() > 0 for c in BG8])
    if not pres.any():
        return None
    trop = [iou[i] for i in (4, 5, 6, 7) if pres[i]]
    return {"miou8": round(float(iou[pres].mean() * 100), 2),
            "trop4": round(float(np.mean(trop) * 100), 2) if trop else None,
            "acc": round(float(tp.sum() / max(cm[np.array(BG8), :].sum(), 1) * 100), 2)}


out = {}
for dim in (64, 128):
    for arm, tok in ARMS.items():
        for X in XS:
            best = None
            for d in sorted(glob.glob(f"{P2}/oe_preddump_fsb-p{dim}-{arm}-lr*")):
                lr = re.search(r"-lr([0-9.]+)$", d)
                lr = lr.group(1) if lr else "?"
                hits = [f for f in os.listdir(d)
                        if f.endswith("_preds.pt") and f"_{tok}{X}_" in f]
                if len(hits) != 1:
                    continue
                s = score(os.path.join(d, hits[0]))
                if s and (best is None or s["miou8"] > best["miou8"]):
                    best = dict(s, lr=lr)
            if best:
                out[f"{dim}|{arm}|{X}"] = best
                print(f"  p{dim} {arm:6s} X={X:<5d} mIoU {best['miou8']:6.2f}  "
                      f"trop-4 {str(best['trop4']):>6s}  acc {best['acc']:6.2f}  "
                      f"(lr={best['lr']})", flush=True)
json.dump(out, open(f"{P2}/fewshot_newckpt.json", "w"), indent=1)
print(f"  wrote fewshot_newckpt.json ({len(out)}/24 cells)")
