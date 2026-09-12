"""Render the misclassified windows for a split, to judge label-noise vs. ambiguity.

Runs the best checkpoint over a split, tracks each window's name, and finds where the
prediction (prob >= threshold) disagrees with the ground-truth label. Each misclassified
window is rendered as a Brovey pan-sharpened tile annotated with the true label and
prob(correct), grouped into false positives (pred pos / true neg) and false negatives
(pred neg / true pos).

Usage:
    python -m rslp.landsat_vessels.evaluation.misclassified --split val --threshold 0.5
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningDataModule, LightningModule
from rslearn.lightning_cli import RslearnArgumentParser, RslearnLightningCLI
from rslearn.train.model_context import ModelContext

from rslp.landsat_vessels.evaluation.visualize_split import DS, GROUPS, rgb_thumb


def main() -> None:
    """Run inference, collect misclassified windows, and render them."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="data/landsat_vessels/config_classifier_20260616.yaml"
    )
    parser.add_argument(
        "--ckpt",
        default="/weka/dfive-default/rslearn-eai/projects/landsat_vessel_classification_v2/olmoearth_base_v1.2_final_20260616/best.ckpt",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", default="./pr_out")
    parser.add_argument("--cols", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cli = RslearnLightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        args=["--config", args.config, "--management_dir=null"],
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_class=RslearnArgumentParser,
        run=False,
    )
    module = cli.model
    datamodule = cli.datamodule
    task = datamodule.task
    pos_id = task.classes.index("correct")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec B614 - loading our own trained checkpoint
    module.load_state_dict(state["state_dict"])
    device = "cpu"
    module.to(device).eval()

    stage = {"train": "fit", "val": "validate", "test": "test"}[args.split]
    datamodule.setup(stage)
    loader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[args.split]()

    # rows: (window_name, prob_correct, true_label_str)
    rows: list[tuple[str, float, str]] = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets, metadatas = batch
            context = ModelContext(inputs=inputs, metadatas=metadatas)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                out = module(context)
            for o, tgt, md in zip(out.outputs, targets, metadatas):
                prob = float(o[pos_id])
                true_lbl = "correct" if int(tgt["class"]) == pos_id else "incorrect"
                rows.append((md.window_name, prob, true_lbl))

    # Find the window directory for each name (search the config groups).
    name_to_dir: dict[str, str] = {}
    for g in GROUPS:
        gdir = os.path.join(DS, "windows", g)
        if os.path.isdir(gdir):
            for w in os.listdir(gdir):
                name_to_dir.setdefault(w, os.path.join(gdir, w))

    # Classify errors.
    false_pos = []  # pred pos, true neg
    false_neg = []  # pred neg, true pos
    for name, prob, true_lbl in rows:
        pred_pos = prob >= args.threshold
        if pred_pos and true_lbl == "incorrect":
            false_pos.append((name, prob, true_lbl))
        elif not pred_pos and true_lbl == "correct":
            false_neg.append((name, prob, true_lbl))

    false_pos.sort(key=lambda r: -r[1])  # most confident FP first
    false_neg.sort(key=lambda r: r[1])  # most confident FN first
    errors = [("FP", *r) for r in false_pos] + [("FN", *r) for r in false_neg]
    print(
        f"split={args.split} thr={args.threshold}: {len(rows)} windows, "
        f"{len(false_pos)} false positives, {len(false_neg)} false negatives"
    )
    if not errors:
        print("No misclassifications at this threshold.")
        return

    n = len(errors)
    cols = args.cols
    rows_grid = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 1.7, rows_grid * 2.0))
    axes = np.atleast_2d(axes)
    for i in range(rows_grid * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        kind, name, prob, true_lbl = errors[i]
        wdir = name_to_dir.get(name)
        # FP border orange, FN border purple.
        border = "#ff7f0e" if kind == "FP" else "#9467bd"
        thumb = rgb_thumb(wdir) if wdir else None
        title = f"{kind}  p={prob:.2f}\ntrue={'pos' if true_lbl=='correct' else 'neg'}"
        if thumb is None:
            ax.text(0.5, 0.5, "no img\n" + title, ha="center", va="center", fontsize=7)
            continue
        ax.imshow(thumb)
        ax.set_title(title, fontsize=7, pad=2, color=border)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{args.split} misclassifications @ thr={args.threshold}  "
        f"(FP=orange pred-pos/true-neg: {len(false_pos)},  "
        f"FN=purple pred-neg/true-pos: {len(false_neg)})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(
        args.out_dir, f"misclassified_{args.split}_thr{args.threshold}.png"
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
