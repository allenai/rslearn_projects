"""Test-set probability analysis + precision-recall curve for the classifier.

Loads the managed best checkpoint, runs inference over the ``test`` split of the
classifier dataset, collects ``prob(correct)`` vs. the ground-truth label for every
window, then:

  * prints a threshold sweep (precision / recall / F1 for the positive "correct" class),
  * writes per-sample probabilities to a CSV,
  * saves a precision-recall curve (with average precision) to a PNG.

Usage:
    python -m rslp.landsat_vessels.evaluation.pr_curve \
        --config data/landsat_vessels/config_classifier_20260616.yaml \
        --out_dir ./pr_out
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningDataModule, LightningModule
from rslearn.lightning_cli import RslearnArgumentParser, RslearnLightningCLI
from rslearn.train.model_context import ModelContext
from sklearn.metrics import average_precision_score, precision_recall_curve


def main() -> None:
    """Run test-set inference and emit the probability analysis + PR curve."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--ckpt",
        default="/weka/dfive-default/rslearn-eai/projects/landsat_vessel_classification_v2/olmoearth_base_v1.2_final_20260616/best.ckpt",
    )
    parser.add_argument("--out_dir", default="./pr_out")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build model + datamodule from the training config (no trainer run).
    cli = RslearnLightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        # Disable managed project mode; we load the checkpoint manually below.
        args=["--config", args.config, "--management_dir=null"],
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_class=RslearnArgumentParser,
        run=False,
    )
    module = cli.model
    datamodule = cli.datamodule
    task = datamodule.task

    # Index of the positive "correct" class.
    pos_id = task.classes.index("correct")

    # Load the best checkpoint weights.
    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec B614 - loading our own trained checkpoint
    module.load_state_dict(state["state_dict"])
    # Run on CPU: the OlmoEarth encoder holds some band-index buffers that .to(cuda)
    # does not relocate, causing a device mismatch. Only 50 test samples, so CPU is fine.
    device = "cpu"
    module.to(device).eval()

    # Map the requested split to the datamodule stage + dataloader.
    stage = {"train": "fit", "val": "validate", "test": "test"}[args.split]
    datamodule.setup(stage)
    loader = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }[args.split]()

    probs: list[float] = []
    labels: list[int] = []  # 1 == positive ("correct"), 0 == negative ("incorrect")
    # ClassificationTask(allow_invalid=True) returns class=0 with valid=0 for a window
    # whose label is not one of `classes` — "unsure" in round1_20260803, "skip" in
    # selected_copy, "unknown" in phase2a_completed. Class 0 is "correct" here, so
    # counting those as targets would invent positives; rslearn's own metrics mask them
    # out the same way.
    n_invalid = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, metadatas = batch
            inputs = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inp.items()}
                for inp in inputs
            ]
            context = ModelContext(inputs=inputs, metadatas=metadatas)
            # The OlmoEarth encoder runs under bfloat16 autocast, so its features are
            # bf16; wrap the whole forward so the fp32 decoder autocasts to match.
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                out = module(context)
            for o, tgt in zip(out.outputs, targets):
                if float(tgt["valid"]) <= 0:
                    n_invalid += 1
                    continue
                p = float(o[pos_id])
                probs.append(p)
                labels.append(1 if int(tgt["class"]) == pos_id else 0)

    probs_a = np.array(probs)
    labels_a = np.array(labels)
    n_pos = int(labels_a.sum())
    n_neg = int((labels_a == 0).sum())
    print(
        f"\nCollected {len(probs)} {args.split} samples: {n_pos} positive, "
        f"{n_neg} negative (skipped {n_invalid} with an out-of-vocabulary label)"
    )

    # ---- Per-sample CSV ----
    csv_path = os.path.join(args.out_dir, f"{args.split}_probs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prob_correct", "true_label"])
        for p, y in zip(probs, labels):
            w.writerow([f"{p:.6f}", "correct" if y == 1 else "incorrect"])
    print(f"Wrote per-sample probabilities: {csv_path}")

    # ---- prob(correct) histogram, split by true label ----
    buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    print("\nprob(correct) histogram (P=true correct, N=true incorrect):")
    print(f"  {'bucket':<14}{'P':>4}{'N':>5}")
    for lo, hi in zip(buckets, buckets[1:]):
        np_ = int(((probs_a >= lo) & (probs_a < hi) & (labels_a == 1)).sum())
        nn_ = int(((probs_a >= lo) & (probs_a < hi) & (labels_a == 0)).sum())
        print(f"  [{lo:.2f},{hi:.2f}){'':<2}{np_:>4}{nn_:>5}")

    # ---- Threshold sweep for the positive class ----
    print("\nThreshold sweep (positive class = 'correct'):")
    print(f"  {'thr':>5}{'prec':>8}{'recall':>8}{'f1':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        pred = probs_a >= thr
        tp = int((pred & (labels_a == 1)).sum())
        fp = int((pred & (labels_a == 0)).sum())
        fn = int((~pred & (labels_a == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {thr:>5.2f}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}{tp:>5}{fp:>5}{fn:>5}")

    # ---- PR curve ----
    precision, recall, thresholds = precision_recall_curve(labels_a, probs_a)
    ap = average_precision_score(labels_a, probs_a)
    print(f"\nAverage precision (AP) for 'correct': {ap:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker=".", label=f"correct (AP={ap:.3f})")
    baseline = n_pos / (n_pos + n_neg)
    plt.axhline(baseline, ls="--", color="gray", lw=1, label=f"baseline={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — landsat classifier ({args.split} split)")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    png_path = os.path.join(args.out_dir, f"pr_curve_{args.split}.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved PR curve: {png_path}")

    # Also dump the PR points to CSV for inspection.
    pr_csv = os.path.join(args.out_dir, f"pr_points_{args.split}.csv")
    with open(pr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall"])
        # thresholds has one fewer element than precision/recall.
        for i, thr in enumerate(thresholds):
            w.writerow([f"{thr:.6f}", f"{precision[i]:.6f}", f"{recall[i]:.6f}"])
    print(f"Wrote PR points: {pr_csv}")


if __name__ == "__main__":
    main()
