"""Verify the round-1 windows are trainable, by reading them the way training will.

Builds the same inputs and task the classifier config uses and pulls real samples through
:class:`rslp.landsat_vessels.transforms.CenterCrop`, which takes the 512 px annotation
window down to the 64 px window the model trains on. This is what catches a window that
looks fine on disk but blows up (or silently mis-crops) in the data loader.

Pass several groups to check that a mixed-window-size run works: the new 512 px group and
an existing 64 px group must both come out as 64 px samples, since round 2 trains on them
together.

Usage:
    python verify_round1_windows.py                      # the new group alone
    python verify_round1_windows.py --group round1_20260803 feedback_20260325
"""

import argparse

import numpy as np
import torch
from rslearn.dataset import Dataset
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from upath import UPath

from rslp.landsat_vessels.transforms import CenterCrop

DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
# Band order from config_classifier_20260616.yaml.
BANDS = ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
WINDOW_SIZE = 512


def main() -> None:
    """Verify the round-1 windows exist and are well-formed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", default=["round1_20260803"])
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--n", type=int, default=8, help="samples to pull")
    parser.add_argument(
        "--split", default=None, help="only windows in this split (train/val/test)"
    )
    args = parser.parse_args()

    print(f"groups {args.group}: centre-cropping each window to {args.crop_size} px")

    inputs = {
        "landsat": DataInput(
            data_type="raster", layers=["landsat"], bands=BANDS, passthrough=True
        ),
        "targets": DataInput(data_type="vector", layers=["label"], is_target=True),
    }
    task = ClassificationTask(
        property_name="label",
        classes=["correct", "incorrect"],
        prob_property="prob",
        positive_class="correct",
        positive_class_threshold=0.85,
        enable_f1_metric=True,
        skip_unknown_categories=True,
        allow_invalid=True,
    )
    split_config = SplitConfig(
        groups=args.group,
        tags={"split": args.split} if args.split else None,
        transforms=[CenterCrop(crop_size=args.crop_size, image_selectors=["landsat"])],
    )

    dataset = ModelDataset(
        dataset=Dataset(DATASET_ROOT),
        split_config=split_config,
        inputs=inputs,
        task=task,
        workers=0,
    )
    print(f"dataset reports {len(dataset)} samples")
    if len(dataset) == 0:
        print(
            "FAIL: no samples — are any windows labelled? (run apply_round1_labels.py)"
        )
        raise SystemExit(1)

    failures = []
    for i in range(min(args.n, len(dataset))):
        sample = dataset[i]
        inputs_dict, targets, metadata = sample[0], sample[1], sample[2]
        image = inputs_dict["landsat"]
        # The loader hands back a RasterImage; single_ts_to_chw_tensor drops the single
        # timestep this dataset has and gives the (bands, height, width) tensor the model
        # sees. A bare tensor is accepted too, in case an rslearn version returns one.
        if hasattr(image, "single_ts_to_chw_tensor"):
            array = image.single_ts_to_chw_tensor()
        else:
            array = image
        shape = (
            tuple(array.shape)
            if isinstance(array, torch.Tensor)
            else tuple(np.asarray(array).shape)
        )

        label = targets.get("class") if isinstance(targets, dict) else None
        name = metadata.get("window_name", "?") if isinstance(metadata, dict) else "?"
        print(
            f"  [{i}] {name[:52]:52s} shape {shape} "
            f"label {label.item() if hasattr(label, 'item') else label}"  # type: ignore[union-attr]
        )
        if len(shape) != 3 or shape[0] != len(BANDS):
            failures.append(
                f"sample {i}: expected {len(BANDS)} bands, got shape {shape}"
            )
        if shape[1:] != (args.crop_size, args.crop_size):
            failures.append(
                f"sample {i}: expected {args.crop_size}x{args.crop_size}, got {shape[1:]}"
            )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print(
        f"\nok: {min(args.n, len(dataset))} samples read as "
        f"{len(BANDS)}x{args.crop_size}x{args.crop_size} with labels"
    )


if __name__ == "__main__":
    main()
