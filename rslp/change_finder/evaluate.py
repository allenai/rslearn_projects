"""Evaluate a trained change finder model on test windows.

For each test window, computes P(class 1) for triplets [0y, Xy, 6y] with X in {1..5},
then writes per-window results to a JSON file.
"""

import argparse
import json

import torch
import tqdm
from rslearn.dataset import Dataset
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import ModelContext, RasterImage
from upath import UPath

from .train import (
    ChangeFinderLightningModule,
    ChangeFinderModel,
    ChangeFinderNormalize,
    ChangeFinderTask,
)

OLMOEARTH_BAND_ORDER = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]

YEAR_OFFSETS = list(range(7))


def load_model(checkpoint_path: str, device: str = "cuda") -> ChangeFinderModel:
    """Load a trained ChangeFinderModel from a Lightning checkpoint.

    Args:
        checkpoint_path: path to the .ckpt file.
        device: torch device string.

    Returns:
        the model in eval mode on the specified device.
    """
    lit_module = ChangeFinderLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model: ChangeFinderModel = lit_module.model
    model.eval()
    model.to(device)
    return model


def evaluate(
    ds_path: str,
    checkpoint_path: str,
    output_path: str,
    device: str = "cuda",
    crop_size: int = 64,
) -> None:
    """Run evaluation and write per-window per-timestep scores.

    Args:
        ds_path: path to the rslearn dataset (must have all 7 sentinel2_yN layers).
        checkpoint_path: path to the trained model checkpoint.
        output_path: path to write the JSON results.
        device: torch device.
        crop_size: spatial crop size for evaluation.
    """
    model = load_model(checkpoint_path, device)

    modality_names = [f"sentinel2_y{i}" for i in YEAR_OFFSETS]
    normalizer = ChangeFinderNormalize(
        modality_names=modality_names,
        band_names=OLMOEARTH_BAND_ORDER,
        skip_missing=True,
    )

    dataset = Dataset(UPath(ds_path))

    inputs_config = {}
    for i in YEAR_OFFSETS:
        inputs_config[f"sentinel2_y{i}"] = DataInput(
            data_type="raster",
            layers=[f"sentinel2_y{i}"],
            bands=OLMOEARTH_BAND_ORDER,
            passthrough=True,
            dtype="FLOAT32",
            load_all_item_groups=True,
            load_all_layers=True,
        )

    split_config = SplitConfig(
        tags={"split": "val"},
        crop_size=crop_size,
    )

    task = ChangeFinderTask()

    model_dataset = ModelDataset(
        dataset=dataset,
        inputs=inputs_config,
        task=task,
        split_config=split_config,
        transforms=normalizer,
    )

    results = []
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(model_dataset)), desc="Evaluating"):
            input_dict, _target_dict, metadata = model_dataset[idx]

            year_images = {}
            for i in YEAR_OFFSETS:
                key = f"sentinel2_y{i}"
                if key in input_dict:
                    year_images[i] = input_dict[key]

            if 0 not in year_images or 6 not in year_images:
                continue

            scores = {}
            for x in range(1, 6):
                if x not in year_images:
                    continue

                anchor1 = RasterImage(
                    year_images[0].image.to(device),
                    timestamps=year_images[0].timestamps,
                )
                query = RasterImage(
                    year_images[x].image.to(device),
                    timestamps=year_images[x].timestamps,
                )
                anchor2 = RasterImage(
                    year_images[6].image.to(device),
                    timestamps=year_images[6].timestamps,
                )

                context = ModelContext(
                    inputs=[{"anchor1": anchor1, "query": query, "anchor2": anchor2}],
                    metadatas=[metadata],
                )
                model_output = model(context)
                logits = model_output.outputs[0]["change"].unsqueeze(0)
                probs = torch.softmax(logits, dim=1)
                scores[f"y{x}"] = probs[0, 1].item()

            window_info = {
                "window_name": metadata.window_name,
                "window_group": metadata.window_group,
            }
            if metadata.time_range is not None:
                window_info["base_time"] = metadata.time_range[0].isoformat()
            window_info["scores"] = scores

            results.append(window_info)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate change finder model")
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--checkpoint_path", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to write JSON results"
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--crop_size", type=int, default=64)
    args = parser.parse_args()
    evaluate(
        ds_path=args.ds_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        device=args.device,
        crop_size=args.crop_size,
    )
