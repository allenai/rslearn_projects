"""Script to collect Helios embeddings for an rslearn dataset."""

import argparse
from typing import Any

import numpy as np
import torch
import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.pad import Pad
from upath import UPath

from rslp.helios.model import Helios
from rslp.helios.norm import HeliosNormalize


def input_to_device(inp: Any, device: torch.DeviceObjType) -> Any:
    """Move input to device if it is a Tensor, otherwise return the input unchanged."""
    if isinstance(inp, torch.Tensor):
        inp = inp.to(device=device)
    return inp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The path to the rslearn dataset",
        required=True,
    )
    # parser.add_argument(
    #     "--num_timesteps",
    #     type=int,
    #     help="Number of timesteps to input",
    #     required=True,
    # )
    parser.add_argument(
        "--patch_size",
        type=int,
        help="Patch size to use for Helios model",
        required=True,
    )
    parser.add_argument(
        "--input_size",
        type=int,
        help="Input crop size",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Helios checkpoint path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes for dataset initialization and data loading",
        default=32,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Model batch size",
        default=32,
    )
    parser.add_argument(
        "--embed_fname",
        type=str,
        help="Filename to use to save the embeddings",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Either center or pool, default center",
        default="center",
    )
    args = parser.parse_args()

    print("Initializing dataset")
    dataset = Dataset(UPath(args.ds_path))
    transforms = [
        HeliosNormalize(
            config_fname="/weka/dfive-default/gabrielt/helios/data/norm_configs/computed.json",
            band_names=dict(
                sentinel2_l2a=[
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
                ],
            ),
        ),
    ]
    if args.input_size is not None:
        transforms.append(
            Pad(
                size=args.input_size,
                mode="center",
                image_selectors=["sentinel2_l2a"],
            ),
        )
    model_dataset = ModelDataset(
        dataset=dataset,
        split_config=SplitConfig(
            transforms=transforms,
            skip_targets=True,
        ),
        inputs=dict(
            sentinel2_l2a=DataInput(
                data_type="raster",
                layers=["sentinel2"],
                bands=[
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
                ],
                passthrough=True,
                load_all_layers=True,
                load_all_item_groups=True,
            ),
            label=DataInput(
                data_type="vector",
                layers=["label"],
                is_target=True,
            ),
        ),
        task=ClassificationTask(property_name="placeholder", classes=["cls0"]),
        workers=args.workers,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=model_dataset,
        num_workers=args.workers,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
    )

    print("Initializing Helios model")
    device = torch.device("cpu")
    helios = Helios(
        checkpoint_path=args.checkpoint_path,
        selector=["encoder"],
        forward_kwargs=dict(patch_size=args.patch_size),
    )
    helios.to(device)
    helios.eval()

    with torch.no_grad():
        for inputs, targets, metadatas in tqdm.tqdm(data_loader):
            # Skip inputs without enough Sentinel-2 images.
            good_indexes = []
            for idx, input_dict in enumerate(inputs):
                if input_dict["sentinel2_l2a"].shape[0] != 144:
                    continue
                good_indexes.append(idx)
            inputs = [inputs[idx] for idx in good_indexes]
            targets = [targets[idx] for idx in good_indexes]
            metadatas = [metadatas[idx] for idx in good_indexes]

            gpu_inputs = [
                {k: input_to_device(v, device) for k, v in input_dict.items()}
                for input_dict in inputs
            ]
            features = helios(gpu_inputs)[0]

            for cur_feat, metadata in zip(features, metadatas):
                if args.mode == "center":
                    selected = cur_feat.cpu().numpy()[
                        :, cur_feat.shape[1] // 2, cur_feat.shape[2] // 2
                    ]
                elif args.mode == "pool":
                    selected = cur_feat.cpu().numpy().mean(axis=(1, 2))
                else:
                    raise ValueError(f"invalid mode {args.mode}")
                out_fname = (
                    dataset.path
                    / "windows"
                    / metadata["group"]
                    / metadata["window_name"]
                    / args.embed_fname
                )
                np.save(out_fname.path, selected)
