"""Script to collect OlmoEarth embeddings for an rslearn dataset."""

import argparse
from typing import Any

import numpy as np
import torch
import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.pad import Pad
from upath import UPath


def input_to_device(inp: Any, device: torch.DeviceObjType) -> Any:
    """Move input to device if it is a Tensor, otherwise return the input unchanged."""
    if isinstance(inp, torch.Tensor):
        inp = inp.to(device=device)
    return inp


def initialize_dataset_for_olmoearth(
    dataset: Dataset, input_size: int | None = None, workers: int = 32
) -> ModelDataset:
    """Create a ModelDataset for predicting OlmoEarth embeddings.

    The ModelDataset is configured to read only the images from the windows in the
    specified dataset. It is assumed that there is a layer "sentinel2" in the dataset
    containing Sentinel-2 L2A image time series with 12 timesteps.

    Args:
        dataset: the rslearn dataset to use.
        input_size: crop down to this size at the center of each window. If not
            specified, the entire window is read.
        workers: the number of workers to use for initializing the ModelDataset.

    Returns:
        the ModelDataset.
    """
    transforms = [
        OlmoEarthNormalize(
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
    if input_size is not None:
        transforms.append(
            Pad(
                size=input_size,
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
        workers=workers,
    )
    return model_dataset


def get_embeddings(
    model: OlmoEarth,
    model_dataset: ModelDataset,
    batch_size: int,
    mode: str,
    embed_fname: str,
    workers: int = 16,
) -> None:
    """Get OlmoEarth embeddings for each window in the ModelDataset.

    Args:
        model: the OlmoEarth model to apply.
        model_dataset: the dataset to apply the model on.
        batch_size: batch size to use for data loading.
        mode: embedding selection mode, "center" to pick the embedding for the patch at
            the center, or "pool" to average pool the embedding across the patches.
        workers: number of data loader worker processes.
        embed_fname: the filename to use to store the embeddings. It should be .npy.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=model_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    with torch.no_grad():
        for inputs, targets, metadatas in tqdm.tqdm(data_loader):
            # Skip inputs without enough Sentinel-2 images.
            # Currently we assume there are twelve timesteps, so the number of channels
            # in the image (which has T*C on first axis since the images are stacked)
            # should be 144 (12 Sentinel-2 L2A bands).
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
            features = model(gpu_inputs)[0]

            for cur_feat, metadata in zip(features, metadatas):
                if mode == "center":
                    selected = cur_feat.cpu().numpy()[
                        :, cur_feat.shape[1] // 2, cur_feat.shape[2] // 2
                    ]
                elif mode == "pool":
                    selected = cur_feat.cpu().numpy().mean(axis=(1, 2))
                else:
                    raise ValueError(f"invalid mode {mode}")
                out_fname = (
                    dataset.path
                    / "windows"
                    / metadata["group"]
                    / metadata["window_name"]
                    / args.embed_fname
                )
                np.save(out_fname.path, selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The path to the rslearn dataset",
        required=True,
    )
    # Currently this assumes there are twelve timesteps and we read all of the timesteps.
    # parser.add_argument(
    #     "--num_timesteps",
    #     type=int,
    #     help="Number of timesteps to input",
    #     required=True,
    # )
    parser.add_argument(
        "--patch_size",
        type=int,
        help="Patch size to use for OlmoEarth model",
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
        help="OlmoEarth checkpoint path",
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
    model_dataset = initialize_dataset_for_olmoearth(
        dataset,
        input_size=args.input_size,
        workers=args.workers,
    )

    print("Initializing OlmoEarth model")
    device = torch.device("cuda")
    model = OlmoEarth(
        checkpoint_path=args.checkpoint_path,
        selector=["encoder"],
        forward_kwargs=dict(patch_size=args.patch_size),
    )
    model.to(device)
    model.eval()

    get_embeddings(
        model=model,
        model_dataset=model_dataset,
        batch_size=args.batch_size,
        mode=args.mode,
        embed_fname=args.embed_fname,
        workers=args.workers,
    )
