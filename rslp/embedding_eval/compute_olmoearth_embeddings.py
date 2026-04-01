"""Script to collect OlmoEarth embeddings for an rslearn dataset.

The dataset must have a "sentinel2" layer with a Sentinel-2 L2A image time series.
"""

import argparse
import os
from typing import Any

import h5py
import numpy as np
import torch
import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.models.olmoearth_pretrain.model import ModelID, OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.crop import Crop
from rslearn.train.transforms.pad import Pad
from upath import UPath


def input_to_device(inp: Any, device: torch.DeviceObjType) -> Any:
    """Move input to device if it is a Tensor or RasterImage."""
    if isinstance(inp, torch.Tensor):
        return inp.to(device=device)
    if isinstance(inp, RasterImage):
        return RasterImage(image=inp.image.to(device=device), timestamps=inp.timestamps)
    return inp


def initialize_dataset_for_olmoearth(
    dataset: Dataset,
    input_size: int | None = None,
    label_position: tuple[int, int] | None = None,
    workers: int = 32,
) -> ModelDataset:
    """Create a ModelDataset for predicting OlmoEarth embeddings.

    The ModelDataset is configured to read only the images from the windows in the
    specified dataset. It is assumed that there is a layer "sentinel2" in the dataset
    containing Sentinel-2 L2A image time series with 12 timesteps.

    Args:
        dataset: the rslearn dataset to use.
        input_size: crop down to this size at the center of each window. If not
            specified, the entire window is read.
        label_position: (x, y) pixel position in the output crop where the window center
            (label point) should land. If not specified, the crop is centered.
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
        if label_position is not None:
            # Two-step crop: first Pad (center crop) to a known intermediate size,
            # then Crop with offset to place the label point at label_position.
            # The intermediate size must be large enough that the crop region fits.
            lx, ly = label_position
            # e.g. if user wants label at (2, 2) in an 8x8 crop then we can center crop
            # down to 12x12 around the window's center and then crop further from
            # offset (4, 4).
            center_crop_size = 2 * max(lx, ly, input_size - lx, input_size - ly)
            transforms.append(
                Pad(
                    size=center_crop_size,
                    mode="center",
                    image_selectors=["sentinel2_l2a"],
                ),
            )
            # After Pad, image center is at (pad_size//2, pad_size//2).
            col1 = center_crop_size // 2 - lx
            row1 = center_crop_size // 2 - ly
            transforms.append(
                Crop(
                    crop_size=input_size,
                    offset=(col1, row1),
                    image_selectors=["sentinel2_l2a"],
                ),
            )
        else:
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
    out_path: str,
    device: torch.device,
    input_size: int | None = None,
    label_position: tuple[int, int] | None = None,
    workers: int = 16,
) -> None:
    """Get OlmoEarth embeddings for each window in the ModelDataset and save to H5.

    Args:
        model: the OlmoEarth model to apply.
        model_dataset: the dataset to apply the model on.
        batch_size: batch size to use for data loading.
        mode: embedding selection mode, "center" to pick the embedding for the patch
            closest to the label point, or "pool" to average pool across patches.
        out_path: path to the output H5 file.
        device: torch device to run the model on.
        input_size: the crop size (needed when label_position is set).
        label_position: (x, y) pixel position of the label in the crop. When set with
            mode="center", picks the patch covering this position instead of the center.
        workers: number of data loader worker processes.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=model_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    all_window_names: list[str] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets, metadatas in tqdm.tqdm(data_loader):
            gpu_inputs = [
                {k: input_to_device(v, device) for k, v in input_dict.items()}
                for input_dict in inputs
            ]
            context = ModelContext(inputs=gpu_inputs, metadatas=list(metadatas))
            feature_maps = model(context)
            features = feature_maps.feature_maps[0]

            for cur_feat, metadata in zip(features, metadatas):
                if mode == "center":
                    if label_position is not None and input_size is not None:
                        # Figure out which patch contains the label_position pixel.
                        # Since in this case the label (which was at center of the
                        # original window) may not be at center of the crop.
                        feat_h, feat_w = cur_feat.shape[1], cur_feat.shape[2]
                        feat_row = min(
                            int(label_position[1] / input_size * feat_h), feat_h - 1
                        )
                        feat_col = min(
                            int(label_position[0] / input_size * feat_w), feat_w - 1
                        )
                    else:
                        feat_row = cur_feat.shape[1] // 2
                        feat_col = cur_feat.shape[2] // 2
                    selected = cur_feat.cpu().numpy()[:, feat_row, feat_col]
                elif mode == "pool":
                    selected = cur_feat.cpu().numpy().mean(axis=(1, 2))
                else:
                    raise ValueError(f"invalid mode {mode}")
                window_name = f"{metadata.window_group}/{metadata.window_name}"
                all_window_names.append(window_name)
                all_embeddings.append(selected)

    print(f"Writing {len(all_embeddings)} embeddings to {out_path}")
    tmp_path = out_path + f".tmp.{os.getpid()}"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset(
            "window_names",
            data=np.array(all_window_names, dtype=h5py.string_dtype()),
        )
        f.create_dataset("embeddings", data=np.stack(all_embeddings))
    os.rename(tmp_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The path to the rslearn dataset",
        required=True,
    )
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
        "--out_path",
        type=str,
        help="Path to the output H5 file",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Either center or pool, default center",
        default="center",
    )
    parser.add_argument(
        "--label_position",
        type=int,
        nargs=2,
        help="(x, y) pixel position where the label point should land in the crop. "
        "Requires --input_size. Default: centered.",
        default=None,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="OlmoEarth model ID",
        default="OlmoEarth-v1-Base",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Checkpoint directory to load (overrides --model_id when set)",
        default=None,
    )
    args = parser.parse_args()

    if args.label_position is not None and args.input_size is None:
        parser.error("--label_position requires --input_size")

    label_position: tuple[int, int] | None = (
        (args.label_position[0], args.label_position[1])
        if args.label_position
        else None
    )

    print("Initializing dataset")
    dataset = Dataset(UPath(args.ds_path))
    model_dataset = initialize_dataset_for_olmoearth(
        dataset,
        input_size=args.input_size,
        label_position=label_position,
        workers=args.workers,
    )

    print("Initializing OlmoEarth model")
    device = torch.device("cuda")
    if args.checkpoint_dir is not None:
        model = OlmoEarth(
            checkpoint_path=args.checkpoint_dir,
            patch_size=args.patch_size,
        )
    else:
        model = OlmoEarth(
            model_id=ModelID(args.model_id),
            patch_size=args.patch_size,
        )
    model.to(device)
    model.eval()

    get_embeddings(
        model=model,
        model_dataset=model_dataset,
        batch_size=args.batch_size,
        mode=args.mode,
        out_path=args.out_path,
        device=device,
        input_size=args.input_size,
        label_position=label_position,
        workers=args.workers,
    )
