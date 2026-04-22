"""Compute per-year land cover probabilities on the ten-year dataset.

For each window, runs a worldcover-style segmentation model on each year's imagery
independently and stacks the softmax probabilities for all ``NUM_YEARS`` years into
a single multi-band GeoTIFF.

Output GeoTIFF layout (uint8, 0-255, year-major) for each window:
    band (y * NUM_CLASSES + c) = probability of class ``c`` in year ``y``.

Downstream scripts consume these per-year probabilities to compute change masks,
pick pivot years, etc.
"""

import argparse
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import tqdm
from rslearn.dataset import Dataset
from rslearn.models.olmoearth_pretrain.model import ModelID, OlmoEarth
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.unet import UNetDecoder
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from .train import ChangeFinderNormalize, ChangeFinderTask

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
NUM_YEARS = 10
NUM_CLASSES = 13


class _ShuffledSkipExistingDataset(torch.utils.data.IterableDataset):
    """Iterate windows in a random order, skipping ones with existing output.

    The shuffle order is fixed at construction so every DataLoader worker on
    one machine sees the same permutation (each worker just takes a different
    slice of it).  Across machines the orders diverge because the seed is
    drawn from the system RNG.

    The output-file check runs just-in-time before yielding, so when machine A
    finishes a window after machine B has started but before B gets there,
    B will skip it rather than recomputing.
    """

    def __init__(
        self,
        base: ModelDataset,
        storage: Any,
        out_filename: str,
        shuffle_seed: int | None = None,
    ) -> None:
        self.base = base
        self.storage = storage
        self.out_filename = out_filename
        if shuffle_seed is None:
            shuffle_seed = random.Random().randint(0, 2**31 - 1)
        self.shuffle_seed = shuffle_seed

    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        windows = self.base.get_dataset_examples()
        indices = list(range(len(windows)))
        random.Random(self.shuffle_seed).shuffle(indices)
        if worker_info is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]
        for idx in indices:
            window = windows[idx]
            root = self.storage.get_window_root(window.group, window.name)
            if (root / self.out_filename).exists():
                continue
            yield self.base[idx]


def _predict_year(
    model: torch.nn.Module,
    input_dicts: list[dict],
    metadatas: list,
    year_key: str,
    device: str,
) -> np.ndarray:
    """Run the segmentation model on one year for a batch of windows.

    Returns:
        (B, num_classes, H, W) numpy array of softmax probabilities.
    """
    inputs = []
    for inp in input_dicts:
        raster: RasterImage = inp[year_key]
        gpu_raster = RasterImage(
            image=raster.image.to(device),
            timestamps=raster.timestamps,
        )
        inputs.append({"sentinel2_l2a": gpu_raster})
    context = ModelContext(inputs=inputs, metadatas=list(metadatas))
    output = model(context)
    return output.outputs.cpu().numpy()


def compute_land_cover_probs(
    ds_path: str,
    checkpoint_path: str,
    out_filename: str = "land_cover_probs.tif",
    batch_size: int = 4,
    device: str = "cuda",
    workers: int = 32,
) -> None:
    """Run per-year land cover prediction on the ten-year dataset.

    Args:
        ds_path: path to the rslearn dataset with sentinel2_y0..y9 layers.
        checkpoint_path: path to the worldcover segmentation model checkpoint.
        out_filename: GeoTIFF filename to write inside each window directory.
        batch_size: number of windows per batch.
        device: torch device string.
        workers: number of dataloader workers.
    """
    dataset = Dataset(UPath(ds_path))

    modality_names = [f"sentinel2_y{i}" for i in range(NUM_YEARS)]
    normalizer = ChangeFinderNormalize(
        modality_names=modality_names,
        band_names=OLMOEARTH_BAND_ORDER,
        skip_missing=True,
    )

    inputs_config = {}
    for i in range(NUM_YEARS):
        inputs_config[f"sentinel2_y{i}"] = DataInput(
            data_type="raster",
            layers=[f"sentinel2_y{i}"],
            bands=OLMOEARTH_BAND_ORDER,
            passthrough=True,
            load_all_item_groups=True,
            load_all_layers=True,
        )

    split_config = SplitConfig(transforms=[normalizer])
    task = ChangeFinderTask()

    model_dataset = ModelDataset(
        dataset=dataset,
        inputs=inputs_config,
        task=task,
        split_config=split_config,
        workers=workers,
    )

    # Wrap the dataset so each machine (a) iterates in its own random order
    # and (b) checks the output file right before yielding, skipping any
    # window another machine has finished in the meantime.
    iter_dataset = _ShuffledSkipExistingDataset(
        base=model_dataset,
        storage=dataset.storage,
        out_filename=out_filename,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=iter_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    print(f"Loading model from {checkpoint_path}")
    model = SingleTaskModel(
        encoder=[OlmoEarth(model_id=ModelID.OLMOEARTH_V1_BASE, patch_size=4)],
        decoder=[
            UNetDecoder(
                in_channels=[(4, 768)],
                out_channels=NUM_CLASSES,
                num_channels={2: 256, 1: 128},
            ),
            SegmentationHead(),
        ],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    raster_format = GeotiffRasterFormat()

    with torch.no_grad():
        for input_dicts, _target_dicts, metadatas in tqdm.tqdm(
            data_loader, desc="Computing land cover probabilities"
        ):
            cur_batch_size = len(input_dicts)

            year_probs: list[np.ndarray] = []
            for year_idx in range(NUM_YEARS):
                key = f"sentinel2_y{year_idx}"
                year_probs.append(
                    _predict_year(model, input_dicts, metadatas, key, device)
                )

            for b in range(cur_batch_size):
                per_year = [
                    np.clip(year_probs[yi][b] * 255, 0, 255).astype(np.uint8)
                    for yi in range(NUM_YEARS)
                ]
                out = np.concatenate(per_year, axis=0)

                metadata = metadatas[b]
                window_root = dataset.storage.get_window_root(
                    metadata.window_group, metadata.window_name
                )
                raster = RasterArray(chw_array=out)
                raster_format.encode_raster(
                    path=window_root,
                    projection=metadata.projection,
                    bounds=metadata.window_bounds,
                    raster=raster,
                    fname=out_filename,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-year land cover probabilities on ten-year dataset"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to worldcover model checkpoint",
    )
    parser.add_argument(
        "--out_filename",
        default="land_cover_probs.tif",
        help="GeoTIFF filename per window",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    compute_land_cover_probs(
        ds_path=args.ds_path,
        checkpoint_path=args.checkpoint_path,
        out_filename=args.out_filename,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
