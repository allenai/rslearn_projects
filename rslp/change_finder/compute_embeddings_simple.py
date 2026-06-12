"""Compute L2 distance between first-year and last-year OlmoEarth embeddings.

For each window, runs OlmoEarth on the first year (y0 / 2016) and last year
(y9 / 2025) independently, then computes the per-patch L2 distance between
the two spatial feature maps and saves a GeoTIFF normalized to 0-255.
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
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.utils.geometry import Projection
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

YEAR_KEYS = ["sentinel2_y0", "sentinel2_y9"]


class _ShuffledSkipExistingDataset(torch.utils.data.IterableDataset):
    """Iterate windows in random order, skipping ones with existing output."""

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
            try:
                example = self.base[idx]
            except Exception as e:
                print(
                    f"[compute_embeddings_simple] skipping window "
                    f"{window.group}/{window.name} due to load error: {e}"
                )
                continue
            yield example


def _embed_year(
    model: OlmoEarth,
    input_dicts: list[dict],
    metadatas: list,
    year_key: str,
    device: str,
) -> np.ndarray:
    """Run OlmoEarth on a batch of single-year RasterImages.

    Returns:
        (B, C, H', W') numpy array of spatial patch embeddings.
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
    feature_maps = model(context)
    feat = feature_maps.feature_maps[0]  # (B, C, H', W')
    return feat.cpu().numpy()


def compute_embeddings_simple(
    ds_path: str,
    out_filename: str = "l2_distance.tif",
    model_id: str = "OlmoEarth-v1-Base",
    patch_size: int = 8,
    batch_size: int = 8,
    device: str = "cuda",
    workers: int = 32,
) -> None:
    """Compute L2 distance between first and last year embeddings.

    Args:
        ds_path: path to the rslearn dataset with sentinel2_y0..y9 layers.
        out_filename: GeoTIFF filename to write inside each window directory.
        model_id: OlmoEarth model identifier.
        patch_size: patch size for OlmoEarth.
        batch_size: number of windows to load per batch.
        device: torch device string.
        workers: number of workers for dataset initialization.
    """
    dataset = Dataset(UPath(ds_path))

    normalizer = ChangeFinderNormalize(
        modality_names=YEAR_KEYS,
        band_names=OLMOEARTH_BAND_ORDER,
        skip_missing=True,
    )

    inputs_config = {}
    for key in YEAR_KEYS:
        inputs_config[key] = DataInput(
            data_type="raster",
            layers=[key],
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

    print(f"Initializing OlmoEarth ({model_id}, patch_size={patch_size})")
    model = OlmoEarth(model_id=ModelID(model_id), patch_size=patch_size)
    model.to(device)
    model.eval()

    # OlmoEarth BASE has 768-dim embeddings. The normalized input values are
    # in [-1, 1], so each embedding dimension is bounded roughly by [-1, 1].
    # Max L2 distance = sqrt(768 * (1 - (-1))^2) = sqrt(768 * 4) = sqrt(3072).
    # In practice values rarely hit this bound, but it gives a fixed scale.
    embed_dim = 768
    max_l2 = np.sqrt(embed_dim * 4.0)

    raster_format = GeotiffRasterFormat()

    with torch.no_grad():
        for input_dicts, _target_dicts, metadatas in tqdm.tqdm(
            data_loader, desc="Computing L2 distance"
        ):
            feats_first = _embed_year(
                model, input_dicts, metadatas, YEAR_KEYS[0], device
            )  # (B, C, H', W')
            feats_last = _embed_year(
                model, input_dicts, metadatas, YEAR_KEYS[1], device
            )  # (B, C, H', W')

            # Per-patch L2 distance: (B, H', W')
            l2_dist = np.sqrt(((feats_first - feats_last) ** 2).sum(axis=1))

            for b in range(len(input_dicts)):
                metadata = metadatas[b]
                window_root = dataset.storage.get_window_root(
                    metadata.window_group, metadata.window_name
                )

                dist_map = l2_dist[b]  # (H', W')
                normalized = np.clip(dist_map / max_l2, 0.0, 1.0)
                gray = (normalized * 255).astype(np.uint8)

                feat_projection = Projection(
                    crs=metadata.projection.crs,
                    x_resolution=metadata.projection.x_resolution * patch_size,
                    y_resolution=metadata.projection.y_resolution * patch_size,
                )
                feat_bounds = (
                    metadata.window_bounds[0] // patch_size,
                    metadata.window_bounds[1] // patch_size,
                    metadata.window_bounds[0] // patch_size + gray.shape[1],
                    metadata.window_bounds[1] // patch_size + gray.shape[0],
                )
                raster = RasterArray(chw_array=gray[np.newaxis, :, :])
                raster_format.encode_raster(
                    path=window_root,
                    projection=feat_projection,
                    bounds=feat_bounds,
                    raster=raster,
                    fname=out_filename,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute L2 distance between first/last year OlmoEarth embeddings"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--out_filename", default="l2_distance.tif", help="GeoTIFF filename per window"
    )
    parser.add_argument(
        "--model_id", default="OlmoEarth-v1-Base", help="OlmoEarth model ID"
    )
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    compute_embeddings_simple(
        ds_path=args.ds_path,
        out_filename=args.out_filename,
        model_id=args.model_id,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
