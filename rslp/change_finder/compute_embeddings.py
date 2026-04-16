"""Compute per-year OlmoEarth embeddings for change finder windows.

For each window, runs OlmoEarth on each year's 6-image stack independently,
spatially mean-pools the feature map, and saves a (NUM_YEARS, embed_dim) array
as an H5 file inside the window directory.
"""

import argparse
import io

import h5py
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
NUM_YEARS = 10


def _embed_year(
    model: OlmoEarth,
    input_dicts: list[dict],
    metadatas: list,
    year_key: str,
    device: str,
) -> np.ndarray:
    """Run OlmoEarth on a batch of single-year RasterImages and return spatial features.

    Returns:
        (B, C, H', W') numpy array of unpooled patch embeddings.
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


def _compute_change_mask(spatial: np.ndarray) -> np.ndarray:
    """Compute a per-patch change heatmap from spatial embeddings.

    Args:
        spatial: (NUM_YEARS, C, H', W') feature array for one window.

    Returns:
        (H', W') uint8 grayscale image where brighter = more change.
    """
    norms = np.linalg.norm(spatial, axis=1, keepdims=True).clip(min=1e-12)
    normed = spatial / norms
    ref_first = normed[0]  # (C, H', W')
    ref_last = normed[-1]

    scores = np.empty((NUM_YEARS, spatial.shape[2], spatial.shape[3]), dtype=np.float32)
    for i in range(NUM_YEARS):
        cos_first = (normed[i] * ref_first).sum(axis=0)
        cos_last = (normed[i] * ref_last).sum(axis=0)
        d_first = 1.0 - cos_first
        d_last = 1.0 - cos_last
        denom = d_first + d_last
        scores[i] = np.where(denom < 1e-12, 0.5, d_first / denom)

    overall = np.minimum.reduce([scores[7], scores[8], scores[9]]) - np.maximum.reduce(
        [scores[2], scores[3], scores[4]]
    )
    overall = np.clip(overall, 0.0, 1.0)
    return (overall * 255).astype(np.uint8)


def compute_embeddings(
    ds_path: str,
    out_filename: str = "embeddings.h5",
    mask_filename: str | None = None,
    model_id: str = "OlmoEarth-v1-Base",
    patch_size: int = 8,
    batch_size: int = 8,
    device: str = "cuda",
    workers: int = 32,
) -> None:
    """Compute and save per-year OlmoEarth embeddings for each window.

    Args:
        ds_path: path to the rslearn dataset with sentinel2_y0..y9 layers.
        out_filename: H5 filename to write inside each window directory.
        mask_filename: when set, also save a per-patch change heatmap GeoTIFF.
        model_id: OlmoEarth model identifier.
        patch_size: patch size for OlmoEarth.
        batch_size: number of windows to load per batch.
        device: torch device string.
        workers: number of workers for dataset initialization.
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

    data_loader = torch.utils.data.DataLoader(
        dataset=model_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    print(f"Initializing OlmoEarth ({model_id}, patch_size={patch_size})")
    model = OlmoEarth(model_id=ModelID(model_id), patch_size=patch_size)
    model.to(device)
    model.eval()

    raster_format = GeotiffRasterFormat()

    with torch.no_grad():
        for input_dicts, _target_dicts, metadatas in tqdm.tqdm(
            data_loader, desc="Computing embeddings"
        ):
            cur_batch_size = len(input_dicts)

            # (NUM_YEARS, batch, C, H', W') spatial features per year.
            year_feats: list[np.ndarray] = []
            for year_idx in range(NUM_YEARS):
                key = f"sentinel2_y{year_idx}"
                feats = _embed_year(model, input_dicts, metadatas, key, device)
                year_feats.append(feats)

            for b in range(cur_batch_size):
                # Pool spatially for the H5 embeddings.
                pooled = np.stack(
                    [year_feats[yi][b].mean(axis=(1, 2)) for yi in range(NUM_YEARS)]
                )  # (NUM_YEARS, C)

                metadata = metadatas[b]
                window_root = dataset.storage.get_window_root(
                    metadata.window_group, metadata.window_name
                )
                out_path = window_root / out_filename
                buf = io.BytesIO()
                with h5py.File(buf, "w") as f:
                    f.create_dataset("embeddings", data=pooled)
                with out_path.open("wb") as fp:
                    fp.write(buf.getvalue())

                if mask_filename:
                    spatial = np.stack(
                        [year_feats[yi][b] for yi in range(NUM_YEARS)]
                    )  # (NUM_YEARS, C, H', W')
                    mask = _compute_change_mask(spatial)  # (H', W')

                    # Write as single-band GeoTIFF with adjusted resolution.
                    feat_projection = Projection(
                        crs=metadata.projection.crs,
                        x_resolution=metadata.projection.x_resolution * patch_size,
                        y_resolution=metadata.projection.y_resolution * patch_size,
                    )
                    feat_bounds = (
                        metadata.window_bounds[0] // patch_size,
                        metadata.window_bounds[1] // patch_size,
                        metadata.window_bounds[0] // patch_size + mask.shape[1],
                        metadata.window_bounds[1] // patch_size + mask.shape[0],
                    )
                    raster = RasterArray(
                        chw_array=mask[np.newaxis, :, :],
                    )
                    raster_format.encode_raster(
                        path=window_root,
                        projection=feat_projection,
                        bounds=feat_bounds,
                        raster=raster,
                        fname=mask_filename,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute OlmoEarth embeddings for change finder"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--out_filename", default="embeddings.h5", help="H5 filename per window"
    )
    parser.add_argument(
        "--mask_filename",
        default=None,
        help="GeoTIFF filename for per-patch change mask",
    )
    parser.add_argument(
        "--model_id", default="OlmoEarth-v1-Base", help="OlmoEarth model ID"
    )
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    compute_embeddings(
        ds_path=args.ds_path,
        out_filename=args.out_filename,
        mask_filename=args.mask_filename,
        model_id=args.model_id,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
