"""Detect land cover change using a segmentation model on the ten-year dataset.

For each window, runs a worldcover-style segmentation model on each year's imagery
independently, then identifies pixels where early years (default 0,1,2) consistently
predict one land cover class and late years (default 7,8,9) consistently predict a
different class (both above a configurable confidence threshold). Writes a multi-band
GeoTIFF per window: B0 = binary change flag, B1 = early dominant class, B2 = late
dominant class, then 13 probability bands (uint8, 0-255) for each early year followed
by 13 probability bands for each late year.
"""

import argparse

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


def _compute_land_cover_change_mask(
    probs: np.ndarray,
    early_years: list[int],
    late_years: list[int],
    threshold: float,
) -> np.ndarray:
    """Compute change mask with per-year probabilities from class predictions.

    Args:
        probs: (N, num_classes, H, W) softmax probabilities for the selected years.
        early_years: indices into probs for the early period.
        late_years: indices into probs for the late period.
        threshold: minimum confidence required in each year.

    Returns:
        (C, H, W) uint8 array with bands:
            B0: binary change flag (0/1)
            B1: early dominant class (argmax of mean early probs)
            B2: late dominant class (argmax of mean late probs)
            B3..: 13 probability bands per early year, then per late year (uint8, 0-255)
    """
    num_classes = probs.shape[1]
    h, w = probs.shape[2], probs.shape[3]

    early_probs = np.stack([probs[y] for y in early_years])  # (E, C, H, W)
    late_probs = np.stack([probs[y] for y in late_years])  # (L, C, H, W)

    early_mean = early_probs.mean(axis=0)  # (C, H, W)
    late_mean = late_probs.mean(axis=0)

    early_class = early_mean.argmax(axis=0)  # (H, W)
    late_class = late_mean.argmax(axis=0)

    early_argmax = np.stack([probs[y].argmax(axis=0) for y in early_years])
    early_maxprob = np.stack([probs[y].max(axis=0) for y in early_years])
    early_consistent = np.ones((h, w), dtype=bool)
    for i in range(len(early_years)):
        early_consistent &= early_argmax[i] == early_argmax[0]
        early_consistent &= early_maxprob[i] >= threshold

    late_argmax = np.stack([probs[y].argmax(axis=0) for y in late_years])
    late_maxprob = np.stack([probs[y].max(axis=0) for y in late_years])
    late_consistent = np.ones((h, w), dtype=bool)
    for i in range(len(late_years)):
        late_consistent &= late_argmax[i] == late_argmax[0]
        late_consistent &= late_maxprob[i] >= threshold

    changed = early_consistent & late_consistent & (early_class != late_class)

    num_prob_bands = (len(early_years) + len(late_years)) * num_classes
    out = np.zeros((3 + num_prob_bands, h, w), dtype=np.uint8)
    out[0] = changed.astype(np.uint8)
    out[1] = early_class.astype(np.uint8)
    out[2] = late_class.astype(np.uint8)

    band = 3
    for i in range(len(early_years)):
        out[band : band + num_classes] = np.clip(early_probs[i] * 255, 0, 255).astype(
            np.uint8
        )
        band += num_classes
    for i in range(len(late_years)):
        out[band : band + num_classes] = np.clip(late_probs[i] * 255, 0, 255).astype(
            np.uint8
        )
        band += num_classes

    return out


def compute_land_cover_change(
    ds_path: str,
    checkpoint_path: str,
    out_filename: str = "land_cover_change.tif",
    threshold: float = 0.75,
    early_years: list[int] | None = None,
    late_years: list[int] | None = None,
    batch_size: int = 4,
    device: str = "cuda",
    workers: int = 32,
) -> None:
    """Run land cover change detection on the ten-year dataset.

    Args:
        ds_path: path to the rslearn dataset with sentinel2_y0..y9 layers.
        checkpoint_path: path to the worldcover segmentation model checkpoint.
        out_filename: GeoTIFF filename to write inside each window directory.
        threshold: minimum class probability required in each year.
        early_years: year indices for the early period (default [0, 1, 2]).
        late_years: year indices for the late period (default [7, 8, 9]).
        batch_size: number of windows per batch.
        device: torch device string.
        workers: number of dataloader workers.
    """
    if early_years is None:
        early_years = [0, 1, 2]
    if late_years is None:
        late_years = [7, 8, 9]

    all_years = sorted(set(early_years) | set(late_years))

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

    print(f"Loading model from {checkpoint_path}")
    model = SingleTaskModel(
        encoder=[OlmoEarth(model_id=ModelID.OLMOEARTH_V1_BASE, patch_size=4)],
        decoder=[
            UNetDecoder(
                in_channels=[(4, 768)],
                out_channels=13,
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
            data_loader, desc="Computing land cover change"
        ):
            cur_batch_size = len(input_dicts)

            year_probs: dict[int, np.ndarray] = {}
            for year_idx in all_years:
                key = f"sentinel2_y{year_idx}"
                probs = _predict_year(model, input_dicts, metadatas, key, device)
                year_probs[year_idx] = probs

            for b in range(cur_batch_size):
                per_year = np.stack(
                    [year_probs[yi][b] for yi in range(NUM_YEARS) if yi in year_probs]
                )
                idx_map = {yi: idx for idx, yi in enumerate(sorted(year_probs.keys()))}
                mapped_early = [idx_map[y] for y in early_years]
                mapped_late = [idx_map[y] for y in late_years]

                mask = _compute_land_cover_change_mask(
                    per_year, mapped_early, mapped_late, threshold
                )

                metadata = metadatas[b]
                window_root = dataset.storage.get_window_root(
                    metadata.window_group, metadata.window_name
                )
                raster = RasterArray(chw_array=mask)
                raster_format.encode_raster(
                    path=window_root,
                    projection=metadata.projection,
                    bounds=metadata.window_bounds,
                    raster=raster,
                    fname=out_filename,
                )


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect land cover change on ten-year dataset"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to worldcover model checkpoint",
    )
    parser.add_argument(
        "--out_filename",
        default="land_cover_change.tif",
        help="GeoTIFF filename per window",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold per year",
    )
    parser.add_argument(
        "--early_years",
        type=_parse_int_list,
        default="0,1,2",
        help="Comma-separated early year indices",
    )
    parser.add_argument(
        "--late_years",
        type=_parse_int_list,
        default="7,8,9",
        help="Comma-separated late year indices",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    compute_land_cover_change(
        ds_path=args.ds_path,
        checkpoint_path=args.checkpoint_path,
        out_filename=args.out_filename,
        threshold=args.threshold,
        early_years=args.early_years,
        late_years=args.late_years,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
