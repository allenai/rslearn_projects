"""Find locations and years with the most change from OlmoEarth embeddings.

For each window, splits each year's 6-image stack into two halves (first 3
months, second 3 months), embeds each half independently, then for every
candidate year Y in 2017-2025 computes change between consecutive years as:
    min(1 - cos_sim_half1(Y-1, Y), 1 - cos_sim_half2(Y-1, Y))
per spatial patch.  Change is attributed to Y.  Picks the year whose maximum
patch score is highest, writes the change mask GeoTIFF, and produces a JSON
summary.
"""

import argparse
import json
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
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from ..train import ChangeFinderNormalize, ChangeFinderTask

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
START_YEAR = 2016
# Candidate years: 2017-2025 (index 1-9); change is attributed to Y based
# on comparing Y-1 vs Y.
CANDIDATE_YEARS = list(range(2017, 2026))
# Number of timesteps per half-year (6-image stack split into two halves).
HALF_T = 3


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
                    f"[from_embeddings] skipping window "
                    f"{window.group}/{window.name} due to load error: {e}"
                )
                continue
            yield example


def _embed_half_years(
    model: OlmoEarth,
    input_dicts: list[dict],
    metadatas: list,
    year_key: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed both halves (first and second 3 months) of a year for a batch.

    Moves each sample's raster to the GPU once, then splits it into the first
    and second halves which are batched together into a single model forward
    pass.  Samples whose year has fewer than 2*HALF_T timesteps are excluded
    for both halves and filled with zeros.

    Returns:
        fh_feat: (B, C, H', W') tensor on device for the first half.
        sh_feat: (B, C, H', W') tensor on device for the second half.
        valid: (B,) bool tensor on device (True iff both halves were run).
    """
    batch_size = len(input_dicts)
    inputs: list[dict] = []
    meta_list: list = []
    valid_indices: list[int] = []

    for i, inp in enumerate(input_dicts):
        raster: RasterImage = inp[year_key]
        t = raster.image.shape[1]
        if t < 2 * HALF_T:
            continue
        image_gpu = raster.image.to(device)
        ts = raster.timestamps
        first_half = RasterImage(
            image=image_gpu[:, :HALF_T, :, :],
            timestamps=ts[:HALF_T] if ts is not None else None,
        )
        second_half = RasterImage(
            image=image_gpu[:, HALF_T : 2 * HALF_T, :, :],
            timestamps=ts[HALF_T : 2 * HALF_T] if ts is not None else None,
        )
        inputs.append({"sentinel2_l2a": first_half})
        meta_list.append(metadatas[i])
        inputs.append({"sentinel2_l2a": second_half})
        meta_list.append(metadatas[i])
        valid_indices.append(i)

    valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for idx in valid_indices:
        valid[idx] = True

    if not inputs:
        # All samples lack enough timesteps; infer output shape from input.
        raster = input_dicts[0][year_key]
        h_out = raster.image.shape[2] // model.patch_size
        w_out = raster.image.shape[3] // model.patch_size
        c_out = model.embedding_size or 768
        empty = torch.zeros(
            (batch_size, c_out, h_out, w_out), dtype=torch.float32, device=device
        )
        return empty, empty.clone(), valid

    context = ModelContext(inputs=inputs, metadatas=meta_list)
    feature_maps = model(context)
    feat = feature_maps.feature_maps[0]  # (2*len(valid_indices), C, H', W')

    fh_out = torch.zeros(
        (batch_size, feat.shape[1], feat.shape[2], feat.shape[3]),
        dtype=torch.float32,
        device=device,
    )
    sh_out = torch.zeros_like(fh_out)
    for j, idx in enumerate(valid_indices):
        fh_out[idx] = feat[2 * j]
        sh_out[idx] = feat[2 * j + 1]
    return fh_out, sh_out, valid


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-patch cosine distance: 1 - cos_sim.

    Args:
        a: (C, H', W') tensor on device.
        b: the other tensor.

    Returns:
        (H', W') cosine distance map on device.
    """
    dot = (a * b).sum(dim=0)
    norm_a = a.norm(dim=0).clamp(min=1e-12)
    norm_b = b.norm(dim=0).clamp(min=1e-12)
    cos_sim = dot / (norm_a * norm_b)
    return 1.0 - cos_sim


def _compute_change_scores(
    first_half_feats: torch.Tensor,
    second_half_feats: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[np.ndarray | None, int | None]:
    """Compute per-patch change scores for each candidate year.

    For each year Y in 2017-2025, change at each spatial patch is:
        min(cosine_dist_half1(Y-1, Y), cosine_dist_half2(Y-1, Y))
    comparing consecutive years.  The change is attributed to the second
    year Y (since data typically starts mid-year).

    Years where Y-1 or Y lacks valid embeddings are skipped.
    Picks the year whose maximum spatial score is highest.

    Args:
        first_half_feats: (NUM_YEARS, C, H', W') tensors on device.
        second_half_feats: (NUM_YEARS, C, H', W') tensors on device.
        valid: (NUM_YEARS,) bool tensor on device.

    Returns:
        best_scores: (H', W') float32 numpy change score map for the best year,
            or None if no valid candidate year exists.
        best_year: the actual year (e.g. 2020) with the highest max score,
            or None if no valid candidate year exists.
    """
    best_overall_max = float("-inf")
    best_year = None
    best_scores_map = None

    for year in CANDIDATE_YEARS:
        yi = year - START_YEAR  # index into the arrays

        if not (valid[yi - 1] and valid[yi]):
            continue

        d1 = _cosine_distance(first_half_feats[yi - 1], first_half_feats[yi])
        d2 = _cosine_distance(second_half_feats[yi - 1], second_half_feats[yi])
        change = torch.minimum(d1, d2)

        year_max = float(change.max().item())
        if year_max > best_overall_max:
            best_overall_max = year_max
            best_year = year
            best_scores_map = change

    if best_scores_map is not None:
        best_scores_map = best_scores_map.cpu().numpy()
    return best_scores_map, best_year


def find_change(
    ds_path: str,
    out_filename: str = "change_scores.tif",
    summary_filename: str = "change_summary.json",
    model_id: str = "OlmoEarth-v1-Base",
    patch_size: int = 8,
    batch_size: int = 8,
    device: str = "cuda",
    workers: int = 32,
) -> None:
    """Find locations and years with the most change.

    Args:
        ds_path: path to the rslearn dataset with sentinel2_y0..y9 layers.
        out_filename: GeoTIFF filename for per-patch change scores (per window).
        summary_filename: JSON filename written inside each window directory.
        model_id: OlmoEarth model identifier.
        patch_size: patch size for OlmoEarth.
        batch_size: number of windows per batch.
        device: torch device string.
        workers: number of DataLoader workers.
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

    raster_format = GeotiffRasterFormat()

    with torch.no_grad():
        for input_dicts, _target_dicts, metadatas in tqdm.tqdm(
            data_loader, desc="Finding change"
        ):
            cur_batch_size = len(input_dicts)

            # Embed each year's first and second halves together in one forward.
            # feats: list of (B, C, H', W'), valid: list of (B,) bool
            first_half_feats: list[torch.Tensor] = []
            second_half_feats: list[torch.Tensor] = []
            year_valid: list[torch.Tensor] = []
            for year_idx in range(NUM_YEARS):
                key = f"sentinel2_y{year_idx}"
                fh_feat, sh_feat, v = _embed_half_years(
                    model, input_dicts, metadatas, key, device
                )
                first_half_feats.append(fh_feat)
                second_half_feats.append(sh_feat)
                year_valid.append(v)

            for b in range(cur_batch_size):
                metadata = metadatas[b]
                window_root = dataset.storage.get_window_root(
                    metadata.window_group, metadata.window_name
                )

                fh = torch.stack(
                    [first_half_feats[yi][b] for yi in range(NUM_YEARS)]
                )  # (NUM_YEARS, C, H', W')
                sh = torch.stack(
                    [second_half_feats[yi][b] for yi in range(NUM_YEARS)]
                )  # (NUM_YEARS, C, H', W')
                v = torch.stack([year_valid[yi][b] for yi in range(NUM_YEARS)])

                scores, best_year = _compute_change_scores(fh, sh, v)
                if scores is None:
                    continue
                max_score = float(scores.max())

                # Write GeoTIFF with change scores for the best year.
                score_uint8 = (np.clip(scores, 0.0, 1.0) * 255).astype(np.uint8)
                feat_projection = Projection(
                    crs=metadata.projection.crs,
                    x_resolution=metadata.projection.x_resolution * patch_size,
                    y_resolution=metadata.projection.y_resolution * patch_size,
                )
                feat_bounds = (
                    metadata.window_bounds[0] // patch_size,
                    metadata.window_bounds[1] // patch_size,
                    metadata.window_bounds[0] // patch_size + scores.shape[1],
                    metadata.window_bounds[1] // patch_size + scores.shape[0],
                )
                raster = RasterArray(chw_array=score_uint8[np.newaxis, :, :])
                raster_format.encode_raster(
                    path=window_root,
                    projection=feat_projection,
                    bounds=feat_bounds,
                    raster=raster,
                    fname=out_filename,
                )

                window_summary = {
                    "window_name": metadata.window_name,
                    "window_group": metadata.window_group,
                    "max_year": best_year,
                    "max_score": max_score,
                }
                with open_atomic(window_root / summary_filename, "w") as f:
                    json.dump(window_summary, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find locations and years with the most change"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--out_filename",
        default="change_scores.tif",
        help="GeoTIFF filename per window",
    )
    parser.add_argument(
        "--summary_filename",
        default="change_summary.json",
        help="JSON filename written per window",
    )
    parser.add_argument(
        "--model_id", default="OlmoEarth-v1-Base", help="OlmoEarth model ID"
    )
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    find_change(
        ds_path=args.ds_path,
        out_filename=args.out_filename,
        summary_filename=args.summary_filename,
        model_id=args.model_id,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
