"""Apply the per-pixel land-cover model to extract ten-year change samples."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import shapely.geometry
import torch
import tqdm
from rslearn.dataset import Dataset
from rslearn.models.olmoearth_pretrain.model import ModelID, OlmoEarth
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.unet import UNetDecoder
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import (
    ModelContext,
    RasterImage,
    SampleMetadata,
)
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from upath import UPath

from rslp.change_finder.train import ChangeFinderNormalize, ChangeFinderTask

from .train import PerPixelModelWrapper

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

BASE_YEAR = 2016
NUM_YEARS = 10
NUM_CLASSES = 13
NUM_CONTEXT_YEARS = 3

CLASS_NAMES = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]


class _ShuffledSkipExistingDataset(torch.utils.data.IterableDataset):
    """Iterate windows in random order, skipping windows with existing JSON."""

    def __init__(
        self,
        base: ModelDataset,
        output_dir: UPath,
        shuffle_seed: int | None = None,
    ) -> None:
        self.base = base
        self.output_dir = output_dir
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
            if _get_output_path(self.output_dir, window.group, window.name).exists():
                continue
            try:
                yield self.base[idx]
            except Exception as e:
                print(
                    "[per_pixel_land_cover] skipping window "
                    f"{window.group}/{window.name} due to load error: {e}"
                )


@dataclass
class _BestChange:
    score: float
    row: int
    col: int
    pivot_year_idx: int
    src_class_id: int
    dst_class_id: int


def _get_output_path(output_dir: UPath, group: str, name: str) -> UPath:
    """Get the per-window output JSON path."""
    return output_dir / group / f"{name}.json"


def _load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Instantiate the per-pixel model and load a Lightning checkpoint."""
    model = PerPixelModelWrapper(
        SingleTaskModel(
            encoder=[OlmoEarth(model_id=ModelID.OLMOEARTH_V1_BASE, patch_size=1)],
            decoder=[
                UNetDecoder(
                    in_channels=[(1, 768)],
                    out_channels=NUM_CLASSES,
                    kernel_size=1,
                ),
                SegmentationHead(),
            ],
        )
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _build_model_dataset(ds_path: str, workers: int) -> tuple[Dataset, ModelDataset]:
    """Create an rslearn ModelDataset for the ten-year Sentinel-2 inputs."""
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

    model_dataset = ModelDataset(
        dataset=dataset,
        inputs=inputs_config,
        task=ChangeFinderTask(),
        split_config=SplitConfig(transforms=[normalizer]),
        workers=workers,
    )
    return dataset, model_dataset


def _get_center_crop_bounds(
    height: int,
    width: int,
    center_crop_size: int,
) -> tuple[int, int, int, int]:
    """Get row/column bounds for a centered square crop."""
    if center_crop_size <= 0:
        raise ValueError("center_crop_size must be positive")
    crop_height = min(center_crop_size, height)
    crop_width = min(center_crop_size, width)
    row_start = (height - crop_height) // 2
    col_start = (width - crop_width) // 2
    return row_start, row_start + crop_height, col_start, col_start + crop_width


def _predict_year_crop(
    model: torch.nn.Module,
    input_dict: dict[str, Any],
    metadata: SampleMetadata,
    year_idx: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    device: str,
) -> torch.Tensor:
    """Predict class probabilities for one year and one spatial crop."""
    raster: RasterImage = input_dict[f"sentinel2_y{year_idx}"]
    crop = RasterImage(
        image=raster.image[..., row_start:row_end, col_start:col_end]
        .contiguous()
        .to(device),
        timestamps=raster.timestamps,
    )
    context = ModelContext(
        inputs=[{"sentinel2_l2a": crop}],
        metadatas=[metadata],
    )
    output = model(context)
    if not isinstance(output.outputs, torch.Tensor):
        raise ValueError("per-pixel land-cover model must return tensor outputs")
    return output.outputs[0]


def _score_crop(
    probs: torch.Tensor,
    row_start: int,
    col_start: int,
    src_threshold: float,
    dst_threshold: float,
) -> _BestChange | None:
    """Find the best class decline in a centered crop.

    A change is any class whose probability stays above src_threshold across all
    pre years (min over time) and drops below dst_threshold across all post years
    (max over time). Among such (class, pixel) candidates the one with the
    largest margin (pre_min - post_max) is returned.

    Args:
        probs: tensor shaped NUM_YEARS x NUM_CLASSES x H x W.
        row_start: row offset of the crop in the source window.
        col_start: column offset of the crop in the source window.
        src_threshold: minimum pre-period probability (min over time) of the
            declining class.
        dst_threshold: maximum post-period probability (max over time) of the
            declining class.
    """
    best: _BestChange | None = None
    best_score = float("-inf")
    valid_classes = probs[:, 1:, :, :]

    min_pivot = NUM_CONTEXT_YEARS
    max_pivot = NUM_YEARS - 1 - NUM_CONTEXT_YEARS
    for pivot_idx in range(min_pivot, max_pivot + 1):
        pre = valid_classes[pivot_idx - NUM_CONTEXT_YEARS : pivot_idx]
        post = valid_classes[pivot_idx + 1 : pivot_idx + 1 + NUM_CONTEXT_YEARS]
        pre_min = pre.min(dim=0).values
        post_max = post.max(dim=0).values
        mask = (pre_min > src_threshold) & (post_max < dst_threshold)
        scores = torch.where(
            mask, pre_min - post_max, torch.full_like(pre_min, float("-inf"))
        )
        crop_best_score = float(scores.max().item())
        if crop_best_score <= best_score:
            continue

        _, crop_height, crop_width = scores.shape
        flat_idx = int(scores.argmax().item())
        class_idx = flat_idx // (crop_height * crop_width)
        remainder = flat_idx % (crop_height * crop_width)
        local_row = remainder // crop_width
        local_col = remainder % crop_width
        row = row_start + local_row
        col = col_start + local_col

        post_avg = probs[
            pivot_idx + 1 : pivot_idx + 1 + NUM_CONTEXT_YEARS,
            1:,
            local_row,
            local_col,
        ].mean(dim=0)
        src_class_id = class_idx + 1
        dst_class_id = int(post_avg.argmax().item()) + 1

        best_score = crop_best_score
        best = _BestChange(
            score=crop_best_score,
            row=row,
            col=col,
            pivot_year_idx=pivot_idx,
            src_class_id=src_class_id,
            dst_class_id=dst_class_id,
        )

    return best


def _find_best_change(
    model: torch.nn.Module,
    input_dict: dict[str, Any],
    metadata: SampleMetadata,
    center_crop_size: int,
    device: str,
    src_threshold: float,
    dst_threshold: float,
) -> _BestChange | None:
    """Scan one window and return its best per-pixel land-cover decline."""
    first_raster: RasterImage = input_dict["sentinel2_y0"]
    height, width = first_raster.image.shape[-2:]
    row_start, row_end, col_start, col_end = _get_center_crop_bounds(
        height, width, center_crop_size
    )
    year_probs = [
        _predict_year_crop(
            model=model,
            input_dict=input_dict,
            metadata=metadata,
            year_idx=year_idx,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            device=device,
        )
        for year_idx in range(NUM_YEARS)
    ]
    probs = torch.stack(year_probs, dim=0)
    return _score_crop(probs, row_start, col_start, src_threshold, dst_threshold)


def _pixel_to_lonlat(
    metadata: SampleMetadata,
    row: int,
    col: int,
) -> tuple[float, float]:
    """Convert a window-local pixel center to lon/lat."""
    px_x = metadata.window_bounds[0] + col + 0.5
    px_y = metadata.window_bounds[1] + row + 0.5
    pt = shapely.geometry.Point(px_x, px_y)
    wgs84_pt = (
        STGeometry(metadata.projection, pt, time_range=None)
        .to_projection(WGS84_PROJECTION)
        .shp
    )
    return float(wgs84_pt.x), float(wgs84_pt.y)


def _format_time_range(
    time_range: tuple[datetime, datetime] | None,
) -> list[str]:
    """Format a V2 annotation time_range."""
    if time_range is None:
        start = datetime(BASE_YEAR, 1, 1, tzinfo=timezone.utc)
        end = datetime(BASE_YEAR + NUM_YEARS, 1, 1, tzinfo=timezone.utc)
        time_range = (start, end)
    return [time_range[0].isoformat(), time_range[1].isoformat()]


def _build_v2_entry(
    metadata: SampleMetadata, best: _BestChange
) -> list[dict[str, Any]]:
    """Build a one-entry V2-compatible annotation JSON list."""
    lon, lat = _pixel_to_lonlat(metadata, best.row, best.col)
    pivot_year = BASE_YEAR + best.pivot_year_idx
    return [
        {
            "projection": metadata.projection.serialize(),
            "bounds": list(metadata.window_bounds),
            "window_name": metadata.window_name,
            "group": metadata.window_group,
            "time_range": _format_time_range(metadata.time_range),
            "positive_points": [
                {
                    "lon": lon,
                    "lat": lat,
                    "pre_change": f"{pivot_year}-01-01",
                    "pre_category": CLASS_NAMES[best.src_class_id],
                    "post_category": CLASS_NAMES[best.dst_class_id],
                }
            ],
            "negative_points": [],
        }
    ]


def apply_per_pixel_land_cover(
    ds_path: str,
    checkpoint_path: str,
    output_dir: str,
    src_threshold: float = 0.75,
    dst_threshold: float = 0.25,
    batch_size: int = 1,
    device: str = "cuda",
    workers: int = 4,
    center_crop_size: int = 64,
    seed: int | None = None,
) -> None:
    """Apply per-pixel land-cover change scoring to a ten-year dataset."""
    output_root = UPath(output_dir)
    _dataset, model_dataset = _build_model_dataset(ds_path, workers)
    iter_dataset = _ShuffledSkipExistingDataset(
        base=model_dataset,
        output_dir=output_root,
        shuffle_seed=seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=iter_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    model = _load_model(checkpoint_path, device)

    written = 0
    skipped_no_change = 0
    with torch.no_grad():
        for input_dicts, _target_dicts, metadatas in tqdm.tqdm(
            data_loader, desc="Finding per-pixel land-cover changes"
        ):
            for input_dict, metadata in zip(input_dicts, metadatas, strict=True):
                out_path = _get_output_path(
                    output_root, metadata.window_group, metadata.window_name
                )
                if out_path.exists():
                    continue

                best = _find_best_change(
                    model=model,
                    input_dict=input_dict,
                    metadata=metadata,
                    center_crop_size=center_crop_size,
                    device=device,
                    src_threshold=src_threshold,
                    dst_threshold=dst_threshold,
                )
                if best is None:
                    skipped_no_change += 1
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open_atomic(out_path, "w") as f:
                    json.dump(_build_v2_entry(metadata, best), f, indent=2)
                written += 1

    print(
        f"Wrote {written} V2 annotation JSON files; "
        f"skipped {skipped_no_change} with no qualifying change"
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Find per-pixel land-cover changes and write V2 JSON entries"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to per-pixel WorldCover model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for per-window V2 JSON list files",
    )
    parser.add_argument("--src_threshold", type=float, default=0.75)
    parser.add_argument("--dst_threshold", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--center_crop_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    apply_per_pixel_land_cover(
        ds_path=args.ds_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        src_threshold=args.src_threshold,
        dst_threshold=args.dst_threshold,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        center_crop_size=args.center_crop_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
