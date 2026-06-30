"""Create rslearn windows for the HLS Burn Scars dataset.

Reads 512x512 (30m) burn scar masks from the HLS dataset, upsamples to
1536x1536 (10m) via nearest neighbor, splits into a 6x6 grid of 256x256
tiles, and writes each tile as an rslearn window with a label_raster layer.

Windows are discarded if the mask is entirely -1 (all nodata).
Time range is set to a 72-hour window around the acquisition date from
the filename.
"""

import multiprocessing as mp
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

SRC = Path("/weka/dfive-default/rslearn-eai/artifacts/hls_burn_scars")
DST = Path("/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/hls_burn_scars")

TILE_SIZE = 256
GRID_SIZE = 6  # 1536 / 256
NUM_PROC = 64

GEOTIFF_FORMAT = GeotiffRasterFormat()

FILENAME_RE = re.compile(
    r"subsetted_512x512_HLS\.S30\.([A-Z0-9]+)\.(\d{4})(\d{3})\.v[\d.]+\.mask\.tif"
)


def get_mask_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.mask.tif"))


def process_mask(args: tuple[Path, str, str]) -> None:
    mask_path, split, ds_path_str = args
    ds_path = UPath(ds_path_str)
    dataset = Dataset(ds_path)

    m = FILENAME_RE.match(mask_path.name)
    if m is None:
        return

    tile_id = m.group(1)
    year = int(m.group(2))
    doy = int(m.group(3))

    acq_date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    time_start = acq_date - timedelta(days=1)
    time_end = acq_date + timedelta(days=1)

    with rasterio.open(mask_path) as src:
        mask_30m = src.read(1)  # (512, 512), int16
        crs = src.crs
        transform = src.transform

    # Upsample mask from 30m (512x512) to 10m (1536x1536) via nearest neighbor
    mask_10m = np.repeat(np.repeat(mask_30m, 3, axis=0), 3, axis=1)

    # Compute top-left corner at 10m
    # Original transform is at 30m; at 10m the pixel size is 10
    origin_x = transform.c  # top-left x
    origin_y = transform.f  # top-left y
    res_x = 10
    res_y = -10

    projection = Projection(crs, res_x, res_y)

    # Convert geographic origin to pixel coordinates in the projection
    # pixel_col = origin_x / res_x, pixel_row = origin_y / res_y
    base_col = int(round(origin_x / res_x))
    base_row = int(round(origin_y / res_y))

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            tile = mask_10m[
                row * TILE_SIZE : (row + 1) * TILE_SIZE,
                col * TILE_SIZE : (col + 1) * TILE_SIZE,
            ]

            # Discard all-nodata tiles
            if (tile == -1).all():
                continue

            window_name = f"HLS_S30_{tile_id}_{year}{doy:03d}_r{row}_c{col}"

            bounds = (
                base_col + col * TILE_SIZE,
                base_row + row * TILE_SIZE,
                base_col + (col + 1) * TILE_SIZE,
                base_row + (row + 1) * TILE_SIZE,
            )

            window = Window(
                storage=dataset.storage,
                group=split,
                name=window_name,
                projection=projection,
                bounds=bounds,
                time_range=(time_start, time_end),
                options={"split": split},
            )
            window.save()

            # Write label_raster
            raster_arr = tile.astype(np.int16)[np.newaxis, :, :]  # (1, 256, 256)
            raster = RasterArray(chw_array=raster_arr)
            layer_dir = window.get_layer_dir("label_raster")
            band_dir = layer_dir / "label"
            band_dir.mkdir(parents=True, exist_ok=True)
            GEOTIFF_FORMAT.encode_raster(band_dir, projection, bounds, raster)
            window.mark_layer_completed("label_raster")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    import shutil
    config_src = Path(__file__).parent / "hls_burn_scars_config.json"
    shutil.copyfile(config_src, DST / "config.json")

    jobs: list[tuple[Path, str, str]] = []

    for split_name, split_group in [("training", "train"), ("validation", "val")]:
        split_dir = SRC / split_name
        if not split_dir.exists():
            print(f"Skipping {split_dir}, does not exist")
            continue
        mask_files = get_mask_files(split_dir)
        print(f"{split_name}: {len(mask_files)} mask files")
        for mask_path in mask_files:
            jobs.append((mask_path, split_group, str(DST)))

    print(f"Total jobs: {len(jobs)}")

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(process_mask, jobs), total=len(jobs)
        ):
            pass


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
