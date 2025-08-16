"""Create windows for crop type mapping."""

import argparse
import hashlib
import multiprocessing
import os
from datetime import datetime, timezone

import rasterio
import tqdm
from rslearn.dataset import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    get_raster_projection_and_bounds,
)
from upath import UPath

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"

# Two long-rain (LR) seasons in Kenya: March to August.
START_TIME_2021 = datetime(2021, 8, 1, tzinfo=timezone.utc)
END_TIME_2021 = datetime(2021, 8, 31, tzinfo=timezone.utc)

START_TIME_2023 = datetime(2023, 8, 1, tzinfo=timezone.utc)
END_TIME_2023 = datetime(2023, 8, 31, tzinfo=timezone.utc)


def create_window(geotiff_path: str, ds_path: UPath, group_name: str) -> None:
    """Create windows for crop type mapping.

    Args:
        geotiff_path: path to the GeoTIFF file
        ds_path: path to the dataset
        group_name: name of the group
    """
    with rasterio.open(geotiff_path) as src:
        projection, bounds = get_raster_projection_and_bounds(src)
        array = src.read(1)
        no_data_value = src.nodata
        if array is None:
            raise ValueError(f"GeoTIFF file {geotiff_path} is empty or invalid")
        if array.shape != (50, 50):
            raise ValueError(
                f"GeoTIFF file {geotiff_path} does not have the expected shape of 50x50"
            )

    window_name = os.path.basename(geotiff_path).split(".")[0]
    is_val = hashlib.sha256(window_name.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"

    # Example: JRC_2023_LR_15250_32000.tif
    window_year = window_name.split("_")[1]
    if window_year == "2021":
        start_time, end_time = START_TIME_2021, END_TIME_2021
    elif window_year == "2023":
        start_time, end_time = START_TIME_2023, END_TIME_2023
    else:
        raise ValueError(f"Unsupported year in window name: {window_year}")

    window = Window(
        path=Window.get_window_root(ds_path, group_name, window_name),
        group=group_name,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,
        },
    )
    window.save()

    array[array != no_data_value] += 1
    array[array == no_data_value] = 0

    raster_dir = window.get_raster_dir("label", ["class"])
    GeotiffRasterFormat().encode_raster(
        raster_dir, projection, bounds, array[None, :, :]
    )
    window.mark_layer_completed("label")


def create_windows_from_geotiffs(
    geotiff_dir: UPath,
    ds_path: UPath,
    group_name: str,
) -> None:
    """Create windows from GeoTIFF.

    Args:
        geotiff_dir: path to the GeoTIFF directory
        ds_path: path to the dataset
        group_name: name of the group
    """
    geotiff_paths = geotiff_dir.glob("*.tif")
    if not geotiff_paths:
        raise ValueError(f"No GeoTIFF files found in {geotiff_dir}")

    jobs = [
        dict(
            geotiff_path=geotiff_path,
            ds_path=ds_path,
            group_name=group_name,
        )
        for geotiff_path in geotiff_paths
    ]
    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, create_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(description="Create windows from Geotiff")
    parser.add_argument(
        "--geotiff_dir",
        type=str,
        required=True,
        help="Path to the GeoTIFF directory",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        required=False,
        help="Window group name",
        default="worldcover",
    )
    args = parser.parse_args()
    create_windows_from_geotiffs(
        UPath(args.geotiff_dir),
        UPath(args.ds_path),
        args.group_name,
    )
