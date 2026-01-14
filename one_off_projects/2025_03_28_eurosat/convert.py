"""Convert EuroSat to rslearn format."""

import argparse
import hashlib
import multiprocessing
import os
from datetime import datetime, timezone

import rasterio
import tqdm
from upath import UPath
from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.raster_format import get_raster_projection_and_bounds, GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from rslearn.utils.mp import star_imap_unordered

GROUP = "default"
START_TIME = datetime(2018, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2019, 1, 1, tzinfo=timezone.utc)
RASTER_LAYER = "sentinel2"
LABEL_LAYER = "label"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


def handle_example(ds_path: UPath, tif_fname: str, category: str) -> None:
    """Convert one EuroSat example.

    Args:
        ds_path: the rslearn dataset path to write to.
        tif_fname: the EuroSat GeoTIFF to read from.
        category: the category of the image.
    """
    # Get the projection and bounds from the GeoTIFF, which we will reuse for the
    # window.
    with rasterio.open(tif_fname) as raster:
        projection, bounds = get_raster_projection_and_bounds(raster)
        array = raster.read()

    # Create the window.
    # We include the category in the tags in case the user wants to do something with
    # that (e.g. train on a subset of categories).
    # We also compute a train/val split based on a hash of the window name.
    window_name = os.path.basename(tif_fname).split(".")[0]
    is_val = hashlib.sha256(window_name.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"
    window = Window(
        path=Window.get_window_root(ds_path, GROUP, window_name),
        group=GROUP,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=(START_TIME, END_TIME),
        options={
            "category": category,
            "split": split,
        }
    )
    window.save()

    # Add the image.
    raster_dir = window.get_raster_dir(RASTER_LAYER, BANDS)
    GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, array)
    window.mark_layer_completed(RASTER_LAYER)

    # Add the label.
    feature = Feature(window.get_geometry(), {
        "category": category,
    })
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def convert_eurosat(eurosat_dir: str, ds_path: UPath, workers: int = 64) -> None:
    """Convert EuroSat dataset downloaded from Zenodo to rslearn format.

    The rslearn dataset should be initialized as a directory containing the config.json
    here.

    Args:
        eurosat_dir: directory where EuroSat has been downloaded and extracted to. It
            can be downloaded from Zenodo at https://zenodo.org/records/7711810
        ds_path: the rslearn dataset to write the EuroSat examples to.
        workers: how many worker processes to use.
    """
    # Get list of args to run handle_example with.
    jobs = []
    for category in os.listdir(eurosat_dir):
        category_dir = os.path.join(eurosat_dir, category)
        for fname in os.listdir(category_dir):
            jobs.append(dict(
                ds_path=ds_path,
                tif_fname=os.path.join(category_dir, fname),
                category=category,
            ))

    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, handle_example, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Convert EuroSat dataset to rslearn format",
    )
    parser.add_argument(
        "--eurosat_dir",
        type=str,
        help="Local directory containing extracted EuroSat dataset",
        required=True,
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="rslearn dataset path to write to (copy config.json to it first)",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="How many worker processes to use",
        default=64,
    )
    args = parser.parse_args()

    convert_eurosat(
        args.eurosat_dir,
        UPath(args.ds_path),
        args.workers,
    )
