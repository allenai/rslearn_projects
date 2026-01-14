"""Convert PASTIS-R to rslearn format."""

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from datetime import datetime, timezone

import rasterio
import numpy as np
import numpy.typing as npt
import tqdm
from upath import UPath
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection
from rasterio.crs import CRS
from rslearn.utils.feature import Feature
from rslearn.utils.raster_format import get_raster_projection_and_bounds, GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from rslearn.utils.mp import star_imap_unordered

START_TIME = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_TIME = datetime(2020, 1, 1, tzinfo=timezone.utc)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["vv", "vh"]


def compute_cloud_score(s2_image: npt.NDArray) -> int:
    """Compute cloud score for the given Sentinel-2 image (CHW)."""
    missing_pixels = np.count_nonzero(s2_image.max(axis=0) <= 0)
    cloudy_pixels = np.count_nonzero(s2_image[(0, 1, 2), :, :].min(axis=0) >= 2000)
    return missing_pixels + cloudy_pixels


def handle_example(src_path: Path, ds_path: UPath, group: str, example_id: str) -> None:
    """Convert one PASTIS-R example.

    Args:
        src_path: the path where the PASTIS-R dataset has been extracted.
        ds_path: the rslearn dataset path to write to.
        group: the fold that this example belongs to.
        example_id: the example ID to process, e.g. "20266".
    """
    # Load the images.
    s2 = np.load(src_path / "DATA_S2" / f"S2_{example_id}.npy")
    s1a = np.load(src_path / "DATA_S1A" / f"S1A_{example_id}.npy")
    s1d = np.load(src_path / "DATA_S1D" / f"S1D_{example_id}.npy")
    target = np.load(src_path / "ANNOTATIONS" / f"TARGET_{example_id}.npy")

    if s2.shape[0] < 12:
        print(f"skipping example {example_id} because there are not enough Sentinel-2 images")
        return
    if s1a.shape[0] < 12 or s1d.shape[0] < 12:
        print(f"skipping example {example_id} because there are not enough Sentinel-1 images")
        return

    # Collapse to 12 images each.
    # For Sentinel-2, we pick the image with the lowest cloud score which attempts to
    # count missing and cloudy pixels.
    s1a = s1a[::s1a.shape[0]//12][0:12]
    s1d = s1d[::s1d.shape[0]//12][0:12]

    s2_new = []
    group_size = s2.shape[0]//12
    for group_idx in range(12):
        cur_group = s2[group_idx*group_size:(group_idx+1)*group_size]
        cur_options = [image for image in cur_group]
        cur_options.sort(key=compute_cloud_score)
        s2_new.append(cur_options[0])
    s2 = np.stack(s2_new, axis=0)

    # Create the window.
    # We use a dummy projection and bounds here since it is difficult to get the actual
    # correct one.
    projection = Projection(CRS.from_epsg(3857), 10, -10)
    bounds = (0, 0, s2.shape[3], s2.shape[2])
    window = Window(
        path=Window.get_window_root(ds_path, group, example_id),
        group=group,
        name=example_id,
        projection=projection,
        bounds=bounds,
        time_range=(START_TIME, END_TIME),
    )
    window.save()

    # Add the images.
    for group_idx, image in enumerate(s2):
        raster_dir = window.get_raster_dir("sentinel2", S2_BANDS, group_idx)
        GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, image)
        window.mark_layer_completed("sentinel2", group_idx)

    for group_idx, image in enumerate(s1a):
        raster_dir = window.get_raster_dir("s1a", S1_BANDS, group_idx)
        GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, image.astype(np.float32))
        window.mark_layer_completed("s1a", group_idx)

    for group_idx, image in enumerate(s1d):
        raster_dir = window.get_raster_dir("s1d", S1_BANDS, group_idx)
        GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, image.astype(np.float32))
        window.mark_layer_completed("s1d", group_idx)

    # In the source data, background is 0 while void is 19.
    # But we want to train on background while masking out void.
    # So we need to set void to 0 (for compatibility with zero_is_invalid=True) and
    # increment the other classes.
    target = target[0]
    target = target + 1
    target[target == 20] = 0
    raster_dir = window.get_raster_dir("label", ["class"])
    GeotiffRasterFormat().encode_raster(raster_dir, projection, bounds, target[None, :, :])
    window.mark_layer_completed("label")


def convert_pastis(src_path: Path, ds_path: UPath, workers: int = 64) -> None:
    """Convert PASTIS-R dataset to rslearn format.

    The rslearn dataset should be initialized as a directory containing the config.json
    here.

    Args:
        src_path: directory where PASTIS-R has been downloaded and extracted to. It can
            be downloaded from https://zenodo.org/records/5735646.
        ds_path: the rslearn dataset to write the PASTIS examples to.
        workers: how many worker processes to use.
    """
    # Get list of args to run handle_example with.
    jobs = []
    with (src_path / "metadata.geojson").open() as f:
        fc = json.load(f)
        for feat in fc["features"]:
            fold = feat["properties"]["Fold"]
            example_id = str(feat["id"])
            jobs.append(dict(
                src_path=src_path,
                ds_path=ds_path,
                group=f"fold{fold}",
                example_id=example_id,
            ))

    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, handle_example, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Convert PASTIS dataset to rslearn format",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        help="Local directory containing extracted PASTIS-R dataset",
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

    convert_pastis(
        Path(args.src_path),
        UPath(args.ds_path),
        args.workers,
    )
