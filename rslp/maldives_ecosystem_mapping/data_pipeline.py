"""Data pipeline for Maldives ecosystem mapping project."""

import argparse
import functools
import hashlib
import io
import json
import multiprocessing
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
import rasterio.features
import shapely
import tqdm
from google.cloud import storage
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, parse_file_api_string
from rslearn.utils.raster_format import GeotiffRasterFormat

from rslp.config import BaseDataPipelineConfig

from .config import CATEGORIES, COLORS


class DataPipelineConfig(BaseDataPipelineConfig):
    """Data pipeline config for Maldives ecosystem mapping."""

    def __init__(
        self,
        ds_root: str | None = None,
        workers: int = 1,
        src_bucket: str = "earthsystem-a1",
        src_prefix: str = "maxar/",
    ):
        """Create a new DataPipelineConfig.

        Args:
            ds_root: optional dataset root to write the dataset. This defaults to GCS.
            workers: number of workers.
            src_bucket: source bucket to read images and labels from.
            src_prefix: prefix within source bucket.
        """
        if ds_root is None:
            ds_root = "s3://rslearn-data/datasets/maldives_ecosystem_mapping/dataset_v1/live/?endpoint_url=https://storage.googleapis.com"
        super().__init__(ds_root, workers)
        self.src_bucket = src_bucket
        self.src_prefix = src_prefix


class ProcessJob:
    """A job to process one Maxar scene (one island).

    Two windows are created for each job:
    (1) A big window corresponding to the entire scene, that is unlabeled.
    (2) A small window corresponding to just the patch that has labels.
    """

    def __init__(self, config: DataPipelineConfig, prefix: str, is_sentinel2: bool):
        """Create a new ProcessJob.

        Args:
            config: the DataPipelineConfig.
            prefix: the prefix for the scene to process in this job.
            is_sentinel2: whether to create a Sentinel-2 window. In this case, we
                populate the scene at 10 m/pixel and without any raster, rather than
                populating with the input Maxar image.
        """
        self.config = config
        self.prefix = prefix
        self.is_sentinel2 = is_sentinel2

    def image_group_name(self):
        """The group where the big window for the entire scene should be written."""
        if self.is_sentinel2:
            return "images_sentinel2"
        else:
            return "images"

    def crop_group_name(self):
        """The group where the window for the labeled portion should be written."""
        if self.is_sentinel2:
            return "crops_sentinel2"
        else:
            return "crops"


def clip(value, lo, hi):
    """Clip the input value to [lo, hi].

    Args:
        value: the value to clip
        lo: the minimum value.
        hi: the maximum value.

    Returns:
        the clipped value.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@functools.cache
def get_bucket(bucket_name: str):
    """Get cached bucket object for the specified bucket.

    Args:
        bucket_name: the name of the Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    return storage_client.bucket(bucket_name)


def process(job: ProcessJob):
    """Process one ProcessJob.

    Args:
        job: the ProcessJob.
    """
    label = job.prefix.split("/")[-1]
    dst_fapi = parse_file_api_string(job.config.ds_root)
    is_val = hashlib.sha256(label.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"

    src_bucket = get_bucket(job.config.src_bucket)
    blob = src_bucket.blob(job.prefix + ".tif")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    raster = rasterio.open(buf)
    projection = Projection(raster.crs, raster.transform.a, raster.transform.e)
    start_col = round(raster.transform.c / raster.transform.a)
    start_row = round(raster.transform.f / raster.transform.e)
    raster_bounds = [
        start_col,
        start_row,
        start_col + raster.width,
        start_row + raster.height,
    ]

    if job.is_sentinel2:
        projection = Projection(raster.crs, 10, -10)
        raster_bounds = [
            int(raster_bounds[0] * raster.transform.a / projection.x_resolution),
            int(raster_bounds[1] * raster.transform.e / projection.y_resolution),
            int(raster_bounds[2] * raster.transform.a / projection.x_resolution),
            int(raster_bounds[3] * raster.transform.e / projection.y_resolution),
        ]

    # Extract datetime.
    parts = job.prefix.split("_")[-1].split("-")
    assert len(parts) == 5
    ts = datetime(
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
        int(parts[4]),
        tzinfo=timezone.utc,
    )

    # First create window for the entire GeoTIFF.
    if job.is_sentinel2:
        window_name = label + "_sentinel2"
    else:
        window_name = label
    window_root = dst_fapi.get_folder("windows", job.image_group_name(), window_name)
    window = Window(
        file_api=window_root,
        group=job.image_group_name(),
        name=window_name,
        projection=projection,
        bounds=raster_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    if not job.is_sentinel2:
        layer_dir = window_root.get_folder("layers", "maxar")
        with layer_dir.get_folder("R_G_B").open("geotiff.tif", "wb") as f:
            f.write(buf.getvalue())
        with layer_dir.open("completed", "w") as f:
            pass

    # Second create a window just for the annotated patch.
    # Start by reading the annotations and mapping polygon.
    def convert_geom(bounding_poly):
        # Convert the boundingPoly from the JSON file into an STGeometry in the
        # coordinates of `projection`.
        exterior = [
            (vertex["x"], vertex["y"])
            for vertex in bounding_poly[0]["normalizedVertices"]
        ]
        interiors = []
        for poly in bounding_poly[1:]:
            interior = [
                (vertex["x"], vertex["y"]) for vertex in poly["normalizedVertices"]
            ]
            interiors.append(interior)
        shp = shapely.Polygon(exterior, interiors)
        src_geom = STGeometry(WGS84_PROJECTION, shp, None)
        dst_geom = src_geom.to_projection(projection)
        return dst_geom

    blob = src_bucket.blob(job.prefix + "_labels.json")
    data = json.loads(blob.download_as_string())
    proj_shapes = []
    for annot in data["annotations"]:
        assert len(annot["categories"]) == 1
        category_id = CATEGORIES.index(annot["categories"][0]["name"])
        dst_geom = convert_geom(annot["boundingPoly"])
        proj_shapes.append((dst_geom.shp, category_id))

    dst_geom = convert_geom(data["mapping_area"][0]["boundingPoly"])
    bounding_shp = dst_geom.shp

    # Convert the bounding shp to int bounds.
    # Also crop the raster.
    proj_bounds = [int(x) for x in bounding_shp.bounds]
    pixel_bounds = [
        proj_bounds[0] - raster_bounds[0],
        proj_bounds[1] - raster_bounds[1],
        proj_bounds[2] - raster_bounds[0],
        proj_bounds[3] - raster_bounds[1],
    ]
    crop = None
    if not job.is_sentinel2:
        array = raster.read()
        clipped_pixel_bounds = [
            clip(pixel_bounds[0], 0, array.shape[2]),
            clip(pixel_bounds[1], 0, array.shape[1]),
            clip(pixel_bounds[2], 0, array.shape[2]),
            clip(pixel_bounds[3], 0, array.shape[1]),
        ]
        if pixel_bounds != clipped_pixel_bounds:
            print(
                f"warning: {label}: clipping pixel bounds from {pixel_bounds} to {clipped_pixel_bounds}"
            )
        pixel_bounds = clipped_pixel_bounds
        crop = array[
            :, pixel_bounds[1] : pixel_bounds[3], pixel_bounds[0] : pixel_bounds[2]
        ]

    # Create window.
    window_name = f"{label}_{pixel_bounds[0]}_{pixel_bounds[1]}"
    if job.is_sentinel2:
        window_name += "_sentinel2"
    window_root = dst_fapi.get_folder("windows", job.crop_group_name(), window_name)
    window = Window(
        file_api=window_root,
        group=job.crop_group_name(),
        name=window_name,
        projection=projection,
        bounds=proj_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    # Write the GeoTIFF.
    if not job.is_sentinel2:
        layer_dir = window_root.get_folder("layers", "maxar")
        GeotiffRasterFormat().encode_raster(
            layer_dir.get_folder("R_G_B"), projection, proj_bounds, crop
        )
        with layer_dir.open("completed", "w") as f:
            pass

    # Render the GeoJSON labels and write that too.
    pixel_shapes = []
    for shp, category_id in proj_shapes:
        shp = shapely.transform(
            shp, lambda coords: coords - [proj_bounds[0], proj_bounds[1]]
        )
        pixel_shapes.append((shp, category_id))
    mask = rasterio.features.rasterize(
        pixel_shapes,
        out_shape=(proj_bounds[3] - proj_bounds[1], proj_bounds[2] - proj_bounds[0]),
    )
    file_api = window.file_api.get_folder("layers", "label", "label")
    GeotiffRasterFormat().encode_raster(
        file_api, projection, proj_bounds, mask[None, :, :]
    )

    # Along with a visualization image.
    label_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for category_id in range(len(CATEGORIES)):
        color = COLORS[category_id % len(COLORS)]
        label_vis[mask == category_id] = color
    layer_dir = window_root.get_folder("layers", "label", "vis")
    GeotiffRasterFormat().encode_raster(
        layer_dir, projection, proj_bounds, label_vis.transpose(2, 0, 1)
    )
    with layer_dir.open("completed", "w") as f:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", help="Input directory containing retrieved images and labels"
    )
    parser.add_argument("--out_dir", help="Output directory")
    args = parser.parse_args()

    jobs = []
    for fname in os.listdir(args.in_dir):
        if not fname.endswith(".tif"):
            continue
        prefix = os.path.join(args.in_dir, fname.split(".tif")[0])
        job = ProcessJob(prefix, args.out_dir, is_sentinel2=True)
        jobs.append(job)
        job = ProcessJob(prefix, args.out_dir, is_sentinel2=False)
        jobs.append(job)
    for job in tqdm.tqdm(jobs):
        process(job)


def data_pipeline(dp_config: DataPipelineConfig):
    """Run the data pipeline for Maldives ecosystem mapping.

    Args:
        dp_config: the pipeline configuration.
    """
    # First copy config.json.
    dst_fapi = parse_file_api_string(dp_config.ds_root)
    with open("data/maldives_ecosystem_mapping/config.json") as f:
        cfg_str = f.read()
    with dst_fapi.open("config.json", "w") as f:
        f.write(cfg_str)

    # Launch jobs to populate windows.
    p = multiprocessing.Pool(dp_config.workers)
    src_bucket = get_bucket(dp_config.src_bucket)
    jobs: list[ProcessJob] = []
    for blob in src_bucket.list_blobs(
        prefix=dp_config.src_prefix, match_glob="**/*_labels.json"
    ):
        prefix = blob.name.split("_labels.json")[0]
        jobs.append(ProcessJob(dp_config, prefix, is_sentinel2=False))
        jobs.append(ProcessJob(dp_config, prefix, is_sentinel2=True))
    outputs = p.imap_unordered(process, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        continue
    p.close()
