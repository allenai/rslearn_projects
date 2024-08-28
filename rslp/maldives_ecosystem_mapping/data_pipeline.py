"""Data pipeline for Maldives ecosystem mapping project."""

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
from rslearn.dataset import Dataset, Window
from rslearn.main import (
    IngestHandler,
    MaterializeHandler,
    PrepareHandler,
    apply_on_windows,
)
from rslearn.utils import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.config import BaseDataPipelineConfig

from .config import CATEGORIES, COLORS


class DataPipelineConfig(BaseDataPipelineConfig):
    """Data pipeline config for Maldives ecosystem mapping."""

    def __init__(
        self,
        ds_root: str | None = None,
        workers: int = 1,
        src_dir: str = "gs://earthsystem-a1/maxar",
        skip_ingest: bool = False,
    ):
        """Create a new DataPipelineConfig.

        Args:
            ds_root: optional dataset root to write the dataset. This defaults to GCS.
            workers: number of workers.
            src_dir: the source directory to read from.
            skip_ingest: whether to skip running prepare/ingest/materialize on the
                dataset.
        """
        if ds_root is None:
            rslp_bucket = os.environ["RSLP_BUCKET"]
            ds_root = f"gcs://{rslp_bucket}/datasets/maldives_ecosystem_mapping/dataset_v1/live/"
        super().__init__(ds_root, workers)
        self.src_dir = src_dir
        self.skip_ingest = skip_ingest


class ProcessJob:
    """A job to process one Maxar scene (one island).

    Two windows are created for each job:
    (1) A big window corresponding to the entire scene, that is unlabeled.
    (2) A small window corresponding to just the patch that has labels.
    """

    def __init__(
        self, config: DataPipelineConfig, path: UPath, prefix: str, is_sentinel2: bool
    ):
        """Create a new ProcessJob.

        Args:
            config: the DataPipelineConfig.
            path: the directory containing the scene to process in this job.
            prefix: the filename prefix of the scene.
            is_sentinel2: whether to create a Sentinel-2 window. In this case, we
                populate the scene at 10 m/pixel and without any raster, rather than
                populating with the input Maxar image.
        """
        self.config = config
        self.path = path
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
    label = job.prefix
    dst_path = UPath(job.config.ds_root)
    is_val = hashlib.sha256(label.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"

    buf = io.BytesIO()
    tif_path = job.path / (job.prefix + ".tif")
    with tif_path.open("rb") as f:
        buf.write(f.read())

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
    window_root = dst_path / "windows" / job.image_group_name() / window_name
    window = Window(
        path=window_root,
        group=job.image_group_name(),
        name=window_name,
        projection=projection,
        bounds=raster_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    if not job.is_sentinel2:
        layer_dir = window_root / "layers" / "maxar"
        out_fname = layer_dir / "R_G_B" / "geotiff.tif"
        out_fname.parent.mkdir(parents=True, exist_ok=True)
        with out_fname.open("wb") as f:
            f.write(buf.getvalue())
        with (layer_dir / "completed").open("w") as f:
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

    json_path = job.path / (job.prefix + "_labels.json")
    with json_path.open("r") as f:
        data = json.load(f)
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
    window_root = dst_path / "windows" / job.crop_group_name() / window_name
    window = Window(
        path=window_root,
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
        layer_dir = window_root / "layers" / "maxar"
        GeotiffRasterFormat().encode_raster(
            layer_dir / "R_G_B", projection, proj_bounds, crop
        )
        with (layer_dir / "completed").open("w") as f:
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
    layer_dir = window_root / "layers" / "label"
    GeotiffRasterFormat().encode_raster(
        layer_dir / "label", projection, proj_bounds, mask[None, :, :]
    )
    with (layer_dir / "completed").open("w") as f:
        pass

    # Along with a visualization image.
    label_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for category_id in range(len(CATEGORIES)):
        color = COLORS[category_id % len(COLORS)]
        label_vis[mask == category_id] = color
    layer_dir = window_root / "layers" / "label"
    GeotiffRasterFormat().encode_raster(
        layer_dir / "vis", projection, proj_bounds, label_vis.transpose(2, 0, 1)
    )
    with (layer_dir / "completed").open("w") as f:
        pass


def data_pipeline(dp_config: DataPipelineConfig):
    """Run the data pipeline for Maldives ecosystem mapping.

    Args:
        dp_config: the pipeline configuration.
    """
    # First copy config.json.
    dst_path = UPath(dp_config.ds_root)
    with open("data/maldives_ecosystem_mapping/config.json") as f:
        cfg_str = f.read()
    with (dst_path / "config.json").open("w") as f:
        f.write(cfg_str)

    # Launch jobs to populate windows.
    print("populate windows")
    p = multiprocessing.Pool(dp_config.workers)
    jobs: list[ProcessJob] = []
    src_path = UPath(dp_config.src_dir)
    for example_path in src_path.glob("**/*_labels.json"):
        prefix = example_path.name.split("_labels.json")[0]
        jobs.append(
            ProcessJob(dp_config, example_path.parent, prefix, is_sentinel2=False)
        )
        jobs.append(
            ProcessJob(dp_config, example_path.parent, prefix, is_sentinel2=True)
        )
    outputs = p.imap_unordered(process, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        continue
    p.close()

    if not dp_config.skip_ingest:
        print("prepare, ingest, materialize")
        dataset = Dataset(dst_path)
        for group in ["images_sentinel2", "crops_sentinel2"]:
            apply_on_windows(
                PrepareHandler(force=False),
                dataset,
                workers=dp_config.workers,
                group=group,
            )
            apply_on_windows(
                IngestHandler(),
                dataset,
                workers=dp_config.workers,
                use_initial_job=False,
                jobs_per_process=1,
                group=group,
            )
            apply_on_windows(
                MaterializeHandler(),
                dataset,
                workers=dp_config.workers,
                use_initial_job=False,
                group=group,
            )
