"""Data pipeline for Maldives ecosystem mapping project."""

import functools
import hashlib
import io
import json
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import rasterio
import rasterio.features
import shapely
import tqdm
from google.cloud import storage
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.main import (
    IngestHandler,
    MaterializeHandler,
    PrepareHandler,
    apply_on_windows,
)
from rslearn.utils import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.config import BaseDataPipelineConfig

from .config import CATEGORIES, COLORS

DEFAULT_TS = datetime(2024, 8, 1, tzinfo=timezone.utc)


class DataPipelineConfig(BaseDataPipelineConfig):
    """Data pipeline config for Maldives ecosystem mapping."""

    def __init__(
        self,
        ds_root: str | None = None,
        workers: int = 1,
        src_dir: str = "gs://earthsystem-a1/maxar",
        islands_fname: str = "gs://earthsystem-a1/resources/maldives_base_layers/island_geo.geojson",
        skip_ingest: bool = True,
    ):
        """Create a new DataPipelineConfig.

        Args:
            ds_root: optional dataset root to write the dataset. This defaults to GCS.
            workers: number of workers.
            src_dir: the source directory to read from.
            islands_fname: path to the GeoJSON file containing the Maldives island polygons.
            skip_ingest: whether to skip running prepare/ingest/materialize on the
                dataset.
        """
        if ds_root is None:
            rslp_bucket = os.environ["RSLP_BUCKET"]
            ds_root = f"gs://{rslp_bucket}/datasets/maldives_ecosystem_mapping/dataset_v1/20240924/"
        super().__init__(ds_root, workers)
        self.src_dir = src_dir
        self.islands_fname = islands_fname
        self.skip_ingest = skip_ingest


class Label:
    """A segmentation label for ecosystem mapping."""

    def __init__(
        self,
        bounding_geom: STGeometry,
        geoms: list[STGeometry],
        classes: list[int],
        ts: datetime,
    ):
        """Create a new Label.

        Args:
            bounding_geom: label footprint.
            geoms: list of polygons in the label.
            classes: corresponding list of ecosystem classes for those polygons.
            ts: the timestamp of the image that this label is based on.
        """
        self.bounding_geom = bounding_geom
        self.geoms = geoms
        self.classes = classes
        self.ts = ts


class MaxarJob:
    """A job to process one Maxar scene (one island).

    Two windows are created for each job:
    (1) A big window corresponding to the entire scene, that is unlabeled.
    (2) A small window corresponding to just the patch that has labels.
    """

    def __init__(
        self, config: DataPipelineConfig, path: UPath, prefix: str, label: Label
    ):
        """Create a new ProcessJob.

        Args:
            config: the DataPipelineConfig.
            path: the directory containing the scene to process in this job.
            prefix: the filename prefix of the scene.
            label: the label for this scene.
        """
        self.config = config
        self.path = path
        self.prefix = prefix
        self.label = label


class BareJob:
    """A job to add window without existing image file.

    This is used for Sentinel-2 and Planet images which are ingested using rslearn.
    """

    def __init__(
        self,
        config: DataPipelineConfig,
        suffix: str,
        window_prefix: str,
        projection: Projection,
        island_geom: STGeometry,
        label: Label | None = None,
    ):
        """Create a new BareJob.

        Args:
            config: the DataPipelineConfig.
            suffix: suffix for the group (images_X or crops_X) and window name
            window_prefix: prefix for the window name.
            projection: projection to use.
            island_geom: geometry of the island.
            label: the label for this window, if any.
        """
        self.config = config
        self.suffix = suffix
        self.window_prefix = window_prefix
        self.projection = projection
        self.island_geom = island_geom
        self.label = label


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


def convert_geom(
    bounding_poly: list[dict[str, Any]], projection: Projection
) -> STGeometry:
    """Convert a bounding polygon to projection coordinates.

    We extract a polygon from the boundingPoly of a Kili JSON export. It should be a
    list of polygons, the first one is exterior while rest are interior holes, and each
    polygon should be represented as a dict with a normalizedVertices key containing
    list of dicts that have x/y keys.

    We convert the polygon into an STGeometry in the coordinates of the given projection.

    Args:
        bounding_poly: the bounding polygon from Kili JSON export. It is a list of
            dicts each represented exterior or interior ring.
        projection: the projection to convert to.

    Returns:
        an STGeometry representing the polygon in the given projection.
    """
    exterior = [
        (vertex["x"], vertex["y"]) for vertex in bounding_poly[0]["normalizedVertices"]
    ]
    interiors = []
    for poly in bounding_poly[1:]:
        interior = [(vertex["x"], vertex["y"]) for vertex in poly["normalizedVertices"]]
        interiors.append(interior)
    shp = shapely.Polygon(exterior, interiors)
    src_geom = STGeometry(WGS84_PROJECTION, shp, None)
    dst_geom = src_geom.to_projection(projection)
    return dst_geom


def load_label(json_path: UPath) -> tuple[UPath, Label]:
    """Load a Label from GCS.

    Args:
        json_path: the path to the JSON file.

    Returns:
        tuple (json_path, label) containing the same path that was passed in along with
            the Label object.
    """
    with json_path.open() as f:
        data = json.load(f)

    geoms = []
    classes = []

    for annot in data["annotations"]:
        assert len(annot["categories"]) == 1
        geom = convert_geom(annot["boundingPoly"], WGS84_PROJECTION)
        category_id = CATEGORIES.index(annot["categories"][0]["name"])
        geoms.append(geom)
        classes.append(category_id)

    bounding_geom = convert_geom(
        data["mapping_area"][0]["boundingPoly"], WGS84_PROJECTION
    )
    return json_path, Label(
        bounding_geom=bounding_geom,
        geoms=geoms,
        classes=classes,
        ts=DEFAULT_TS,
    )


def process_maxar(job: MaxarJob) -> tuple[str, datetime]:
    """Process one MaxarJob.

    Args:
        job: the MaxarJob.

    Returns:
        a tuple (prefix, timestamp) containing the filename prefix along with timestamp
            of the Maxar image.
    """
    window_prefix = job.prefix
    dst_path = UPath(job.config.ds_root)
    is_val = hashlib.sha256(window_prefix.encode()).hexdigest()[0] in ["0", "1"]
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
    window_name = window_prefix
    window_root = dst_path / "windows" / "images" / window_name
    window = Window(
        path=window_root,
        group="images",
        name=window_name,
        projection=projection,
        bounds=raster_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    layer_dir = window_root / "layers" / "maxar"
    out_fname = layer_dir / "R_G_B" / "geotiff.tif"
    out_fname.parent.mkdir(parents=True, exist_ok=True)
    with out_fname.open("wb") as f:
        f.write(buf.getvalue())
    (layer_dir / "completed").touch()

    # Second create a window just for the annotated patch.
    # Starting by converting the label bounding geometry and polygons to the projection
    # of the Maxar image.
    bounding_shp = job.label.bounding_geom.to_projection(projection).shp
    proj_bounds = [int(x) for x in bounding_shp.bounds]
    pixel_bounds = [
        proj_bounds[0] - raster_bounds[0],
        proj_bounds[1] - raster_bounds[1],
        proj_bounds[2] - raster_bounds[0],
        proj_bounds[3] - raster_bounds[1],
    ]
    array = raster.read()
    clipped_pixel_bounds = [
        clip(pixel_bounds[0], 0, array.shape[2]),
        clip(pixel_bounds[1], 0, array.shape[1]),
        clip(pixel_bounds[2], 0, array.shape[2]),
        clip(pixel_bounds[3], 0, array.shape[1]),
    ]
    if pixel_bounds != clipped_pixel_bounds:
        print(
            f"warning: {window_prefix}: clipping pixel bounds from {pixel_bounds} to {clipped_pixel_bounds}"
        )
    pixel_bounds = clipped_pixel_bounds

    proj_shapes = []
    for geom, category_id in zip(job.label.geoms, job.label.classes):
        geom = geom.to_projection(projection)
        proj_shapes.append((geom.shp, category_id))

    # Crop the raster to the bounds of the label.
    crop = array[
        :, pixel_bounds[1] : pixel_bounds[3], pixel_bounds[0] : pixel_bounds[2]
    ]

    # Create window.
    window_name = f"{window_prefix}_{pixel_bounds[0]}_{pixel_bounds[1]}"
    window_root = dst_path / "windows" / "crops" / window_name
    window = Window(
        path=window_root,
        group="crops",
        name=window_name,
        projection=projection,
        bounds=proj_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    # Write the Maxar GeoTIFF.
    layer_dir = window_root / "layers" / "maxar"
    GeotiffRasterFormat(always_enable_tiling=True).encode_raster(
        layer_dir / "R_G_B", projection, proj_bounds, crop
    )
    (layer_dir / "completed").touch()

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
    GeotiffRasterFormat(always_enable_tiling=True).encode_raster(
        layer_dir / "label", projection, proj_bounds, mask[None, :, :]
    )
    (layer_dir / "completed").touch()

    # Along with a visualization image.
    label_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for category_id in range(len(CATEGORIES)):
        color = COLORS[category_id % len(COLORS)]
        label_vis[mask == category_id] = color
    layer_dir = window_root / "layers" / "label"
    GeotiffRasterFormat(always_enable_tiling=True).encode_raster(
        layer_dir / "vis", projection, proj_bounds, label_vis.transpose(2, 0, 1)
    )
    (layer_dir / "completed").touch()

    return job.prefix, ts


def process_bare(job: BareJob) -> None:
    """Process one BareJob.

    Args:
        job: the BareJob.
    """
    window_prefix = job.window_prefix
    dst_path = UPath(job.config.ds_root)
    is_val = hashlib.sha256(window_prefix.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"

    projection = job.projection
    island_bounds = [int(value) for value in job.island_geom.shp.bounds]

    if job.label and job.label.ts:
        ts = job.label.ts
    else:
        ts = DEFAULT_TS

    # First create window for the entire GeoTIFF.
    window_name = f"{window_prefix}_{job.suffix}"
    group_name = f"images_{job.suffix}"
    window_root = dst_path / "windows" / group_name / window_name
    window = Window(
        path=window_root,
        group=group_name,
        name=window_name,
        projection=projection,
        bounds=island_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    if not job.label:
        return

    # Second create a window just for the annotated patch.
    bounding_shp = job.label.bounding_geom.to_projection(projection).shp
    proj_bounds = [int(x) for x in bounding_shp.bounds]
    proj_shapes = []
    for geom, category_id in zip(job.label.geoms, job.label.classes):
        geom = geom.to_projection(projection)
        proj_shapes.append((geom.shp, category_id))

    # Create window.
    window_name = f"{window_prefix}_{proj_bounds[0]}_{proj_bounds[1]}_{job.suffix}"
    group_name = f"crops_{job.suffix}"
    window_root = dst_path / "windows" / group_name / window_name
    window = Window(
        path=window_root,
        group=group_name,
        name=window_name,
        projection=projection,
        bounds=proj_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        options={"split": split},
    )
    window.save()

    # Render the GeoJSON labels and write it to layer.
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
    GeotiffRasterFormat(always_enable_tiling=True).encode_raster(
        layer_dir / "label", projection, proj_bounds, mask[None, :, :]
    )
    (layer_dir / "completed").touch()

    # Along with a visualization image.
    label_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
    for category_id in range(len(CATEGORIES)):
        color = COLORS[category_id % len(COLORS)]
        label_vis[mask == category_id] = color
    layer_dir = window_root / "layers" / "label"
    GeotiffRasterFormat(always_enable_tiling=True).encode_raster(
        layer_dir / "vis", projection, proj_bounds, label_vis.transpose(2, 0, 1)
    )
    (layer_dir / "completed").touch()


def get_bare_jobs(
    dp_config: DataPipelineConfig,
    island_feature: dict[str, Any],
    src_projection: Projection,
    labels: list[Label],
) -> list[BareJob]:
    """Get the bare jobs needed for this island.

    First it finds a matching label from the list of labels, if any. If there is a
    label, then a crops_X window would be created, otherwise only an images_X window.

    Then it creates the Sentinel-2, SkySat, and PlanetScope jobs.

    Args:
        dp_config: the data pipeline config to include in the job data.
        island_feature: the feature from islands GeoJSON. It should have atoll and
            islandName properties, and the projection should correspond to
            src_projection.
        src_projection: the projection that the island features are in.
        labels: list of labels available. If there's one that intersects this island
            then we include in with the job so the crops_X window can be created.

    Returns:
        a list of BareJob objects.
    """
    props = island_feature["properties"]
    atoll = props["atoll"].replace(" ", "")
    island_name = props["islandName"]
    if island_name is None:
        island_name = "none"
    island_fcode = props["FCODE"]
    window_prefix = f"{atoll}_{island_name}_{island_fcode}"

    island_src_shp = shapely.geometry.shape(island_feature["geometry"])
    island_src_geom = STGeometry(src_projection, island_src_shp, None)
    island_wgs84_shp = island_src_geom.to_projection(WGS84_PROJECTION).shp

    # Find matching label if any.
    matching_label = None
    for label in labels.values():
        label_shp = label.bounding_geom.shp
        if not label_shp.intersects(island_wgs84_shp):
            continue
        fraction_contained = (
            label_shp.intersection(island_wgs84_shp).area / label_shp.area
        )
        if fraction_contained < 0.1:
            continue
        matching_label = label
        break

    bare_jobs = []

    for suffix, resolution in [("sentinel2", 10), ("skysat", 0.5), ("planetscope", 2)]:
        dst_projection = get_utm_ups_projection(
            island_wgs84_shp.centroid.x,
            island_wgs84_shp.centroid.y,
            resolution,
            -resolution,
        )
        dst_geom = island_src_geom.to_projection(dst_projection)
        bare_jobs.append(
            BareJob(
                config=dp_config,
                suffix=suffix,
                window_prefix=window_prefix,
                projection=dst_projection,
                island_geom=dst_geom,
                label=matching_label,
            )
        )

    return bare_jobs


def data_pipeline(dp_config: DataPipelineConfig):
    """Run the data pipeline for Maldives ecosystem mapping.

    Args:
        dp_config: the pipeline configuration.
    """
    # First copy config.json.
    dst_path = UPath(dp_config.ds_root)
    with open("data/maldives_ecosystem_mapping/config_sentinel2.json") as f:
        cfg_str = f.read()
    with (dst_path / "config.json").open("w") as f:
        f.write(cfg_str)

    # Download all of the labels.
    # As we obtain them, we also populate Maxar jobs in case the corresponding Maxar
    # image exists.
    src_path = UPath(dp_config.src_dir)
    json_paths = list(src_path.glob("**/*_labels.json"))
    p = multiprocessing.Pool(dp_config.workers)
    outputs = p.imap_unordered(load_label, json_paths)

    labels: dict[str, Label] = {}
    maxar_jobs: list[MaxarJob] = []
    for json_path, label in tqdm.tqdm(
        outputs, total=len(json_paths), desc="download labels"
    ):
        if len(label.geoms) == 0:
            print(f"warning: {json_path} contains zero annotations")
            continue

        prefix = json_path.name.split("_labels.json")[0]
        labels[prefix] = label

        maxar_path = json_path.parent / (prefix + ".tif")
        if not maxar_path.exists():
            continue
        maxar_jobs.append(
            MaxarJob(
                config=dp_config,
                path=json_path.parent,
                prefix=prefix,
                label=label,
            )
        )

    # Now we can run the Maxar jobs to populate those windows.
    # This step also returns the timestamp of the Maxar images so we can update the
    # label objects accordingly, that way when we get Sentinel-2 / Planet images we can
    # have the timestamps aligned.
    outputs = p.imap_unordered(process_maxar, maxar_jobs)
    for prefix, ts in tqdm.tqdm(
        outputs, total=len(maxar_jobs), desc="populate maxar windows"
    ):
        labels[prefix].ts = ts

    # Populate the Sentinel-2 and Planet windows.
    # These are populated based on a GeoJSON file that includes the bounds of all of
    # the islands in the Maldives.
    # We also need to match labels to island polygons so that the labels can be
    # included in case they exist.
    islands_path = UPath(dp_config.islands_fname)
    with islands_path.open() as f:
        fc = json.load(f)
        src_crs = CRS.from_string(fc["crs"]["properties"]["name"])
        src_projection = Projection(src_crs, 1, 1)

    bare_jobs = []
    outputs = star_imap_unordered(
        p,
        get_bare_jobs,
        [
            dict(
                dp_config=dp_config,
                island_feature=feat,
                src_projection=src_projection,
                labels=labels,
            )
            for feat in fc["features"]
        ],
    )
    for job_list in tqdm.tqdm(
        outputs, total=len(fc["features"]), desc="Matching labels to islands"
    ):
        bare_jobs.extend(job_list)

    outputs = p.imap_unordered(process_bare, bare_jobs)
    for _ in tqdm.tqdm(outputs, total=len(bare_jobs), desc="populating other windows"):
        pass
    p.close()

    # TODO: this doesn't really work because we need to use different dataset
    # configuration files for Sentinel-2, SkySat, and PlanetScope.
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
