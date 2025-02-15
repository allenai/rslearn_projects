"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item, data_source_from_config
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils.fsspec import open_rasterio_upath_reader
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.log_utils import get_logger
from rslp.utils.filter import NearInfraFilter
from rslp.utils.rslearn import (
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
    materialize_dataset,
    run_model_predict,
)
from rslp.vessels import VesselDetection

logger = get_logger(__name__)

# Name to use for source attribute in VesselDetection.
SENTINEL2_SOURCE = "sentinel2"

# Name to use in rslearn dataset for layer containing Sentinel-2 images.
SENTINEL2_LAYER_NAME = "sentinel2"

# Name of layer containing the output.
OUTPUT_LAYER_NAME = "output"

SCENE_ID_DATASET_CONFIG = "data/sentinel2_vessels/config.json"
IMAGE_FILES_DATASET_CONFIG = "data/sentinel2_vessels/config_local_files.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"

SENTINEL2_RESOLUTION = 10
CROP_WINDOW_SIZE = 128

# Distance threshold for near marine infrastructure filter in km.
# 0.05 km = 50 m
INFRA_DISTANCE_THRESHOLD = 0.05

# The bands of the TCI image. It must be provided when using ImageFiles mode.
TCI_BANDS = ["R", "G", "B"]


@dataclass
class ImageFile:
    """An image file provided for inference.

    Args:
        bands: the list of bands contained in this file.
        fname: the filename.
    """

    bands: list[str]
    fname: str


@dataclass
class PredictionTask:
    """A task to predict vessels in one Sentinel-2 scene.

    Args:
        scene_id: the Sentinel-2 scene ID. One of scene_id or image_files must be set.
        image_files: a list of ImageFiles.
        json_path: optional path to write the JSON of vessel detections.
        crop_path: optional path to write the vessel crop images.
        geojson_path: optional path to write GeoJSON of detections.
    """

    scene_id: str | None = None
    image_files: list[ImageFile] | None = None
    json_path: str | None = None
    crop_path: str | None = None
    geojson_path: str | None = None


@dataclass
class SceneData:
    """Data about the Sentinel-2 scene that the prediction should be applied on.

    This can come from scene ID or provided GeoTIFFs.
    """

    # Basic required information.
    projection: Projection
    bounds: PixelBounds

    # Optional time range -- if provided, it is passed on to the Window.
    time_range: tuple[datetime, datetime] | None = None

    # Optional Item -- if provided, we write the layer metadatas (items.json) in the
    # window to skip prepare step so that the correct item is always read.
    item: Item | None = None


def setup_dataset_with_scene_ids(
    ds_path: UPath, scene_ids: list[str]
) -> list[SceneData]:
    """Initialize an rslearn dataset for prediction with scene IDs.

    Args:
        ds_path: the dataset path to write to.
        scene_ids: list of scene IDs to apply the vessel detection model on.

    Returns:
        a list of SceneData corresponding to the list of scene IDs.
    """
    # Write dataset configuration file (which is set up to get Sentinel-2 images from
    # GCP.)
    with open(SCENE_ID_DATASET_CONFIG, "rb") as src:
        with (ds_path / "config.json").open("wb") as dst:
            shutil.copyfileobj(src, dst)

    # Initialize Sentinel2 data source object so we can lookup the Item that each scene
    # ID corresponds to.
    dataset = Dataset(ds_path)
    data_source: Sentinel2 = data_source_from_config(
        dataset.layers[SENTINEL2_LAYER_NAME], dataset.path
    )

    # Get the SceneData based on looking up the scene.
    scene_datas: list[SceneData] = []
    for scene_id in scene_ids:
        # Lookup item.
        item = data_source.get_item_by_name(scene_id)

        # Get the projection to use for the window.
        wgs84_geom = item.geometry.to_projection(WGS84_PROJECTION)
        projection = get_utm_ups_projection(
            wgs84_geom.shp.centroid.x,
            wgs84_geom.shp.centroid.y,
            SENTINEL2_RESOLUTION,
            -SENTINEL2_RESOLUTION,
        )
        dst_geom = item.geometry.to_projection(projection)
        bounds = (
            int(dst_geom.shp.bounds[0]),
            int(dst_geom.shp.bounds[1]),
            int(dst_geom.shp.bounds[2]),
            int(dst_geom.shp.bounds[3]),
        )

        scene_data = SceneData(
            projection=projection,
            bounds=bounds,
            time_range=item.geometry.time_range,
            item=item,
        )
        scene_datas.append(scene_data)

    return scene_datas


def setup_dataset_with_image_files(
    ds_path: UPath, image_files_list: list[list[ImageFile]]
) -> list[SceneData]:
    """Initialize an rslearn dataset for prediction with image files.

    Args:
        ds_path: the dataset path to write to.
        image_files_list: list of ImageFile lists needed for each task.

    Returns:
        a list of SceneData corresponding to image_files_list.
    """
    # Write dataset configuration file.
    # We need to override the item_specs placeholders.
    # src_dir is a placeholder too but we don't need to override it since we use
    # absolute paths only here.
    with open(IMAGE_FILES_DATASET_CONFIG) as f:
        cfg = json.load(f)
    item_specs = []
    for image_files in image_files_list:
        item_spec: dict = {
            "fnames": [],
            "bands": [],
        }
        for image_file in image_files:
            # Ensure the filename is a URI so it won't be treated as a relative path.
            item_spec["fnames"].append(UPath(image_file.fname).absolute().as_uri())
            item_spec["bands"].append(image_file.bands)
        item_specs.append(item_spec)
    cfg["layers"][SENTINEL2_LAYER_NAME]["data_source"]["item_specs"] = item_specs

    with (ds_path / "config.json").open("w") as f:
        json.dump(cfg, f)

    # Get the projection and scene bounds for each task from the TCI image.
    scene_datas: list[SceneData] = []
    for image_files in image_files_list:
        # Look for TCI image. It is required.
        tci_fname: UPath | None = None
        for image_file in image_files:
            if image_file.bands != TCI_BANDS:
                continue
            tci_fname = UPath(image_file.fname)

        if tci_fname is None:
            raise ValueError("provided list of image files does not have TCI image")

        with open_rasterio_upath_reader(tci_fname) as raster:
            projection = Projection(raster.crs, raster.transform.a, raster.transform.e)
            left = int(raster.transform.c / projection.x_resolution)
            top = int(raster.transform.f / projection.y_resolution)
            scene_bounds = (
                left,
                top,
                left + int(raster.width),
                top + int(raster.height),
            )

        scene_datas.append(
            SceneData(
                projection=projection,
                bounds=scene_bounds,
            )
        )

    return scene_datas


def get_vessel_detections(
    ds_path: UPath,
    scene_datas: list[SceneData],
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Sentinel-2 images.

    Args:
        ds_path: the dataset path that will be populated with a new window to apply the
            detector.
        scene_datas: the SceneDatas to apply the detector on.
    """
    # Create a window for each SceneData.
    windows: list[Window] = []
    group = "detector_predict"
    for scene_idx, scene_data in enumerate(scene_datas):
        window_name = str(scene_idx)
        window_path = ds_path / "windows" / group / window_name
        window = Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=scene_data.projection,
            bounds=scene_data.bounds,
            time_range=scene_data.time_range,
        )
        window.save()
        windows.append(window)

        if scene_data.item:
            layer_data = WindowLayerData(
                SENTINEL2_LAYER_NAME, [[scene_data.item.serialize()]]
            )
            window.save_layer_datas({SENTINEL2_LAYER_NAME: layer_data})

    # Materialize the windows.
    logger.info("Materialize dataset for Sentinel-2 Vessel Detection")
    apply_windows_args = ApplyWindowsArgs(group=group, workers=32)
    materialize_pipeline_args = MaterializePipelineArgs(
        disabled_layers=[],
        prepare_args=PrepareArgs(apply_windows_args=apply_windows_args),
        ingest_args=IngestArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
        materialize_args=MaterializeArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
    )
    materialize_dataset(ds_path, materialize_pipeline_args)
    for window in windows:
        if not window.is_layer_completed(SENTINEL2_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have Sentinel-2 layer completed"
            )

    # Run object detector.
    run_model_predict(DETECT_MODEL_CONFIG, ds_path)

    # Read the detections.
    detections: list[VesselDetection] = []
    for task_idx, (window, scene_data) in enumerate(zip(windows, scene_datas)):
        layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
        features = GeojsonVectorFormat().decode_vector(layer_dir, window.bounds)
        for feature in features:
            geometry = feature.geometry
            score = feature.properties["score"]

            detection = VesselDetection(
                source=SENTINEL2_SOURCE,
                col=int(geometry.shp.centroid.x),
                row=int(geometry.shp.centroid.y),
                projection=geometry.projection,
                score=score,
                # We use this metadata to keep track of which window/scene each
                # detection came from.
                metadata={"task_idx": task_idx},
            )

            if scene_data.item:
                detection.scene_id = scene_data.item.name
                detection.ts = scene_data.item.geometry.time_range[0]

            detections.append(detection)

    return detections


def get_vessel_crop_windows(
    ds_path: UPath, detections: list[VesselDetection], scene_datas: list[SceneData]
) -> list[Window]:
    """Create a window for each vessel to obtain a cropped image for it.

    Args:
        ds_path: the rslearn dataset path (same one used for object detector -- we will
            put the crop windows in a different group).
        detections: list of vessel detections.
        scene_datas: list of SceneDatas that we are processing.

    Returns:
        list of windows corresponding to the detection list, where cropped images have
            been materialized.
    """
    # Create the windows.
    group = "crops"
    crop_windows: list[UPath] = []
    for detection in detections:
        window_name = f"{detection.scene_id}_{detection.col}_{detection.row}"
        window_path = Window.get_window_root(ds_path, group, window_name)
        bounds = (
            detection.col - CROP_WINDOW_SIZE // 2,
            detection.row - CROP_WINDOW_SIZE // 2,
            detection.col + CROP_WINDOW_SIZE // 2,
            detection.row + CROP_WINDOW_SIZE // 2,
        )

        # task_idx metadata is always set in sentinel2_vessels.
        task_idx = detection.metadata["task_idx"]
        scene_data = scene_datas[task_idx]
        window = Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=detection.projection,
            bounds=bounds,
            time_range=scene_data.time_range,
        )
        window.save()

        if scene_data.item:
            layer_data = WindowLayerData(
                SENTINEL2_LAYER_NAME, [[scene_data.item.serialize()]]
            )
            window.save_layer_datas({SENTINEL2_LAYER_NAME: layer_data})

        crop_windows.append(window)

    # Materialize the windows.
    apply_windows_args = ApplyWindowsArgs(group=group, workers=32)
    materialize_pipeline_args = MaterializePipelineArgs(
        disabled_layers=[],
        prepare_args=PrepareArgs(apply_windows_args=apply_windows_args),
        ingest_args=IngestArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
        materialize_args=MaterializeArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
    )
    # Avoid error in case of no detections.
    if len(detections) > 0:
        materialize_dataset(ds_path, materialize_pipeline_args)

    return crop_windows


def predict_pipeline(
    tasks: list[PredictionTask], scratch_path: str
) -> list[list[VesselDetection]]:
    """Run the Sentinel-2 vessel prediction pipeline.

    Given a Sentinel-2 scene ID, the pipeline produces the vessel detections.
    Specifically, it outputs a CSV containing the vessel detection locations along with
    crops of each detection.

    Args:
        tasks: prediction tasks to execute. They must all specify scene IDs or all
            specify image files to process.
        scratch_path: directory to use to store temporary dataset.

    Returns:
        list of vessel detections for each task.
    """
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    if tasks[0].scene_id is not None:
        scene_ids: list[str] = []
        for task in tasks:
            if task.scene_id is None:
                raise ValueError(
                    "all tasks must specify scene IDs or all tasks must specify image files"
                )
            scene_ids.append(task.scene_id)
        scene_datas = setup_dataset_with_scene_ids(ds_path, scene_ids)

    else:
        image_files_list: list[list[ImageFile]] = []
        for task in tasks:
            if task.image_files is None:
                raise ValueError(
                    "all tasks must specify scene IDs or all tasks must specify image files"
                )
            image_files_list.append(task.image_files)
        scene_datas = setup_dataset_with_image_files(ds_path, image_files_list)

    # Apply the vessel detection model.
    detections = get_vessel_detections(ds_path, scene_datas)

    # Create and materialize windows that correspond to a crop of each detection.
    crop_windows = get_vessel_crop_windows(ds_path, detections, scene_datas)

    # Write crops and prepare the JSON data.
    json_vessels_by_task: list[list[dict[str, Any]]] = [[] for _ in tasks]
    geojson_vessels_by_task: list[list[dict[str, Any]]] = [[] for _ in tasks]
    detections_by_task: list[list[VesselDetection]] = [[] for _ in tasks]

    near_infra_filter = NearInfraFilter(
        infra_distance_threshold=INFRA_DISTANCE_THRESHOLD
    )
    raster_format = GeotiffRasterFormat()
    for detection, crop_window in zip(detections, crop_windows):
        # Apply near infra filter (True -> filter out, False -> keep)
        lon, lat = detection.get_lon_lat()
        if near_infra_filter.should_filter(lon, lat):
            continue

        # Get which task this is for.
        task_idx = detection.metadata["task_idx"]
        task = tasks[task_idx]

        if task.crop_path is not None:
            crop_upath = UPath(task.crop_path)
            crop_upath.mkdir(parents=True, exist_ok=True)

            # Get RGB crop.
            raster_dir = crop_window.get_raster_dir(
                SENTINEL2_LAYER_NAME, ["R", "G", "B"]
            )
            raster_bounds = raster_format.get_raster_bounds(raster_dir)
            image = GeotiffRasterFormat().decode_raster(raster_dir, raster_bounds)

            # And save it under the specified crop path.
            detection.crop_fname = crop_upath / f"{detection.col}_{detection.row}.png"
            with detection.crop_fname.open("wb") as f:
                Image.fromarray(image.transpose(1, 2, 0)).save(f, format="PNG")

        json_vessels_by_task[task_idx].append(detection.to_dict())
        geojson_vessels_by_task[task_idx].append(detection.to_feature())
        detections_by_task[task_idx].append(detection)

    for task_idx, (json_data, geojson_features) in enumerate(
        zip(json_vessels_by_task, geojson_vessels_by_task)
    ):
        task = tasks[task_idx]

        if task.json_path is not None:
            json_upath = UPath(task.json_path)
            json_upath.parent.mkdir(parents=True, exist_ok=True)
            with json_upath.open("w") as f:
                json.dump(json_data, f)

        if task.geojson_path is not None:
            geojson_upath = UPath(task.geojson_path)
            geojson_upath.parent.mkdir(parents=True, exist_ok=True)
            with geojson_upath.open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "properties": {},
                        "features": geojson_features,
                    },
                    f,
                )

    return detections_by_task
