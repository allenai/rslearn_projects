"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime

import numpy as np
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
from rslp.sentinel2_vessels.prom_metrics import TimerOperations, time_operation
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
from rslp.vessels import VesselAttributes, VesselDetection, VesselDetectionSource

logger = get_logger(__name__)

# Name to use in rslearn dataset for layer containing Sentinel-2 images.
SENTINEL2_LAYER_NAME = "sentinel2"

# Name of layer containing the output.
OUTPUT_LAYER_NAME = "output"

SCENE_ID_DATASET_CONFIG = "data/sentinel2_vessels/config_predict_gcp.json"
IMAGE_FILES_DATASET_CONFIG = "data/sentinel2_vessels/config_predict_local_files.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"
ATTRIBUTE_MODEL_CONFIG = "data/sentinel2_vessel_attribute/config.yaml"

SENTINEL2_RESOLUTION = 10
CROP_WINDOW_SIZE = 128

# Distance threshold for near marine infrastructure filter in km.
# 0.05 km = 50 m
INFRA_DISTANCE_THRESHOLD = 0.05

# The bands in the 10 m/pixel band set.
# The first band here is also used to determine projection/bounds in ImageFiles mode.
# B02/B03/B04 are also used to create the RGB image.
HIGH_RES_BAND_SET = ["B02", "B03", "B04", "B08"]
RGB_BAND_INDICES = (2, 1, 0)

# How much to divide B04/B03/B02 by to get 8-bit image.
RGB_NORM_FACTOR = 10


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
    # We need to override the item_specs and src_dir placeholders.
    # The src_dir is only used to store summary.json, since we require absolute paths
    # for the actual image files and they will be set directly.
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

    src_dir = ds_path / "source_dir"
    src_dir.mkdir(parents=True)
    cfg["layers"][SENTINEL2_LAYER_NAME]["data_source"]["src_dir"] = src_dir.name

    with (ds_path / "config.json").open("w") as f:
        json.dump(cfg, f)

    # Get the projection and scene bounds for each task from the TCI image.
    scene_datas: list[SceneData] = []
    for image_files in image_files_list:
        # Look for an image at the highest resolution. It is required.
        hr_fname: UPath | None = None
        for image_file in image_files:
            if image_file.bands != [HIGH_RES_BAND_SET[0]]:
                continue
            hr_fname = UPath(image_file.fname)

        if hr_fname is None:
            raise ValueError(
                f"provided list of image files does not have band {HIGH_RES_BAND_SET[0]}"
            )

        with open_rasterio_upath_reader(hr_fname) as raster:
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
    with time_operation(TimerOperations.MaterializeDataset):
        materialize_dataset(ds_path, materialize_pipeline_args)
    for window in windows:
        if not window.is_layer_completed(SENTINEL2_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have Sentinel-2 layer completed"
            )

    # Run object detector.
    with time_operation(TimerOperations.RunModelPredict):
        run_model_predict(DETECT_MODEL_CONFIG, ds_path)

    # Read the detections.
    detections: list[VesselDetection] = []
    for task_idx, (window, scene_data) in enumerate(zip(windows, scene_datas)):
        layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
        features = GeojsonVectorFormat().decode_vector(
            layer_dir, window.projection, window.bounds
        )
        for feature in features:
            geometry = feature.geometry
            score = feature.properties["score"]

            detection = VesselDetection(
                source=VesselDetectionSource.SENTINEL2,
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


def run_attribute_model(
    ds_path: UPath,
    detections: list[VesselDetection],
    scene_datas: list[SceneData],
) -> list[Window]:
    """Run the attribute prediction model.

    Args:
        ds_path: the dataset path that will be populated with new windows to apply the
            attribute model.
        detections: the detections from the detector.
        scene_datas: the list of SceneDatas.

    Returns:
        the new windows. The detections will also be updated with the predicted
            attributes.
    """
    # Avoid errors with materialize_dataset and run_model_predict when there are no
    # detections to process.
    if len(detections) == 0:
        return []

    # Create windows for applying attribute prediction model.
    group = "attribute_predict"
    windows: list[Window] = []
    for detection in detections:
        window_name = (
            f"{detection.metadata['task_idx']}_{detection.col}_{detection.row}"
        )
        window_path = Window.get_window_root(ds_path, group, window_name)
        bounds = [
            detection.col - CROP_WINDOW_SIZE // 2,
            detection.row - CROP_WINDOW_SIZE // 2,
            detection.col + CROP_WINDOW_SIZE // 2,
            detection.row + CROP_WINDOW_SIZE // 2,
        ]

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
        windows.append(window)
        detection.metadata["crop_window"] = window

        if scene_data.item:
            layer_data = WindowLayerData(
                SENTINEL2_LAYER_NAME, [[scene_data.item.serialize()]]
            )
            window.save_layer_datas({SENTINEL2_LAYER_NAME: layer_data})

    # Materialize the dataset.
    logger.info("materialize dataset")
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

    # Verify that no window is unmaterialized.
    for window in windows:
        if not window.is_layer_completed(SENTINEL2_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have materialized Sentinel-2 image"
            )

    # Run classification model.
    run_model_predict(ATTRIBUTE_MODEL_CONFIG, ds_path, groups=[group])

    # Read the results.
    for detection, window in zip(detections, windows):
        layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
        features = GeojsonVectorFormat().decode_vector(
            layer_dir, window.projection, window.bounds
        )
        properties = features[0].properties
        detection.attributes = VesselAttributes(
            length=properties["length"],
            width=properties["width"],
            speed=properties["sog"],
            heading=properties["heading"],
            vessel_type=properties["type"],
        )

    return windows


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
    if len(tasks) == 0:
        # Avoid error with below with checking tasks[0].scene_id.
        return []

    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    with time_operation(TimerOperations.SetupDataset):
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
    with time_operation(TimerOperations.GetVesselDetections):
        detections = get_vessel_detections(ds_path, scene_datas)

    # Apply the attribute prediction model.
    # This also collects vessel crop windows.
    with time_operation(TimerOperations.RunAttributeModel):
        crop_windows = run_attribute_model(ds_path, detections, scene_datas)

    # Write crops and prepare the JSON data.
    with time_operation(TimerOperations.BuildPredictionsAndCrops):
        detections_by_task = _build_predictions_and_crops(
            detections, crop_windows, tasks
        )

    for task_idx in range(0, len(tasks)):
        task = tasks[task_idx]
        detections = detections_by_task[task_idx]

        if task.json_path is not None:
            json_upath = UPath(task.json_path)
            json_upath.parent.mkdir(parents=True, exist_ok=True)
            json_data = [d.to_dict() for d in detections]
            with json_upath.open("w") as f:
                json.dump(json_data, f)

        if task.geojson_path is not None:
            geojson_upath = UPath(task.geojson_path)
            geojson_upath.parent.mkdir(parents=True, exist_ok=True)
            geojson_features = [d.to_feature() for d in detections]
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


def _build_predictions_and_crops(
    detections: list[VesselDetection],
    crop_windows: list[Window],
    tasks: list[PredictionTask],
) -> list[list[VesselDetection]]:
    detections_by_task: list[list[VesselDetection]] = [[] for _ in tasks]

    near_infra_filter = NearInfraFilter(
        infra_distance_threshold=INFRA_DISTANCE_THRESHOLD
    )
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
                SENTINEL2_LAYER_NAME,
                HIGH_RES_BAND_SET,
            )
            image = GeotiffRasterFormat().decode_raster(
                raster_dir, crop_window.projection, crop_window.bounds
            )
            rgb_image = image[RGB_BAND_INDICES, :, :]
            rgb_image = np.clip(rgb_image // RGB_NORM_FACTOR, 0, 255).astype(np.uint8)

            # And save it under the specified crop path.
            detection.crop_fname = crop_upath / f"{detection.col}_{detection.row}.png"
            with detection.crop_fname.open("wb") as f:
                Image.fromarray(rgb_image.transpose(1, 2, 0)).save(f, format="PNG")

        detections_by_task[task_idx].append(detection)
    return detections_by_task
