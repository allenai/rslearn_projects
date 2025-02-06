"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from typing import Any

from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item, data_source_from_config
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.dataset import Dataset, Window, WindowLayerData
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

DATASET_CONFIG = "data/sentinel2_vessels/config.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"

SENTINEL2_RESOLUTION = 10
CROP_WINDOW_SIZE = 64

# Distance threshold for near marine infrastructure filter in km.
# 0.05 km = 50 m
INFRA_DISTANCE_THRESHOLD = 0.05


class PredictionTask:
    """A task to predict vessels in one Sentinel-2 scene."""

    def __init__(
        self,
        scene_id: str,
        json_path: str | None = None,
        crop_path: str | None = None,
        geojson_path: str | None = None,
    ):
        """Create a new PredictionTask.

        Args:
            scene_id: the Sentinel-2 scene ID.
            json_path: path to write the JSON of vessel detections.
            crop_path: path to write the vessel crop images.
            geojson_path: path to write GeoJSON of detections.
        """
        self.scene_id = scene_id
        self.json_path = json_path
        self.crop_path = crop_path
        self.geojson_path = geojson_path


def get_vessel_detections(
    ds_path: UPath,
    items: list[Item],
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Sentinel-2 images.

    Args:
        ds_path: the dataset path that will be populated with a new window to apply the
            detector.
        items: the items (scenes) in Sentinel-2 data source to apply the detector on.
    """
    # Create a window corresponding to each item.
    windows: list[Window] = []
    for item in items:
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

        group = "detector_predict"
        window_path = ds_path / "windows" / group / item.name
        window = Window(
            path=window_path,
            group=group,
            name=item.name,
            projection=projection,
            bounds=bounds,
            time_range=item.geometry.time_range,
        )
        window.save()
        windows.append(window)

        layer_data = WindowLayerData(SENTINEL2_LAYER_NAME, [[item.serialize()]])
        window.save_layer_datas({SENTINEL2_LAYER_NAME: layer_data})

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
    for window in windows:
        layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
        features = GeojsonVectorFormat().decode_vector(layer_dir, window.bounds)
        for feature in features:
            geometry = feature.geometry
            score = feature.properties["score"]
            detections.append(
                VesselDetection(
                    source=SENTINEL2_SOURCE,
                    scene_id=window.name,
                    col=int(geometry.shp.centroid.x),
                    row=int(geometry.shp.centroid.y),
                    projection=geometry.projection,
                    score=score,
                    ts=window.time_range[0],
                )
            )

    return detections


def get_vessel_crop_windows(
    ds_path: UPath, detections: list[VesselDetection], items_by_scene: dict[str, Item]
) -> list[Window]:
    """Create a window for each vessel to obtain a cropped image for it.

    Args:
        ds_path: the rslearn dataset path (same one used for object detector -- we will
            put the crop windows in a different group).
        detections: list of vessel detections.
        items_by_scene: scene ID -> Item map, same one used to get the big image to run
            detector over.

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

        # scene_id attribute is always set in sentinel2_vessels.
        assert detection.scene_id is not None
        item = items_by_scene[detection.scene_id]
        window = Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=detection.projection,
            bounds=bounds,
            time_range=item.geometry.time_range,
        )
        window.save()

        layer_data = WindowLayerData(SENTINEL2_LAYER_NAME, [[item.serialize()]])
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


def predict_pipeline(tasks: list[PredictionTask], scratch_path: str) -> None:
    """Run the Sentinel-2 vessel prediction pipeline.

    Given a Sentinel-2 scene ID, the pipeline produces the vessel detections.
    Specifically, it outputs a CSV containing the vessel detection locations along with
    crops of each detection.

    Args:
        tasks: prediction tasks to execute.
        scratch_path: directory to use to store temporary dataset.
    """
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    # Write dataset configuration file (which is set up to get Sentinel-2 images from
    # GCP.)
    with open(DATASET_CONFIG, "rb") as src:
        with (ds_path / "config.json").open("wb") as dst:
            shutil.copyfileobj(src, dst)

    # Determine the bounds and timestamp of this scene using the data source.
    dataset = Dataset(ds_path)
    data_source: Sentinel2 = data_source_from_config(
        dataset.layers[SENTINEL2_LAYER_NAME], dataset.path
    )
    items_by_scene: dict[str, Item] = {}
    tasks_by_scene: dict[str, PredictionTask] = {}
    for task in tasks:
        item = data_source.get_item_by_name(task.scene_id)
        items_by_scene[item.name] = item
        tasks_by_scene[item.name] = task

        # Also make sure crop directory exists here.
        if task.crop_path is not None:
            UPath(task.crop_path).mkdir(parents=True, exist_ok=True)

    # Apply the vessel detection model.
    detections = get_vessel_detections(ds_path, list(items_by_scene.values()))

    # Create and materialize windows that correspond to a crop of each detection.
    crop_windows = get_vessel_crop_windows(ds_path, detections, items_by_scene)

    # Write JSON and crops.
    json_vessels_by_scene: dict[str, list[dict[str, Any]]] = {}
    geojson_vessels_by_scene: dict[str, list[dict[str, Any]]] = {}
    # Populate the dict so all JSONs are written including empty ones (this way their
    # presence can be used to check for task completion).
    for scene_id in tasks_by_scene.keys():
        json_vessels_by_scene[scene_id] = []
        geojson_vessels_by_scene[scene_id] = []

    near_infra_filter = NearInfraFilter(
        infra_distance_threshold=INFRA_DISTANCE_THRESHOLD
    )
    raster_format = GeotiffRasterFormat()
    for detection, crop_window in zip(detections, crop_windows):
        # Apply near infra filter (True -> filter out, False -> keep)
        lon, lat = detection.get_lon_lat()
        if near_infra_filter.should_filter(lon, lat):
            continue

        assert detection.scene_id is not None
        scene_id = detection.scene_id

        if tasks_by_scene[scene_id].crop_path is not None:
            crop_upath = UPath(tasks_by_scene[scene_id].crop_path)

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

        json_vessels_by_scene[scene_id].append(detection.to_dict())
        geojson_vessels_by_scene[scene_id].append(detection.to_feature())

    for scene_id, json_data in json_vessels_by_scene.items():
        if tasks_by_scene[scene_id].json_path is not None:
            json_upath = UPath(tasks_by_scene[scene_id].json_path)
            with json_upath.open("w") as f:
                json.dump(json_data, f)

    for scene_id, geojson_features in geojson_vessels_by_scene.items():
        if tasks_by_scene[scene_id].geojson_path is not None:
            geojson_upath = UPath(tasks_by_scene[scene_id].geojson_path)
            with geojson_upath.open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "properties": {},
                        "features": geojson_features,
                    },
                    f,
                )
