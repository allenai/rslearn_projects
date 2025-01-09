"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from datetime import datetime
from typing import Any

import rasterio
import shapely
from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item, data_source_from_config
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
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

logger = get_logger(__name__)

SENTINEL2_LAYER_NAME = "sentinel2"
DATASET_CONFIG = "data/sentinel2_vessels/config.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"
SENTINEL2_RESOLUTION = 10
CROP_WINDOW_SIZE = 64
INFRA_DISTANCE_THRESHOLD = 0.05  # unit: km, 50 meters


class PredictionTask:
    """A task to predict vessels in one Sentinel-2 scene."""

    def __init__(self, scene_id: str, json_path: str, crop_path: str):
        """Create a new PredictionTask.

        Args:
            scene_id: the Sentinel-2 scene ID.
            json_path: path to write the JSON of vessel detections.
            crop_path: path to write the vessel crop images.
        """
        self.scene_id = scene_id
        self.json_path = json_path
        self.crop_path = crop_path


class VesselDetection:
    """A vessel detected in a Sentinel-2 window."""

    def __init__(
        self,
        scene_id: str,
        col: int,
        row: int,
        projection: Projection,
        score: float,
        ts: datetime,
        crop_window_dir: UPath | None = None,
    ) -> None:
        """Create a new VesselDetection.

        Args:
            scene_id: the scene ID that the vessel was detected in.
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
            ts: datetime fo the window.
            crop_window_dir: the crop window directory.
        """
        self.scene_id = scene_id
        self.col = col
        self.row = row
        self.projection = projection
        self.score = score
        self.ts = ts
        self.crop_window_dir = crop_window_dir


# TODO: make a simple class to store bounds
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
        window.save_layer_datas(dict(SENTINEL2_LAYER_NAME=layer_data))

    logger.info("Materialize dataset for Sentinel-2 Vessel Detection")
    apply_windows_args = ApplyWindowsArgs(group=group, workers=1)
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
        assert (
            window.path / "layers" / SENTINEL2_LAYER_NAME / "R_G_B" / "geotiff.tif"
        ).exists()

    # Run object detector.
    run_model_predict(DETECT_MODEL_CONFIG, ds_path)

    # Read the detections.
    detections: list[VesselDetection] = []
    for window in windows:
        output_fname = window.path / "layers" / "output" / "data.geojson"
        with output_fname.open() as f:
            feature_collection = json.load(f)
        for feature in feature_collection["features"]:
            shp = shapely.geometry.shape(feature["geometry"])
            col = int(shp.centroid.x)
            row = int(shp.centroid.y)
            score = feature["properties"]["score"]
            detections.append(
                VesselDetection(
                    scene_id=window.name,
                    col=col,
                    row=row,
                    projection=projection,
                    score=score,
                    ts=window.time_range[0],
                )
            )

    return detections


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

    detections = get_vessel_detections(ds_path, list(items_by_scene.values()))

    # Create windows just to collect crops for each detection.
    group = "crops"
    window_paths: list[UPath] = []
    for detection in detections:
        window_name = f"{detection.scene_id}_{detection.col}_{detection.row}"
        window_path = ds_path / "windows" / group / window_name
        detection.crop_window_dir = window_path
        bounds = (
            detection.col - CROP_WINDOW_SIZE // 2,
            detection.row - CROP_WINDOW_SIZE // 2,
            detection.col + CROP_WINDOW_SIZE // 2,
            detection.row + CROP_WINDOW_SIZE // 2,
        )

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
        window.save_layer_datas(dict(SENTINEL2_LAYER_NAME=layer_data))

        window_paths.append(window_path)

    apply_windows_args = ApplyWindowsArgs(group=group, workers=1)
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
    if len(detections) > 0:
        materialize_dataset(ds_path, materialize_pipeline_args)

    # Write JSON and crops.
    json_vessels_by_scene: dict[str, list[dict[str, Any]]] = {}
    # Populate the dict so all JSONs are written including empty ones (this way their
    # presence can be used to check for task completion).
    for scene_id in tasks_by_scene.keys():
        json_vessels_by_scene[scene_id] = []

    near_infra_filter = NearInfraFilter(
        infra_distance_threshold=INFRA_DISTANCE_THRESHOLD
    )
    for detection, crop_window_path in zip(detections, window_paths):
        # Get longitude/latitude.
        src_geom = STGeometry(
            detection.projection, shapely.Point(detection.col, detection.row), None
        )
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        lon = dst_geom.shp.x
        lat = dst_geom.shp.y

        # Apply near infra filter (True -> filter out, False -> keep)
        if near_infra_filter.should_filter(lat, lon):
            continue

        scene_id = detection.scene_id
        crop_upath = UPath(tasks_by_scene[scene_id].crop_path)

        # Get RGB crop.
        image_fname = (
            crop_window_path / "layers" / SENTINEL2_LAYER_NAME / "R_G_B" / "geotiff.tif"
        )
        with image_fname.open("rb") as f:
            with rasterio.open(f) as src:
                image = src.read()
        crop_fname = crop_upath / f"{detection.col}_{detection.row}.png"
        with crop_fname.open("wb") as f:
            Image.fromarray(image.transpose(1, 2, 0)).save(f, format="PNG")

        if scene_id not in json_vessels_by_scene:
            json_vessels_by_scene[scene_id] = []

        json_vessels_by_scene[scene_id].append(
            dict(
                longitude=lon,
                latitude=lat,
                score=detection.score,
                ts=detection.ts.isoformat(),
                scene_id=scene_id,
                crop_fname=str(crop_fname),
            )
        )

    for scene_id, json_data in json_vessels_by_scene.items():
        json_upath = UPath(tasks_by_scene[scene_id].json_path)
        with json_upath.open("w") as f:
            json.dump(json_data, f)
