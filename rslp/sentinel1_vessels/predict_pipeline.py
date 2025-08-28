"""Sentinel-1 vessel prediction pipeline."""

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import shapely
from PIL import Image
from rslearn.config.dataset import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.data_sources.copernicus import (
    Sentinel1,
    Sentinel1Polarisation,
    Sentinel1ProductType,
)
from rslearn.dataset import Window, WindowLayerData
from rslearn.utils.fsspec import open_rasterio_upath_reader
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.log_utils import get_logger
from rslp.sentinel1_vessels.prom_metrics import TimerOperations, time_operation
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
from rslp.vessels import VesselDetection, VesselDetectionSource

logger = get_logger(__name__)

# Layer name for the Sentinel-1 image in which we want to detect vessels.
SENTINEL1_LAYER_NAME = "sentinel1"

# Layer names for historical Sentinel-1 images.
HISTORICAL_LAYER_NAME = "sentinel1_historical"

# Name of layer containing the output.
OUTPUT_LAYER_NAME = "output"

SCENE_ID_DATASET_CONFIG = "data/sentinel1_vessels/config.json"
IMAGE_FILES_DATASET_CONFIG = "data/sentinel1_vessels/config_predict_local_files.json"
DETECT_MODEL_CONFIG = "data/sentinel1_vessels/config.yaml"

RESOLUTION = 10
CROP_WINDOW_SIZE = 128
# Band order as configured in dataset.
BAND_NAMES = ["vv", "vh"]

# Factor to multiply by when converting bands to 8-bit image.
NORM_FACTOR = 1.0

# Distance threshold for near marine infrastructure filter in km.
# 0.05 km = 50 m
INFRA_DISTANCE_THRESHOLD = 0.05


@dataclass
class Sentinel1Image:
    """A scene provided for inference as file paths.

    Args:
        vv: filename for vv band.
        vh: filename for vh band.
    """

    vv: str
    vh: str


@dataclass
class PredictionTask:
    """A task to predict vessels in one Sentinel-1 scene.

    Args:
        scene_id: the Sentinel-1 scene ID. One of scene_id or image/historical1/historical2 must be set.
        image: the Sentinel1Image to detect vessels in.
        historical1: the Sentinel1Image to use as first historical image.
        historical2: the Sentinel1Image to use as second historical image.
        json_path: optional path to write the JSON of vessel detections.
        crop_path: optional path to write the vessel crop images.
        geojson_path: optional path to write GeoJSON of detections.
    """

    scene_id: str | None = None
    image: Sentinel1Image | None = None
    historical1: Sentinel1Image | None = None
    historical2: Sentinel1Image | None = None
    json_path: str | None = None
    crop_path: str | None = None
    geojson_path: str | None = None


@dataclass
class SceneData:
    """Data about the Sentinel-1 scene that the prediction should be applied on.

    This can come from scene ID or provided GeoTIFFs.
    """

    # Basic required information.
    projection: Projection
    bounds: PixelBounds

    # Optional time range -- if provided, it is passed on to the Window.
    time_range: tuple[datetime, datetime] | None = None

    # Optional Item -- if provided, we write the layer metadatas (items.json) in the
    # window to skip prepare step so that the correct item is always read.
    # This is a map from layer name to item.
    layer_to_item_groups: dict[str, list[list[Item]]] | None = None

    def get_layer_datas(self) -> dict[str, WindowLayerData]:
        """Get map that can be passed to Window.save_layer_datas."""
        assert self.layer_to_item_groups is not None
        layer_datas = {}
        for layer_name, item_groups in self.layer_to_item_groups.items():
            layer_data = WindowLayerData(
                layer_name,
                [[item.serialize() for item in group] for group in item_groups],
            )
            layer_datas[layer_name] = layer_data
        return layer_datas


def setup_dataset_with_scene_ids(
    ds_path: UPath, scene_ids: list[str]
) -> list[SceneData]:
    """Initialize an rslearn dataset for prediction with scene IDs.

    Args:
        ds_path: the dataset path to write to.
        scene_ids: list of scene IDs to apply the vessel detection model on. The scenes
            must share the same orbit direction.

    Returns:
        a list of SceneData corresponding to the list of scene IDs.
    """
    # Initialize Sentinel-1 data source so we can lookup the Item that each scene ID corresponds to.
    data_source = Sentinel1(
        product_type=Sentinel1ProductType.IW_GRDH,
        polarisation=Sentinel1Polarisation.VV_VH,
    )

    # Get the SceneData based on looking up the scene.
    scene_datas: list[SceneData] = []
    orbit_direction: str | None = None
    for scene_id in scene_ids:
        # Lookup the Sentinel-1 scene.
        product = data_source._get_product(scene_id, expand_attributes=True)
        item = data_source.get_item_by_name(scene_id)

        # Use the product to check the orbit direction.
        product_attributes = {
            attr["Name"]: attr["Value"] for attr in product["Attributes"]
        }
        product_orbit_direction = product_attributes["orbitDirection"]
        if orbit_direction is None:
            orbit_direction = product_orbit_direction
        elif orbit_direction != product_orbit_direction:
            raise ValueError("all products must have the same orbit direction")

        # Get the projection to use for the window.
        wgs84_geom = item.geometry.to_projection(WGS84_PROJECTION)
        projection = get_utm_ups_projection(
            wgs84_geom.shp.centroid.x,
            wgs84_geom.shp.centroid.y,
            RESOLUTION,
            -RESOLUTION,
        )
        dst_geom = item.geometry.to_projection(projection)
        bounds = (
            int(dst_geom.shp.bounds[0]),
            int(dst_geom.shp.bounds[1]),
            int(dst_geom.shp.bounds[2]),
            int(dst_geom.shp.bounds[3]),
        )

        # Find historical images to use. We look for the ones with the best overlap
        # against wgs84_geom above. We don't use rslearn prepare for these images
        # because it will find too many images in mosaic mode, or not the best
        # overlapping image in intersect/contain mode.
        historical_layer_item_groups: list[list[Item]] = []
        for hist_group_idx, hist_time_offset in enumerate(
            [timedelta(days=-60), timedelta(days=-90)]
        ):
            hist_time_range = (
                wgs84_geom.time_range[0] + hist_time_offset,
                wgs84_geom.time_range[0] + hist_time_offset + timedelta(days=30),
            )
            hist_query_geometry = STGeometry(
                wgs84_geom.projection,
                wgs84_geom.shp,
                hist_time_range,
            )

            logger.debug(
                f"Searching for historical images in group {hist_group_idx} corresponding to time range {hist_time_range}"
            )
            result_item_groups = data_source.get_items(
                [hist_query_geometry],
                QueryConfig(max_matches=9999, space_mode=SpaceMode.INTERSECTS),
            )[0]
            logger.debug(
                f"Got {len(result_item_groups)} options for the historical image"
            )
            best_hist_option: Item | None = None
            best_intersect_area = 0
            for group in result_item_groups:
                hist_option = group[0]
                hist_option_geom = hist_option.geometry.to_projection(WGS84_PROJECTION)
                intersect_area = hist_option_geom.shp.intersection(wgs84_geom.shp).area
                if intersect_area > best_intersect_area:
                    best_hist_option = hist_option
                    best_intersect_area = intersect_area

            if best_hist_option is None:
                raise ValueError(
                    f"No historical image option found matching geometry {hist_query_geometry}"
                )

            logger.debug(
                f"Got scene {best_hist_option.name} with intersect area {best_intersect_area} / {wgs84_geom.shp.area}"
            )
            historical_layer_item_groups.append([best_hist_option])

        scene_data = SceneData(
            projection=projection,
            bounds=bounds,
            time_range=item.geometry.time_range,
            layer_to_item_groups={
                SENTINEL1_LAYER_NAME: [[item]],
                HISTORICAL_LAYER_NAME: historical_layer_item_groups,
            },
        )
        scene_datas.append(scene_data)

    # Now that we know the orbit direction, write the rslearn dataset configuration
    # file (which is set up to get Sentinel-1 images from AWS.)
    with open(SCENE_ID_DATASET_CONFIG) as f:
        ds_config = json.load(f)
    ds_config["layers"][HISTORICAL_LAYER_NAME]["data_source"]["orbit_direction"] = (
        orbit_direction
    )
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_config, f)

    return scene_datas


def setup_dataset_with_image_files(
    ds_path: UPath, tasks: list[PredictionTask]
) -> list[SceneData]:
    """Initialize an rslearn dataset for prediction with image files.

    Args:
        ds_path: the dataset path to write to.
        tasks: the list of prediction tasks that have image/historical1/historical2 set.

    Returns:
        a list of SceneData corresponding to the specified images.
    """
    # Write dataset configuration file.
    # We need to override the item_specs and src_dir placeholders.
    # The src_dir is only used to store summary.json, since we require absolute paths
    # for the actual image files and they will be set directly.
    with open(IMAGE_FILES_DATASET_CONFIG) as f:
        cfg = json.load(f)
    item_specs = []
    for task in tasks:
        for image in [task.image, task.historical1, task.historical2]:
            assert image is not None
            # Ensure the filename is a URI so it won't be treated as a relative path.
            item_specs.append(
                {
                    "fnames": [
                        UPath(image.vv).absolute().as_uri(),
                        UPath(image.vh).absolute().as_uri(),
                    ],
                    "bands": [["vv"], ["vh"]],
                }
            )
    cfg["layers"][SENTINEL1_LAYER_NAME]["data_source"]["item_specs"] = item_specs

    src_dir = ds_path / "source_dir"
    src_dir.mkdir(parents=True)
    cfg["layers"][SENTINEL1_LAYER_NAME]["data_source"]["src_dir"] = src_dir.name

    with (ds_path / "config.json").open("w") as f:
        json.dump(cfg, f)

    # Get the projection and scene bounds for each task from the vv image.
    # We need to do it based on the ground control points though.
    scene_datas: list[SceneData] = []
    for task in tasks:
        assert task.image is not None
        with open_rasterio_upath_reader(task.image.vv) as raster:
            gcps, gcp_crs = raster.gcps
            xs = [gcp.x for gcp in gcps]
            ys = [gcp.y for gcp in gcps]
            src_geom = STGeometry(
                Projection(gcp_crs, 1, 1),
                shapely.box(min(xs), min(ys), max(xs), max(ys)),
                None,
            )
            wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
            dst_proj = get_utm_ups_projection(
                wgs84_geom.shp.centroid.x,
                wgs84_geom.shp.centroid.y,
                RESOLUTION,
                -RESOLUTION,
            )
            dst_geom = src_geom.to_projection(dst_proj)

            scene_datas.append(
                SceneData(
                    projection=dst_proj,
                    bounds=dst_geom.shp.bounds,
                )
            )

    return scene_datas


def get_vessel_detections(
    ds_path: UPath,
    scene_datas: list[SceneData],
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Sentinel-1 images.

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

        if scene_data.layer_to_item_groups is not None:
            window.save_layer_datas(scene_data.get_layer_datas())

    # Materialize the windows.
    logger.info("Materialize dataset for Sentinel-1 Vessel Detection")
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
        if not window.is_layer_completed(SENTINEL1_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have Sentinel-1 layer completed"
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
                source=VesselDetectionSource.SENTINEL1,
                col=int(geometry.shp.centroid.x),
                row=int(geometry.shp.centroid.y),
                projection=geometry.projection,
                score=score,
                # We use this metadata to keep track of which window/scene each
                # detection came from.
                metadata={"task_idx": task_idx},
            )

            if scene_data.layer_to_item_groups is not None:
                item = scene_data.layer_to_item_groups[SENTINEL1_LAYER_NAME][0][0]
                detection.scene_id = item.name
                detection.ts = item.geometry.time_range[0]

            detections.append(detection)

    return detections


def get_vessel_crop_windows(
    ds_path: UPath,
    detections: list[VesselDetection],
    scene_datas: list[SceneData],
) -> list[Window]:
    """Create a window for each vessel to obtain a cropped image for it.

    Args:
        ds_path: the dataset path that will be populated with new windows.
        detections: the detections from the detector.
        scene_datas: the list of SceneDatas.

    Returns:
        the new windows.
    """
    # Avoid errors with materialize_dataset when there are no detections to process.
    if len(detections) == 0:
        return []

    # Create windows for collecting the cropped images.
    group = "crops"
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

        # task_idx metadata is always set in sentinel1_vessels.
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

        if scene_data.layer_to_item_groups is not None:
            window.save_layer_datas(scene_data.get_layer_datas())

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
        if not window.is_layer_completed(SENTINEL1_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have materialized Sentinel-1 image"
            )

    return windows


def predict_pipeline(
    tasks: list[PredictionTask],
    scratch_path: str | None = None,
) -> list[list[VesselDetection]]:
    """Run the Sentinel-1 vessel prediction pipeline.

    Given a Sentinel-1 scene ID, the pipeline produces the vessel detections.
    Specifically, it outputs a list of the vessel detection locations along with crops
    of each detection.

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

    if scratch_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name

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
            scene_datas = setup_dataset_with_image_files(ds_path, tasks)

    # Apply the vessel detection model.
    with time_operation(TimerOperations.GetVesselDetections):
        detections = get_vessel_detections(ds_path, scene_datas)

    # Apply the attribute prediction model.
    # This also collects vessel crop windows.
    with time_operation(TimerOperations.RunAttributeModel):
        crop_windows = get_vessel_crop_windows(ds_path, detections, scene_datas)

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

            # Get vv and vh crops.
            crop_fnames = {}
            raster_dir = crop_window.get_raster_dir(
                SENTINEL1_LAYER_NAME,
                BAND_NAMES,
            )
            image = GeotiffRasterFormat().decode_raster(
                raster_dir, crop_window.projection, crop_window.bounds
            )
            for band_idx, band_name in enumerate(BAND_NAMES):
                band_image = np.clip(
                    image[band_idx, :, :] * NORM_FACTOR, 0, 255
                ).astype(np.uint8)

                # And save it under the specified crop path.
                crop_fnames[band_name] = (
                    crop_upath / f"{detection.col}_{detection.row}_{band_name}.png"
                )
                with crop_fnames[band_name].open("wb") as f:
                    Image.fromarray(band_image).save(f, format="PNG")

        detections_by_task[task_idx].append(detection)
    return detections_by_task
