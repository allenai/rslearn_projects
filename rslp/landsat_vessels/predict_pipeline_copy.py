"""Landsat vessel prediction pipeline."""

import json
import tempfile
from datetime import datetime

import dateutil.parser
import shapely
from rasterio.crs import CRS
from rslearn.data_sources import Item, data_source_from_config
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils import Projection
from typing_extensions import TypedDict
from upath import UPath

from rslp.utils.rslearn import materialize_dataset, run_model_predict

LANDSAT_LAYER_NAME = "landsat"
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]
LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"
DETECT_MODEL_CONFIG = "data/landsat_vessels/config.yaml"
CLASSIFY_MODEL_CONFIG = "landsat/recheck_landsat_labels/phase123_config.yaml"
LANDSAT_RESOLUTION = 15
CLASSIFY_WINDOW_SIZE = 64
INFRA_DISTANCE_THRESHOLD = 0.1  # unit: km, 100 meters


class VesselDetection:
    """A vessel detected in a Landsat scene."""

    def __init__(
        self,
        col: int,
        row: int,
        projection: Projection,
        time_range: tuple[datetime, datetime],
        score: float,
        crop_window_dir: UPath | None = None,
    ) -> None:
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            time_range: the time range of the scene.
            score: confidence score from object detector.
            crop_window_dir: the path to the window used for classifying the crop.
        """
        self.col = col
        self.row = row
        self.projection = projection
        self.time_range = time_range
        self.score = score
        self.crop_window_dir = crop_window_dir


class FormattedPrediction(TypedDict):
    """Formatted prediction for a single vessel detection."""

    latitude: float
    longitude: float
    score: float
    rgb_fname: str
    b8_fname: str


def get_vessel_detections(
    ds_path: UPath,
    window_name: str,
) -> tuple[list[VesselDetection], str]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Landsat images.

    Args:
        ds_path: the dataset path that will be populated with a new window to apply the
            detector.
        window_name: the name of the window to apply the detector to.
        projection: the projection to apply the detector in.
        bounds: the bounds to apply the detector in.
        time_range: optional time range to apply the detector in (in case the data
            source needs an actual time range).
        item: only ingest this item. This is set if we are getting the scene directly
            from a Landsat data source, not local file.
    """
    # Create a window for applying detector.
    group = "labels_utm"
    window_path = ds_path / "windows" / group / window_name

    # Read the detections.
    output_fname = window_path / "layers" / "output" / "data.geojson"
    metadata_fname = window_path / "metadata.json"
    items_fname = window_path / "items.json"
    detections: list[VesselDetection] = []
    with output_fname.open() as f:
        feature_collection = json.load(f)
    with metadata_fname.open() as f:
        metadata = json.load(f)
    with items_fname.open() as f:
        items = json.load(f)
    time_range = (
        dateutil.parser.isoparse(metadata["time_range"][0]),
        dateutil.parser.isoparse(metadata["time_range"][1]),
    )

    # get first name from this file: [{"layer_name": "landsat", "serialized_item_groups": [[{"name": "LC08_L1TP_022047_20211224_20211230_02_T1", "geometry": {"projection": {"crs": "EPSG:4326", "x_resolution": 1, "y_resolution": 1}, "shp": "POLYGON ((-94.01605 19.83281, -91.83882 19.83192, -91.85315 17.72833, -94.00351 17.72911, -94.01605 19.83281))", "time_range": ["2021-12-24T16:35:35.175076+00:00", "2021-12-24T16:35:35.175076+00:00"]}, "blob_path": "collection02/level-1/standard/oli-tirs/2021/022/047/LC08_L1TP_022047_20211224_20211230_02_T1/LC08_L1TP_022047_20211224_20211230_02_T1_", "cloud_cover": "4.02"}]], "materialized": false}]
    scene_id = items[0]["serialized_item_groups"][0][0]["name"]

    # Get projection directly from the geojson
    crs = CRS.from_string(feature_collection["properties"]["crs"])
    x_resolution = feature_collection["properties"]["x_resolution"]
    y_resolution = feature_collection["properties"]["y_resolution"]
    projection = Projection(
        crs=crs,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
    )
    for feature in feature_collection["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        col = int(shp.centroid.x)
        row = int(shp.centroid.y)
        score = feature["properties"]["score"]
        detections.append(
            VesselDetection(
                col=col,
                row=row,
                projection=projection,
                time_range=time_range,
                score=score,
            )
        )

    return detections, scene_id


def run_classifier(
    ds_path: UPath,
    detections: list[VesselDetection],
    window_name: str,
    item: Item | None = None,
) -> list[VesselDetection]:
    """Run the classifier to try to prune false positive detections.

    Args:
        ds_path: the dataset path that will be populated with new windows to apply the
            classifier.
        detections: the detections from the detector.
        window_name: the name of the window to apply the classifier to.
        item: only ingest this item. This is set if we are getting the scene directly
            from a Landsat data source, not local file.

    Returns:
        the subset of detections that pass the classifier.
    """
    # Avoid error materializing empty group.
    if len(detections) == 0:
        return []

    # Create windows for applying classifier.
    group = "classify_predict"
    window_paths: list[UPath] = []
    for detection in detections:
        window_name_full = (
            f"{window_name}_{detection.col}_{detection.row}"  # add a window name here!!
        )
        window_path = ds_path / "windows" / group / window_name_full
        detection.crop_window_dir = window_path
        bounds = [
            detection.col - CLASSIFY_WINDOW_SIZE // 2,
            detection.row - CLASSIFY_WINDOW_SIZE // 2,
            detection.col + CLASSIFY_WINDOW_SIZE // 2,
            detection.row + CLASSIFY_WINDOW_SIZE // 2,
        ]
        window = Window(
            path=window_path,
            group=group,
            name=window_name_full,
            projection=detection.projection,
            bounds=bounds,
            time_range=detection.time_range,
        )
        window.save()
        window_paths.append(window_path)

        if item:
            layer_data = WindowLayerData(LANDSAT_LAYER_NAME, [[item.serialize()]])
            window.save_layer_datas(dict(LANDSAT_LAYER_NAME=layer_data))

    print("materialize dataset")
    materialize_dataset(ds_path, group=group)
    for window_path in window_paths:
        assert (
            window_path / "layers" / LANDSAT_LAYER_NAME / "B8" / "geotiff.tif"
        ).exists()

    # Run classification model.
    run_model_predict(CLASSIFY_MODEL_CONFIG, ds_path)  # output for all windows

    # Read the results.
    good_detections = []
    for detection, window_path in zip(detections, window_paths):
        output_fname = window_path / "layers" / "output" / "data.geojson"
        with output_fname.open() as f:
            feature_collection = json.load(f)
        category = feature_collection["features"][0]["properties"]["label"]
        if category == "correct":
            good_detections.append(detection)

    return good_detections


# def load_detection_gt(ds_path: UPath, window_name: str) -> list[VesselDetection]:


def predict_pipeline(
    scratch_path: str | None = None,
) -> None:
    """Run the Landsat vessel prediction pipeline.

    This inputs a Landsat scene (consisting of per-band GeoTIFFs) and produces the
    vessel detections. It produces a CSV containing the vessel detection locations
    along with crops of each detection.

    Args:
        scratch_path: directory to use to store temporary dataset.
        crop_path: path to write the vessel crop images.
    """
    if scratch_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name
    else:
        tmp_dir = None

    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    # # Loop through each detection window
    # group = "labels_utm"
    # window_path = ds_path / "windows" / group

    visualization_dir = "/home/yawenz/visualizations/"
    # for png in visualization_dir:, get the name of the file as window_name
    window_names = [
        png.name.split(".")[0]
        for png in UPath(visualization_dir).iterdir()
        if png.is_file()
    ]

    # # Get all windows under the window_path
    # window_names = [window.name for window in window_path.iterdir() if window.is_dir()]
    for window_name in window_names:
        print(f"processing window {window_name}")
        detections, scene_id = get_vessel_detections(ds_path, window_name)
        # Get the projection and scene bounds using the Landsat data source.
        dataset = Dataset(ds_path)
        data_source: LandsatOliTirs = data_source_from_config(
            dataset.layers[LANDSAT_LAYER_NAME], dataset.path
        )
        item = data_source.get_item_by_name(scene_id)
        detections = run_classifier(ds_path, detections, window_name, item)


if __name__ == "__main__":
    # TODO: remove temp path
    predict_pipeline(
        "gs://rslearn-eai/datasets/landsat_vessel_detection/detector/dataset_20240924/"
    )
