"""Landsat vessel prediction pipeline."""

import json
from datetime import datetime

import rasterio
import rasterio.features
import shapely
from rslearn.dataset import Window
from rslearn.utils import Projection
from upath import UPath

from rslp.utils.rslearn import materialize_dataset, run_model_predict

DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
DETECT_MODEL_CONFIG = "data/landsat_vessels/config.yaml"
CLASSIFY_MODEL_CONFIG = "landsat/recheck_landsat_labels/phase2_config.yaml"

CLASSIFY_WINDOW_SIZE = 64
"""The size of windows expected by the classifier."""


class VesselDetection:
    """A vessel detected in a Landsat scene."""

    def __init__(self, col: int, row: int, projection: Projection, score: float):
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
        """
        self.col = col
        self.row = row
        self.projection = projection
        self.score = score


def get_vessel_detections(
    ds_path: UPath,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    time_range: tuple[datetime, datetime] | None = None,
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Landsat images.

    Args:
        ds_path: the dataset path that will be populated with a new window to apply the
            detector.
        projection: the projection to apply the detector in.
        bounds: the bounds to apply the detector in.
        time_range: optional time range to apply the detector in (in case the data
            source needs an actual time range).
    """
    # Create a window for applying detector.
    group = "default"
    window_path = ds_path / "windows" / group / "default"
    Window(
        path=window_path,
        group=group,
        name="default",
        projection=projection,
        bounds=bounds,
        time_range=time_range,
    ).save()

    print("materialize dataset")
    materialize_dataset(ds_path, group=group)
    assert (window_path / "layers" / "landsat" / "B8" / "geotiff.tif").exists()

    # Run object detector.
    run_model_predict(DETECT_MODEL_CONFIG, ds_path)

    # Read the detections.
    output_fname = window_path / "layers" / "output" / "data.geojson"
    detections: list[VesselDetection] = []
    with output_fname.open() as f:
        feature_collection = json.load(f)
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
                score=score,
            )
        )

    return detections


def run_classifier(
    ds_path: UPath,
    detections: list[VesselDetection],
    time_range: tuple[datetime, datetime] | None = None,
) -> list[VesselDetection]:
    """Run the classifier to try to prune false positive detections.

    Args:
        ds_path: the dataset path that will be populated with new windows to apply the
            classifier.
        detections: the detections from the detector.
        time_range: optional time range to apply the detector in (in case the data
            source needs an actual time range).

    Returns:
        the subset of detections that pass the classifier.
    """
    # Create windows for applying classifier.
    group = "classify_predict"
    window_paths: list[UPath] = []
    for detection in detections:
        window_name = f"{detection.col}_{detection.row}"
        window_path = ds_path / "windows" / group / window_name
        bounds = [
            detection.col - CLASSIFY_WINDOW_SIZE // 2,
            detection.row - CLASSIFY_WINDOW_SIZE // 2,
            detection.col + CLASSIFY_WINDOW_SIZE // 2,
            detection.row + CLASSIFY_WINDOW_SIZE // 2,
        ]
        Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=detection.projection,
            bounds=bounds,
            time_range=time_range,
        ).save()
        window_paths.append(window_path)

    print("materialize dataset")
    materialize_dataset(ds_path, group=group)
    for window_path in window_paths:
        assert (window_path / "layers" / "landsat" / "B8" / "geotiff.tif").exists()

    # Run classification model.
    run_model_predict(CLASSIFY_MODEL_CONFIG, ds_path)

    # Read the results.
    good_detections = []
    for detection, window_path in zip(detections, window_paths):
        output_fname = window_path / "layers" / "output" / "data.geojson"
        with output_fname.open() as f:
            feature_collection = json.load(f)
        category = feature_collection[0]["properties"]["label"]
        if category == "correct":
            good_detections.append(detection)

    return good_detections


def predict_pipeline(
    image_files: dict[str, str], scratch_path: str, csv_path: str, crop_path: str
):
    """Run the Landsat vessel prediction pipeline.

    This inputs a Landsat scene (consisting of per-band GeoTIFFs) and produces the
    vessel detections. It produces a CSV containing the vessel detection locations
    along with crops of each detection.

    Args:
        image_files: map from band name like "B8" to the path of the image. The path
            will be converted to UPath so it can include protocol like gs://...
        scratch_path: directory to use to store temporary dataset.
        csv_path: path to write the CSV.
        crop_path: path to write the vessel crop images.
    """
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    # Setup the dataset configuration file with the provided image files.
    with open(DATASET_CONFIG) as f:
        cfg = json.load(f)
    item_spec = {
        "fnames": [],
        "bands": [],
    }
    for band, image_path in image_files.items():
        cfg["src_dir"] = str(UPath(image_path).parent)
        item_spec["fnames"].append(image_path)
        item_spec["bands"].append([band])
    cfg["layers"]["landsat"]["data_source"]["item_specs"] = [item_spec]

    with (ds_path / "config.json").open("w") as f:
        json.dump(cfg, f)

    # Get the projection and scene bounds from the B8 image.
    with UPath(image_files["B8"]).open("rb") as f:
        with rasterio.open(f) as raster:
            projection = Projection(raster.crs, raster.transform.a, raster.transform.e)
            left = int(raster.transform.c / projection.x_resolution)
            top = int(raster.transform.f / projection.y_resolution)
            scene_bounds = [left, top, left + raster.width, top + raster.height]

    detections = get_vessel_detections(ds_path, projection, scene_bounds)
    detections = run_classifier(ds_path, detections)


"""
then run model predict with the vessel detection model (may need to add some object detection post-processing to merge predictions or something; also stride for the patches that we read?).
then create another dataset of windows based on the vessel positions.
again run prepare/ingest/materialize (using the LocalFiles data source).
and then run model predict with the vessel classification model.
"""
