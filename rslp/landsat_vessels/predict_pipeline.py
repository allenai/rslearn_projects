"""Landsat vessel prediction pipeline."""

import json
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import rasterio
import rasterio.features
import shapely
from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import data_source_from_config
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from typing_extensions import TypedDict
from upath import UPath

from rslp.utils.rslearn import materialize_dataset, run_model_predict

LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"
DETECT_MODEL_CONFIG = "data/landsat_vessels/config.yaml"
CLASSIFY_MODEL_CONFIG = "landsat/recheck_landsat_labels/phase123_config.yaml"
LANDSAT_RESOLUTION = 15

CLASSIFY_WINDOW_SIZE = 64


class VesselDetection:
    """A vessel detected in a Landsat scene."""

    def __init__(
        self,
        col: int,
        row: int,
        projection: Projection,
        score: float,
        crop_window_dir: UPath | None = None,
    ) -> None:
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
            crop_window_dir: the path to the window used for classifying the crop.
        """
        self.col = col
        self.row = row
        self.projection = projection
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
    # Avoid error materializing empty group.
    if len(detections) == 0:
        return []

    # Create windows for applying classifier.
    group = "classify_predict"
    window_paths: list[UPath] = []
    for detection in detections:
        window_name = f"{detection.col}_{detection.row}"
        window_path = ds_path / "windows" / group / window_name
        detection.crop_window_dir = window_path
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
        category = feature_collection["features"][0]["properties"]["label"]
        if category == "correct":
            good_detections.append(detection)

    return good_detections


def predict_pipeline(
    crop_path: str | None = None,
    scratch_path: str | None = None,
    json_path: str | None = None,
    image_files: dict[str, str] | None = None,
    scene_id: str | None = None,
) -> list[FormattedPrediction]:
    """Run the Landsat vessel prediction pipeline.

    This inputs a Landsat scene (consisting of per-band GeoTIFFs) and produces the
    vessel detections. It produces a CSV containing the vessel detection locations
    along with crops of each detection.

    Args:
        scratch_path: directory to use to store temporary dataset.
        json_path: path to write vessel detections as JSON file.
        crop_path: path to write the vessel crop images.
        image_files: map from band name like "B8" to the path of the image. The path
            will be converted to UPath so it can include protocol like gs://...
        scene_id: Landsat scene ID. Exactly one of image_files or scene_id should be
            specified.
    """
    start_time = time.time()  # Start the timer
    time_profile = {}

    if scratch_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name
    else:
        tmp_dir = None

    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    if image_files:
        # Setup the dataset configuration file with the provided image files.
        with open(LOCAL_FILES_DATASET_CONFIG) as f:
            cfg = json.load(f)
        item_spec: dict = {
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
                projection = Projection(
                    raster.crs, raster.transform.a, raster.transform.e
                )
                left = int(raster.transform.c / projection.x_resolution)
                top = int(raster.transform.f / projection.y_resolution)
                scene_bounds = [left, top, left + raster.width, top + raster.height]

        time_range = None

    else:
        with open(AWS_DATASET_CONFIG) as f:
            cfg = json.load(f)
        with (ds_path / "config.json").open("w") as f:
            json.dump(cfg, f)

        # Get the projection and scene bounds using the Landsat data source.
        dataset = Dataset(ds_path)
        data_source: LandsatOliTirs = data_source_from_config(
            dataset.layers["landsat"], dataset.path
        )
        item = data_source.get_item_by_name(scene_id)
        wgs84_geom = item.geometry.to_projection(WGS84_PROJECTION)
        projection = get_utm_ups_projection(
            wgs84_geom.shp.centroid.x,
            wgs84_geom.shp.centroid.y,
            LANDSAT_RESOLUTION,
            -LANDSAT_RESOLUTION,
        )
        dst_geom = item.geometry.to_projection(projection)
        scene_bounds = [int(value) for value in dst_geom.shp.bounds]
        time_range = (
            dst_geom.time_range[0] - timedelta(minutes=30),
            dst_geom.time_range[1] + timedelta(minutes=30),
        )

    time_profile["setup"] = time.time() - start_time

    # Run pipeline.
    step_start_time = time.time()
    print("run detector")
    detections = get_vessel_detections(
        ds_path,
        projection,
        scene_bounds,  # type: ignore
        time_range=time_range,
    )
    time_profile["get_vessel_detections"] = time.time() - step_start_time

    step_start_time = time.time()
    print("run classifier")
    detections = run_classifier(ds_path, detections, time_range=time_range)
    time_profile["run_classifier"] = time.time() - step_start_time

    # Write JSON and crops.
    step_start_time = time.time()
    if crop_path:
        crop_upath = UPath(crop_path)
        crop_upath.mkdir(parents=True, exist_ok=True)

    json_data = []
    for idx, detection in enumerate(detections):
        # Load crops from the window directory.
        images = {}
        if detection.crop_window_dir is None:
            raise ValueError("Crop window directory is None")
        for band in ["B2", "B3", "B4", "B8"]:
            image_fname = (
                detection.crop_window_dir / "layers" / "landsat" / band / "geotiff.tif"
            )
            with image_fname.open("rb") as f:
                with rasterio.open(f) as src:
                    images[band] = src.read(1)

        # Apply simple pan-sharpening for the RGB.
        # This is just linearly scaling RGB bands to add up to B8, which is captured at
        # a higher resolution.
        for band in ["B2", "B3", "B4"]:
            sharp = images[band].astype(np.int32)
            sharp = sharp.repeat(repeats=2, axis=0).repeat(repeats=2, axis=1)
            images[band + "_sharp"] = sharp
        total = np.clip(
            (images["B2_sharp"] + images["B3_sharp"] + images["B4_sharp"]) // 3, 1, 255
        )
        for band in ["B2", "B3", "B4"]:
            images[band + "_sharp"] = np.clip(
                images[band + "_sharp"] * images["B8"] // total, 0, 255
            ).astype(np.uint8)
        rgb = np.stack(
            [images["B4_sharp"], images["B3_sharp"], images["B2_sharp"]], axis=2
        )

        if crop_path:
            rgb_fname = crop_upath / f"{idx}_rgb.png"
            with rgb_fname.open("wb") as f:
                Image.fromarray(rgb).save(f, format="PNG")

            b8_fname = crop_upath / f"{idx}_b8.png"
            with b8_fname.open("wb") as f:
                Image.fromarray(images["B8"]).save(f, format="PNG")
        else:
            rgb_fname = ""
            b8_fname = ""

        # Get longitude/latitude.
        src_geom = STGeometry(
            detection.projection, shapely.Point(detection.col, detection.row), None
        )
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        lon = dst_geom.shp.x
        lat = dst_geom.shp.y

        json_data.append(
            FormattedPrediction(
                longitude=lon,
                latitude=lat,
                score=detection.score,
                rgb_fname=rgb_fname,
                b8_fname=b8_fname,
            ),
        )

    time_profile["write_json_and_crops"] = time.time() - step_start_time

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    time_profile["total"] = elapsed_time

    # Clean up any temporary directories.
    if tmp_dir:
        tmp_dir.cleanup()

    if json_path:
        json_upath = UPath(json_path)
        with json_upath.open("w") as f:
            json.dump(json_data, f)

    print(f"Prediction pipeline completed in {elapsed_time:.2f} seconds")
    for step, duration in time_profile.items():
        print(f"{step} took {duration:.2f} seconds")

    return json_data
