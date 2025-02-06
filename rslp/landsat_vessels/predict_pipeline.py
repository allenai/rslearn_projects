"""Landsat vessel prediction pipeline."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import rasterio
import rasterio.features
from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item, data_source_from_config
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from typing_extensions import TypedDict
from upath import UPath

from rslp.landsat_vessels.config import (
    AWS_DATASET_CONFIG,
    CLASSIFY_MODEL_CONFIG,
    CLASSIFY_WINDOW_SIZE,
    DETECT_MODEL_CONFIG,
    INFRA_THRESHOLD_KM,
    LANDSAT_BANDS,
    LANDSAT_LAYER_NAME,
    LANDSAT_RESOLUTION,
    LANDSAT_SOURCE,
    LOCAL_FILES_DATASET_CONFIG,
    OUTPUT_LAYER_NAME,
)
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
    item: Item | None = None,
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
        item: only ingest this item. This is set if we are getting the scene directly
            from a Landsat data source, not local file.
    """
    # Create a window for applying detector.
    group = "default"
    window_name = "default"
    window_path = Window.get_window_root(ds_path, group, window_name)
    window = Window(
        path=window_path,
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # Restrict to the item if set.
    if item:
        layer_data = WindowLayerData(LANDSAT_LAYER_NAME, [[item.serialize()]])
        window.save_layer_datas(dict(LANDSAT_LAYER_NAME=layer_data))

    logger.info("materialize dataset")
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

    # Sanity check that the layer is completed.
    if not window.is_layer_completed(LANDSAT_LAYER_NAME):
        raise ValueError("landsat layer did not get materialized")

    # Run object detector.
    run_model_predict(DETECT_MODEL_CONFIG, ds_path)

    # Read the detections.
    layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
    features = GeojsonVectorFormat().decode_vector(layer_dir)
    detections: list[VesselDetection] = []
    for feature in features:
        geometry = feature.geometry
        score = feature["properties"]["score"]

        detection = VesselDetection(
            source=LANDSAT_SOURCE,
            col=int(geometry.shp.centroid.x),
            row=int(geometry.shp.centroid.y),
            projection=geometry.projection,
            score=score,
        )
        if item:
            detection.scene_id = item.name
            detection.ts = item.geometry.time_range[0]
        detections.append(detection)

    return detections


def run_classifier(
    ds_path: UPath,
    detections: list[VesselDetection],
    time_range: tuple[datetime, datetime] | None = None,
    item: Item | None = None,
) -> list[VesselDetection]:
    """Run the classifier to try to prune false positive detections.

    Args:
        ds_path: the dataset path that will be populated with new windows to apply the
            classifier.
        detections: the detections from the detector.
        time_range: optional time range to apply the detector in (in case the data
            source needs an actual time range).
        item: only ingest this item.

    Returns:
        the subset of detections that pass the classifier.
    """
    # Avoid error materializing empty group.
    if len(detections) == 0:
        return []

    # Create windows for applying classifier.
    group = "classify_predict"
    windows: list[Window] = []
    for detection in detections:
        window_name = f"{detection.col}_{detection.row}"
        window_path = ds_path / "windows" / group / window_name
        bounds = [
            detection.col - CLASSIFY_WINDOW_SIZE // 2,
            detection.row - CLASSIFY_WINDOW_SIZE // 2,
            detection.col + CLASSIFY_WINDOW_SIZE // 2,
            detection.row + CLASSIFY_WINDOW_SIZE // 2,
        ]
        window = Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=detection.projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()
        windows.append(window)
        detection.metadata["crop_window"] = window

        if item:
            layer_data = WindowLayerData(LANDSAT_LAYER_NAME, [[item.serialize()]])
            window.save_layer_datas(dict(LANDSAT_LAYER_NAME=layer_data))

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
        if not window.is_layer_completed(LANDSAT_LAYER_NAME):
            raise ValueError(f"window {window.name} does not have materialized Landsat")

    # Run classification model.
    run_model_predict(CLASSIFY_MODEL_CONFIG, ds_path, groups=[group])

    # Read the results.
    good_detections = []
    for detection, window in zip(detections, windows):
        layer_dir = window.get_layer_dir(OUTPUT_LAYER_NAME)
        features = GeojsonVectorFormat().decode_vector(layer_dir, window.bounds)
        category = features[0].properties["label"]
        if category == "correct":
            good_detections.append(detection)

    return good_detections


def download_and_unzip_scene(
    remote_zip_path: str, local_zip_path: str, extract_to: str
) -> None:
    """Download a zip file to the local path and unzip it to the extraction path.

    Args:
        remote_zip_path: the path to the zip file.
        local_zip_path: the path to the local directory to download the zip file to.
        extract_to: the path to the extraction directory.
    """
    remote_zip_upath = UPath(remote_zip_path)
    with remote_zip_upath.open("rb") as f:
        with open(local_zip_path, "wb") as f_out:
            shutil.copyfileobj(f, f_out)
    shutil.unpack_archive(local_zip_path, extract_to)
    print(f"unzipped {local_zip_path} to {extract_to}")


def predict_pipeline(
    scene_id: str | None = None,
    scene_zip_path: str | None = None,
    image_files: dict[str, str] | None = None,
    window_path: str | None = None,
    json_path: str | None = None,
    scratch_path: str | None = None,
    crop_path: str | None = None,
    geojson_path: str | None = None,
) -> list[FormattedPrediction]:
    """Run the Landsat vessel prediction pipeline.

    This inputs a Landsat scene (consisting of per-band GeoTIFFs) and produces the
    vessel detections. It produces a CSV containing the vessel detection locations
    along with crops of each detection.

    Args:
        scene_id: Landsat scene ID. Exactly one of image_files or scene_id should be
            specified.
        scene_zip_path: path to the zip file containing the Landsat scene.
        image_files: map from band name like "B8" to the path of the image. The path
            will be converted to UPath so it can include protocol like gs://...
        window_path: path to the metadata.json file for the window.
        json_path: path to write vessel detections as JSON file.
        scratch_path: directory to use to store temporary dataset.
        crop_path: path to write the vessel crop images.
        geojson_path: path to write vessel detections as GeoJSON file.
    """
    if scratch_path is None:
        tmp_scratch_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_scratch_dir.name
    else:
        tmp_scratch_dir = None

    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)
    item = None

    if (
        scene_id is None
        and scene_zip_path is None
        and image_files is None
        and window_path is None
    ):
        raise ValueError(
            "One of scene_id, scene_zip_path, image_files, or window_path must be specified."
        )

    if scene_zip_path:
        # Download the zip file and create image_files locally.
        tmp_zip_dir = tempfile.TemporaryDirectory()
        zip_dir = tmp_zip_dir.name
        scene_id = scene_zip_path.split("/")[-1].split(".")[0]
        local_zip_path = os.path.join(zip_dir, scene_id + ".zip")
        download_and_unzip_scene(scene_zip_path, local_zip_path, zip_dir)
        image_files = {}
        for band in LANDSAT_BANDS:
            image_files[band] = f"{zip_dir}/{scene_id}/{scene_id}_{band}.TIF"
    else:
        tmp_zip_dir = None

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
        cfg["layers"][LANDSAT_LAYER_NAME]["data_source"]["item_specs"] = [item_spec]

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
                scene_bounds = (
                    left,
                    top,
                    left + int(raster.width),
                    top + int(raster.height),
                )
        time_range = None

    else:
        # Load the AWS dataset configuration file.
        with open(AWS_DATASET_CONFIG) as f:
            cfg = json.load(f)
        with (ds_path / "config.json").open("w") as f:
            json.dump(cfg, f)

        if scene_id:
            # Get the projection and scene bounds using the Landsat data source.
            dataset = Dataset(ds_path)
            data_source: LandsatOliTirs = data_source_from_config(
                dataset.layers[LANDSAT_LAYER_NAME], dataset.path
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
            scene_bounds = (
                int(dst_geom.shp.bounds[0]),
                int(dst_geom.shp.bounds[1]),
                int(dst_geom.shp.bounds[2]),
                int(dst_geom.shp.bounds[3]),
            )
            time_range = (
                dst_geom.time_range[0] - timedelta(minutes=30),
                dst_geom.time_range[1] + timedelta(minutes=30),
            )
        elif window_path:
            # Load the window metadata.
            metadata_fname = UPath(window_path) / "metadata.json"
            with metadata_fname.open("r") as f:
                window_metadata = json.load(f)
            scene_bounds = window_metadata["bounds"]
            projection = Projection.deserialize(window_metadata["projection"])
            time_range = (
                (
                    datetime.fromisoformat(window_metadata["time_range"][0]),
                    datetime.fromisoformat(window_metadata["time_range"][1]),
                )
                if window_metadata["time_range"]
                else None
            )

    # Run pipeline.
    print("run detector")
    detections = get_vessel_detections(
        ds_path,
        projection,
        scene_bounds,
        time_range=time_range,
        item=item,
    )
    print("run classifier")
    detections = run_classifier(
        ds_path,
        detections=detections,
        time_range=time_range,
        item=item,
    )

    # Write JSON and crops.
    if crop_path:
        crop_upath = UPath(crop_path)
        crop_upath.mkdir(parents=True, exist_ok=True)

    json_data = []
    geojson_features = []
    near_infra_filter = NearInfraFilter(infra_distance_threshold=INFRA_THRESHOLD_KM)
    raster_format = GeotiffRasterFormat()
    infra_detections = 0
    for idx, detection in enumerate(detections):
        # Apply near infra filter (True -> filter out, False -> keep)
        lon, lat = detection.get_lon_lat()
        if near_infra_filter.should_filter(lon, lat):
            infra_detections += 1
            continue

        # Load crops from the window directory for writing output PNGs.
        # We create two PNGs:
        # - b8.png: just has B8 (panchromatic band).
        # - rgb.png: true color with pan-sharpening. The RGB is from B4, B3, and B2
        #   respectively while B8 is used for pan-sharpening.
        images = {}
        crop_window: Window = detection.metadata["crop_window"]
        if crop_window is None:
            raise ValueError("Crop window is None")
        for band in ["B2", "B3", "B4", "B8"]:
            raster_dir = crop_window.get_raster_dir(LANDSAT_LAYER_NAME, [band])

            # Different bands are in different resolutions so get the bounds from the
            # raster since anyway we just want to read the whole raster.
            band_bounds = raster_format.get_raster_bounds(raster_dir)
            image = raster_format.decode_raster(raster_dir, band_bounds)
            if image.shape[0] != 1:
                raise ValueError(
                    f"expected single-band image for {band} but got {image.shape[0]} bands"
                )

            images[band] = image[0, :, :]

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

        json_data.append(
            FormattedPrediction(
                longitude=lon,
                latitude=lat,
                score=detection.score,
                rgb_fname=str(rgb_fname),
                b8_fname=str(b8_fname),
            ),
        )
        geojson_features.append(detection.to_feature())

    if json_path:
        json_upath = UPath(json_path)
        with json_upath.open("w") as f:
            json.dump(json_data, f)

    if geojson_path:
        geojson_upath = UPath(geojson_path)
        with geojson_upath.open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": geojson_features,
                },
                f,
            )

    return json_data
