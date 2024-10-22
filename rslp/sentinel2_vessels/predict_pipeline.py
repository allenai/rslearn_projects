"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from datetime import datetime, timedelta

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

from rslp.utils.rslearn import materialize_dataset, run_model_predict

SENTINEL2_LAYER_NAME = "sentinel2"
DATASET_CONFIG = "data/sentinel2_vessels/config.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"
SENTINEL2_RESOLUTION = 10
CROP_WINDOW_SIZE = 64


class VesselDetection:
    """A vessel detected in a Sentinel-2 window."""

    def __init__(
        self,
        col: int,
        row: int,
        projection: Projection,
        score: float,
        ts: datetime,
        crop_window_dir: UPath | None = None,
    ) -> None:
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
            ts: datetime fo the window.
            crop_window_dir: the crop window directory.
        """
        self.col = col
        self.row = row
        self.projection = projection
        self.score = score
        self.ts = ts
        self.crop_window_dir = crop_window_dir


# TODO: make a simple class to store bounds
def get_vessel_detections(
    ds_path: UPath,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    ts: datetime,
    item: Item,
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    Sentinel-2 images.

    Args:
        ds_path: the dataset path that will be populated with a new window to apply the
            detector.
        projection: the projection to apply the detector in.
        bounds: the bounds to apply the detector in.
        ts: timestamp to apply the detector on.
        item: the item to ingest.
    """
    # Create a window for applying detector.
    group = "detector_predict"
    window_path = ds_path / "windows" / group / "default"
    window = Window(
        path=window_path,
        group=group,
        name="default",
        projection=projection,
        bounds=bounds,
        time_range=(ts - timedelta(minutes=20), ts + timedelta(minutes=20)),
    )
    window.save()

    if item:
        layer_data = WindowLayerData(SENTINEL2_LAYER_NAME, [[item.serialize()]])
        window.save_layer_datas(dict(SENTINEL2_LAYER_NAME=layer_data))

    print("materialize dataset")
    materialize_dataset(ds_path, group=group, workers=1)
    assert (
        window_path / "layers" / SENTINEL2_LAYER_NAME / "R_G_B" / "geotiff.tif"
    ).exists()

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
                ts=ts,
            )
        )

    return detections


def predict_pipeline(
    scene_id: str, scratch_path: str, json_path: str, crop_path: str
) -> None:
    """Run the Sentinel-2 vessel prediction pipeline.

    Given a Sentinel-2 scene ID, the pipeline produces the vessel detections.
    Specifically, it outputs a CSV containing the vessel detection locations along with
    crops of each detection.

    Args:
        scene_id: the Sentinel-2 scene ID.
        scratch_path: directory to use to store temporary dataset.
        json_path: path to write the JSON of vessel detections.
        crop_path: path to write the vessel crop images.
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
    item = data_source.get_item_by_name(scene_id)
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
    ts = item.geometry.time_range[0]

    detections = get_vessel_detections(ds_path, projection, bounds, ts, item)

    # Create windows just to collect crops for each detection.
    group = "crops"
    window_paths: list[UPath] = []
    for detection in detections:
        window_name = f"{detection.col}_{detection.row}"
        window_path = ds_path / "windows" / group / window_name
        detection.crop_window_dir = window_path
        bounds = (
            detection.col - CROP_WINDOW_SIZE // 2,
            detection.row - CROP_WINDOW_SIZE // 2,
            detection.col + CROP_WINDOW_SIZE // 2,
            detection.row + CROP_WINDOW_SIZE // 2,
        )
        Window(
            path=window_path,
            group=group,
            name=window_name,
            projection=detection.projection,
            bounds=bounds,
            time_range=(ts - timedelta(minutes=20), ts + timedelta(minutes=20)),
        ).save()
        window_paths.append(window_path)
    if len(detections) > 0:
        materialize_dataset(ds_path, group=group, workers=4)

    # Write JSON and crops.
    json_upath = UPath(json_path)
    crop_upath = UPath(crop_path)
    json_data = []
    for detection, crop_window_path in zip(detections, window_paths):
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

        # Get longitude/latitude.
        src_geom = STGeometry(
            detection.projection, shapely.Point(detection.col, detection.row), None
        )
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        lon = dst_geom.shp.x
        lat = dst_geom.shp.y

        json_data.append(
            dict(
                longitude=lon,
                latitude=lat,
                score=detection.score,
                ts=detection.ts.isoformat(),
                scene_id=scene_id,
                crop_fname=str(crop_fname),
            )
        )

    with json_upath.open("w") as f:
        json.dump(json_data, f)
