"""Sentinel-2 vessel prediction pipeline."""

import json
import shutil
from datetime import datetime, timedelta

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import data_source_from_config
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from rslp.utils.rslearn import materialize_dataset, run_model_predict

DATASET_CONFIG = "data/sentinel2_vessels/config.json"
DETECT_MODEL_CONFIG = "data/sentinel2_vessels/config.yaml"
SENTINEL2_RESOLUTION = 10


class VesselDetection:
    """A vessel detected in a Sentinel-2 window."""

    def __init__(
        self, col: int, row: int, projection: Projection, score: float, ts: datetime
    ):
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
            ts: datetime fo the window.
        """
        self.col = col
        self.row = row
        self.projection = projection
        self.score = score
        self.ts = ts


def get_vessel_detections(
    ds_path: UPath,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    ts: datetime,
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
    """
    # Create a window for applying detector.
    group = "detector_predict"
    window_path = ds_path / "windows" / group / "default"
    Window(
        path=window_path,
        group=group,
        name="default",
        projection=projection,
        bounds=bounds,
        time_range=(ts - timedelta(minutes=20), ts + timedelta(minutes=20)),
    ).save()

    print("materialize dataset")
    materialize_dataset(ds_path, group=group)
    assert (window_path / "layers" / "sentinel2" / "B02" / "geotiff.tif").exists()

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


def predict_pipeline(scene_id: str, scratch_path: str, csv_path: str, crop_path: str):
    """Run the Sentinel-2 vessel prediction pipeline.

    Given a Sentinel-2 scene ID, the pipeline produces the vessel detections.
    Specifically, it outputs a CSV containing the vessel detection locations along with
    crops of each detection.

    Args:
        scene_id: the Sentinel-2 scene ID.
        scratch_path: directory to use to store temporary dataset.
        csv_path: path to write the CSV.
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
        dataset.layers["sentinel2"], dataset.path
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
    bounds = [int(value) for value in dst_geom.bounds]

    detections = get_vessel_detections(
        ds_path, projection, bounds, item.geometry.time_range[0]
    )

    print(detections)
