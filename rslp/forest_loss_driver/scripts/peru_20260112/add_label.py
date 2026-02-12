"""Add the label polygon since we forgot to include it initially."""

import multiprocessing
from datetime import datetime, timedelta

import tqdm
from rasterio.crs import CRS
from rslearn.dataset import Dataset
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

PREDICTION_FNAME = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/inference/dataset_20260109/events_from_studio_jobs.geojson"
OUTPUT_DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/"
NUM_WORKERS = 128

# Web Mercator projection that all windows are in.
PROJECTION = Projection(CRS.from_epsg(3857), 9.554628535647032, -9.554628535647032)


def reproject_feature(feat: Feature) -> Feature:
    """Helper function to re-project a feature to the WebMercator projection."""
    return Feature(feat.geometry.to_projection(PROJECTION), feat.properties)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    # Load features (predictions) and windows.
    features = GeojsonVectorFormat().decode_from_file(UPath(PREDICTION_FNAME))
    dataset = Dataset(UPath(OUTPUT_DATASET_PATH))
    windows = dataset.load_windows(show_progress=True, workers=128)

    # We need to find the feature that corresponds to each window so we can add it as
    # the label layer. So we create a grid index over the features. We use Web Mercator
    # for the grid index since the index needs everything in one projection.
    p = multiprocessing.Pool(NUM_WORKERS)
    reprojected_features = p.imap_unordered(reproject_feature, features)
    grid_index = GridIndex(size=100)
    for feat in tqdm.tqdm(
        reprojected_features, desc="Creating grid index", total=len(features)
    ):
        grid_index.insert(feat.geometry.shp.bounds, feat)
    p.close()

    # Now iterate over windows and find the closest feature.
    # We make sure that the dates line up.
    for window in tqdm.tqdm(windows, desc="Adding labels"):
        candidates: list[Feature] = grid_index.query(window.bounds)
        best_feat = None
        best_distance: int | None = None
        for candidate in candidates:
            candidate_point = candidate.geometry.to_projection(PROJECTION).shp.centroid
            distance = window.get_geometry().shp.centroid.distance(candidate_point)
            if best_distance is None or distance < best_distance:
                best_feat = candidate
                best_distance = distance

        # The rslearn windows were created using select_examples_for_annotation.py
        # based on the centroid of the GeoJSON featuers, so if there is large distance
        # then it must mean we matched to the wrong feature.
        if best_feat is None or best_distance is None or best_distance > 10:
            raise ValueError(f"no spatially matching feature for window {window.name}")

        feat_datetime = datetime.fromisoformat(best_feat.properties["oe_start_time"])
        if abs(feat_datetime - window.time_range[0]) > timedelta(days=1):
            raise ValueError(f"no tempoarlly matching feature for window {window.name}")

        layer_dir = window.get_layer_dir("label")
        # Reset the label so it is marked unlabeled.
        best_feat.properties["new_label"] = "unlabeled"
        GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84).encode_vector(
            layer_dir, [best_feat]
        )
        window.mark_layer_completed("label")
