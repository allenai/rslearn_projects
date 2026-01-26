"""Fix coordinates of labels in the naip group of the wind turbine dataset.

Somehow these labels were re-projected incorrectly when the dataset was first converted
from multisat format to rslearn dataset: although the "label" group seems okay, the
"naip" group still has coordinates in WebMercator. This script fixes it. See
data/satlas/wind_turbine/README.md for details.
"""

import argparse
import json
import math
import multiprocessing

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat, GeojsonCoordinateMode
from upath import UPath

# Compute the WebMercator projection for the source coordinates.
SRC_CRS = CRS.from_epsg(3857)
SRC_ZOOM = 13
WEB_MERCATOR_M = 2 * math.pi * 6378137
TOTAL_PIXELS_AT_ZOOM_LEVEL = (2 ** SRC_ZOOM) * 512
SRC_PIXEL_SIZE = WEB_MERCATOR_M / TOTAL_PIXELS_AT_ZOOM_LEVEL
SRC_PROJECTION = Projection(SRC_CRS, SRC_PIXEL_SIZE, -SRC_PIXEL_SIZE)

DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/wind_turbine/dataset_v1/20260122/"
NUM_WORKERS = 64


def fix_window(window: Window) -> None:
    """Apply the fix to the specified window."""
    label_dir = window.get_layer_dir("label")

    # We read directly as JSON since the projection/coordinates aren't right, meaning
    # that GeojsonVectorFormat won't work correctly.
    with (label_dir / "data.geojson").open() as f:
        fc = json.load(f)
    if len(fc["features"]) == 0:
        # We don't need to overwrite it since there are no features.
        return

    # Re-project each feature. We need to subtract TOTAL_PIXELS_AT_ZOOM_LEVEL because
    # in rslearn the coordinate system starts from (-TOTAL_PIXELS_AT_ZOOM_LEVEL // 2),
    # while in multisat it starts from 0.
    new_features = []
    for feat in fc["features"]:
        col, row = feat["geometry"]["coordinates"]
        src_geom = STGeometry(SRC_PROJECTION, shapely.Point(col - TOTAL_PIXELS_AT_ZOOM_LEVEL // 2, row - TOTAL_PIXELS_AT_ZOOM_LEVEL // 2), None)
        dst_geom = src_geom.to_projection(window.projection)
        new_feat = Feature(dst_geom, feat["properties"])

        # Just double check that the resulting coordinates are reasonable. They might not
        # be exactly contained within the window bounds, but it should be close.
        window_geom = window.get_geometry()
        distance = window_geom.shp.distance(dst_geom.shp)
        if distance > 100:
            raise ValueError(f"expected window {window.bounds} to contain re-projected geometry {dst_geom} but got distance {distance}")

        new_features.append(new_feat)

    # Now we can write out the GeoJSON. We use WGS84 so it is easy to verify in qgis.
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    vector_format.encode_vector(label_dir, new_features)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    dataset = Dataset(UPath(DATASET_PATH))
    windows = dataset.load_windows(groups=["naip"], workers=NUM_WORKERS, show_progress=True)
    p = multiprocessing.Pool(128)
    outputs = p.imap_unordered(fix_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
