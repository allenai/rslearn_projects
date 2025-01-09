"""This provides a generic way to convert a label that was made in WebMercator into a
window with UTM projection in an rslearn dataset.
"""

import json
import math
from datetime import datetime
from typing import Any

import numpy as np
import shapely
import skimage.draw
from PIL import Image
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from rslearn.utils.raster_format import SingleImageRasterFormat
from upath import UPath

src_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137


def convert_window(
    root_dir: UPath,
    group: str,
    zoom: int,
    bounds: tuple[int, int, int, int],
    labels: list[tuple[shapely.Geometry, dict[str, Any]]],
    time_range: tuple[datetime, datetime],
    dst_pixel_size: float = 10,
    window_name: str | None = None,
) -> Window:
    """Create an rslearn window from a multisat window with the specified properties.

    Args:
        root_dir: the dataset root dir
        group: the window's group
        zoom: the zoom level of this window.
        bounds: the bounds in pixel coordinates at the zoom level. The pixel
            coordinates are of the multisat format where they start from (0, 0).
        labels: list of (geom, properties) label tuples for this window.
        dst_pixel_size: resolution of the output window
        window_name: override the default window name (which uses the topleft
            coordinates of the window in multisat pixel coordinates)
    """
    total_pixels = (2**zoom) * 512
    src_pixel_size = web_mercator_m / total_pixels
    src_projection = Projection(src_crs, src_pixel_size, -src_pixel_size)

    if window_name is None:
        window_name = f"{bounds[0]}_{bounds[1]}"

    # Apply an offset on bounds to go from pixel coordinates that assign (0, 0) to
    # topleft pixel, to ones that assign (0, 0) to center pixel.
    bounds = [val - total_pixels // 2 for val in bounds]

    # Compute polygon in source projection coordinates.
    src_polygon = shapely.Polygon(
        [
            [bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]],
        ]
    )

    # Now identify the appropriate UTM projection for the polygon, and transform it.
    src_geom = STGeometry(src_projection, src_polygon, None)
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    # We apply abs() on the latitude because Landsat only uses northern UTM zones.
    dst_crs = get_utm_ups_crs(wgs84_geom.shp.centroid.x, abs(wgs84_geom.shp.centroid.y))
    dst_projection = Projection(dst_crs, dst_pixel_size, -dst_pixel_size)
    dst_geom = src_geom.to_projection(dst_projection)
    dst_polygon = dst_geom.shp

    # (1) Write the window itself.
    bounds = [
        int(dst_polygon.bounds[0]),
        int(dst_polygon.bounds[1]),
        int(dst_polygon.bounds[2]),
        int(dst_polygon.bounds[3]),
    ]
    window_root = Window.get_window_root(root_dir, group, window_name)
    window = Window(
        path=window_root,
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # (2) Write the turbine positions.
    features: list[Feature] = []
    for shp, properties in labels:
        # Similar to with bounds, subtract the WebMercator pixel offset between
        # multisat and rslearn.
        shp = shapely.transform(shp, lambda coords: coords - total_pixels // 2)

        src_geom = STGeometry(src_projection, shp, None)
        dst_geom = src_geom.to_projection(dst_projection)
        features.append(Feature(dst_geom, properties))

    layer_name = "label"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat().encode_vector(layer_dir, dst_projection, features)
    window.mark_layer_completed(layer_name)

    # (3) Write mask corresponding to old window projected onto new window.
    mask = np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0]), dtype=np.uint8)
    assert len(dst_polygon.exterior.coords) == 5
    assert len(dst_polygon.interiors) == 0
    polygon_rows = [coord[1] - bounds[1] for coord in dst_polygon.exterior.coords]
    polygon_cols = [coord[0] - bounds[0] for coord in dst_polygon.exterior.coords]
    rr, cc = skimage.draw.polygon(polygon_rows, polygon_cols, shape=mask.shape)
    mask[rr, cc] = 255
    layer_name = "mask"
    layer_dir = window.get_layer_dir(layer_name)
    SingleImageRasterFormat().encode_raster(layer_dir, dst_projection, bounds, mask[None, :, :])
    window.mark_layer_completed(layer_name)

    return window
