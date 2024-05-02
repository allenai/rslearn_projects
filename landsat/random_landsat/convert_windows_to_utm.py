"""
Convert the random windows we created from WebMercator projection to appropriate UTM projection.
"""
import os
import shapely

from rasterio.crs import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_projection


out_dir = "/data/favyenb/rslearn_landsat/windows/"
IN_GROUP = "default"
OUT_GROUP = "utm"

pixel_size = 15

for window_name in os.listdir(os.path.join(out_dir, IN_GROUP)):
    window = Window.load(os.path.join(out_dir, IN_GROUP, window_name))
    point = shapely.Point((window.bounds[0] + window.bounds[2]) // 2, (window.bounds[1] + window.bounds[3]) // 2)
    geom1 = STGeometry(window.projection, point, None)
    geom2 = geom1.to_projection(WGS84_PROJECTION)
    utm_crs = get_utm_ups_projection(geom2.shp.x, geom2.shp.y)
    dst_projection = Projection(utm_crs, pixel_size, -pixel_size)
    geom3 = geom1.to_projection(dst_projection)
    bounds = [
        int(geom3.shp.x) - 512,
        int(geom3.shp.y) - 512,
        int(geom3.shp.x) + 512,
        int(geom3.shp.y) + 512,
    ]
    new_window = Window(
        window_root=os.path.join(out_dir, OUT_GROUP, window_name + "_utm"),
        group=OUT_GROUP,
        name=window_name + "_utm",
        projection=dst_projection,
        bounds=bounds,
        time_range=window.time_range,
    )
    new_window.save()
