"""Convert the solar farm labels in siv to rslearn format while also switching to using
UTM projection.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio.features
import shapely
from upath import UPath

from rslearn.utils.vector_format import GeojsonVectorFormat
from rslearn.utils.raster_format import SingleImageRasterFormat
from ..lib import convert_window

db_path = "/home/ubuntu/siv_renewable/data/siv.sqlite3"
out_dir = "/multisat/datasets/rslearn_datasets_satlas/solar_farm/"
group = "default"

conn = sqlite3.connect(db_path)
conn.isolation_level = None
db = conn.cursor()

# Get the windows.
db.execute("""
    SELECT w.id, im.time, w.column, w.row, w.width, w.height
    FROM windows AS w, images AS im
    WHERE dataset_id = 2 AND w.image_id = im.id
    AND split in ('2023mar16-flagged-done', '2023apr10-flagged', 'pick01', 'fp01-done', 'fp02-done', 'fp03-done', 'fp04-done', 'fp05', '2023sep06', 'fp06')
""")
for w_id, im_time, w_col, w_row, w_width, w_height in db.fetchall():
    bounds = [w_col, w_row, w_col + w_width, w_row + w_height]

    ts = datetime.fromisoformat(im_time)
    if not ts.tzinfo:
        ts = ts.replace(tzinfo=timezone.utc)
    time_range = (
        ts - timedelta(days=120),
        ts + timedelta(days=60),
    )

    db.execute(
        """
        SELECT extent FROM labels WHERE window_id = ?
    """,
        (w_id,),
    )
    labels = []
    for (extent,) in db.fetchall():
        extent = json.loads(extent)
        if len(extent) < 3:
            continue
        polygon = shapely.Polygon(extent)
        properties = {"category": "solar_farm"}
        labels.append((polygon, properties))

    window = convert_window(
        root_dir=UPath(out_dir),
        group=group,
        zoom=15,
        bounds=bounds,
        labels=labels,
        time_range=time_range,
    )

    # Create raster version of the label.
    layer_dir = window.get_layer_dir("label")
    features = GeojsonVectorFormat().decode_vector(layer_dir, bounds)

    shapes = []
    for feat in features:
        assert feat.geometry.projection == window.projection
        geometry = json.loads(shapely.to_geojson(feat.geometry.shp))
        assert geometry["type"] == "Polygon"
        geometry["coordinates"] = (
            np.array(geometry["coordinates"]) - [window.bounds[0], window.bounds[1]]
        ).tolist()
        shapes.append((geometry, 1))
    if shapes:
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(
                window.bounds[3] - window.bounds[1],
                window.bounds[2] - window.bounds[0],
            ),
            dtype=np.uint8,
        )
    else:
        mask = np.zeros(
            (window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
            dtype=np.uint8,
        )
    layer_name = "label_raster"
    raster_dir = window.get_raster_dir(layer_name, ["label"])
    SingleImageRasterFormat().encode_raster(raster_dir, window.projection, window.bounds, mask[None, :, :])
    window.mark_layer_completed(layer_name)
