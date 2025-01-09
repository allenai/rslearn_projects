"""Convert the solar farm labels in siv to rslearn format while also switching to using
UTM projection.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio.features
import shapely
from PIL import Image

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
        root_dir=out_dir,
        group=group,
        zoom=15,
        bounds=bounds,
        labels=labels,
        time_range=time_range,
    )

    # Create raster version of the label.
    layer_name = "label"
    layer_dir = window.get_layer_dir(layer_name)
    features = GeojsonVectorFormat().decode_vector(layer_dir, bounds)
    shapes = []
    for feat in features:
        geometry = feat.geometry


        geometry = feat["geometry"]
        assert geometry["type"] == "Polygon"
        geometry["coordinates"] = (
            np.array(geometry["coordinates"]) - [window.bounds[0], window.bounds[1]]
        ).tolist()
        shapes.append((geometry, 255))
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
    with window.file_api.get_folder("layers/label_raster").open("image.png", "wb") as f:
        Image.fromarray(mask).save(f, format="PNG")
