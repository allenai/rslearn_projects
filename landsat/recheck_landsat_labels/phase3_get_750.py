"""Get a sample of ~700 vessel labels."""

import csv
import json
import os
import re
from datetime import datetime, timezone

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from upath import UPath

in_fname = "phase3a_selected.csv"
out_ds_dir = (
    "gcs://rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20240905/"
)
out_ds_dir = UPath(out_ds_dir)
out_group = "phase3a_selected"

candidates = []
with open(in_fname) as f:
    reader = csv.DictReader(f)
    for idx, csv_row in enumerate(reader):
        lon = float(csv_row["lon"])
        lat = float(csv_row["lat"])
        # get UTC date
        scene_id = csv_row["scene_id"]
        date_str = re.search(r"(\d{8})", scene_id).group(1)
        ts = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        candidates.append((idx, lon, lat, ts))
selected = candidates

for idx, lon, lat, ts in selected:
    # Find appropriate UTM projection.
    dst_crs = get_utm_ups_crs(lon, lat)
    dst_projection = Projection(dst_crs, 15, -15)

    # Get window bounds centered at the vessel in dst_crs.
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(dst_projection)
    bounds = [
        int(dst_geom.shp.x) - 32,
        int(dst_geom.shp.y) - 32,
        int(dst_geom.shp.x) + 32,
        int(dst_geom.shp.y) + 32,
    ]

    # Update to UTC 00:00 to 23:59
    start_time = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = ts.replace(hour=23, minute=59, second=59, microsecond=999999)
    time_range = (start_time, end_time)

    format_string = "%Y-%m-%dT%H:%M:%SZ"
    from_time = time_range[0].strftime(format_string)
    to_time = time_range[1].strftime(format_string)
    url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=16&lat={lat}&lng={lon}&fromTime={from_time}&toTime={to_time}&datasetId=AWS_LOTL1&layerId=2_TRUE_COLOR_PANSHARPENED"

    window_name = f"vessel_{idx}"
    window_root = out_ds_dir / "windows" / out_group / window_name
    os.makedirs(window_root, exist_ok=True)

    window = Window(
        path=window_root,
        group=out_group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options={"split": "train", "weight": 1},
    )
    window.save()

    label_layer_dir = os.path.join(window_root, "layers", "label")
    os.makedirs(label_layer_dir, exist_ok=True)
    with open(os.path.join(label_layer_dir, "completed"), "w") as f:
        pass
    if not os.path.exists(os.path.join(label_layer_dir, "data.geojson")):
        with open(os.path.join(label_layer_dir, "data.geojson"), "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [[32, 32]],
                            },
                            "properties": {
                                "label": "incorrect",
                                "url": url,
                                "idx": idx,
                                "lon": lon,
                                "lat": lat,
                                "ts": ts.isoformat(),
                            },
                        }
                    ],
                    "properties": window.projection.serialize(),
                },
                f,
            )
