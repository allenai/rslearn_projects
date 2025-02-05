"""Populate rslearn dataset with windows from the source CSVs."""

import csv
import hashlib
import json
import multiprocessing
import shutil
from datetime import datetime, timedelta
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from .ship_types import SHIP_TYPES, VESSEL_CATEGORIES

PIXEL_SIZE = 10
WINDOW_SIZE = 128
GROUP = "default"
DATASET_CONFIG_FNAME = "data/sentinel2_vessel_attribute/config.json"


def process_row(ds_upath: UPath, csv_row: dict[str, str]) -> None:
    """Create a window from one row in the vessel CSV.

    Args:
        ds_upath: the path of the output rslearn dataset.
        csv_row: the row from vessel CSV.
    """

    def get_optional_float(k: str) -> float | None:
        if csv_row[k]:
            return float(csv_row[k])
        else:
            return None

    if "event_time" in csv_row:
        event_id = csv_row["event_id"]
        ts = datetime.fromisoformat(csv_row["event_time"])
        lat = float(csv_row["lat"])
        lon = float(csv_row["lon"])
        if csv_row["vessel_category"]:
            ship_type = csv_row["vessel_category"]
        else:
            ship_type = "unknown"
        vessel_length = get_optional_float("vessel_length")
        vessel_width = get_optional_float("vessel_width")
        vessel_cog = get_optional_float("ais_course")
        vessel_cog_avg = get_optional_float("course")
        vessel_sog = get_optional_float("ais_speed")
    else:
        ts = datetime.fromisoformat(csv_row["timestamp"])
        lat = float(csv_row["latitude"])
        lon = float(csv_row["longitude"])
        if csv_row["ship_type"]:
            ship_type = SHIP_TYPES.get(int(csv_row["ship_type"]), "unknown")
        else:
            ship_type = "unknown"

        vessel_length = get_optional_float("length")
        vessel_width = get_optional_float("width")
        vessel_cog = get_optional_float("cog")
        vessel_cog_avg = None
        vessel_sog = get_optional_float("sog")
        event_id = (
            f"{csv_row['timestamp']}_{csv_row['longitude']}_{csv_row['latitude']}"
        )

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_projection = get_utm_ups_projection(lon, lat, PIXEL_SIZE, -PIXEL_SIZE)
    dst_geometry = src_geometry.to_projection(dst_projection)

    bounds = (
        int(dst_geometry.shp.x) - WINDOW_SIZE // 2,
        int(dst_geometry.shp.y) - WINDOW_SIZE // 2,
        int(dst_geometry.shp.x) + WINDOW_SIZE // 2,
        int(dst_geometry.shp.y) + WINDOW_SIZE // 2,
    )
    time_range = (ts - timedelta(hours=1), ts + timedelta(hours=1))

    # Check if train or val.
    is_val = hashlib.sha256(event_id.encode()).hexdigest()[0] in ["0", "1"]
    split = "val" if is_val else "train"

    window_name = event_id
    window_root = Window.get_window_root(ds_upath, GROUP, window_name)
    window = Window(
        path=window_root,
        group=GROUP,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(
            split=split,
        ),
    )
    window.save()

    # Save metadata.
    with (window_root / "info.json").open("w") as f:
        json.dump(
            {
                "event_id": event_id,
                "length": vessel_length,
                "width": vessel_width,
                "cog": vessel_cog,
                "cog_avg": vessel_cog_avg,
                "sog": vessel_sog,
                "type": ship_type,
            },
            f,
        )

    info_dir = window_root / "layers" / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    gt_layer_fname = info_dir / "data.geojson"
    properties: dict[str, Any] = {
        "event_id": event_id,
    }
    if vessel_length and vessel_length >= 5 and vessel_length < 460:
        properties["length"] = vessel_length
    if vessel_width and vessel_width >= 2 and vessel_width < 120:
        properties["width"] = vessel_width
    if (
        vessel_cog
        and vessel_sog
        and vessel_sog > 5
        and vessel_sog < 50
        and vessel_cog >= 0
        and vessel_cog < 360
    ):
        properties["cog"] = vessel_cog
    if vessel_sog and vessel_sog > 0 and vessel_sog < 60:
        properties["sog"] = vessel_sog
    if ship_type and ship_type in VESSEL_CATEGORIES:
        properties["type"] = VESSEL_CATEGORIES[ship_type]
    feat = Feature(dst_geometry, properties)
    with gt_layer_fname.open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [feat.to_geojson()],
            },
            f,
        )


def create_windows(csv_dir: str, ds_path: str) -> None:
    """Initialize an rslearn dataset at the specified path.

    Args:
        csv_dir: path containing CSVs with AIS-correlated vessel detections, e.g.
            gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_bigtable/.
        ds_path: path to write the dataset, e.g.
            gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20241212/
    """
    csv_upath = UPath(csv_dir)
    ds_upath = UPath(ds_path)

    # Copy dataset configuration first.
    with open(DATASET_CONFIG_FNAME, "rb") as src:
        with (ds_upath / "config.json").open("wb") as dst:
            shutil.copyfileobj(src, dst)

    jobs = []
    for fname in csv_upath.iterdir():
        with fname.open() as f:
            reader = csv.DictReader(f)
            for csv_row in reader:
                jobs.append(
                    dict(
                        ds_upath=ds_upath,
                        csv_row=csv_row,
                    )
                )

    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, process_row, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
