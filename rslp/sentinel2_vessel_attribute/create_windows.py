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
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from .ship_types import VESSEL_CATEGORIES

PIXEL_SIZE = 10
WINDOW_SIZE = 128
DATASET_CONFIG_FNAME = "data/sentinel2_vessel_attribute/config.json"


def process_row(group: str, ds_upath: UPath, csv_row: dict[str, str]) -> None:
    """Create a window from one row in the vessel CSV.

    Args:
        group: the rslearn group to add the window to.
        ds_upath: the path of the output rslearn dataset.
        csv_row: the row from vessel CSV.
    """

    def get_optional_float(k: str) -> float | None:
        if csv_row[k]:
            return float(csv_row[k])
        else:
            return None

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
    vessel_sog_variance = get_optional_float("ais_speed_variance")
    if "time_to_closest_position" in csv_row:
        time_to_closest_position = get_optional_float("time_to_closest_position")
    else:
        time_to_closest_position = None

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
    window_root = Window.get_window_root(ds_upath, group, window_name)
    window = Window(
        path=window_root,
        group=group,
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
                "sog_variance": vessel_sog_variance,
                "time_to_closest_position": time_to_closest_position,
            },
            f,
        )

    info_dir = window.get_layer_dir("info")
    info_dir.mkdir(parents=True, exist_ok=True)
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
    GeojsonVectorFormat().encode_vector(info_dir, [feat])
    window.mark_layer_completed("info")


def create_windows(group: str, csv_dir: str, ds_path: str, workers: int = 32) -> None:
    """Initialize an rslearn dataset at the specified path.

    Args:
        group: which group to use for these windows.
        csv_dir: path containing CSVs with AIS-correlated vessel detections, e.g.
            gs://rslearn-eai/datasets/sentinel2_vessel_attribute/artifacts/sentinel2_correlated_detections_bigtable/.
        ds_path: path to write the dataset, e.g.
            gs://rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20241212/
        workers: number of worker processes to use
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
                        group=group,
                        ds_upath=ds_upath,
                        csv_row=csv_row,
                    )
                )

    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, process_row, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
