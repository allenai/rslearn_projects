"""Create rslearn windows for the Kenya Intercropping dataset.

Reads /tmp/monocrop_intercrop.csv, creates 64x64 windows at 10m in UTM with:
- label_raster: 64x64 GeoTIFF with class index at center pixel, 255 elsewhere
- label: vector layer with {"category": lcct}
- Split: 50/25/25 train/val/test via hash of window name
- Time range: 2022-10-01 to 2023-03-31 (growing season)
"""

import csv
import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

CSV_PATH = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/kenya_intercropping/data.csv"
DST = Path(
    "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/kenya_intercropping"
)

WINDOW_SIZE = 64
RESOLUTION = 10
START_TIME = datetime(2022, 10, 1, tzinfo=timezone.utc)
END_TIME = datetime(2023, 3, 31, tzinfo=timezone.utc)
NUM_PROC = 64

LCCT_CLASSES = ["intercrop", "monocrop", "other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(LCCT_CLASSES)}

GEOTIFF_FORMAT = GeotiffRasterFormat()


def get_split(window_name: str) -> str:
    h = hashlib.sha256(window_name.encode()).hexdigest()
    val = int(h[:8], 16) % 4
    if val == 0:
        return "val"
    elif val == 1:
        return "test"
    else:
        return "train"


def process_row(args: tuple[dict, str]) -> None:
    row, ds_path_str = args
    ds_path = UPath(ds_path_str)
    dataset = Dataset(ds_path)

    lat = float(row["latitude"])
    lon = float(row["longitude"])
    label = row["lcct"]
    window_name = row["task_name"]

    class_idx = CLASS_TO_IDX[label]
    split = get_split(window_name)

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(lon, lat)
    dst_projection = Projection(dst_crs, RESOLUTION, -RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    center_col = int(dst_geometry.shp.x)
    center_row = int(dst_geometry.shp.y)
    bounds = (
        center_col - WINDOW_SIZE // 2,
        center_row - WINDOW_SIZE // 2,
        center_col + WINDOW_SIZE // 2,
        center_row + WINDOW_SIZE // 2,
    )

    window = Window(
        storage=dataset.storage,
        group=split,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(START_TIME, END_TIME),
        options={"split": split, "category": label},
    )
    window.save()

    # Write label_raster: center pixel = class_idx, rest = 255
    arr = np.full((1, WINDOW_SIZE, WINDOW_SIZE), 255, dtype=np.uint8)
    arr[0, WINDOW_SIZE // 2, WINDOW_SIZE // 2] = class_idx
    raster = RasterArray(chw_array=arr)
    layer_dir = window.get_layer_dir("label_raster")
    band_dir = layer_dir / "label"
    band_dir.mkdir(parents=True, exist_ok=True)
    GEOTIFF_FORMAT.encode_raster(band_dir, dst_projection, bounds, raster)
    window.mark_layer_completed("label_raster")

    # Write label vector layer
    feature = Feature(window.get_geometry(), {"category": label})
    label_layer_dir = window.get_layer_dir("label")
    GeojsonVectorFormat().encode_vector(label_layer_dir, [feature])
    window.mark_layer_completed("label")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    # Copy config.json
    config_src = Path(__file__).parent / "kenya_intercropping_config.json"
    import shutil
    shutil.copyfile(config_src, DST / "config.json")

    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Processing {len(rows)} rows")
    jobs = [(row, str(DST)) for row in rows]

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(process_row, jobs), total=len(jobs)
        ):
            pass


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
