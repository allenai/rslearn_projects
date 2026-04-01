"""Create rslearn datasets from the AlphaEarth supplemental evaluation datasets."""

import csv
import multiprocessing
import os
import shutil
from datetime import datetime, timezone

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

DATASET_PATH = "/weka/dfive-default/rslearn-eai/artifacts/deepmind_alphaearth_supplemental_evaluation_datasets/"
DATASET_CSVS = [
    "africa_crop_mask/africa_crop_mask.csv",
    "aster_ged/aster_ged.csv",
    "canada_crops/canada_crops_coarse/canada_crops_coarse.csv",
    "canada_crops/canada_crops_fine/canada_crops_fine.csv",
    "descals/descals.csv",
    "ethiopia_crops/ethiopia_crops.csv",
    "glance/glance.csv",
    "lcmap/lcmap_lcc/lcmap_lcc.csv",
    "lcmap/lcmap_lc/lcmap_lc.csv",
    "lcmap/lcmap_luc/lcmap_luc.csv",
    "lcmap/lcmap_lu/lcmap_lu.csv",
    "lucas/lucas_lc/lucas_lc.csv",
    "lucas/lucas_lu/lucas_lu.csv",
    "openet/openet_ensemble.csv",
    "us_trees/us_trees.csv",
]
OUT_DIR = "/weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations"
RESOLUTION = 10
WINDOW_SIZE = 256


def process_row(ds_path: UPath, example_idx: int, csv_row: dict[str, str]) -> None:
    """Process one row in the CSV into a Window."""
    lon = float(csv_row["x"])
    lat = float(csv_row["y"])
    start_ms = int(csv_row["support_time_start_ms"])
    end_ms = int(csv_row["support_time_end_ms"])
    split = csv_row["split"]
    if "partition" in csv_row:
        partition = float(csv_row["partition"])
    else:
        partition = None
    if "label_name" in csv_row:
        label = csv_row["label_name"]
    else:
        label = csv_row["label"]

    # Get projection and bounding box to use.
    wgs84_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_proj = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    dst_geom = wgs84_geom.to_projection(dst_proj)
    bounds = (
        int(dst_geom.shp.x) - WINDOW_SIZE // 2,
        int(dst_geom.shp.y) - WINDOW_SIZE // 2,
        int(dst_geom.shp.x) + WINDOW_SIZE // 2,
        int(dst_geom.shp.y) + WINDOW_SIZE // 2,
    )

    # Get time range.
    # Note that lcmap_lcc and lcmap_luc are bitemporal and user will need to
    # adjust dataset configuration to config_bitemporal.json. The time range
    # will cover both sub-ranges and the dataset config will have separate
    # layers to pick out the before sub-range and the after sub-range.
    time_range = (
        datetime.fromtimestamp(start_ms // 1000).replace(tzinfo=timezone.utc),
        datetime.fromtimestamp(end_ms // 1000).replace(tzinfo=timezone.utc),
    )

    group = split
    window_name = f"sample_{example_idx+1}"
    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=dst_proj,
        bounds=bounds,
        time_range=time_range,
        options=dict(
            lon=lon,
            lat=lat,
            split=split,
            partition=partition,
            label=label,
        ),
    )
    window.save()

    # Add the label to a vector layer.
    # This gets loaded by the classification task configured in the model
    # configuration file.
    feature = Feature(dst_geom, dict(label=label))
    layer_name = "label"
    layer_dir = window.get_layer_dir(layer_name)
    GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84).encode_vector(
        layer_dir, [feature]
    )


def process_dataset(csv_fname: str) -> None:
    """Process the raw dataset into rslearn dataset."""
    dataset_name = csv_fname.split(".csv")[0].split("/")[-1]
    ds_path = UPath(OUT_DIR) / dataset_name
    ds_path.mkdir(parents=True, exist_ok=True)
    print(f"creating windows for dataset {dataset_name} in {ds_path}")
    shutil.copyfile(
        "one_off_projects/2025_08_01_alphaearth_eval_to_rslearn/config.json",
        (ds_path / "config.json").path,
    )

    # Get list of arguments to call process_row with.
    jobs = []
    with open(csv_fname) as f:
        reader = csv.DictReader(f)
        for example_idx, csv_row in enumerate(reader):
            jobs.append(
                dict(
                    ds_path=ds_path,
                    example_idx=example_idx,
                    csv_row=csv_row,
                )
            )

    p = multiprocessing.Pool(128)
    outputs = star_imap_unordered(p, process_row, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    for csv_name in DATASET_CSVS:
        csv_fname = os.path.join(DATASET_PATH, csv_name)
        process_dataset(csv_fname)
