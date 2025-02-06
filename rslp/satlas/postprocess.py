"""Postprocessing outputs from Satlas models."""

import json
import math
import multiprocessing
import shutil
import subprocess  # nosec
import tempfile
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.grid_index import GridIndex
from upath import UPath

from rslp.log_utils import get_logger

from .predict_pipeline import Application

# Approximate maximum meters in one degree latitude/longitude.
# Above/below the equator there will be fewer meters.
# This is used to compute the NMS_DISTANCE_THRESHOLD below.
MAX_METERS_PER_DEGREE = 111111

# Threshold on Euclidean distance between lat/lon for NMS.
# We just do Euclidean distance for speed/simplicity since NMS doesn't need to be super
# exact (instead of spherical distance).
NMS_DISTANCE_THRESHOLD = 100 / MAX_METERS_PER_DEGREE

# Individual Satlas applications use different category names than the global ones that
# we want to serve. We should adjust this but for now this map helps to rename the
# categories.
APP_CATEGORY_MAPS = {
    Application.MARINE_INFRA: {
        "platform": "offshore_platform",
        "turbine": "offshore_wind_turbine",
    },
    Application.WIND_TURBINE: {
        "turbine": "wind_turbine",
    },
}

logger = get_logger(__name__)


def _get_fc(fname: UPath) -> tuple[UPath, dict[str, Any]]:
    """Read the FeatureCollection from the specified file.

    This is intended to be used as a handler for multiprocessing.

    Args:
        fname: the filename to read.

    Returns:
        a tuple (fname, fc) of the filename and the decoded FeatureCollection JSON.
    """
    with fname.open() as f:
        return fname, json.load(f)


def apply_nms(
    features: list[dict[str, Any]],
    distance_threshold: float,
) -> list[dict[str, Any]]:
    """Apply non-maximum suppression over the points.

    Although we run NMS inside the object detector, we need to run a global NMS again
    because there two levels where we are dividing into patches -- at the global level,
    where we start different prediction tasks for every 32768x32768 patch, and again
    within the tasks, where we process each 2048x2048 sub-patch. So there can be
    redundant detections across these boundaries.

    Args:
        features: the list of JSON Feature objects.
        distance_threshold: the distance threshold to match points.

    Returns:
        new Features with NMS applied.
    """
    # A few multiples of the distance threshold is generally a good grid size.
    grid_index = GridIndex(distance_threshold * 10)

    # Insert features into the index.
    for idx, feat in enumerate(features):
        coordinates = feat["geometry"]["coordinates"]
        box = (coordinates[0], coordinates[1], coordinates[0], coordinates[1])
        grid_index.insert(box, idx)

    # Now we iterate over the features and use the index to identify other features
    # that are nearby. If the other feature has a higher score then we delete the
    # feature.
    good_features = []
    for idx, feat in enumerate(features):
        coordinates = feat["geometry"]["coordinates"]
        # Create search box with distance threshold padding.
        box = (
            coordinates[0] - distance_threshold,
            coordinates[1] - distance_threshold,
            coordinates[0] + distance_threshold,
            coordinates[1] + distance_threshold,
        )
        is_feat_okay = True
        for other_idx in grid_index.query(box):
            other_feat = features[other_idx]
            if idx == other_idx:
                continue
            if feat["properties"]["score"] < other_feat["properties"]["score"]:
                continue
            other_coordinates = other_feat["geometry"]["coordinates"]
            distance = math.sqrt(
                (coordinates[0] - other_coordinates[0]) ** 2
                + (coordinates[1] - other_coordinates[1]) ** 2
            )
            if distance > distance_threshold:
                continue
            is_feat_okay = False
            break

        if is_feat_okay:
            good_features.append(feat)

    return good_features


def merge_points(
    application: Application,
    label: str,
    predict_path: str,
    merged_path: str,
    workers: int = 32,
) -> None:
    """Merge Satlas point outputs.

    This merges the outputs across different prediction tasks for this timestamp.

    Args:
        application: the application.
        label: YYYY-MM representation of the time range used for this prediction run.
        predict_path: output path of the prediction pipeline where GeoJSONs from all
            the different tasks have been written.
        merged_path: folder to write merged predictions. The filename will be
            YYYY-MM.geojson.
        workers: number of worker processes.
    """
    predict_upath = UPath(predict_path)
    merged_features = []
    merged_patches: dict[str, list[tuple[int, int]]] = {}

    fnames = [fname for fname in predict_upath.iterdir() if fname.name != "index"]
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(_get_fc, fnames)

    # Get category remapping in case one is specified for this application.
    category_map = APP_CATEGORY_MAPS.get(application, {})

    # Iterate over each of the files produced by a prediction task.
    # We merge both the predicted points along with the valid patches (patches
    # processed by the task that had available input images).
    for fname, cur_fc in tqdm.tqdm(outputs, total=len(fnames)):
        # The projection information may be missing if there are no valid patches.
        # In that case we can skip the file since it has neither valid patches that we
        # need to track nor any predicted points.
        if "crs" not in cur_fc["properties"]:
            # Just do some sanity checks, there should be no features and no valid
            # patches.
            assert len(cur_fc["features"]) == 0
            patch_list = list(cur_fc["properties"]["valid_patches"].values())
            assert len(patch_list) == 1 and len(patch_list[0]) == 0
            continue

        src_projection = Projection.deserialize(cur_fc["properties"])
        crs_str = str(src_projection.crs)

        # We ultimately want to store longitude/latitude but
        # smooth_point_labels_viterbi.go needs to know the projection and x/y so we
        # write them as properties of the feature, while converting the geometry
        # coordinates to WGS84.
        for feat in cur_fc["features"]:
            col, row = feat["geometry"]["coordinates"]
            feat["properties"]["col"] = int(col)
            feat["properties"]["row"] = int(row)
            feat["properties"]["projection"] = crs_str

            src_geom = STGeometry(src_projection, shapely.Point(col, row), None)
            dst_geom = src_geom.to_projection(WGS84_PROJECTION)
            feat["geometry"]["coordinates"] = [dst_geom.shp.x, dst_geom.shp.y]

            category = feat["properties"]["category"]
            if category in category_map:
                feat["properties"]["category"] = category_map[category]

            merged_features.append(feat)

        # Merge the valid patches too, these indicate which portions of the world
        # actually had image content for the current timestep.
        assert len(cur_fc["properties"]["valid_patches"]) == 1
        if crs_str not in merged_patches:
            merged_patches[crs_str] = []
        merged_patches[crs_str].extend(cur_fc["properties"]["valid_patches"][crs_str])

    p.close()

    merged_upath = UPath(merged_path)
    merged_fname = merged_upath / f"{label}.geojson"
    with merged_fname.open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": merged_features,
                "properties": {
                    "valid_patches": merged_patches,
                },
            },
            f,
        )


def smooth_points(
    application: Application,
    label: str,
    merged_path: str,
    smoothed_path: str,
) -> None:
    """Smooth the Satlas point outputs.

    It applies Viterbi smoothing that takes into account merged outputs from previous
    time ranges, and uploads the results.

    Args:
        application: the application.
        label: YYYY-MM representation of the time range used for this prediction run.
        merged_path: folder to write merged predictions. The filename will be
            YYYY-MM.geojson.
        smoothed_path: folder to write smoothed predictions. The filename will be
            YYYY-MM.geojson.
    """
    merged_upath = UPath(merged_path)
    # Download the merged prediction history (ending with the one we just wrote) and
    # run smoothing.
    smoothed_upath = UPath(smoothed_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_upath = UPath(tmp_dir)
        tmp_merged_dir = tmp_upath / "merged"
        tmp_smoothed_dir = tmp_upath / "smoothed"
        tmp_hist_fname = tmp_upath / "history.geojson"

        tmp_merged_dir.mkdir()
        tmp_smoothed_dir.mkdir()

        labels: list[str] = []
        for merged_fname in merged_upath.iterdir():
            # Get the label like 2024-01 from 2024-01.geojson.
            if not merged_fname.name.endswith(".geojson"):
                continue
            label = merged_fname.name.split(".")[0]

            local_fname = tmp_merged_dir / merged_fname.name
            with merged_fname.open("rb") as src:
                with local_fname.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            labels.append(label)

        # Sort by YYYY-MM, since the smoothing function expects us to provide all of
        # the labels in temporal order.
        labels.sort()

        # Smoothing is handled by a Go script.
        subprocess.check_call(
            [
                "rslp/satlas/scripts/smooth_point_labels_viterbi",
                "--labels",
                ",".join(labels),
                "--fname",
                (tmp_merged_dir / "LABEL.geojson").path,
                "--out",
                (tmp_smoothed_dir / "LABEL.geojson").path,
                "--hist",
                tmp_hist_fname.path,
            ],
        )  # nosec

        # Now we can upload the smoothed per-timestep files.
        for label in labels:
            src_path = tmp_smoothed_dir / f"{label}.geojson"
            dst_path = smoothed_upath / f"{label}.geojson"
            with src_path.open("rb") as src:
                with dst_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        # The smoothing also produces a history GeoJSON containing all of the points
        # annotated with start/end properties indicating the first and last timesteps
        # when the point was detected. (In this case, points detected over time are
        # merged into a single GeoJSON feature.) So we upload that too.
        # This history file is the one that used to create vector tiles for the web
        # application.
        dst_path = smoothed_upath / "history.geojson"
        with tmp_hist_fname.open("rb") as src:
            with dst_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
