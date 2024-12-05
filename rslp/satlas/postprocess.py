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
MAX_METERS_PER_DEGREE = 111111

# Threshold on Euclidean distance between lat/lon for NMS.
# We just do Euclidean distance for speed/simplicity since NMS doesn't need to be super
# exact.
NMS_DISTANCE_THRESHOLD = 100 / MAX_METERS_PER_DEGREE

logger = get_logger(__name__)


def _get_fc(fname: UPath) -> dict[str, Any]:
    with fname.open() as f:
        return json.load(f)


def apply_nms(
    features: list[dict[str, Any]],
    distance_threshold: float,
) -> list[dict[str, Any]]:
    """Apply non-maximum suppression over the points.

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


def postprocess_points(
    application: Application,
    label: str,
    predict_path: str,
    merged_path: str,
    smoothed_path: str,
    workers: int = 32,
) -> None:
    """Post-process Satlas point outputs.

    This merges the outputs across different prediction tasks for this timestamp and
    spatial tile. Then it applies Viterbi smoothing that takes into account merged
    outputs from previous time ranges, and uploads the results.

    Args:
        application: the application.
        label: YYYY-MM representation of the time range used for this prediction run.
        predict_path: output path of the prediction pipeline where GeoJSONs from all
            the different tasks have been written.
        merged_path: folder to write merged predictions. The filename will be
            YYYY-MM.geojson.
        smoothed_path: folder to write smoothed predictions. The filename will be
            YYYY-MM.geojson.
        workers: number of worker processes.
    """
    # Merge the predictions.
    predict_upath = UPath(predict_path)
    merged_features = []
    merged_patches: dict[str, list[tuple[int, int]]] = {}

    fnames = [fname for fname in predict_upath.iterdir() if fname.name != "index"]
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(_get_fc, fnames)

    for cur_fc in tqdm.tqdm(outputs, total=len(fnames)):
        # The projection information may be missing if there are no valid patches.
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

            merged_features.append(feat)

        # Merge the valid patches too, these indicate which portions of the world
        # actually had image content for the current timestep.
        assert len(cur_fc["properties"]["valid_patches"]) == 1
        if crs_str not in merged_patches:
            merged_patches[crs_str] = []
        merged_patches[crs_str].extend(cur_fc["properties"]["valid_patches"][crs_str])

    p.close()

    nms_features = apply_nms(merged_features, distance_threshold=NMS_DISTANCE_THRESHOLD)
    logger.info(
        "NMS filtered from %d -> %d features", len(merged_features), len(nms_features)
    )

    merged_upath = UPath(merged_path)
    merged_fname = merged_upath / f"{label}.geojson"
    with merged_fname.open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": nms_features,
                "properties": {
                    "valid_patches": merged_patches,
                },
            },
            f,
        )

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

        # Sort by YYYY-MM.
        labels.sort()

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

        for label in labels:
            src_path = tmp_smoothed_dir / f"{label}.geojson"
            dst_path = smoothed_upath / f"{label}.geojson"
            with src_path.open("rb") as src:
                with dst_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        dst_path = smoothed_upath / "history.geojson"
        with tmp_hist_fname.open("rb") as src:
            with dst_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
