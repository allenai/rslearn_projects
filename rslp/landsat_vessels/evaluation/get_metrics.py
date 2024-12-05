"""This script is used to compute the evaluation metrics for the vessel detection pipeline."""

import argparse
import hashlib
import json
from functools import partial
from multiprocessing import Pool

import shapely
from haversine import Unit, haversine
from rslearn.const import WGS84_PROJECTION
from rslearn.utils import Projection, STGeometry
from upath import UPath

from rslp.landsat_vessels.config import MATCH_THRESHOLD_KM
from rslp.utils.mp import init_mp


def process_window(
    window_name: str,
    ground_truth_dir: UPath,
    predictions_dir: UPath,
) -> tuple[int, int, int]:
    """Process a single window to get matches, missed expected, and unmatched predicted.

    Args:
        window_name: the name of the window.
        ground_truth_dir: UPath to the ground-truth window directory.
        predictions_dir: UPath to the directory containing the result JSON files.

    Returns:
        matches: the number of matches.
        missed_expected: the number of missed expected.
        unmatched_predicted: the number of unmatched predicted.
    """
    matches = 0
    missed_expected = 0
    unmatched_predicted = 0

    print(f"processing window {window_name}")
    expected_detections = []
    predicted_detections = []

    # Get the ground-truth detections
    window_path = ground_truth_dir / window_name
    label_fname = window_path / "layers" / "label" / "data.geojson"
    with label_fname.open() as f:
        label_data = json.load(f)
    # Convert coordinates to longitude/latitude
    projection = Projection.deserialize(label_data["properties"])
    for feature in label_data["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        col = int(shp.centroid.x)
        row = int(shp.centroid.y)
        src_geom = STGeometry(projection, shapely.Point(col, row), None)
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        lon = dst_geom.shp.x
        lat = dst_geom.shp.y
        expected_detections.append((lat, lon))

    # Get the predicted detections
    json_path = predictions_dir / f"{window_name}.json"
    if not json_path.exists():
        print(f"no prediction file for window {window_name}")
        return matches, missed_expected, unmatched_predicted
    with json_path.open() as f:
        predicted_data = json.load(f)
    for item in predicted_data:
        predicted_detections.append((item["latitude"], item["longitude"]))

    # Compute the metrics
    current_missed_expected = set(expected_detections)
    for pred in predicted_detections:
        matched = False
        for exp in expected_detections:
            distance = haversine(pred, exp, unit=Unit.KILOMETERS)
            if distance <= MATCH_THRESHOLD_KM:
                matches += 1
                if exp in current_missed_expected:
                    current_missed_expected.remove(exp)
                matched = True
                break
        if not matched:
            unmatched_predicted += 1
    missed_expected += len(current_missed_expected)
    print(
        f"window {window_name}, "
        f"matches: {matches}, "
        f"missed_expected: {missed_expected}, "
        f"unmatched_predicted: {unmatched_predicted}"
    )

    return matches, missed_expected, unmatched_predicted


def compute_metrics(
    ground_truth_dir: UPath,
    predictions_dir: UPath,
) -> tuple[float, float, float]:
    """Compute the evaluation metrics for the landsat vessel detection pipeline.

    This function collects the ground-truth from the detector dataset and
    the predicted results from the result JSON files to compute the metrics.

    Args:
        ground_truth_dir: UPath to the ground-truth window directory.
        predictions_dir: UPath to the directory containing the result JSON files.

    Returns:
        recall: the recall of the pipeline.
        precision: the precision of the pipeline.
        f1_score: the f1 score of the pipeline.
    """
    window_names = []
    for dir in ground_truth_dir.iterdir():
        if dir.is_dir():
            if hashlib.sha256(dir.name.encode()).hexdigest()[0] in ["0", "1"]:
                window_names.append(str(dir.name))

    with Pool() as pool:
        results = pool.map(
            partial(
                process_window,
                ground_truth_dir=ground_truth_dir,
                predictions_dir=predictions_dir,
            ),
            window_names,
        )
    # Aggregate results
    matches = sum(res[0] for res in results)
    missed_expected = sum(res[1] for res in results)
    unmatched_predicted = sum(res[2] for res in results)
    print(
        f"matches: {matches}, "
        f"missed_expected: {missed_expected}, "
        f"unmatched_predicted: {unmatched_predicted}"
    )
    recall = matches / (matches + missed_expected)
    precision = matches / (matches + unmatched_predicted)
    f1_score = 2 * precision * recall / (precision + recall)

    return recall, precision, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    args = parser.parse_args()

    init_mp()
    recall, precision, f1_score = compute_metrics(
        ground_truth_dir=UPath(args.ground_truth_dir),
        predictions_dir=UPath(args.predictions_dir),
    )
    print(f"recall: {recall}, precision: {precision}, f1_score: {f1_score}")
