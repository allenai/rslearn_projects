"""This script is used to compute the evaluation metrics for the vessel detection pipeline."""

import argparse
import hashlib
import json
from functools import partial
from multiprocessing import Pool

import dateutil.parser
import shapely
from rasterio.crs import CRS
from rslearn.utils import Projection
from upath import UPath

from rslp.landsat_vessels.config import (
    CLASSIFY_MODEL_CONFIG,
    CLASSIFY_WINDOW_SIZE,
    DETECT_MODEL_EVAL_CONFIG,
)
from rslp.landsat_vessels.predict_pipeline import VesselDetection
from rslp.landsat_vessels.predict_pipeline import (
    run_classifier as materialize_and_run_classifier,
)
from rslp.utils.mp import init_mp
from rslp.utils.rslearn import materialize_dataset, run_model_predict


def is_match(
    point1: tuple[int, int],
    point2: tuple[int, int],
    buffer: int = CLASSIFY_WINDOW_SIZE,
) -> bool:
    """Check if two points are within the buffer distance.

    Args:
        point1: the first point.
        point2: the second point.
        buffer: the buffer distance.

    Returns:
        True if the points are within the buffer distance, False otherwise.
    """
    return abs(point1[0] - point2[0]) <= buffer and abs(point1[1] - point2[1]) <= buffer


def process_window(
    window_name: str,
    detector_ds_path: UPath,
    detector_group: str,
    classifier_ds_path: UPath,
) -> tuple[int, int, int]:
    """Process a single window to get matches, missed expected, and unmatched predicted.

    Args:
        window_name: the name of the window.
        detector_ds_path: path to the detector dataset.
        detector_group: the group of the detector dataset.
        classifier_ds_path: path to the classifier dataset.

    Returns:
        matches: the number of matches.
        missed_expected: the number of missed expected.
        unmatched_predicted: the number of unmatched predicted.
    """
    matches = 0
    missed_expected = 0
    unmatched_predicted = 0

    # Only use the validation set
    if hashlib.sha256(window_name.encode()).hexdigest()[0] not in ["0", "1"]:
        return matches, missed_expected, unmatched_predicted

    print(f"processing window {window_name}")
    expected_detections = []
    predicted_detections = []

    # Get the expected detections from the detector dataset
    window_path = detector_ds_path / "windows" / detector_group / window_name
    label_fname = window_path / "layers" / "label" / "data.geojson"
    with label_fname.open() as f:
        feature_collection = json.load(f)
    for feature in feature_collection["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        col = int(shp.centroid.x)
        row = int(shp.centroid.y)
        expected_detections.append((col, row))

    # Get the predicted detections from the classifier dataset
    classifier_group_path = classifier_ds_path / "windows" / window_name
    if classifier_group_path.exists():
        classifier_window_names = [
            dir.name for dir in classifier_group_path.iterdir() if dir.is_dir()
        ]
        for classifier_window_name in classifier_window_names:
            output_fname = (
                classifier_group_path
                / classifier_window_name
                / "layers"
                / "output"
                / "data.geojson"
            )
            with output_fname.open() as f:
                output_feature_collection = json.load(f)
            for feature in output_feature_collection["features"]:
                label = feature["properties"]["label"]
                if label != "correct":
                    continue
                shp = shapely.geometry.shape(feature["geometry"])
                col = int(shp.centroid.x)
                row = int(shp.centroid.y)
                predicted_detections.append((col, row))

    # Compute the metrics
    current_missed_expected = set(expected_detections)
    for pred in predicted_detections:
        matched = False
        for exp in expected_detections:
            if is_match(pred, exp):
                matches += 1
                if exp in current_missed_expected:
                    current_missed_expected.remove(exp)
                matched = True
                break
        if not matched:
            unmatched_predicted += 1
    missed_expected += len(current_missed_expected)

    return matches, missed_expected, unmatched_predicted


def compute_metrics(
    detector_ds_path: UPath,
    detector_group: str,
    classifier_ds_path: UPath,
) -> tuple[float, float, float]:
    """Compute the evaluation metrics for the vessel detection pipeline.

    This function collects the ground-truth from the detector dataset and
    the predicted results from the classifier dataset to compute the metrics.

    Args:
        detector_ds_path: UPath to the detector dataset.
        detector_group: the group of the detector dataset.
        classifier_ds_path: UPath to the classifier dataset.

    Returns:
        recall: the recall of the pipeline.
        precision: the precision of the pipeline.
        f1_score: the f1 score of the pipeline.
    """
    detector_group_path = detector_ds_path / "windows" / detector_group
    window_names = [dir.name for dir in detector_group_path.iterdir() if dir.is_dir()]

    with Pool() as pool:
        results = pool.map(
            partial(
                process_window,
                detector_ds_path=detector_ds_path,
                detector_group=detector_group,
                classifier_ds_path=classifier_ds_path,
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
    # Compute final metrics
    recall = matches / (matches + missed_expected)
    precision = matches / (matches + unmatched_predicted)
    f1_score = 2 * precision * recall / (precision + recall)

    return recall, precision, f1_score


def evaluate_pipeline(
    detector_ds_path: UPath,
    detector_group: str,
    materialize_detector_ds: bool,
    run_detector: bool,
    classifier_ds_path: UPath,
    materialize_classifier_ds: bool,
    run_classifier: bool,
) -> tuple[float, float, float]:
    """Run the evaluation pipeline.

    Args:
        detector_ds_path: UPath to the detector dataset.
        detector_group: the group of the detector dataset.
        materialize_detector_ds: whether to materialize the detector dataset.
        run_detector: whether to run the detector.
        classifier_ds_path: UPath to the classifier dataset.
        materialize_classifier_ds: whether to materialize the classifier dataset.
        run_classifier: whether to run the classifier.

    Returns:
        recall: the recall of the pipeline.
        precision: the precision of the pipeline.
        f1_score: the f1 score of the pipeline.
    """
    if materialize_detector_ds:
        materialize_dataset(detector_ds_path, group=detector_group)
    if run_detector:
        run_model_predict(DETECT_MODEL_EVAL_CONFIG, detector_ds_path)

    # Read the detector output
    detector_group_path = detector_ds_path / "windows" / detector_group
    window_names = [dir.name for dir in detector_group_path.iterdir() if dir.is_dir()]
    for window_name in window_names:
        # Only use the validation set
        if hashlib.sha256(window_name.encode()).hexdigest()[0] not in ["0", "1"]:
            continue
        # Check if the output file exists
        window_path = detector_ds_path / "windows" / detector_group / window_name
        output_fname = window_path / "layers" / "output" / "data.geojson"
        if not output_fname.exists():
            continue
        # Read the metadata and output file
        metadata_fname = window_path / "metadata.json"
        detections: list[VesselDetection] = []
        with output_fname.open() as f:
            feature_collection = json.load(f)
        with metadata_fname.open() as f:
            metadata = json.load(f)
        time_range = (
            dateutil.parser.isoparse(metadata["time_range"][0]),
            dateutil.parser.isoparse(metadata["time_range"][1]),
        )
        crs = CRS.from_string(feature_collection["properties"]["crs"])
        x_resolution = feature_collection["properties"]["x_resolution"]
        y_resolution = feature_collection["properties"]["y_resolution"]
        projection = Projection(
            crs=crs,
            x_resolution=x_resolution,
            y_resolution=y_resolution,
        )
        for feature in feature_collection["features"]:
            shp = shapely.geometry.shape(feature["geometry"])
            col = int(shp.centroid.x)
            row = int(shp.centroid.y)
            score = feature["properties"]["score"]
            detections.append(
                VesselDetection(
                    col=col,
                    row=row,
                    projection=projection,
                    score=score,
                )
            )
        if len(detections) == 0:
            continue

        if materialize_classifier_ds and run_classifier:
            materialize_and_run_classifier(
                ds_path=classifier_ds_path,
                detections=detections,
                time_range=time_range,
                item=None,
                group=window_name,
            )
        elif not materialize_classifier_ds and run_classifier:
            run_model_predict(
                CLASSIFY_MODEL_CONFIG,
                classifier_ds_path,
                groups=[window_name],
            )
        else:
            break

    # Compute final metrics after running the detector and classifier
    recall, precision, f1_score = compute_metrics(
        detector_ds_path,
        detector_group,
        classifier_ds_path,
    )
    return recall, precision, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Detector config
    parser.add_argument("--detector_ds_path", type=str, required=True)
    parser.add_argument("--detector_group", type=str, required=True)
    parser.add_argument("--materialize_detector_ds", action="store_true")
    parser.add_argument("--run_detector", action="store_true")
    # Classifier config
    parser.add_argument("--classifier_ds_path", type=str, required=True)
    parser.add_argument("--materialize_classifier_ds", action="store_true")
    parser.add_argument("--run_classifier", action="store_true")
    # Output config
    parser.add_argument("--output_path", type=str, required=False)
    args = parser.parse_args()

    init_mp()
    # If both detector and classifier are already run, compute the metrics directly
    recall, precision, f1_score = evaluate_pipeline(
        UPath(args.detector_ds_path),
        args.detector_group,
        args.materialize_detector_ds,
        args.run_detector,
        UPath(args.classifier_ds_path),
        args.materialize_classifier_ds,
        args.run_classifier,
    )
    print(f"recall: {recall}, precision: {precision}, f1_score: {f1_score}")
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(
                {"recall": recall, "precision": precision, "f1_score": f1_score}, f
            )
