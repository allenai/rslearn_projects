"""This script is used to compute the evaluation metrics for the vessel detection pipeline."""

import argparse
import hashlib
import json
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import dateutil.parser
import shapely
from rasterio.crs import CRS
from rslearn.data_sources import data_source_from_config
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils import Projection
from upath import UPath

from rslp.landsat_vessels.config import (
    CLASSIFY_MODEL_CONFIG,
    CLASSIFY_WINDOW_SIZE,
    LANDSAT_LAYER_NAME,
)
from rslp.utils.mp import init_mp
from rslp.utils.rslearn import materialize_dataset, run_model_predict


class VesselDetection:
    """A vessel detected in a Landsat scene."""

    def __init__(
        self,
        col: int,
        row: int,
        score: float,
        projection: Projection,
        time_range: tuple[datetime, datetime],
    ) -> None:
        """Create a new VesselDetection.

        Args:
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            score: confidence score from object detector.
            projection: the projection used.
            time_range: the time range of the scene.
        """
        self.col = col
        self.row = row
        self.score = score
        self.projection = projection
        self.time_range = time_range


def process_detector_output(
    detector_dataset_path: UPath,
    group: str,
    window_name: str,
) -> tuple[list[VesselDetection], str]:
    """Get the output of the detector.

    Args:
        detector_dataset_path: path to the detector dataset.
        group: the group of the detector dataset.
        window_name: the name of the window to check.

    Returns:
        detections: the detections in the window.
        scene_id: the scene id of the window.
    """
    window_path = detector_dataset_path / "windows" / group / window_name
    # Check if the output file exists
    output_fname = window_path / "layers" / "output" / "data.geojson"
    if not output_fname.exists():
        return [], ""

    # Get more details about the window
    metadata_fname = window_path / "metadata.json"
    items_fname = window_path / "items.json"
    detections: list[VesselDetection] = []
    with output_fname.open() as f:
        feature_collection = json.load(f)
    with metadata_fname.open() as f:
        metadata = json.load(f)
    with items_fname.open() as f:
        items = json.load(f)

    scene_id = items[0]["serialized_item_groups"][0][0]["name"]
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

    # Prepare the detections with the details
    for feature in feature_collection["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        col = int(shp.centroid.x)
        row = int(shp.centroid.y)
        score = feature["properties"]["score"]
        detections.append(
            VesselDetection(
                col=col,
                row=row,
                score=score,
                projection=projection,
                time_range=time_range,
            )
        )

    return detections, scene_id


def materialize_classifier_dataset(
    detector_dataset_path: UPath,
    detector_group: str,
    classifier_dataset_path: UPath,
) -> None:
    """Materialize the classifier dataset from the detector output.

    Args:
        detector_dataset_path: path to the detector dataset.
        detector_group: the group of the detector dataset.
        classifier_dataset_path: path to the classifier dataset.
    """
    detector_dataset_path = UPath(detector_dataset_path)
    classifier_dataset_path = UPath(classifier_dataset_path)
    classifier_dataset_path.mkdir(parents=True, exist_ok=True)

    # Go through windows within the detector group
    detector_window_path = detector_dataset_path / "windows" / detector_group
    window_names = [dir.name for dir in detector_window_path.iterdir() if dir.is_dir()]
    for window_name in window_names:
        # We only use the validation set
        if hashlib.sha256(window_name.encode()).hexdigest()[0] not in ["0", "1"]:
            continue
        detections, scene_id = process_detector_output(
            detector_dataset_path,
            detector_group,
            window_name,
        )
        if len(detections) == 0:
            continue
        print(f"processing window {window_name}")
        dataset = Dataset(classifier_dataset_path)
        data_source: LandsatOliTirs = data_source_from_config(
            dataset.layers[LANDSAT_LAYER_NAME], dataset.path
        )
        item = data_source.get_item_by_name(scene_id)
        window_paths: list[UPath] = []
        for detection in detections:
            cur_window_name = f"{detection.col}_{detection.row}"
            window_path = (
                classifier_dataset_path / "windows" / window_name / cur_window_name
            )
            bounds = [
                detection.col - CLASSIFY_WINDOW_SIZE // 2,
                detection.row - CLASSIFY_WINDOW_SIZE // 2,
                detection.col + CLASSIFY_WINDOW_SIZE // 2,
                detection.row + CLASSIFY_WINDOW_SIZE // 2,
            ]
            window = Window(
                path=window_path,
                group=window_name,
                name=cur_window_name,
                projection=detection.projection,
                bounds=bounds,
                time_range=detection.time_range,
            )
            window.save()
            window_paths.append(window_path)
            if item:
                layer_data = WindowLayerData(LANDSAT_LAYER_NAME, [[item.serialize()]])
                window.save_layer_datas(dict(LANDSAT_LAYER_NAME=layer_data))

        materialize_dataset(classifier_dataset_path, group=window_name)


def run_classifier(
    classifier_dataset_path: UPath,
) -> None:
    """Run the classifier on the classifier dataset.

    Args:
        classifier_dataset_path: path to the classifier dataset.
    """
    classifier_dataset_path = UPath(classifier_dataset_path)
    # Check if the classifier dataset is materialized
    classifier_window_path = classifier_dataset_path / "windows"
    window_names = [
        dir.name for dir in classifier_window_path.iterdir() if dir.is_dir()
    ]
    if len(window_names) == 0:
        raise ValueError("Classifier dataset is not materialized")

    run_model_predict(CLASSIFY_MODEL_CONFIG, classifier_dataset_path, window_names)


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
    detector_dataset_path: UPath,
    detector_group: str,
    classifier_dataset_path: UPath,
) -> tuple[int, int, int]:
    """Process a single window to get matches, missed expected, and unmatched predicted.

    Args:
        window_name: the name of the window.
        detector_dataset_path: path to the detector dataset.
        detector_group: the group of the detector dataset.
        classifier_dataset_path: path to the classifier dataset.

    Returns:
        matches: the number of matches.
        missed_expected: the number of missed expected.
        unmatched_predicted: the number of unmatched predicted.
    """
    matches = 0
    missed_expected = 0
    unmatched_predicted = 0

    # We only use the validation set
    if hashlib.sha256(window_name.encode()).hexdigest()[0] not in ["0", "1"]:
        return matches, missed_expected, unmatched_predicted

    print(f"processing window {window_name}")

    expected_detections = []
    predicted_detections = []

    # Get the expected detections from the detector dataset
    window_path = detector_dataset_path / "windows" / detector_group / window_name
    label_fname = window_path / "layers" / "label" / "data.geojson"
    with label_fname.open() as f:
        feature_collection = json.load(f)
    for feature in feature_collection["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        col = int(shp.centroid.x)
        row = int(shp.centroid.y)
        expected_detections.append((col, row))

    # Get the predicted detections from the classifier dataset
    classifier_window_path = classifier_dataset_path / "windows" / window_name
    if classifier_window_path.exists():
        classifier_window_names = [
            dir.name for dir in classifier_window_path.iterdir() if dir.is_dir()
        ]
        for classifier_window_name in classifier_window_names:
            output_fname = (
                classifier_window_path
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
    detector_dataset_path: UPath,
    detector_group: str,
    classifier_dataset_path: UPath,
) -> tuple[float, float, float]:
    """Compute the evaluation metrics for the vessel detection pipeline.

    This function collects the ground-truth from the detector dataset and
    the predicted results from the classifier dataset to compute the metrics.

    Args:
        detector_dataset_path: path to the detector dataset.
        detector_group: the group of the detector dataset.
        classifier_dataset_path: path to the classifier dataset.

    Returns:
        recall: the recall of the pipeline.
        precision: the precision of the pipeline.
        f1_score: the f1 score of the pipeline.
    """
    detector_dataset_path = UPath(detector_dataset_path)
    classifier_dataset_path = UPath(classifier_dataset_path)
    classifier_dataset_path.mkdir(parents=True, exist_ok=True)

    detector_window_path = detector_dataset_path / "windows" / detector_group
    window_names = [dir.name for dir in detector_window_path.iterdir() if dir.is_dir()]

    with Pool() as pool:
        results = pool.map(
            partial(
                process_window,
                detector_dataset_path=detector_dataset_path,
                detector_group=detector_group,
                classifier_dataset_path=classifier_dataset_path,
            ),
            window_names,
        )

    # Aggregate results
    matches = sum(res[0] for res in results)
    missed_expected = sum(res[1] for res in results)
    unmatched_predicted = sum(res[2] for res in results)
    print(
        f"matches: {matches}, missed_expected: {missed_expected}, unmatched_predicted: {unmatched_predicted}"
    )

    # Compute the final metrics
    recall = matches / (matches + missed_expected)
    precision = matches / (matches + unmatched_predicted)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"recall: {recall}, precision: {precision}, f1_score: {f1_score}")

    return recall, precision, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_dataset_path", type=str, required=True)
    parser.add_argument("--detector_group", type=str, required=True)
    parser.add_argument("--classifier_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    args = parser.parse_args()

    init_mp()
    # materialize_classifier_dataset needs to be run only when we update the detector
    # materialize_classifier_dataset(
    #     UPath(args.detector_dataset_path),
    #     args.detector_group,
    #     UPath(args.classifier_dataset_path),
    # )
    run_classifier(UPath(args.classifier_dataset_path))
    recall, precision, f1_score = compute_metrics(
        UPath(args.detector_dataset_path),
        args.detector_group,
        UPath(args.classifier_dataset_path),
    )
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(
                {"recall": recall, "precision": precision, "f1_score": f1_score}, f
            )
