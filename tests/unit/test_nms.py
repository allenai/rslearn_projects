from pathlib import Path

import numpy as np
import pytest
import shapely
from rslearn.config import LayerConfig, LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.train.prediction_writer import PendingCropOutput
from rslearn.utils import Feature, STGeometry
from upath import UPath

from rslp.utils.nms import NMSDistanceMerger, distance_nms


class TestDistanceNms:
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    @pytest.fixture
    def nms_window(self, tmp_path: Path) -> Window:
        return Window(
            storage=FileWindowStorage(UPath(tmp_path)),
            group="fake",
            name="fake",
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            time_range=None,
        )

    def test_no_boxes(self, nms_window: Window) -> None:
        """Test with no boxes provided.

        In this case, the merger should return no features.
        """
        merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=[],
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )
        # Expected: No boxes, so the result should be an empty list.
        assert len(merged_features) == 0

    def test_one_duplicate(self, nms_window: Window) -> None:
        """Test NMS with 3 boxes, single class, where one box overlaps another."""
        features = []
        for box, score, class_id in zip(
            [
                [10, 10, 20, 20],  # Box 0
                [12, 12, 22, 22],  # Box 1, overlaps with Box 0
                [30, 30, 40, 40],  # Box 2, separate
            ],
            [0.9, 0.85, 0.8],
            [0, 0, 0],
        ):
            shp = shapely.box(
                self.BOUNDS[0] + float(box[0]),
                self.BOUNDS[1] + float(box[1]),
                self.BOUNDS[0] + float(box[2]),
                self.BOUNDS[1] + float(box[3]),
            )
            geom = STGeometry(self.PROJECTION, shp, None)
            properties = {"score": float(score), "category": class_id}
            features.append(Feature(geom, properties))

        # With distance threshold 5, box 0 (highest score) and box 2 should be kept.
        merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )
        assert set(merged_features) == {features[0], features[2]}

        # With a smaller distance threshold, all of the boxes should be kept.
        merger = NMSDistanceMerger(grid_size=10, distance_threshold=1)
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )
        assert set(merged_features) == set(features)

    def test_negative_coordinates(self, nms_window: Window) -> None:
        """Test with the y coordinates being negative.

        I'm not sure if this test is very useful.
        """
        features = []
        for box, score, class_id in zip(
            [
                [10, -20, 20, -10],  # Box 0
                [12, -22, 22, -12],  # Box 1, overlaps with Box 0
                [30, -40, 40, -30],  # Box 2, separate
            ],
            [0.9, 0.85, 0.8],
            [0, 0, 0],
        ):
            shp = shapely.box(
                self.BOUNDS[0] + float(box[0]),
                self.BOUNDS[1] + float(box[1]),
                self.BOUNDS[0] + float(box[2]),
                self.BOUNDS[1] + float(box[3]),
            )
            geom = STGeometry(self.PROJECTION, shp, None)
            properties = {"score": float(score), "category": class_id}
            features.append(Feature(geom, properties))

        merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )

        # Expected: Box 0 (highest score) and Box 2 (no overlap) should be kept.
        assert set(merged_features) == {features[0], features[2]}

    def test_separate_classes(self, nms_window: Window) -> None:
        """Test with multiple classes where NMS should be performed per class."""
        features = []
        for box, score, class_id in zip(
            [
                [10, 10, 20, 20],  # Class 0, Box 0
                [12, 12, 22, 22],  # Class 0, Box 1 (overlapping with Box 0)
                [10, 10, 20, 20],  # Class 1, Box 2
                [12, 12, 22, 22],  # Class 1, Box 3 (overlapping with Box 2)
            ],
            [0.9, 0.85, 0.8, 0.95],
            [0, 0, 1, 1],
        ):
            shp = shapely.box(
                self.BOUNDS[0] + float(box[0]),
                self.BOUNDS[1] + float(box[1]),
                self.BOUNDS[0] + float(box[2]),
                self.BOUNDS[1] + float(box[3]),
            )
            geom = STGeometry(self.PROJECTION, shp, None)
            properties = {"score": float(score), "category": class_id}
            features.append(Feature(geom, properties))

        merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )
        # Expected: For Class 0, Box 0 kept (higher score); Box 1 suppressed.
        # For Class 1, Box 3 kept (higher score); Box 2 suppressed.
        assert set(merged_features) == {features[0], features[3]}

    def test_class_agnostic(self, nms_window: Window) -> None:
        """Test with multiple classes where NMS should be performed class-agnostic."""
        features = []
        for box, score, class_id in zip(
            [
                [10, 10, 20, 20],  # Class 0, Box 0
                [12, 12, 22, 22],  # Class 0, Box 1 (overlapping with Box 0)
                [10, 10, 20, 20],  # Class 1, Box 2
                [12, 12, 22, 22],  # Class 1, Box 3 (overlapping with Box 2)
            ],
            [0.9, 0.85, 0.8, 0.95],
            [0, 0, 1, 1],
        ):
            shp = shapely.box(
                self.BOUNDS[0] + float(box[0]),
                self.BOUNDS[1] + float(box[1]),
                self.BOUNDS[0] + float(box[2]),
                self.BOUNDS[1] + float(box[3]),
            )
            geom = STGeometry(self.PROJECTION, shp, None)
            properties = {"score": float(score), "category": class_id}
            features.append(Feature(geom, properties))

        merger = NMSDistanceMerger(
            grid_size=10, distance_threshold=5, class_agnostic=True
        )
        merged_features = merger.merge(
            nms_window,
            [
                PendingCropOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
            LayerConfig(type=LayerType.VECTOR),
        )
        # Expected: Box 3 kept (highest score); Box 0, Box 1, and Box 2 suppressed.
        assert set(merged_features) == {features[3]}


class TestDistanceNmsFunction:
    """Tests for the suppression pass itself, shared by the merger and by callers that
    reduce detections across separately-predicted windows."""

    THRESHOLD = 10

    def _keep(
        self, centers: list[tuple[float, float]], scores: list[float]
    ) -> list[int]:
        return distance_nms(
            np.array(centers, dtype=float),
            np.array(scores, dtype=float),
            self.THRESHOLD,
        )

    def test_no_detections(self) -> None:
        assert distance_nms(np.zeros((0, 2)), np.zeros(0), self.THRESHOLD) == []

    def test_single_detection_survives(self) -> None:
        assert self._keep([(5.0, 5.0)], [0.5]) == [0]

    def test_close_pair_keeps_the_higher_score(self) -> None:
        assert self._keep([(100.0, 100.0), (103.0, 102.0)], [0.4, 0.9]) == [1]

    def test_order_does_not_decide_the_winner(self) -> None:
        """Whichever tile reported first, the better-scoring detection is the survivor."""
        assert self._keep([(100.0, 100.0), (103.0, 102.0)], [0.9, 0.4]) == [0]

    def test_distant_pair_both_survive(self) -> None:
        assert sorted(self._keep([(0.0, 0.0), (500.0, 500.0)], [0.9, 0.8])) == [0, 1]

    def test_distance_is_euclidean_not_per_axis(self) -> None:
        """A diagonal 12.7px apart exceeds the threshold, though each axis is under it."""
        assert sorted(self._keep([(0.0, 0.0), (9.0, 9.0)], [0.9, 0.8])) == [0, 1]

    def test_threshold_is_inclusive(self) -> None:
        assert self._keep([(0.0, 0.0), (0.0, 10.0)], [0.9, 0.8]) == [0]

    def test_equal_scores_are_broken_deterministically(self) -> None:
        """Ties must not depend on input order, or reruns disagree."""
        forward = self._keep([(0.0, 0.0), (2.0, 2.0)], [0.7, 0.7])
        backward = self._keep([(2.0, 2.0), (0.0, 0.0)], [0.7, 0.7])

        assert len(forward) == len(backward) == 1

    def test_suppression_does_not_chain(self) -> None:
        """B is near A and C is near B, but C is far from A, so C is not suppressed."""
        keep = self._keep([(0.0, 0.0), (8.0, 0.0), (16.0, 0.0)], [0.9, 0.5, 0.8])

        assert sorted(keep) == [0, 2]

    def test_indices_are_returned_from_the_given_set(self) -> None:
        """Callers pass their own indices when suppressing a subset."""
        keep = distance_nms(
            np.array([(100.0, 100.0), (103.0, 102.0)]),
            np.array([0.4, 0.9]),
            self.THRESHOLD,
            indices=np.array([7, 9]),
        )

        assert keep == [9]
