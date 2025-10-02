from pathlib import Path

import pytest
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.train.prediction_writer import PendingPatchOutput
from rslearn.utils import Feature, STGeometry
from upath import UPath

from rslp.utils.nms import NMSDistanceMerger


class TestDistanceNms:
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    @pytest.fixture
    def nms_window(self, tmp_path: Path) -> Window:
        return Window(
            path=UPath(tmp_path),
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
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=[],
                )
            ],
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
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
        )
        assert set(merged_features) == {features[0], features[2]}

        # With a smaller distance threshold, all of the boxes should be kept.
        merger = NMSDistanceMerger(grid_size=10, distance_threshold=1)
        merged_features = merger.merge(
            nms_window,
            [
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
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
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
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
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
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
                PendingPatchOutput(
                    bounds=self.BOUNDS,
                    output=features,
                )
            ],
        )
        # Expected: Box 3 kept (highest score); Box 0, Box 1, and Box 2 suppressed.
        assert set(merged_features) == {features[3]}
