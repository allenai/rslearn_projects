"""Unit tests for PointToRasterWindowPreparer."""

import uuid
from datetime import UTC, datetime

import pytest
from esrun.runner.models.training.labeled_data import (
    AnnotationTask,
    LabeledSTGeometry,
)
from rslearn.config import DType
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from shapely.geometry import Point, Polygon

from rslp.esrun.tools.labeled_window_preparers.point_to_raster_window_preparer import (
    PointToRasterWindowPreparer,
)


class TestPointToRasterWindowPreparer:
    """Test cases for PointToRasterWindowPreparer."""

    def _create_annotation_task(
        self, task_id_str: str, annotations: list
    ) -> AnnotationTask:
        """Helper to create a valid AnnotationTask."""
        # Create a task geometry that encompasses all annotations
        task_geometry = STGeometry(
            WGS84_PROJECTION,
            Point(10.0, 60.0),  # Default location in Norway
            None,
        )

        return AnnotationTask(
            task_id=uuid.uuid4(),
            task_st_geometry=task_geometry,
            annotations=annotations,
        )

    def test_init_valid_parameters(self) -> None:
        """Test initialization with valid parameters."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=31,
            window_resolution=10.0,
            dtype="float32",
            nodata_value=-1,
        )
        assert preparer.window_buffer == 31
        assert preparer.window_resolution == 10.0
        assert preparer.dtype == DType("float32")
        assert preparer.nodata_value == -1

    def test_init_different_dtypes(self) -> None:
        """Test initialization with different data types."""
        # Test uint8
        preparer = PointToRasterWindowPreparer(
            window_buffer=15, window_resolution=5.0, dtype="uint8", nodata_value=255
        )
        assert preparer.dtype == DType("uint8")

        # Test int16
        preparer = PointToRasterWindowPreparer(
            window_buffer=15, window_resolution=5.0, dtype="int16", nodata_value=-32768
        )
        assert preparer.dtype == DType("int16")

        # Test case insensitive
        preparer = PointToRasterWindowPreparer(
            window_buffer=15, window_resolution=5.0, dtype="FLOAT32", nodata_value=0
        )
        assert preparer.dtype == DType("float32")

    def test_prepare_labeled_windows_empty_annotations(self) -> None:
        """Test handling of empty annotation tasks."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=31, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create empty annotation task
        annotation_task = self._create_annotation_task("empty_task", [])

        result = preparer.prepare_labeled_windows(annotation_task)
        assert result == []

    def test_prepare_labeled_windows_non_point_geometry_raises_error(self) -> None:
        """Test that non-point geometries raise ValueError."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=31, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create annotation with polygon instead of point
        polygon = Polygon([(10.0, 60.0), (10.1, 60.0), (10.1, 60.1), (10.0, 60.1)])
        st_geometry = STGeometry(WGS84_PROJECTION, polygon, None)

        annotation = LabeledSTGeometry(st_geometry=st_geometry, labels={"class": 1.0})

        annotation_task = self._create_annotation_task("polygon_task", [annotation])

        with pytest.raises(ValueError, match="Expected Point geometry"):
            preparer.prepare_labeled_windows(annotation_task)

    def test_project_geometry_to_utm(self) -> None:
        """Test UTM projection of geometries."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=31, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Test with a point in Norway (should use UTM Zone 33N)
        point = Point(10.0, 60.0)
        result = preparer._project_geometry_to_utm(point)

        # Verify the result is an STGeometry with UTM coordinates
        assert isinstance(result, STGeometry)
        assert isinstance(result.shp, Point)

        # The UTM coordinates should be reasonable for this location
        utm_point = result.shp
        # Just verify we got actual UTM coordinates (not the original lat/lon)
        assert abs(utm_point.x) > 1000  # Should be much larger than lat/lon values
        assert abs(utm_point.y) > 1000  # Should be much larger than lat/lon values
        assert utm_point.x != 10.0  # Should not be the original longitude
        assert utm_point.y != 60.0  # Should not be the original latitude

        # Check that projection has the expected resolution
        assert result.projection.x_resolution == 10.0
        assert result.projection.y_resolution == -10.0

    def test_create_window_geometry_from_point(self) -> None:
        """Test creation of window geometry from projected point."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=2, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create a real projected point using actual UTM projection
        original_point = Point(10.0, 60.0)  # Norway
        projected_point = preparer._project_geometry_to_utm(original_point)

        time_range = (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 12, 31, tzinfo=UTC),
        )
        projected_point = STGeometry(
            projected_point.projection, projected_point.shp, time_range
        )

        result = preparer._create_window_geometry_from_point(
            projected_point, time_range
        )

        # Check that the window is correctly sized and positioned
        bounds = result.shp.bounds
        point_x, point_y = projected_point.shp.x, projected_point.shp.y

        expected_minx = point_x - 2  # point.x - buffer
        expected_miny = point_y - 2  # point.y - buffer
        expected_maxx = point_x + 2 + 1  # point.x + buffer + 1
        expected_maxy = point_y + 2 + 1  # point.y + buffer + 1

        assert bounds[0] == expected_minx
        assert bounds[1] == expected_miny
        assert bounds[2] == expected_maxx
        assert bounds[3] == expected_maxy
        assert result.time_range == time_range
        assert result.projection == projected_point.projection

    def test_compute_window_bounds(self) -> None:
        """Test computation of integer window bounds."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=31, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create window geometry using real projection
        original_point = Point(10.0, 60.0)  # Norway
        projected_point = preparer._project_geometry_to_utm(original_point)

        # Create a window polygon with fractional coordinates
        point_x, point_y = projected_point.shp.x, projected_point.shp.y
        window_polygon = Polygon(
            [
                (point_x - 10.5, point_y - 10.5),
                (point_x + 10.7, point_y - 10.5),
                (point_x + 10.7, point_y + 10.3),
                (point_x - 10.5, point_y + 10.3),
            ]
        )
        window_geometry = STGeometry(projected_point.projection, window_polygon, None)

        result = preparer._compute_window_bounds(window_geometry)

        # Should convert to integer bounds
        expected_minx = int(point_x - 10.5)
        expected_miny = int(point_y - 10.5)
        expected_maxx = int(point_x + 10.7)
        expected_maxy = int(point_y + 10.3)

        assert result == (expected_minx, expected_miny, expected_maxx, expected_maxy)

    def test_window_buffer_affects_window_size(self) -> None:
        """Test that window_buffer parameter affects the resulting window size."""
        preparer_small = PointToRasterWindowPreparer(
            window_buffer=1, window_resolution=10.0, dtype="float32", nodata_value=-1
        )
        preparer_large = PointToRasterWindowPreparer(
            window_buffer=5, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create a real projected point
        original_point = Point(10.0, 60.0)  # Norway
        projected_point = preparer_small._project_geometry_to_utm(original_point)

        # Create windows with different buffer sizes
        window_small = preparer_small._create_window_geometry_from_point(
            projected_point, None
        )
        window_large = preparer_large._create_window_geometry_from_point(
            projected_point, None
        )

        bounds_small = window_small.shp.bounds
        bounds_large = window_large.shp.bounds

        # Large buffer should create larger window
        small_width = bounds_small[2] - bounds_small[0]
        large_width = bounds_large[2] - bounds_large[0]
        small_height = bounds_small[3] - bounds_small[1]
        large_height = bounds_large[3] - bounds_large[1]

        assert large_width > small_width
        assert large_height > small_height

        # Check specific sizes: buffer=1 should give 3x3, buffer=5 should give 11x11
        assert small_width == 3  # 2*1 + 1
        assert small_height == 3
        assert large_width == 11  # 2*5 + 1
        assert large_height == 11

    def test_prepare_labeled_windows_integration(self) -> None:
        """Integration test for preparing windows from point annotations."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=1, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create annotation task with multiple points in different locations
        points_data = [
            (Point(10.0, 60.0), {"class": 1.0}),
            (Point(11.0, 61.0), {"class": 2.0}),
            (Point(12.0, 62.0), {"class": 3.0}),
        ]

        annotations = []
        for i, (point, labels) in enumerate(points_data):
            st_geometry = STGeometry(WGS84_PROJECTION, point, None)
            annotation = LabeledSTGeometry(st_geometry=st_geometry, labels=labels)
            annotations.append(annotation)

        annotation_task = self._create_annotation_task("multi_test", annotations)

        # This test focuses on the structure and basic functionality
        # without testing the complex rasterization logic
        result = preparer.prepare_labeled_windows(annotation_task)

        # Should create one window per point
        assert len(result) == 3

        # Check window names follow the expected pattern
        for i, window in enumerate(result):
            assert window.name.startswith("task_")
            assert window.name.endswith(f"_point_{i}")

        # Each window should have proper geometry
        for window in result:
            assert window.st_geometry is not None
            assert hasattr(window.st_geometry, "shp")
            assert hasattr(window.st_geometry, "projection")

    def test_multiple_labels_handling(self) -> None:
        """Test handling of multiple labels per annotation."""
        preparer = PointToRasterWindowPreparer(
            window_buffer=1, window_resolution=10.0, dtype="float32", nodata_value=-1
        )

        # Create annotation with multiple labels (including None)
        point = Point(10.0, 60.0)  # Norway
        st_geometry = STGeometry(WGS84_PROJECTION, point, None)
        annotation = LabeledSTGeometry(
            st_geometry=st_geometry,
            labels={"class1": 1.0, "class2": None, "class3": 2.0},
        )

        annotation_task = self._create_annotation_task("multi_label_test", [annotation])

        # Test the geometry processing part
        result = preparer.prepare_labeled_windows(annotation_task)
        assert len(result) == 1
        window = result[0]
        assert window.name.startswith("task_")
        assert window.name.endswith("_point_0")
