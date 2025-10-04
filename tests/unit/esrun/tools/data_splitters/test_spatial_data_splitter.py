"""Unit tests for SpatialDataSplitter."""

import math
import random

import numpy as np
import pytest
from esrun.runner.models.training.labeled_data import LabeledWindow, RasterLabel
from esrun.shared.models.data_split_type import DataSplitType
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from shapely.geometry import Point, Polygon

from rslp.esrun.tools.data_splitters.spatial_data_splitter import SpatialDataSplitter


class TestSpatialDataSplitter:
    """Test cases for SpatialDataSplitter."""

    def _create_labeled_window(
        self, name: str, st_geometry: STGeometry
    ) -> LabeledWindow:
        """Helper to create a valid LabeledWindow with proper raster dimensions."""
        bounds = st_geometry.shp.bounds
        width = int(bounds[2] - bounds[0])
        height = int(bounds[3] - bounds[1])

        # Create raster with correct dimensions (0x0 for points, actual size for polygons)
        if width == 0 and height == 0:
            # Point geometry - use empty raster
            raster = np.array([], dtype=np.float32).reshape(0, 0)
        else:
            # Polygon geometry - use actual dimensions
            raster = np.ones((height, width), dtype=np.float32)

        label = RasterLabel(key="test_class", value=raster)
        return LabeledWindow(name=name, st_geometry=st_geometry, labels=[label])

    def test_init_valid_proportions(self) -> None:
        """Test initialization with valid proportions."""
        splitter = SpatialDataSplitter(
            train_prop=0.7, val_prop=0.2, test_prop=0.1, grid_size=1000.0
        )
        assert splitter.train_prop == 0.7
        assert splitter.val_prop == 0.2
        assert splitter.test_prop == 0.1
        assert splitter.grid_size == 1000.0

    def test_init_proportions_sum_to_one(self) -> None:
        """Test initialization with proportions that sum to 1.0."""
        # Test exact sum
        SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        # Test with small floating point error (should be allowed)
        SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1000000001, grid_size=1000.0
        )

    def test_init_negative_proportions_raises_error(self) -> None:
        """Test that negative proportions raise ValueError."""
        with pytest.raises(ValueError, match="All proportions must be non-negative"):
            SpatialDataSplitter(
                train_prop=-0.1, val_prop=0.6, test_prop=0.5, grid_size=1000.0
            )

        with pytest.raises(ValueError, match="All proportions must be non-negative"):
            SpatialDataSplitter(
                train_prop=0.5, val_prop=-0.1, test_prop=0.6, grid_size=1000.0
            )

        with pytest.raises(ValueError, match="All proportions must be non-negative"):
            SpatialDataSplitter(
                train_prop=0.5, val_prop=0.6, test_prop=-0.1, grid_size=1000.0
            )

    def test_init_proportions_not_sum_to_one_raises_error(self) -> None:
        """Test that proportions not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="Proportions must sum to 1.0"):
            SpatialDataSplitter(
                train_prop=0.5, val_prop=0.3, test_prop=0.1, grid_size=1000.0
            )

        with pytest.raises(ValueError, match="Proportions must sum to 1.0"):
            SpatialDataSplitter(
                train_prop=0.8, val_prop=0.3, test_prop=0.1, grid_size=1000.0
            )

    def test_choose_split_for_window_deterministic(self) -> None:
        """Test that the same window always gets the same split."""
        splitter = SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        # Create a test window in Norway (UTM Zone 33N)
        geometry = Point(10.0, 60.0)  # Longitude, Latitude
        st_geometry = STGeometry(WGS84_PROJECTION, geometry, None)

        # Create a real LabeledWindow
        labeled_window = self._create_labeled_window("test_window", st_geometry)

        # Call multiple times and ensure consistent result
        split1 = splitter.choose_split_for_window(labeled_window)
        split2 = splitter.choose_split_for_window(labeled_window)
        split3 = splitter.choose_split_for_window(labeled_window)

        assert split1 == split2 == split3
        assert split1 in [
            DataSplitType.TRAIN,
            DataSplitType.VAL,
            DataSplitType.TEST,
        ]

    def test_choose_split_respects_proportions_approximately(self) -> None:
        """Test that split assignment respects proportions over many samples."""
        splitter = SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        splits = []

        # Generate random points across the entire globe to avoid spatial correlation
        # Use a fixed seed for reproducible tests
        random.seed(42)

        n_samples = 400  # Minimum for 5% tolerance with 95% confidence

        for i in range(n_samples):
            # Generate random longitude: -180 to 180
            lon = random.uniform(-180, 180)

            # Generate random latitude with proper spherical distribution
            # Use inverse transform sampling for uniform distribution on sphere
            lat_rad = random.uniform(-1, 1)  # Uniform in sin(lat)
            lat = math.degrees(math.asin(lat_rad))

            geometry = Point(lon, lat)
            st_geometry = STGeometry(WGS84_PROJECTION, geometry, None)

            # Create a real LabeledWindow and call the actual method
            labeled_window = self._create_labeled_window(
                f"test_window_{i}", st_geometry
            )
            split = splitter.choose_split_for_window(labeled_window)
            splits.append(split)

        train_count = splits.count(DataSplitType.TRAIN)
        val_count = splits.count(DataSplitType.VAL)
        test_count = splits.count(DataSplitType.TEST)
        total = len(splits)

        train_prop = train_count / total
        val_prop = val_count / total
        test_prop = test_count / total

        assert (
            abs(train_prop - 0.6) < 0.05
        ), f"Train proportion {train_prop:.3f} not close to 0.6"
        assert (
            abs(val_prop - 0.3) < 0.05
        ), f"Val proportion {val_prop:.3f} not close to 0.3"
        assert (
            abs(test_prop - 0.1) < 0.05
        ), f"Test proportion {test_prop:.3f} not close to 0.1"

        # Ensure all three splits are represented
        assert train_count > 0, "No training samples"
        assert val_count > 0, "No validation samples"
        assert test_count > 0, "No test samples"

    def test_same_utm_tile_gets_same_split(self) -> None:
        """Test that windows in the same UTM tile get the same split."""
        splitter = SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        # Create two windows that should be in the same UTM tile
        # Using small differences that should result in the same 1km tile
        geometry1 = Point(10.0, 60.0)
        geometry2 = Point(10.0001, 60.0001)  # Very close, should be same tile

        st_geometry1 = STGeometry(WGS84_PROJECTION, geometry1, None)
        st_geometry2 = STGeometry(WGS84_PROJECTION, geometry2, None)

        labeled_window1 = self._create_labeled_window("test_window_1", st_geometry1)
        labeled_window2 = self._create_labeled_window("test_window_2", st_geometry2)

        split1 = splitter.choose_split_for_window(labeled_window1)
        split2 = splitter.choose_split_for_window(labeled_window2)

        assert (
            split1 == split2
        ), "Windows in the same UTM tile should get the same split"

    def test_different_utm_tiles_can_get_different_splits(self) -> None:
        """Test that windows in different UTM tiles can get different splits."""
        splitter = SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        # Create windows in different locations across different UTM zones
        splits = set()
        locations = [
            (10.0, 60.0),  # Norway - UTM 33N
            (-74.0, 40.7),  # New York - UTM 18N
            (139.7, 35.7),  # Tokyo - UTM 54N
            (2.3, 48.9),  # Paris - UTM 31N
            (-122.4, 37.8),  # San Francisco - UTM 10N
        ]

        for lon, lat in locations:
            geometry = Point(lon, lat)
            st_geometry = STGeometry(WGS84_PROJECTION, geometry, None)

            labeled_window = self._create_labeled_window(
                f"test_window_{lon}_{lat}", st_geometry
            )

            split = splitter.choose_split_for_window(labeled_window)
            splits.add(split)

        # We should get at least 2 different splits across different locations
        assert len(splits) >= 2, "Different locations should produce different splits"

    def test_edge_case_zero_proportions(self) -> None:
        """Test edge cases with zero proportions."""
        # Test with zero test proportion
        splitter = SpatialDataSplitter(
            train_prop=0.7, val_prop=0.3, test_prop=0.0, grid_size=1000.0
        )
        assert splitter.test_prop == 0.0

        # Test with zero val proportion
        splitter = SpatialDataSplitter(
            train_prop=0.8, val_prop=0.0, test_prop=0.2, grid_size=1000.0
        )
        assert splitter.val_prop == 0.0

    def test_polygon_geometry_uses_centroid(self) -> None:
        """Test that polygon geometries use their centroid for tile assignment."""
        splitter = SpatialDataSplitter(
            train_prop=0.6, val_prop=0.3, test_prop=0.1, grid_size=1000.0
        )

        # Create a polygon geometry in Norway
        polygon = Polygon([(10.0, 60.0), (10.1, 60.0), (10.1, 60.1), (10.0, 60.1)])
        st_geometry = STGeometry(WGS84_PROJECTION, polygon, None)

        labeled_window = self._create_labeled_window("test_polygon_window", st_geometry)

        # Should not raise an error and should return a valid split
        split = splitter.choose_split_for_window(labeled_window)
        assert split in [DataSplitType.TRAIN, DataSplitType.VAL, DataSplitType.TEST]

        # Test that the same polygon always gets the same split (deterministic)
        split2 = splitter.choose_split_for_window(labeled_window)
        assert split == split2
