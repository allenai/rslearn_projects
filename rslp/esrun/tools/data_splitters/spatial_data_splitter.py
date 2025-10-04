"""Splitter for assigning splits based on spatial grid cell location."""

import hashlib
from enum import StrEnum
from functools import cache
from typing import cast

from esrun.runner.models.training.labeled_data import LabeledWindow
from esrun.runner.tools.data_splitters.data_splitter_interface import (
    DataSplitterInterface,
)
from esrun.shared.models.data_split_type import DataSplitType
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils import get_utm_ups_crs
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import UPS_NORTH_THRESHOLD, UPS_SOUTH_THRESHOLD
from shapely.geometry.base import BaseGeometry


class ProjectionZone(StrEnum):
    """Projection zone types for caching UTM/UPS CRS lookups."""

    UTM_NORTH = "utm_north"
    UTM_SOUTH = "utm_south"
    UPS_NORTH = "ups_north"
    UPS_SOUTH = "ups_south"


class SpatialDataSplitter(DataSplitterInterface):
    """Data splitter that assigns splits based on spatial grid cell location.

    This splitter ensures that all windows within the same UTM tile receive
    the same split assignment, maintaining geographic coherence while respecting
    the specified proportions across the entire dataset.
    """

    def __init__(
        self, train_prop: float, val_prop: float, test_prop: float, grid_size: float
    ):
        """Initialize spatial data splitter with proportions for each split.

        Args:
            train_prop: Proportion of data for training (0.0 to 1.0)
            val_prop: Proportion of data for validation (0.0 to 1.0)
            test_prop: Proportion of data for testing (0.0 to 1.0)
            grid_size: Size of the UTM grid cells in meters

        Raises:
            ValueError: If proportions are negative or don't sum to 1.0
        """
        # Validate proportions
        if train_prop < 0 or val_prop < 0 or test_prop < 0:
            raise ValueError("All proportions must be non-negative")

        total = train_prop + val_prop + test_prop
        if abs(total - 1.0) > 1e-9:  # Allow for small floating point errors
            raise ValueError(f"Proportions must sum to 1.0, got {total}")

        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.grid_size = grid_size

    def choose_split_for_window(self, labeled_window: LabeledWindow) -> DataSplitType:
        """Choose a data split based on the UTM tile containing the window's centroid.

        All windows within the same UTM tile will receive the same split assignment.
        The split is determined by hashing the tile coordinates and using the hash
        to assign splits according to the configured proportions.

        Args:
            labeled_window: Complete labeled window with spatial, temporal and label information

        Returns:
            The assigned data split (train/val/test) based on UTM tile
        """
        # Get geometry and compute centroid
        geometry = cast(
            BaseGeometry, labeled_window.st_geometry.to_projection(WGS84_PROJECTION).shp
        )

        # Convert to UTM coordinates (using cached lookup for performance)
        utm_crs = self._get_utm_ups_crs(geometry.centroid.x, geometry.centroid.y)
        utm_projection = Projection(utm_crs, self.grid_size, self.grid_size)
        utm_geometry = labeled_window.st_geometry.to_projection(utm_projection)
        utm_centroid = cast(BaseGeometry, utm_geometry.shp).centroid

        # Calculate UTM tile coordinates
        utm_tile = (
            int(utm_centroid.x // self.grid_size),
            int(utm_centroid.y // self.grid_size),
        )

        # Create deterministic hash from UTM CRS and tile coordinates
        tile_str = f"{utm_crs.to_epsg()}_{utm_tile[0]}_{utm_tile[1]}"
        sha_hash = hashlib.sha256(tile_str.encode()).hexdigest()

        # Convert full hash to integer and normalize to [0, 1)
        hash_int = int(sha_hash, 16)
        normalized_hash = hash_int / (2**256 - 1)

        # Assign split based on cumulative proportions
        if normalized_hash < self.train_prop:
            return DataSplitType.TRAIN
        elif normalized_hash < self.train_prop + self.val_prop:
            return DataSplitType.VAL
        else:
            return DataSplitType.TEST

    def _get_utm_zone_key(self, lon: float, lat: float) -> tuple[int, ProjectionZone]:
        """Get UTM zone key for caching purposes.

        Returns:
            (utm_zone_number, projection_zone) where utm_zone_number is 1-60 for UTM,
            0 for UPS, and projection_zone indicates the type and hemisphere
        """
        # Check for UPS zones first
        if lat > UPS_NORTH_THRESHOLD:
            return (0, ProjectionZone.UPS_NORTH)
        if lat < UPS_SOUTH_THRESHOLD:
            return (0, ProjectionZone.UPS_SOUTH)

        # Calculate UTM zone (1-60)
        utm_zone = int((lon + 180) / 6) + 1
        if utm_zone > 60:
            utm_zone = 60
        elif utm_zone < 1:
            utm_zone = 1

        # Determine UTM hemisphere
        projection_zone = (
            ProjectionZone.UTM_NORTH if lat >= 0 else ProjectionZone.UTM_SOUTH
        )

        return (utm_zone, projection_zone)

    @cache
    def _get_utm_ups_crs_by_zone(
        self, utm_zone: int, projection_zone: ProjectionZone
    ) -> CRS:
        """Get the UTM/UPS CRS for a given UTM zone and projection zone.

        This caches by actual UTM zone rather than coordinates, so all points
        in the same zone share the same cache entry.
        """
        # Handle UPS zones
        if projection_zone == ProjectionZone.UPS_NORTH:
            sample_lon, sample_lat = 0.0, 85.0
        elif projection_zone == ProjectionZone.UPS_SOUTH:
            sample_lon, sample_lat = 0.0, -85.0
        else:
            # For UTM zones, use a representative point in the middle of the zone
            sample_lon = -180 + (utm_zone - 1) * 6 + 3  # Middle of 6-degree zone
            sample_lat = 45.0 if projection_zone == ProjectionZone.UTM_NORTH else -45.0

        return get_utm_ups_crs(sample_lon, sample_lat)

    def _get_utm_ups_crs(self, lon: float, lat: float) -> CRS:
        """Get the UTM/UPS CRS for a given longitude and latitude with caching."""
        utm_zone, projection_zone = self._get_utm_zone_key(lon, lat)
        return self._get_utm_ups_crs_by_zone(utm_zone, projection_zone)
