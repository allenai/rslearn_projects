"""Test Satlas data_sources.py."""

import pathlib
from datetime import datetime, timedelta, timezone

import shapely
from rslearn.config import (
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource
from rslearn.data_sources.gcp_public_data import Sentinel2 as Sentinel2
from rslearn.data_sources.planetary_computer import Sentinel1
from rslearn.data_sources.planetary_computer import Sentinel2 as AzureSentinel2
from rslearn.utils.geometry import STGeometry
from upath import UPath

from rslp.satlas.data_sources import (
    MonthlyAzureSentinel2,
    MonthlySentinel1,
    MonthlySentinel2,
)

PERIOD_DAYS = 30


class TestGetItems:
    """Test the get_items method in per-period data source."""

    def apply_test(self, data_source: DataSource) -> None:
        """Test that the data source successfully returns per-period mosaics.

        We apply it on a bbox of Seattle for three-month period with period equal to one
        month.

        Args:
            data_source: the data source to test.
        """
        # Create a 0.002x0.002 degree bbox near Seattle for three-month time range.
        seattle_point = (-122.33, 47.61)
        shp = shapely.box(
            seattle_point[0] - 0.001,
            seattle_point[1] - 0.001,
            seattle_point[0] + 0.001,
            seattle_point[1] + 0.001,
        )
        time_range = (
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        geometry = STGeometry(WGS84_PROJECTION, shp, time_range)

        # Look for 2 per-month mosaics.
        # The first month in the three-month time range should not yield a mosaic since
        # it is not needed (it would only be used if the more recent months do not have
        # any scenes, but that is not the case here).
        query_config = QueryConfig(
            space_mode=SpaceMode.MOSAIC,
            max_matches=2,
        )
        groups = data_source.get_items([geometry], query_config)[0]

        # We expect to get two groups, and each one should be in a different period.
        # The groups should be ordered from most recent to least recent.
        # There should not be any group for the first period in the three-month time
        # range since we expect there to be a mosaic available for the more recent
        # periods and max_matches=2.
        expected_time_ranges = [
            (time_range[1] - timedelta(days=PERIOD_DAYS), time_range[1]),
            (
                time_range[1] - timedelta(days=PERIOD_DAYS * 2),
                time_range[1] - timedelta(days=PERIOD_DAYS),
            ),
        ]
        assert len(groups) == len(expected_time_ranges)
        for expected_time_range, group in zip(expected_time_ranges, groups):
            assert len(group) > 0
            for item in group:
                item_ts = item.geometry.time_range[0]
                assert expected_time_range[0] <= item_ts <= expected_time_range[1]

    def test_sentinel1(self) -> None:
        """Run apply_test with MonthlySentinel1."""
        sentinel1 = MonthlySentinel1(
            sentinel1=Sentinel1(
                RasterLayerConfig(LayerType.RASTER, []),
            ),
            period_days=PERIOD_DAYS,
        )
        self.apply_test(sentinel1)

    def test_sentinel2(self, tmp_path: pathlib.Path) -> None:
        """Run apply_test with MonthlySentinel2."""
        sentinel2 = MonthlySentinel2(
            sentinel2=Sentinel2(
                RasterLayerConfig(LayerType.RASTER, []),
                index_cache_dir=UPath(tmp_path),
                use_rtree_index=False,
            ),
            period_days=PERIOD_DAYS,
        )
        self.apply_test(sentinel2)

    def test_azure_sentinel2(self) -> None:
        """Run apply_test with MonthlyAzureSentinel2."""
        sentinel2 = MonthlyAzureSentinel2(
            sentinel2=AzureSentinel2(
                RasterLayerConfig(LayerType.RASTER, []),
            ),
            period_days=PERIOD_DAYS,
        )
        self.apply_test(sentinel2)
