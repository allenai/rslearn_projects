"""Customized data sources for Satlas models."""

from datetime import timedelta
from typing import Any

from rslearn.config import QueryConfig, RasterLayerConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSource, Item
from rslearn.data_sources.gcp_public_data import Sentinel2, Sentinel2Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import TileStore
from rslearn.utils.geometry import STGeometry
from upath import UPath


class MonthlySentinel2(DataSource):
    """Sentinel2 data source where each match is a mosaic from a different month.

    It looks at the geometry time range, identifies matching items within each 30-day
    period, and then picks the most recent {num_matches} months that have at least one
    item.

    It also imposes a scene-level cloud cover limit.
    """

    def __init__(
        self,
        sentinel2: Sentinel2,
        max_cloud_cover: float | None = None,
        period_days: int = 30,
    ):
        """Create a new MonthlySentinel2.

        Args:
            sentinel2: the Sentinel2 data source to wrap.
            max_cloud_cover: cloud cover limit for scenes.
            period_days: create mosaics for intervals of this many days within the
                geometry time range.
        """
        self.sentinel2 = sentinel2
        self.max_cloud_cover = max_cloud_cover
        self.period_days = period_days

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "MonthlySentinel2":
        """Creates a new MonthlySentinel2 instance from a configuration dictionary."""
        sentinel2 = Sentinel2.from_config(config, ds_path)
        kwargs = {}
        d = config.data_source.config_dict
        for k in ["max_cloud_cover", "period_days"]:
            if k not in d:
                continue
            kwargs[k] = d[k]
        return MonthlySentinel2(sentinel2, **kwargs)

    def deserialize_item(self, serialized_item: Any) -> Sentinel2Item:
        """Deserializes an item from JSON-decoded data."""
        return self.sentinel2.deserialize_item(serialized_item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Sentinel2Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # This only makes sense for mosaic space mode.
        assert query_config.space_mode == SpaceMode.MOSAIC

        # This part is the same as in base Sentinel2 class.
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]

        if self.sentinel2.rtree_index:
            candidates = self.sentinel2._get_candidate_items_index(wgs84_geometries)
        else:
            candidates = self.sentinel2._get_candidate_items_direct(wgs84_geometries)

        groups = []

        for geometry, item_list in zip(wgs84_geometries, candidates):
            item_list.sort(key=lambda item: item.cloud_cover)

            # Apply cloud cover limit.
            if self.max_cloud_cover is not None:
                item_list = [
                    item
                    for item in item_list
                    if item.cloud_cover <= self.max_cloud_cover
                ]

            # Find matches across the periods.
            # For each period, we create an STGeometry with modified time range
            # matching the period, and obtain matching mosaic.
            # We start from the end of the time range because we care more about recent
            # periods and so we want to make sure that they align correctly with the
            # end.
            cur_groups: list[Item] = []
            period_end = geometry.time_range[1]
            while (
                period_end > geometry.time_range[0]
                and len(cur_groups) < query_config.max_matches
            ):
                period_time_range = (
                    period_end - timedelta(days=self.period_days),
                    period_end,
                )
                period_end -= timedelta(self.period_days)
                period_geom = STGeometry(
                    geometry.projection, geometry.shp, period_time_range
                )

                # We modify the QueryConfig here since caller should be asking for
                # multiple mosaics, but we just want one mosaic per period.
                period_groups = match_candidate_items_to_window(
                    period_geom,
                    item_list,
                    QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1),
                )

                # There should be zero on one groups depending on whether there were
                # any items that matched. We keep the group if it is there.
                if len(period_groups) == 0 or len(period_groups[0]) == 0:
                    # No matches for this period.
                    continue
                cur_groups.append(period_groups[0])

            # If there are not enough matching mosaics, then we eliminate all the
            # matches since we aren't going to use this window then anyway.
            if len(cur_groups) < query_config.max_matches:
                cur_groups = []

            groups.append(cur_groups)

        return groups

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Sentinel2Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        self.sentinel2.ingest(tile_store, items, geometries)
