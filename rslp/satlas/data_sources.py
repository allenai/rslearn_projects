"""Customized data sources for Satlas models."""

from datetime import timedelta
from typing import Any

import shapely
from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.azure_sentinel1 import Sentinel1
from rslearn.data_sources.azure_sentinel2 import Sentinel2 as AzureSentinel2
from rslearn.data_sources.data_source import DataSource, Item
from rslearn.data_sources.gcp_public_data import Sentinel2 as GcpSentinel2
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Window
from rslearn.tile_stores import TileStore
from rslearn.utils.geometry import STGeometry
from upath import UPath


def _find_monthly_matches(
    geometry: STGeometry, item_list: list[Item], period_days: int, max_matches: int
) -> list[list[Item]]:
    """Match items to the geometry with one mosaic per period.

    We divide the time range of the geometry into shorter periods. Within each period,
    we use the items corresponding to that period to create a mosaic. The returned item
    groups include one group per period, starting from the most recent periods, up to
    the provided max_matches.

    This is used e.g. when a model should process three mosaics, where each mosaic
    should come from a different month. This gives more diversity of images, since
    simply searching for the least cloudy images could result in selecting all of the
    images from the same month.

    max_matches may be smaller than the total number of periods in the given time
    range. In this case, we prefer to use mosaics of the most recent periods. However,
    sometimes there may be no items in a period; in that case, the older periods are
    used as a fallback.

    Args:
        geometry: the window geometry to match items to.
        item_list: the list of items.
        period_days: the length of one period in days.
        max_matches: the number of per-period mosaics to create.

    Returns:
        the matched item groups, where each group contains items that yield a
            per-period mosaic.
    """
    # For each period, we create an STGeometry with modified time range matching that
    # period, and use it with match_candidate_items_to_window to get a mosaic.
    cur_groups: list[list[Item]] = []
    period_end = geometry.time_range[1]
    while period_end > geometry.time_range[0] and len(cur_groups) < max_matches:
        period_time_range = (
            period_end - timedelta(days=period_days),
            period_end,
        )
        period_end -= timedelta(period_days)
        period_geom = STGeometry(geometry.projection, geometry.shp, period_time_range)

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
    if len(cur_groups) < max_matches:
        return []

    return cur_groups


class MonthlySentinel2(DataSource):
    """Sentinel2 data source where each match is a mosaic from a different month.

    It looks at the geometry time range, identifies matching items within each 30-day
    period, and then picks the most recent {num_matches} months that have at least one
    item.

    It also imposes a scene-level cloud cover limit.
    """

    def __init__(
        self,
        sentinel2: GcpSentinel2,
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
        sentinel2 = GcpSentinel2.from_config(config, ds_path)
        kwargs = {}
        d = config.data_source.config_dict
        for k in ["max_cloud_cover", "period_days"]:
            if k not in d:
                continue
            kwargs[k] = d[k]
        return MonthlySentinel2(sentinel2, **kwargs)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return self.sentinel2.deserialize_item(serialized_item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
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

            cur_groups = _find_monthly_matches(
                geometry=geometry,
                item_list=item_list,
                period_days=self.period_days,
                max_matches=query_config.max_matches,
            )
            groups.append(cur_groups)

        return groups

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        self.sentinel2.ingest(tile_store, items, geometries)


class MonthlyAzureSentinel2(DataSource):
    """Similar to MonthlySentinel2 but for Sentinel-2 L2A on Azure."""

    def __init__(
        self,
        sentinel2: AzureSentinel2,
        period_days: int = 30,
    ):
        """Create a new MonthlyAzureSentinel2.

        Args:
            sentinel2: the Sentinel2 data source to wrap.
            period_days: create mosaics for intervals of this many days within the
                geometry time range.
        """
        self.sentinel2 = sentinel2
        self.period_days = period_days

    @staticmethod
    def from_config(
        config: RasterLayerConfig, ds_path: UPath
    ) -> "MonthlyAzureSentinel2":
        """Creates a new MonthlyAzureSentinel2 instance from a configuration dictionary."""
        sentinel2 = AzureSentinel2.from_config(config, ds_path)
        kwargs = {}
        d = config.data_source.config_dict
        for k in ["period_days"]:
            if k not in d:
                continue
            kwargs[k] = d[k]
        return MonthlyAzureSentinel2(sentinel2, **kwargs)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return self.sentinel2.deserialize_item(serialized_item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # This only makes sense for mosaic space mode.
        assert query_config.space_mode == SpaceMode.MOSAIC

        groups = []
        for geometry in geometries:
            # This part is the same as in base Sentinel2 class.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            result = self.sentinel2.client.search(
                collections=[self.sentinel2.COLLECTION_NAME],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=self.sentinel2.query,
            )
            stac_items = [item for item in result.item_collection()]

            if self.sentinel2.sort_by is not None:
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[self.sentinel2.sort_by],
                    reverse=not self.sentinel2.sort_ascending,
                )

            candidate_items = [
                self.sentinel2._stac_item_to_item(stac_item) for stac_item in stac_items
            ]

            # Now we use _find_monthly_matches.
            cur_groups = _find_monthly_matches(
                geometry, candidate_items, self.period_days, query_config.max_matches
            )
            groups.append(cur_groups)

        return groups

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        self.sentinel2.ingest(tile_store, items, geometries)

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        self.sentinel2.materialize(window, item_groups, layer_name, layer_cfg)


class MonthlySentinel1(DataSource):
    """Similar to MonthlySentinel2 but for Sentinel-1 on Azure."""

    def __init__(
        self,
        sentinel1: Sentinel1,
        period_days: int = 30,
    ):
        """Create a new MonthlySentinel1.

        Args:
            sentinel1: the Sentinel1 data source to wrap.
            period_days: create mosaics for intervals of this many days within the
                geometry time range.
        """
        self.sentinel1 = sentinel1
        self.period_days = period_days

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "MonthlySentinel1":
        """Creates a new MonthlySentinel1 instance from a configuration dictionary."""
        sentinel1 = Sentinel1.from_config(config, ds_path)
        kwargs = {}
        d = config.data_source.config_dict
        for k in ["period_days"]:
            if k not in d:
                continue
            kwargs[k] = d[k]
        return MonthlySentinel1(sentinel1, **kwargs)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return self.sentinel1.deserialize_item(serialized_item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # This only makes sense for mosaic space mode.
        assert query_config.space_mode == SpaceMode.MOSAIC

        groups = []
        for geometry in geometries:
            # This part is the same as in base Sentinel1 class.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            result = self.sentinel1.client.search(
                collections=[self.sentinel1.COLLECTION_NAME],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=self.sentinel1.query,
            )
            stac_items = [item for item in result.item_collection()]

            if self.sentinel1.sort_by is not None:
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[self.sentinel1.sort_by],
                    reverse=not self.sentinel1.sort_ascending,
                )

            candidate_items = [
                self.sentinel1._stac_item_to_item(stac_item) for stac_item in stac_items
            ]

            # Now we use _find_monthly_matches.
            cur_groups = _find_monthly_matches(
                geometry, candidate_items, self.period_days, query_config.max_matches
            )
            groups.append(cur_groups)

        return groups

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        self.sentinel1.ingest(tile_store, items, geometries)

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        self.sentinel1.materialize(window, item_groups, layer_name, layer_cfg)
