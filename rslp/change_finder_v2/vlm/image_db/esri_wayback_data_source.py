"""rslearn data source for ArcGIS World Imagery Wayback (Esri) imagery.

The Wayback archive exposes ~200 dated "releases" of the Esri World Imagery
basemap. The imagery shown at a given location only changes on a handful of
those releases. For a point (and its window time range) we:

1. Prune to releases that could hold in-range imagery: a release published
   before the window start can never contain imagery captured in range (a
   release's capture date is never newer than its publish date), so it is
   dropped with no network request.
2. Resolve acquisition dates at the point via the per-release metadata
   ``identify`` endpoint (``SRC_DATE``). Acquisition dates are (almost always)
   monotonically non-increasing along the newest-first release order, so we
   binary-search for the band of releases whose capture date falls in the
   window time range instead of resolving every release. Consecutive releases
   that share a capture date (the same image) are collapsed.
3. Expose each distinct in-range release as an rslearn :class:`Item` whose time
   range is the acquisition date, so that rslearn's ``MOSAIC`` +
   ``period_duration`` matching can select e.g. one release per quarter.

The data source only supports direct materialization (no ingestion). During
materialization it fetches the covering tiles for a release at a fixed "native"
zoom (default 18), falling back to progressively coarser zooms (with bilinear
upsampling) where high-zoom tiles are missing so the rendered raster is always
at a consistent resolution, and reprojects to the requested window grid (which
is the window projection adjusted by the band set's zoom offset).

This module is intentionally self-contained: the Wayback client logic is copied
here (adapted from rslp.change_finder_v2.llm_categorize.wayback) so that this
module has no dependency on llm_categorize.

No authentication is required; all endpoints are public.
"""

from __future__ import annotations

import io
import math
import re
import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio.warp
import requests
import shapely
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rslearn.config import LayerConfig, QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import (
    MatchedItemGroup,
    match_candidate_items_to_window,
)
from rslearn.data_sources.xyz_tiles import read_from_tile_callback
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils import PixelBounds, Projection, STGeometry, get_global_raster_bounds
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

logger = get_logger(__name__)

WEB_MERCATOR_EPSG = 3857
WEB_MERCATOR_UNITS = 2 * math.pi * 6378137
TILE_SIZE = 256

CONFIG_URL = (
    "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
)
TILE_SERVICE_BASE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer"
)

_TITLE_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class TileNotFoundError(Exception):
    """Raised when a Wayback tile is not available at a release/zoom (HTTP 404)."""


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Return (col, row) Web Mercator tile indices for a lon/lat at a zoom."""
    n = 2**zoom
    col = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    row = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return col, row


def _parse_capture_date(attrs: dict) -> date | None:
    """Parse an acquisition date from identify attributes (SRC_DATE/SRC_DATE2)."""
    raw2 = attrs.get("SRC_DATE2")
    if raw2:
        # Format like "11/23/2025".
        try:
            month, day, year = (int(p) for p in raw2.split("/"))
            return date(year, month, day)
        except (ValueError, TypeError):
            pass
    raw = attrs.get("SRC_DATE")
    if raw:
        # Format like "20251123".
        s = str(raw)
        if len(s) == 8 and s.isdigit():
            try:
                return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
            except ValueError:
                pass
    return None


class WaybackClient:
    """Client for the ArcGIS World Imagery Wayback archive.

    Adapted from rslp.change_finder_v2.llm_categorize.wayback so that this module
    is self-contained.
    """

    def __init__(self, timeout: float = 60.0, max_retries: int = 4) -> None:
        """Create the client and load the release config.

        Args:
            timeout: per-request timeout in seconds.
            max_retries: number of attempts per request before giving up.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._local = threading.local()
        self._config = self._get_json(CONFIG_URL)
        # Releases sorted by published release date, newest first.
        self._releases: list[tuple[str, dict]] = sorted(
            self._config.items(),
            key=lambda kv: self._release_date(kv[1]),
            reverse=True,
        )
        self._release_index: dict[str, int] = {
            rel: i for i, (rel, _) in enumerate(self._releases)
        }

    @staticmethod
    def _release_date(value: dict) -> str:
        match = _TITLE_DATE_RE.search(value["itemTitle"])
        return match.group(1) if match else "0000-00-00"

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            logger.debug(
                "Wayback request (attempt %d/%d): GET %s params=%s",
                attempt + 1,
                self.max_retries,
                url,
                params,
            )
            start = time.monotonic()
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                logger.debug(
                    "Wayback request ok in %.2fs (%d bytes): GET %s",
                    time.monotonic() - start,
                    len(resp.content),
                    url,
                )
                return resp
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                logger.debug(
                    "Wayback request HTTP %s in %.2fs: GET %s",
                    status,
                    time.monotonic() - start,
                    url,
                )
                # Client errors (other than rate limiting) are not transient,
                # so do not waste time retrying them.
                if status is not None and 400 <= status < 500 and status != 429:
                    break
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                last_exc = exc
                logger.debug(
                    "Wayback request error in %.2fs: GET %s: %r",
                    time.monotonic() - start,
                    url,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def _get_json(self, url: str, params: dict | None = None) -> dict:
        return self._get(url, params=params).json()

    def enumerate_distinct_releases(
        self, lon: float, lat: float, zoom: int
    ) -> list[str]:
        """Return release numbers with distinct imagery at the point, newest first.

        Uses the ``tilemap`` ``select`` field: for the tile under each candidate
        release, ``select`` names the release that actually owns the tile data,
        so we jump directly between distinct images instead of scanning all ~200
        releases.
        """
        col, row = _lonlat_to_tile(lon, lat, zoom)
        distinct: list[str] = []
        i = 0
        requests_made = 0
        start = time.monotonic()
        while i < len(self._releases):
            release_num = self._releases[i][0]
            url = f"{TILE_SERVICE_BASE}/tilemap/{release_num}/{zoom}/{row}/{col}/1/1"
            data = self._get_json(url)
            requests_made += 1
            if not data.get("valid") or not data.get("select"):
                i += 1
                continue
            owner = str(data["select"][0])
            if owner not in distinct:
                distinct.append(owner)
            # Continue from the release immediately older than the owning one.
            i = self._release_index.get(owner, i) + 1
        logger.debug(
            "enumerate_distinct_releases(lon=%.5f, lat=%.5f, zoom=%d): %d distinct "
            "releases via %d tilemap requests in %.2fs",
            lon,
            lat,
            zoom,
            len(distinct),
            requests_made,
            time.monotonic() - start,
        )
        return distinct

    def get_capture_date(self, release_num: str, lon: float, lat: float) -> date | None:
        """Return the acquisition date of ``release_num``'s imagery at the point."""
        value = self._config[release_num]
        metadata_url = value["metadataLayerUrl"] + "/identify"
        params = {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "sr": "4326",
            "tolerance": "1",
            "mapExtent": f"{lon - 0.005},{lat - 0.005},{lon + 0.005},{lat + 0.005}",
            "imageDisplay": "256,256,96",
            "returnGeometry": "false",
            "f": "json",
            "layers": "all",
        }
        logger.debug(
            "get_capture_date: resolving release %s at lon=%.5f lat=%.5f via %s",
            release_num,
            lon,
            lat,
            metadata_url,
        )
        start = time.monotonic()
        data = self._get_json(metadata_url, params=params)
        results = data.get("results") or []
        capture = _parse_capture_date(results[0].get("attributes", {})) if results else None
        logger.debug(
            "get_capture_date: release %s -> %s (%d results) in %.2fs",
            release_num,
            capture,
            len(results),
            time.monotonic() - start,
        )
        return capture

    def find_releases_in_time_range(
        self,
        lon: float,
        lat: float,
        start: datetime | None,
        end: datetime | None,
    ) -> list[tuple[str, date]]:
        """Return distinct releases whose imagery was captured in ``[start, end)``.

        Returns a list of ``(release_num, capture_date)`` tuples, newest first, with
        consecutive releases that share a capture date (i.e. the same image) collapsed
        to one entry.

        Acquisition dates are (almost always) monotonically non-increasing along the
        newest-first release order, because a location's imagery is only ever updated
        to newer captures. We exploit that to binary-search for the band of in-range
        releases, so only O(log n) slow ``identify`` lookups are needed instead of one
        per release. Releases *published* before ``start`` are pruned with no request
        at all, since a release's capture date can never be newer than its publish date.

        Boundary releases whose acquisition date cannot be resolved are skipped by
        probing their neighbors, tolerating sparse metadata and minor non-monotonicity.

        Args:
            lon: longitude of the point.
            lat: latitude of the point.
            start: inclusive lower bound on capture date (None for no lower bound).
            end: exclusive upper bound on capture date (None for no upper bound).
        """
        if start is not None:
            start_iso = start.date().isoformat()
            candidates = [
                rel
                for rel, value in self._releases
                if self._release_date(value) >= start_iso
            ]
        else:
            candidates = [rel for rel, _ in self._releases]
        n = len(candidates)
        logger.debug(
            "find_releases_in_time_range(lon=%.5f, lat=%.5f, start=%s, end=%s): "
            "%d candidate releases after publish-date pruning",
            lon,
            lat,
            start.date() if start else None,
            end.date() if end else None,
            n,
        )
        if n == 0:
            return []

        date_cache: dict[int, date | None] = {}

        def cap(i: int) -> date | None:
            if i not in date_cache:
                try:
                    date_cache[i] = self.get_capture_date(candidates[i], lon, lat)
                except Exception:  # noqa: BLE001 - skip unresolvable releases
                    date_cache[i] = None
            return date_cache[i]

        def cap_probe(i: int) -> date | None:
            """Capture date at ``i``, probing neighbors if it cannot be resolved."""
            d = cap(i)
            if d is not None:
                return d
            for off in range(1, n):
                for j in (i - off, i + off):
                    if 0 <= j < n:
                        d = cap(j)
                        if d is not None:
                            return d
            return None

        def leftmost_below(threshold: date) -> int:
            """Smallest index whose capture date is < ``threshold`` (else ``n``)."""
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                d = cap_probe(mid)
                if d is None:
                    break
                if d < threshold:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        top = leftmost_below(end.date()) if end is not None else 0
        bottom = leftmost_below(start.date()) if start is not None else n

        # Walk the in-range band one distinct capture date at a time: capture dates
        # are non-increasing, so all releases sharing a date are contiguous and we
        # can binary-search (leftmost_below) to jump past each run instead of
        # resolving every release in the band.
        results: list[tuple[str, date]] = []
        i = top
        while i < bottom:
            d = cap_probe(i)
            if d is None:
                i += 1
                continue
            if end is not None and d >= end.date():
                i += 1
                continue
            if start is not None and d < start.date():
                break
            results.append((candidates[i], d))
            # Jump to the first release older than this capture date.
            i = max(leftmost_below(d), i + 1)
        logger.debug(
            "find_releases_in_time_range(lon=%.5f, lat=%.5f): %d in-range distinct "
            "releases via %d identify requests",
            lon,
            lat,
            len(results),
            len(date_cache),
        )
        return results

    def fetch_tile(self, release_num: str, zoom: int, col: int, row: int) -> Image.Image:
        """Fetch one 256x256 RGB tile for a release at (zoom, col, row).

        Raises:
            TileNotFoundError: if the tile is missing at this release/zoom (404).
        """
        url = (
            self._config[release_num]["itemURL"]
            .replace("{level}", str(zoom))
            .replace("{row}", str(row))
            .replace("{col}", str(col))
        )
        try:
            resp = self._get(url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise TileNotFoundError(url) from exc
            raise
        return Image.open(io.BytesIO(resp.content)).convert("RGB")


class EsriWayback(DataSource[Item], TileStore):
    """A data source for Esri World Imagery Wayback historical imagery.

    Each distinct release at a point is exposed as an item whose time range is the
    release's acquisition date there, so rslearn's ``MOSAIC`` + ``period_duration``
    matching can pick e.g. one release per quarter. Imagery is read on-demand during
    direct materialization (this source does not implement ingestion).
    """

    def __init__(
        self,
        zoom: int = 18,
        max_fallback_levels: int = 3,
        band_names: list[str] = ["R", "G", "B"],
        timeout: float = 60.0,
        max_retries: int = 4,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize an EsriWayback data source.

        Args:
            zoom: the native Web Mercator zoom level at which tiles are fetched and
                rendered before reprojecting to the window grid. Determines the
                resolution of the data this source produces.
            max_fallback_levels: how many coarser zoom levels to fall back to (with
                bilinear upsampling) when a release lacks tiles at the native zoom.
            band_names: names of the bands this source produces (RGB).
            timeout: per-request timeout in seconds.
            max_retries: number of attempts per request before giving up.
            context: the data source context.
        """
        self.zoom = zoom
        self.min_zoom = max(zoom - max_fallback_levels, 1)
        self.band_names = band_names
        self.client = WaybackClient(timeout=timeout, max_retries=max_retries)

        # Web Mercator projection at the native zoom level (no offset; the tile/grid
        # offset is applied inside read_bounds when fetching tiles).
        self.crs = CRS.from_epsg(WEB_MERCATOR_EPSG)
        self.total_pixels = TILE_SIZE * (2**zoom)
        self.pixel_size = WEB_MERCATOR_UNITS / self.total_pixels
        self.pixel_offset = self.total_pixels // 2
        self.projection = Projection(self.crs, self.pixel_size, -self.pixel_size)
        self.shp = shapely.box(
            -self.total_pixels // 2,
            -self.total_pixels // 2,
            self.total_pixels // 2,
            self.total_pixels // 2,
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Get items (Wayback releases) intersecting each geometry.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            wgs84_geom = geometry.to_projection(WGS84_PROJECTION)
            point = wgs84_geom.shp.centroid
            lon, lat = float(point.x), float(point.y)
            logger.debug(
                "get_items: processing geometry at lon=%.5f lat=%.5f time_range=%s",
                lon,
                lat,
                geometry.time_range,
            )
            geom_start = time.monotonic()

            items: list[Item] = []
            start = geometry.time_range[0] if geometry.time_range else None
            end = geometry.time_range[1] if geometry.time_range else None
            try:
                release_dates = self.client.find_releases_in_time_range(
                    lon, lat, start, end
                )
            except Exception:  # noqa: BLE001 - skip points we cannot enumerate
                logger.warning(
                    "failed to enumerate Wayback releases at lon=%.5f lat=%.5f",
                    lon,
                    lat,
                    exc_info=True,
                )
                release_dates = []

            for release_num, capture in release_dates:
                capture_dt = datetime(
                    capture.year, capture.month, capture.day, tzinfo=UTC
                )
                time_range = (capture_dt, capture_dt + timedelta(days=1))
                item_geom = STGeometry(self.projection, self.shp, time_range)
                items.append(Item(release_num, item_geom))

            projected_geometry = geometry.to_projection(self.projection)
            cur_groups = match_candidate_items_to_window(
                projected_geometry, items, query_config
            )
            logger.debug(
                "get_items: lon=%.5f lat=%.5f -> %d candidate items (%d matched "
                "groups) in %.2fs",
                lon,
                lat,
                len(items),
                len(cur_groups),
                time.monotonic() - geom_start,
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def _fetch_tile_chw(
        self, release_num: str, col: int, row: int
    ) -> npt.NDArray[Any] | None:
        """Fetch a native-zoom tile as a CHW uint8 array, with coarse-zoom fallback.

        Tries the native zoom first; if the release lacks the tile, falls back to
        progressively coarser zooms and bilinearly upsamples the covering sub-region
        back to a full native-zoom tile, so the output is always at native resolution.
        Returns None if no tile is available at any zoom.
        """
        for current_zoom in range(self.zoom, self.min_zoom - 1, -1):
            factor = 2 ** (self.zoom - current_zoom)
            parent_col = col // factor
            parent_row = row // factor
            try:
                img = self.client.fetch_tile(
                    release_num, current_zoom, parent_col, parent_row
                )
            except TileNotFoundError:
                continue
            except Exception:  # noqa: BLE001 - omit tile on any fetch error
                logger.warning(
                    "Wayback tile fetch failed for release %s z=%d col=%d row=%d",
                    release_num,
                    current_zoom,
                    parent_col,
                    parent_row,
                    exc_info=True,
                )
                return None
            if factor > 1:
                block = TILE_SIZE // factor
                sub_col = col % factor
                sub_row = row % factor
                crop = img.crop(
                    (
                        sub_col * block,
                        sub_row * block,
                        sub_col * block + block,
                        sub_row * block + block,
                    )
                )
                img = crop.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
            return np.array(img).transpose(2, 0, 1)
        logger.warning(
            "Wayback release %s has no tile at col=%d row=%d for zooms %d-%d",
            release_num,
            col,
            row,
            self.min_zoom,
            self.zoom,
        )
        return None

    def read_bounds(
        self, release_num: str, bounds: PixelBounds
    ) -> npt.NDArray[Any] | None:
        """Read native-zoom raster data for a release within the given bounds."""
        bounds = (
            bounds[0] + self.pixel_offset,
            bounds[1] + self.pixel_offset,
            bounds[2] + self.pixel_offset,
            bounds[3] + self.pixel_offset,
        )
        return read_from_tile_callback(
            bounds,
            lambda col, row: self._fetch_tile_chw(release_num, col, row),
            TILE_SIZE,
        )

    # --- TileStore implementation ---

    def is_raster_ready(self, layer_name: str, item: Item, bands: list[str]) -> bool:
        """Always ready since we wrap on-demand accesses to the Wayback tiles."""
        return True

    def get_raster_bands(self, layer_name: str, item: Item) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item."""
        return [self.band_names]

    def get_raster_bounds(
        self, layer_name: str, item: Item, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection (global)."""
        return get_global_raster_bounds(projection)

    def get_raster_metadata(
        self, layer_name: str, item: Item, bands: list[str]
    ) -> RasterMetadata:
        """Wayback tiles have no file-header nodata or similar metadata."""
        return RasterMetadata()

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read raster data for a release, reprojected to the requested grid.

        Args:
            layer_name: the layer name or alias.
            item: the item (its name is the Wayback release number).
            bands: the bands to read (must match the configured band names).
            projection: the projection to read in (window projection adjusted by the
                band set's zoom offset).
            bounds: the bounds to read.
            resampling: the resampling method to use during reprojection.

        Returns:
            the raster data.
        """
        if bands != self.band_names:
            raise ValueError(
                f"expected request for bands {self.band_names} but requested {bands}"
            )

        # Compute the bounds to read in this source's native projection.
        request_geometry = STGeometry(projection, shapely.box(*bounds), None)
        projected_geometry = request_geometry.to_projection(self.projection)
        projected_bounds = (
            math.floor(projected_geometry.shp.bounds[0]),
            math.floor(projected_geometry.shp.bounds[1]),
            math.ceil(projected_geometry.shp.bounds[2]),
            math.ceil(projected_geometry.shp.bounds[3]),
        )
        array = self.read_bounds(item.name, projected_bounds)
        if array is None:
            array = np.zeros(
                (
                    len(self.band_names),
                    projected_bounds[3] - projected_bounds[1],
                    projected_bounds[2] - projected_bounds[0],
                ),
                dtype=np.uint8,
            )

        # Reproject from the native projection to the requested grid.
        src_transform = get_transform_from_projection_and_bounds(
            self.projection, projected_bounds
        )
        dst_transform = get_transform_from_projection_and_bounds(projection, bounds)
        dst_array = np.zeros(
            (array.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
            dtype=array.dtype,
        )
        rasterio.warp.reproject(
            source=array,
            src_crs=self.projection.crs,
            src_transform=src_transform,
            destination=dst_array,
            dst_crs=projection.crs,
            dst_transform=dst_transform,
            resampling=resampling,
        )
        return RasterArray(chw_array=dst_array, time_range=item.geometry.time_range)

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
        group_time_ranges: list[tuple[datetime, datetime] | None] | None = None,
    ) -> None:
        """Materialize data for the window via direct materialization.

        Args:
            window: the window to materialize.
            item_groups: the items from get_items.
            layer_name: the name of this layer.
            layer_cfg: the config of this layer.
            group_time_ranges: optional request time range for each item group.
        """
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
            group_time_ranges=group_time_ranges,
        )
