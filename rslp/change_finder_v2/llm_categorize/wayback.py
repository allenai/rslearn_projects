"""Fetch high-resolution historical imagery from ArcGIS World Imagery Wayback.

The Wayback archive exposes ~200 dated "releases" of the Esri World Imagery
basemap. The imagery shown at a given location only changes on a handful of
those releases, so for any point we:

1. Enumerate the *distinct* releases at the point's tile using the WMTS
   ``tilemap`` endpoint, whose ``select`` field names the release that actually
   holds the tile data (Esri's own change-detection mechanism).
2. Resolve each distinct release's true acquisition date at the point via the
   per-release metadata ``identify`` endpoint (``SRC_DATE``).
3. Pick the release closest at/before a target ``pre`` date and closest
   at/after a target ``post`` date, and stitch a centered crop from its tiles.

No authentication is required; all endpoints are public.
"""

from __future__ import annotations

import io
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import date

import requests
from PIL import Image

logger = logging.getLogger(__name__)

CONFIG_URL = (
    "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
)
TILE_SERVICE_BASE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer"
)
TILE_SIZE = 256

_TITLE_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class TileNotFoundError(Exception):
    """Raised when a Wayback tile is not available at a release/zoom (HTTP 404)."""


@dataclass
class WaybackImage:
    """A fetched Wayback image and its provenance."""

    release_num: str
    release_date: str
    capture_date: date
    zoom: int
    png_bytes: bytes


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Return (col, row) Web Mercator tile indices for a lon/lat at a zoom."""
    n = 2**zoom
    col = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    row = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return col, row


def _lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Return (x, y) global pixel coordinates for a lon/lat at a zoom."""
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


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
    """Client for the ArcGIS World Imagery Wayback archive."""

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
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                last_exc = exc
                status = (
                    exc.response.status_code if exc.response is not None else None
                )
                # Client errors (other than rate limiting) are not transient,
                # so do not waste time retrying them.
                if status is not None and 400 <= status < 500 and status != 429:
                    break
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                last_exc = exc
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
        while i < len(self._releases):
            release_num = self._releases[i][0]
            url = f"{TILE_SERVICE_BASE}/tilemap/{release_num}/{zoom}/{row}/{col}/1/1"
            data = self._get_json(url)
            if not data.get("valid") or not data.get("select"):
                i += 1
                continue
            owner = str(data["select"][0])
            if owner not in distinct:
                distinct.append(owner)
            # Continue from the release immediately older than the owning one.
            i = self._release_index.get(owner, i) + 1
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
        data = self._get_json(metadata_url, params=params)
        results = data.get("results") or []
        if not results:
            return None
        return _parse_capture_date(results[0].get("attributes", {}))

    def _fetch_tile(self, release_num: str, zoom: int, col: int, row: int) -> Image.Image:
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

    def fetch_image(
        self, release_num: str, lon: float, lat: float, zoom: int, size: int = 512
    ) -> bytes:
        """Fetch a ``size`` x ``size`` PNG centered on the point from a release.

        Stitches the covering tiles into a mosaic and crops a centered window.
        """
        px, py = _lonlat_to_global_pixel(lon, lat, zoom)
        left = px - size / 2.0
        top = py - size / 2.0

        min_col = math.floor(left / TILE_SIZE)
        min_row = math.floor(top / TILE_SIZE)
        max_col = math.floor((left + size - 1) / TILE_SIZE)
        max_row = math.floor((top + size - 1) / TILE_SIZE)

        mosaic_w = (max_col - min_col + 1) * TILE_SIZE
        mosaic_h = (max_row - min_row + 1) * TILE_SIZE
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h))
        for c in range(min_col, max_col + 1):
            for r in range(min_row, max_row + 1):
                tile = self._fetch_tile(release_num, zoom, c, r)
                mosaic.paste(
                    tile,
                    ((c - min_col) * TILE_SIZE, (r - min_row) * TILE_SIZE),
                )

        crop_left = int(round(left - min_col * TILE_SIZE))
        crop_top = int(round(top - min_row * TILE_SIZE))
        crop = mosaic.crop(
            (crop_left, crop_top, crop_left + size, crop_top + size)
        )
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    def find_images(
        self,
        lon: float,
        lat: float,
        pre_date: date,
        post_date: date,
        zoom: int = 18,
        size: int = 512,
    ) -> tuple[WaybackImage | None, WaybackImage | None]:
        """Find Wayback images at/before ``pre_date`` and at/after ``post_date``.

        The distinct releases at a point are ordered newest-first, and their
        acquisition dates are (almost always) monotonically non-increasing along
        that order because a location's imagery is only ever updated to newer
        captures. We exploit that to binary-search for the pre/post boundaries
        instead of resolving the slow per-release acquisition date for every
        release, then refine with a small local window to correct for any minor
        non-monotonicity.

        Returns:
            a (pre_image, post_image) tuple; either element may be None if no
            release with a suitable acquisition date is available at the point.
        """
        distinct = self.enumerate_distinct_releases(lon, lat, zoom)
        n = len(distinct)
        cache: dict[int, date | None] = {}

        def cap(i: int) -> date | None:
            if i not in cache:
                try:
                    cache[i] = self.get_capture_date(distinct[i], lon, lat)
                except Exception:  # noqa: BLE001 - skip unresolvable releases
                    cache[i] = None
            return cache[i]

        def cap_probe(i: int) -> date | None:
            """Acquisition date at ``i``, probing neighbors if unresolvable."""
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

        def select(target: date, is_pre: bool) -> tuple[str, date] | None:
            if n == 0:
                return None
            # Binary search the boundary index. For pre we want the leftmost
            # index whose capture <= target; for post the leftmost whose
            # capture < target (so the index before it is the last >= target).
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                d = cap_probe(mid)
                if d is None:
                    break
                below = d <= target if is_pre else d < target
                if below:
                    hi = mid
                else:
                    lo = mid + 1
            center = lo if is_pre else lo - 1
            # Refine within a small window around the boundary to correct for
            # any minor non-monotonicity or an unresolvable boundary release.
            best: tuple[int, date] | None = None
            for i in range(max(0, center - 1), min(n, center + 2)):
                d = cap(i)
                if d is None:
                    continue
                if is_pre:
                    if d <= target and (best is None or d > best[1]):
                        best = (i, d)
                else:
                    if d >= target and (best is None or d < best[1]):
                        best = (i, d)
            if best is None:
                return None
            return distinct[best[0]], best[1]

        pre_choice = select(pre_date, is_pre=True)
        post_choice = select(post_date, is_pre=False)

        def build(choice: tuple[str, date] | None) -> WaybackImage | None:
            if choice is None:
                return None
            release_num, capture = choice
            # A release may report imagery at the point via the metadata layer
            # but still lack tiles at the requested zoom there, so fall back to
            # progressively coarser zooms before giving up.
            min_zoom = max(zoom - 3, 1)
            for current_zoom in range(zoom, min_zoom - 1, -1):
                try:
                    png = self.fetch_image(
                        release_num, lon, lat, current_zoom, size=size
                    )
                except TileNotFoundError:
                    logger.warning(
                        "Wayback release %s has no tile at zoom %d for "
                        "lon=%.6f lat=%.6f; trying a coarser zoom",
                        release_num,
                        current_zoom,
                        lon,
                        lat,
                    )
                    continue
                except Exception:  # noqa: BLE001 - omit image on any fetch error
                    logger.warning(
                        "Wayback image fetch failed for release %s at "
                        "lon=%.6f lat=%.6f; omitting",
                        release_num,
                        lon,
                        lat,
                        exc_info=True,
                    )
                    return None
                return WaybackImage(
                    release_num=release_num,
                    release_date=self._release_date(self._config[release_num]),
                    capture_date=capture,
                    zoom=current_zoom,
                    png_bytes=png,
                )
            logger.warning(
                "Wayback release %s has no usable tiles at zooms %d-%d for "
                "lon=%.6f lat=%.6f; omitting",
                release_num,
                min_zoom,
                zoom,
                lon,
                lat,
            )
            return None

        return build(pre_choice), build(post_choice)
