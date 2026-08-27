"""Fetch Sentinel-2 L2A RGB chips for the LLM categorization pipeline.

Uses the olmoearth_datasets Sentinel-2 L2A data source to read a 64x64 RGB chip
centered on a point at the scene nearest a target date, rendered as true color
(B04/B03/B02 divided by a brightness divisor, clipped to 0-255), then upsampled
to 128x128 with nearest-neighbor interpolation.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import shapely.geometry
from olmoearth_run.runner.tools.rslearn_data_sources.olmoearth_datasets.sentinel2_l2a import (
    Sentinel2L2A,
)
from olmoearth_shared.models.datasets.bands.sentinel2 import Sentinel2L2ABand
from PIL import Image
from rasterio.enums import Resampling
from rslearn.config import QueryConfig
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

logger = logging.getLogger(__name__)

# Native Sentinel-2 resolution and chip geometry.
S2_RESOLUTION = 10.0
CHIP_SIZE = 64
UPSAMPLED_SIZE = 128
LAYER_NAME = "sentinel2"

_RGB_BANDS = [Sentinel2L2ABand.B04, Sentinel2L2ABand.B03, Sentinel2L2ABand.B02]

# Divisor mapping raw reflectance (nominal 0-10000) to 0-255 for display. A
# value of 13 maps reflectance ~3315 to white, close to a standard Sentinel-2
# true-color stretch and less prone to blowing out bright sandy/arid areas than
# a smaller divisor.
_BRIGHTNESS_DIVISOR = 13.0

# SCL (scene classification) class values treated as cloud / invalid:
# 0 no-data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus.
_SCL_BAD_CLASSES = (0, 1, 3, 8, 9, 10)


@dataclass
class S2Chip:
    """A rendered Sentinel-2 RGB chip."""

    item_name: str
    item_date: date
    target_date: date
    cloud_fraction: float | None
    png_bytes: bytes


class S2Fetcher:
    """Fetches Sentinel-2 RGB chips centered on a point near a target date."""

    def __init__(
        self,
        max_matches: int = 32,
        harmonize: bool = True,
        clear_threshold: float = 0.05,
        max_candidates: int = 8,
    ) -> None:
        """Create the fetcher and its underlying data source.

        Args:
            max_matches: maximum number of candidate scenes to consider per query.
            harmonize: harmonize pixel values across processing baselines so chips
                from different dates are radiometrically consistent.
            clear_threshold: chip cloud/invalid fraction at or below which a scene
                is considered clear enough to accept immediately (the closest such
                scene wins).
            max_candidates: maximum number of candidate scenes (nearest in date)
                whose cloud cover is evaluated before settling on the clearest.
        """
        self.max_matches = max_matches
        self.clear_threshold = clear_threshold
        self.max_candidates = max_candidates
        self._data_source = Sentinel2L2A(
            harmonize=harmonize,
            band_sets=[[band] for band in _RGB_BANDS]
            + [[Sentinel2L2ABand.SCL]],
        )

    def fetch_chip(
        self,
        lon: float,
        lat: float,
        target_date: date,
        tolerance_days: int = 45,
    ) -> S2Chip | None:
        """Fetch the RGB chip nearest ``target_date`` centered on the point.

        Args:
            lon: longitude of the chip center.
            lat: latitude of the chip center.
            target_date: the desired acquisition date.
            tolerance_days: half-width of the search window around target_date.

        Returns:
            the rendered S2Chip, or None if no scene is available in the window.
        """
        projection = get_utm_ups_projection(
            lon, lat, S2_RESOLUTION, -S2_RESOLUTION
        )
        point = shapely.geometry.Point(lon, lat)

        # Center pixel in the chosen projection (already in pixel units).
        center = (
            STGeometry(WGS84_PROJECTION, point, None)
            .to_projection(projection)
            .shp
        )
        center_col = int(round(center.x))
        center_row = int(round(center.y))
        half = CHIP_SIZE // 2
        bounds = (
            center_col - half,
            center_row - half,
            center_col + half,
            center_row + half,
        )

        target_dt = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            12,
            tzinfo=timezone.utc,
        )
        time_range = (
            target_dt - timedelta(days=tolerance_days),
            target_dt + timedelta(days=tolerance_days),
        )
        # Query with a box covering the chip extent (a zero-area point geometry
        # breaks the mosaic matching logic).
        query_box = shapely.geometry.box(
            bounds[0], bounds[1], bounds[2], bounds[3]
        )
        geometry = STGeometry(projection, query_box, time_range)

        groups = self._data_source.get_items(
            [geometry], QueryConfig(max_matches=self.max_matches)
        )[0]
        items = [item for group in groups for item in group.items]
        if not items:
            return None

        best_item, best_fraction = self._select_clearest(
            items, projection, bounds, target_dt
        )

        channels = []
        for band in _RGB_BANDS:
            raster = self._data_source.read_raster(
                layer_name=LAYER_NAME,
                item=best_item,
                bands=[band],
                projection=projection,
                bounds=bounds,
            )
            channels.append(raster.get_chw_array()[0])

        rgb = np.stack(channels, axis=-1).astype(np.float32)
        rgb = np.clip(rgb / _BRIGHTNESS_DIVISOR, 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb, mode="RGB").resize(
            (UPSAMPLED_SIZE, UPSAMPLED_SIZE), resample=Image.NEAREST
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return S2Chip(
            item_name=best_item.name,
            item_date=_item_datetime(best_item).date(),
            target_date=target_date,
            cloud_fraction=best_fraction,
            png_bytes=buf.getvalue(),
        )

    def _select_clearest(
        self,
        items: list,
        projection: object,
        bounds: tuple[int, int, int, int],
        target_dt: datetime,
    ) -> tuple[object, float | None]:
        """Pick the least-cloudy scene near the target date using the SCL band.

        Candidates are evaluated in order of date proximity; the first scene at or
        below ``clear_threshold`` wins, otherwise the clearest among the nearest
        ``max_candidates`` is returned. Falls back to the nearest scene by date if
        no SCL could be read.
        """
        candidates = sorted(
            items, key=lambda item: abs(_item_datetime(item) - target_dt)
        )
        # Collapse duplicate granules that share a capture date so the candidate
        # budget covers distinct dates rather than repeated views of the same one.
        seen_dates: set[date] = set()
        unique: list[object] = []
        for item in candidates:
            item_date = _item_datetime(item).date()
            if item_date in seen_dates:
                continue
            seen_dates.add(item_date)
            unique.append(item)
        candidates = unique
        best_item: object | None = None
        best_fraction: float | None = None
        for item in candidates[: self.max_candidates]:
            try:
                fraction = self._cloud_fraction(item, projection, bounds)
            except Exception:  # noqa: BLE001 - skip scenes we cannot score
                logger.warning(
                    "Failed to read SCL for %s; skipping for cloud scoring",
                    item.name,
                    exc_info=True,
                )
                continue
            if best_fraction is None or fraction < best_fraction:
                best_item, best_fraction = item, fraction
            if fraction <= self.clear_threshold:
                break
        if best_item is None:
            # No SCL could be read for any candidate; fall back to nearest date.
            return candidates[0], None
        return best_item, best_fraction

    def _cloud_fraction(
        self,
        item: object,
        projection: object,
        bounds: tuple[int, int, int, int],
    ) -> float:
        """Return the fraction of cloud/invalid SCL pixels over the chip bounds."""
        raster = self._data_source.read_raster(
            layer_name=LAYER_NAME,
            item=item,
            bands=[Sentinel2L2ABand.SCL],
            projection=projection,
            bounds=bounds,
            resampling=Resampling.nearest,
        )
        scl = raster.get_chw_array()[0]
        if scl.size == 0:
            return 1.0
        bad = np.isin(scl, _SCL_BAD_CLASSES).sum()
        return float(bad) / float(scl.size)


def _item_datetime(item: object) -> datetime:
    """Return a representative acquisition datetime for an rslearn item."""
    time_range = item.geometry.time_range  # type: ignore[attr-defined]
    if time_range is None:
        raise ValueError("Sentinel-2 item is missing a time range")
    return time_range[0]
