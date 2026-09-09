"""An rslearn-backed database of imagery for VLM use cases.

The database is keyed by ``(longitude, latitude, year)`` (longitude/latitude rounded
to 5 decimal places). Each key corresponds to a single rslearn window named
``{lon:.5f}_{lat:.5f}_{year}`` (e.g. ``-122.33210_47.60620_2023``), centered on the
point, with a time range spanning two years before to two years after the key year.

This module provides two operations:

- :func:`add_points`: create the windows for a set of points. The caller is expected to
  have already set up the dataset ``config.json`` (this module does not write it). The
  config defines the layers that get materialized into each window, for example a
  monthly Sentinel-2 RGB layer and a quarterly Esri Wayback layer.
- :func:`list_available_images`: after the caller has run ``rslearn dataset prepare`` and
  ``rslearn dataset materialize``, list the materialized images for a point.

Typical flow::

    add_points(ds_path, [(-122.3321, 47.6062, 2023)])
    # rslearn dataset prepare --root <ds_path> --group default
    # rslearn dataset materialize --root <ds_path> --group default \
    #     --enabled-layers sentinel2 --workers 32
    # rslearn dataset materialize --root <ds_path> --group default \
    #     --enabled-layers esri --workers 1
    images = list_available_images(ds_path, -122.3321, 47.6062, 2023)

A reference ``config.json`` to copy into ``<ds_path>`` lives at
``data/change_finder_v2/vlm/image_db/config.json``. The base window resolution is Web
Mercator zoom 14 so that the Esri layer's ``zoom_offset`` of 3 stores it at Web Mercator
zoom 17 (a 64x64 window becomes a 512x512 Esri raster).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy.typing as npt
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.add_windows import add_windows_from_geometries
from rslearn.utils import Projection, STGeometry
from upath import UPath

# Web Mercator constants for the base window projection.
WEB_MERCATOR_EPSG = 3857
WEB_MERCATOR_UNITS = 2 * math.pi * 6378137

# Base window resolution (Web Mercator zoom 14) and size in pixels. With an Esri layer
# zoom_offset of 3, the 64x64 base window becomes a 512x512 raster at Web Mercator
# zoom 17.
BASE_ZOOM = 14
WINDOW_SIZE = 64

# Number of years before and after the key year to include in the window time range.
YEARS_BEFORE = 2
YEARS_AFTER = 2

DEFAULT_GROUP = "default"


@dataclass
class AvailableImage:
    """One materialized image available for a point.

    Attributes:
        layer_name: the dataset layer the image belongs to (e.g. "sentinel2", "esri").
        group_idx: the item group index within the layer (one per time period).
        item_name: the underlying source item name (e.g. a Sentinel-2 scene id or an
            Esri Wayback release number), or None if unavailable.
        time_range: the acquisition time range of the source item, or None.
        array: the image as a CHW uint8/whatever-dtype numpy array.
    """

    layer_name: str
    group_idx: int
    item_name: str | None
    time_range: tuple[datetime, datetime] | None
    array: npt.NDArray[Any]


def window_name(lon: float, lat: float, year: int) -> str:
    """Return the window name for a point key."""
    return f"{lon:.5f}_{lat:.5f}_{year}"


def _base_projection() -> Projection:
    """Return the base window projection (Web Mercator at BASE_ZOOM)."""
    total_pixels = 256 * (2**BASE_ZOOM)
    pixel_size = WEB_MERCATOR_UNITS / total_pixels
    return Projection(CRS.from_epsg(WEB_MERCATOR_EPSG), pixel_size, -pixel_size)


def _window_time_range(year: int) -> tuple[datetime, datetime]:
    """Return the (start, end) time range for a key year.

    Spans Jan 1 of (year - YEARS_BEFORE) through Jan 1 of (year + YEARS_AFTER + 1),
    i.e. the full set of months from two years before to two years after.
    """
    start = datetime(year - YEARS_BEFORE, 1, 1, tzinfo=UTC)
    end = datetime(year + YEARS_AFTER + 1, 1, 1, tzinfo=UTC)
    return start, end


def _require_dataset(ds_path: str | UPath) -> Dataset:
    """Open the dataset, raising a clear error if its config.json is missing."""
    path = UPath(ds_path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"dataset config not found at {path / 'config.json'}; set up the dataset "
            "config.json before adding points"
        )
    return Dataset(path)


def add_points(
    ds_path: str | UPath,
    points: list[tuple[float, float, int]],
    group: str = DEFAULT_GROUP,
) -> list[Window]:
    """Add windows for a set of ``(lon, lat, year)`` points to the dataset.

    Existing windows (by name) are left untouched. The dataset config.json must already
    exist; it defines which layers get materialized into each window.

    Args:
        ds_path: the root path of the rslearn dataset.
        points: list of ``(longitude, latitude, year)`` tuples.
        group: the window group to add the windows to.

    Returns:
        the list of newly created windows (excludes points whose windows already
        existed).
    """
    dataset = _require_dataset(ds_path)
    projection = _base_projection()

    existing = {
        window.name for window in dataset.load_windows(groups=[group])
    }

    created: list[Window] = []
    for lon, lat, year in points:
        name = window_name(lon, lat, year)
        if name in existing:
            continue
        geometry = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
        windows = add_windows_from_geometries(
            dataset=dataset,
            group=group,
            geometries=[geometry],
            projection=projection,
            name=name,
            window_size=WINDOW_SIZE,
            time_range=_window_time_range(year),
        )
        created.extend(windows)
        existing.add(name)

    return created


def list_available_images(
    ds_path: str | UPath,
    lon: float,
    lat: float,
    year: int,
    group: str = DEFAULT_GROUP,
) -> list[AvailableImage]:
    """List the materialized images available for a ``(lon, lat, year)`` point.

    This reads whatever layers have been materialized for the point's window, so it
    works for any layers defined in the dataset config (e.g. monthly Sentinel-2 and
    quarterly Esri). The caller must have already run rslearn prepare/materialize.

    Args:
        ds_path: the root path of the rslearn dataset.
        lon: the longitude of the point.
        lat: the latitude of the point.
        year: the key year of the point.
        group: the window group the point's window belongs to.

    Returns:
        a list of :class:`AvailableImage`, one per materialized item group, sorted by
        layer name then time range.

    Raises:
        FileNotFoundError: if the dataset config or the point's window does not exist.
    """
    dataset = _require_dataset(ds_path)
    name = window_name(lon, lat, year)
    windows = dataset.load_windows(groups=[group], names=[name])
    if not windows:
        raise FileNotFoundError(
            f"window {name} not found in group {group}; call add_points and materialize "
            "first"
        )
    window = windows[0]
    layer_datas = window.load_layer_datas()

    images: list[AvailableImage] = []
    for layer_name, group_idx in window.list_completed_layers():
        layer_cfg = dataset.layers.get(layer_name)
        if layer_cfg is None or not layer_cfg.band_sets:
            continue
        band_cfg = layer_cfg.band_sets[0]
        projection, bounds = band_cfg.get_final_projection_and_bounds(
            window.projection, window.bounds
        )
        raster_format = band_cfg.instantiate_raster_format()
        raster = window.data.read_raster(
            layer_name,
            band_cfg.bands,
            raster_format,
            projection=projection,
            bounds=bounds,
            group_idx=group_idx,
        )

        item_name, time_range = _item_provenance(layer_datas, layer_name, group_idx)
        images.append(
            AvailableImage(
                layer_name=layer_name,
                group_idx=group_idx,
                item_name=item_name,
                time_range=time_range,
                array=raster.get_chw_array(),
            )
        )

    images.sort(
        key=lambda image: (
            image.layer_name,
            image.time_range[0] if image.time_range is not None else datetime.min,
        )
    )
    return images


def _item_provenance(
    layer_datas: dict[str, Any], layer_name: str, group_idx: int
) -> tuple[str | None, tuple[datetime, datetime] | None]:
    """Extract the source item name and acquisition time range for an item group."""
    layer_data = layer_datas.get(layer_name)
    if layer_data is None:
        return None, None
    if group_idx >= len(layer_data.serialized_item_groups):
        return None, None
    item_group = layer_data.serialized_item_groups[group_idx]
    if not item_group:
        return None, None
    item = item_group[0]
    item_name = item.get("name")
    time_range = None
    raw_time_range = item.get("geometry", {}).get("time_range")
    if raw_time_range is not None:
        time_range = (
            datetime.fromisoformat(raw_time_range[0]),
            datetime.fromisoformat(raw_time_range[1]),
        )
    return item_name, time_range
