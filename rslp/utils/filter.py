"""Filters for vessel detection projects."""

import functools
import json

import numpy as np
from upath import UPath


class Filter:
    """Base class for filters."""

    def should_filter(self, lon: float, lat: float) -> bool:
        """Check if the input (longitude and latitude) should be filtered.

        Args:
            lon: latitude of the target point.
            lat: longitude of the target point.

        Returns:
            True to filter out, False to keep.
        """
        raise NotImplementedError


# URL to the marine infrastructure GeoJSON file.
DEFAULT_INFRA_URL = (
    "https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/outputs/marine/latest.geojson"
)
DEFAULT_DISTANCE_THRESHOLD = 0.1  # unit: km, 100 meters


@functools.cache
def get_infra_lonlats(infra_path: UPath) -> tuple[np.ndarray, np.ndarray]:
    """Fetch and cache the infrastructure longitudes and latitudes.

    Args:
        infra_path: path to the marine infrastructure GeoJSON file.

    Returns:
        A tuple of arrays: (longitudes, latitudes).
    """
    with infra_path.open("r") as f:
        geojson_data = json.load(f)

    lons = np.array(
        [feature["geometry"]["coordinates"][0] for feature in geojson_data["features"]]
    )
    lats = np.array(
        [feature["geometry"]["coordinates"][1] for feature in geojson_data["features"]]
    )

    return lons, lats


class NearInfraFilter(Filter):
    """Filter out vessel detection that are too close to marine infrastructure."""

    def __init__(
        self,
        infra_url: str = DEFAULT_INFRA_URL,
        infra_distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> None:
        """Initialize marine infrastructure filter.

        Args:
            infra_url: url to the marine infrastructure GeoJSON file.
            infra_distance_threshold: distance threshold for marine infrastructure.
        """
        self.infra_url = infra_url
        self.infra_lonlats = get_infra_lonlats(UPath(self.infra_url))
        self.infra_distance_threshold = infra_distance_threshold

    def _get_haversine_distances(
        self,
        target_lon: float,
        target_lat: float,
        lonlats: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Get the haversine distances between a target point and a set of points.

        Args:
            target_lon: longitude of the target point.
            target_lat: latitude of the target point.
            lonlats: a tuple of longitude and latitude arrays.

        Returns:
            distances: an array of haversine distances (unit: km).
        """
        # Convert longitude and latitude from degrees to radians
        target_lon = np.radians(target_lon)
        target_lat = np.radians(target_lat)
        lons = np.radians(lonlats[0])
        lats = np.radians(lonlats[1])

        # Haversine formula
        dlon = lons - target_lon
        dlat = lats - target_lat
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(target_lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        # Radius of the Earth in kilometers (mean value)
        earth_radius_km = 6371.0
        # Calculate the great circle distances
        distances = earth_radius_km * c

        return distances

    def should_filter(self, lon: float, lat: float) -> bool:
        """Check if the input is too close to marine infrastructure.

        Args:
            lon: longitude of the target point.
            lat: latitude of the target point.

        Returns:
            True if it is too close to marine infrastructure, False otherwise.
        """
        distances = self._get_haversine_distances(lon, lat, self.infra_lonlats)
        return np.any(distances < self.infra_distance_threshold)


# TODO: add distance_to_coast filter
