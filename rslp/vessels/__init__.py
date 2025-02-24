"""Shared classes across vessel detection tasks."""

from datetime import datetime
from enum import Enum
from typing import Any

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from typing_extensions import TypedDict
from upath import UPath


class VesselDetectionSource(str, Enum):
    """The sensor that the vessel detection came from."""

    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"


class VesselDetectionDict(TypedDict):
    """Serializable metadata about a VesselDetection.

    Args:
        source: the type of satellite imagery used.
        col: the column in projection coordinates.
        row: the row in projection coordinates.
        projection: the projection used.
        score: confidence score from object detector.
        ts: datetime fo the window (if known).
        scene_id: the scene ID that the vessel was detected in (if known).
        crop_fname: filename where crop image for this vessel is stored.
        longitude: the longitude position of the vessel detection.
        latitude: the latitude position of the vessel detection.
    """

    source: VesselDetectionSource
    col: int
    row: int
    projection: dict[str, Any]
    score: float
    ts: str | None
    scene_id: str | None
    crop_fname: str | None
    longitude: float
    latitude: float


class VesselDetection:
    """A vessel detected in a satellite image scene."""

    def __init__(
        self,
        source: VesselDetectionSource,
        col: int,
        row: int,
        projection: Projection,
        score: float,
        ts: datetime | None = None,
        scene_id: str | None = None,
        crop_fname: UPath | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new VesselDetection.

        Args:
            source: the type of satellite imagery used.
            col: the column in projection coordinates.
            row: the row in projection coordinates.
            projection: the projection used.
            score: confidence score from object detector.
            ts: datetime fo the window (if known).
            scene_id: the scene ID that the vessel was detected in (if known).
            crop_fname: filename where crop image for this vessel is stored.
            metadata: additional metadata that caller wants to store with this detection.
        """
        self.source = source
        self.col = col
        self.row = row
        self.projection = projection
        self.score = score
        self.ts = ts
        self.scene_id = scene_id
        self.crop_fname = crop_fname

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def get_lon_lat(self) -> tuple[float, float]:
        """Get the longitude and latitude of this detection.

        Returns:
            (longitude, latitude) tuple.
        """
        src_geom = STGeometry(self.projection, shapely.Point(self.col, self.row), None)
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        return (dst_geom.shp.x, dst_geom.shp.y)

    def to_dict(self) -> VesselDetectionDict:
        """Serialize this detection to a JSON-encodable dictionary."""
        lon, lat = self.get_lon_lat()
        return VesselDetectionDict(
            source=self.source,
            col=self.col,
            row=self.row,
            projection=self.projection.serialize(),
            score=self.score,
            ts=self.ts.isoformat() if self.ts else None,
            scene_id=self.scene_id,
            crop_fname=str(self.crop_fname) if self.crop_fname else None,
            longitude=lon,
            latitude=lat,
        )

    def to_feature(self) -> dict[str, Any]:
        """Serialize this detection to a GeoJSON feature dictionary."""
        lon, lat = self.get_lon_lat()
        return {
            "type": "Feature",
            "properties": self.to_dict(),
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
        }
