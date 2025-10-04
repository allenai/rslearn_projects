"""Window preparer for creating windows with raster labels from point annotations."""

from collections import defaultdict
from datetime import datetime
from typing import cast

from esrun.runner.models.training.labeled_data import (
    AnnotationTask,
    LabeledSTGeometry,
    LabeledWindow,
    RasterLabel,
)
from esrun.runner.tools.labeled_window_preparers.labeled_window_preparer import (
    RasterLabelsWindowPreparer,
)
from esrun.runner.tools.labeled_window_preparers.rasterization_utils import (
    rasterize_shapes_to_mask,
    transform_geometries_to_pixel_coordinates,
)
from rslearn.config import DType
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.geometry import WGS84_PROJECTION
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry


class PointToRasterWindowPreparer(RasterLabelsWindowPreparer):
    """Point to raster window preparer.

    Creates windows of specified size centered on each point annotation.
    Each point annotation becomes a separate window with the point at the center.
    """

    def __init__(
        self,
        window_buffer: int,
        window_resolution: float,
        dtype: str,
        nodata_value: int,
    ):
        """Initialize point to raster window preparer.

        Args:
            window_buffer: Buffer around the point in pixels (e.g., 31 creates 63x63 pixel windows)
            window_resolution: Resolution in meters per pixel (e.g., 10.0 for 10m/pixel)
            dtype: Data type for the raster labels (e.g., "float32", "uint8", "int16")
            nodata_value: Nodata value for the raster labels
        """
        self.window_buffer = window_buffer
        self.window_resolution = window_resolution
        self.dtype = DType(dtype.lower())
        self.nodata_value = nodata_value

    def prepare_labeled_windows(
        self, annotation_task: AnnotationTask
    ) -> list[LabeledWindow[list[RasterLabel]]]:
        """Prepare labeled windows from point annotation tasks.

        Creates one window per point annotation, with the point at the center
        of a window with window_buffer pixels around the point on each side.

        Args:
            annotation_task: Single AnnotationTask object containing point annotations

        Returns:
            List of LabeledWindow objects, one per point annotation
        """
        if not annotation_task.annotations:
            return []

        labeled_windows = []

        for i, annotation in enumerate(annotation_task.annotations):
            # Get the point geometry
            point_geom = annotation.st_geometry.shp
            if not isinstance(point_geom, Point):
                raise ValueError(
                    f"Expected Point geometry, got {type(point_geom)} with geom_type {getattr(point_geom, 'geom_type', 'unknown')}"
                )

            # Project to UTM
            projected_point = self._project_geometry_to_utm(point_geom)

            # Create window geometry centered on the point
            window_geometry = self._create_window_geometry_from_point(
                projected_point, annotation.st_geometry.time_range
            )

            # Create raster labels for this window
            raster_labels = self._create_raster_labels_for_point(
                annotation, window_geometry
            )

            labeled_window = LabeledWindow(
                name=f"task_{annotation_task.task_id}_point_{i}",
                st_geometry=window_geometry,
                labels=raster_labels,
            )

            labeled_windows.append(labeled_window)

        return labeled_windows

    def _project_geometry_to_utm(self, geometry: BaseGeometry) -> STGeometry:
        """Project a geometry to an appropriate UTM coordinate system.

        Args:
            geometry: The geometry to project

        Returns:
            STGeometry projected to UTM coordinates
        """
        # Create source geometry in WGS84
        src_geometry = STGeometry(WGS84_PROJECTION, geometry, None)

        # Get the centroid coordinates to determine appropriate UTM zone
        lon, lat = geometry.x, geometry.y

        # Get appropriate UTM/UPS CRS for this location
        destination_crs = get_utm_ups_crs(lon, lat)

        # Create projection with specified resolution
        destination_projection = Projection(
            destination_crs,
            self.window_resolution,
            -self.window_resolution,
        )

        # Project the geometry
        return src_geometry.to_projection(destination_projection)

    def _create_window_geometry_from_point(
        self, projected_point: STGeometry, time_range: tuple[datetime, datetime] | None
    ) -> STGeometry:
        """Create a square window geometry centered on the projected point.

        Args:
            projected_point: The projected point geometry
            time_range: Time range for the window

        Returns:
            STGeometry representing the window bounds
        """
        point = cast(Point, projected_point.shp)

        # Create square polygon centered on the point using box
        minx = point.x - self.window_buffer
        miny = point.y - self.window_buffer
        maxx = point.x + self.window_buffer + 1
        maxy = point.y + self.window_buffer + 1

        window_polygon = box(minx, miny, maxx, maxy)

        return STGeometry(projected_point.projection, window_polygon, time_range)

    def _compute_window_bounds(
        self, window_geometry: STGeometry
    ) -> tuple[int, int, int, int]:
        """Compute integer bounds for the window from the window geometry.

        Args:
            window_geometry: The window geometry

        Returns:
            Tuple of (minx, miny, maxx, maxy) in pixel coordinates
        """
        bounds = cast(BaseGeometry, window_geometry.shp).bounds

        minx = int(bounds[0])
        miny = int(bounds[1])
        maxx = int(bounds[2])
        maxy = int(bounds[3])

        return (minx, miny, maxx, maxy)

    def _create_raster_labels_for_point(
        self, annotation: LabeledSTGeometry, window_geometry: STGeometry
    ) -> list[RasterLabel]:
        """Create raster label by rasterizing point annotations.

        Args:
            annotation: The annotation to create raster labels for
            window_geometry: The window geometry of the annotation

        Returns:
            List of RasterLabel objects with rasterized labels
        """
        # Collect labeled shapes for rasterization
        keys = sorted(annotation.labels.keys())
        shape_labels: dict[str, list[tuple[BaseGeometry, int | float]]] = defaultdict(
            list
        )

        projected_annotation = self._project_geometry_to_utm(
            cast(BaseGeometry, annotation.st_geometry.shp)
        )

        for key in keys:
            label_value = annotation.labels[key]
            if label_value is not None:
                shape_labels[key].append(
                    (cast(BaseGeometry, projected_annotation.shp), label_value)
                )

        window_bounds = self._compute_window_bounds(window_geometry)
        width = window_bounds[2] - window_bounds[0]  # maxx - minx
        height = window_bounds[3] - window_bounds[1]  # maxy - miny

        raster_labels: list[RasterLabel] = []
        for key in keys:
            pixel_shapes = transform_geometries_to_pixel_coordinates(
                shape_labels[key], window_bounds
            )
            label_mask = rasterize_shapes_to_mask(
                width=width,
                height=height,
                shape_labels=pixel_shapes,
                dtype=self.dtype.get_numpy_dtype(),
                nodata_value=self.nodata_value,
            )
            raster_labels.append(RasterLabel(key=key, value=label_mask))

        return raster_labels
