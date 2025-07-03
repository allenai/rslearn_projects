"""Utils for creating windows."""

from rslearn.utils import STGeometry


def calculate_bounds(
    geometry: STGeometry, window_size: int
) -> tuple[int, int, int, int]:
    """Calculate the bounds of a window around a geometry.

    Args:
        geometry: the geometry to calculate the bounds of.
        window_size: the size of the window.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")

    # This is specific for window size = 1.
    if window_size == 1:
        bounds = (
            int(geometry.shp.x),
            int(geometry.shp.y) - window_size,
            int(geometry.shp.x) + window_size,
            int(geometry.shp.y),
        )
    else:
        bounds = (
            int(geometry.shp.x) - window_size // 2,
            int(geometry.shp.y) - window_size // 2,
            int(geometry.shp.x) + window_size // 2,
            int(geometry.shp.y) + window_size // 2,
        )

    return bounds
