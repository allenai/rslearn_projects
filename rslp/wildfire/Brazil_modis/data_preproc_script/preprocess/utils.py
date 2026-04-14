"""Set of utility functions for preprocessing."""

import geopandas as gpd
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon


def grid(
    minx: float, maxx: float, miny: float, maxy: float, stepx: float, stepy: float
) -> gpd.GeoDataFrame:
    """Create a grid of square polygons.

    Args:
        minx (float): Minimum x coordinate of the grid
        maxx (float): Maximum x coordinate of the grid
        miny (float): Minimum y coordinate of the grid
        maxy (float): Maximum y coordinate of the grid
        stepx (float): Horizontal step size of the grid.
        stepy (float): Vertical step size of the grid.

    Returns:
        gpd.GeoSeries: A GeoSeries of square polygons
    """
    x_coords = np.arange(minx, maxx, stepx)
    y_coords = np.arange(miny, maxy, stepy)

    # Create a list to store the polygons
    polygons = []

    # Loop over the x and y coordinates to create the polygons
    for x in x_coords:
        for y in y_coords:
            polygons.append(
                Polygon(
                    [(x, y), (x + stepx, y), (x + stepx, y + stepy), (x, y + stepy)]
                )
            )

    # Create a GeoSeries from the polygons
    grid_series = gpd.GeoSeries(polygons)
    canada_grid_df = gpd.GeoDataFrame(geometry=grid_series)
    canada_grid_df["id"] = range(1, len(canada_grid_df) + 1)

    return canada_grid_df


def adaptive_grid(
    minx: float, maxx: float, miny: float, maxy: float, stepx: int, stepy: int
) -> gpd.GeoDataFrame:
    """Create an adaptive grid of square polygons.

    Args:
        minx (float): Minimum x coordinate of the grid
        maxx (float): Maximum x coordinate of the grid
        miny (float): Minimum y coordinate of the grid
        maxy (float): Maximum y coordinate of the grid
        stepx (int): East-west grid step size in meters.
        stepy (int): North-south grid step size in meters.

    Returns:
        gpd.GeoSeries: A GeoSeries of square polygons
    """
    geod = Geod(ellps="WGS84")

    lat_grid = [miny]
    while lat_grid[-1] < maxy:
        _, lat, _ = geod.fwd(minx, lat_grid[-1], 0, stepy)
        lat_grid.append(lat)

    lon_grid = []
    for lat in lat_grid:
        lon_row = [minx]
        while lon_row[-1] < maxx:
            lon, _, _ = geod.fwd(lon_row[-1], lat, 90, stepx)
            lon_row.append(lon)
        lon_grid.append(lon_row)

    # Create a list to store the polygons
    polygons = []
    for i in range(len(lat_grid) - 1):
        for j in range(len(lon_grid[i]) - 1):
            polygons.append(
                Polygon(
                    [
                        (lon_grid[i][j], lat_grid[i]),
                        (lon_grid[i][j + 1], lat_grid[i]),
                        (lon_grid[i][j + 1], lat_grid[i + 1]),
                        (lon_grid[i][j], lat_grid[i + 1]),
                    ]
                )
            )

    # Create a GeoSeries from the polygons
    grid_series = gpd.GeoSeries(polygons)
    canada_grid_df = gpd.GeoDataFrame(geometry=grid_series)
    canada_grid_df["id"] = range(1, len(canada_grid_df) + 1)

    return canada_grid_df
