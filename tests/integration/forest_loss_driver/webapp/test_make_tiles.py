"""Test rslp.forest_loss_driver.webapp.make_tiles module."""

import json
import math
import pathlib
from datetime import UTC, datetime

import mapbox_vector_tile
import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

from rslp.forest_loss_driver.const import WINDOWS_FNAME
from rslp.forest_loss_driver.webapp.make_tiles import (
    OUTPUT_GEOJSON_SUFFIX,
    MakeTilesArgs,
    make_tiles,
)

# Check output at this zoom level.
ZOOM_LEVEL = 0

# WebMercator projection properties.
WEBMERCATOR_CRS = CRS.from_epsg(3857)
WEBMERCATOR_TOTAL_METERS = 2 * math.pi * 6378137
TILE_SIZE = 4096

# The resolution at ZOOM_LEVEL for WebMercator projection.
RESOLUTION = WEBMERCATOR_TOTAL_METERS / (2**ZOOM_LEVEL) / TILE_SIZE


def test_make_tiles(tmp_path: pathlib.Path) -> None:
    """Test make_tiles.

    We create the needed files in a single window and verify that make_tiles produces
    vector tile with the correct polygon.
    """
    ds_path = UPath(tmp_path) / "dataset"
    tile_path = UPath(tmp_path) / "tiles"

    # Set the polygon and category we want to use.
    webmercator_geom = STGeometry(
        Projection(WEBMERCATOR_CRS, RESOLUTION, RESOLUTION),
        shapely.box(0, 0, 5, 5),
        None,
    )
    wgs84_geom = webmercator_geom.to_projection(WGS84_PROJECTION)
    category = "agriculture"

    # Write an info.json file, it just needs to have the wkt/date keys for make_tiles.
    # See output_forest_event_metadata in extract_alerts.py for details.
    window_path = Window.get_window_root(ds_path, "default", "default")
    window_path.mkdir(parents=True, exist_ok=True)
    with (window_path / "info.json").open("w") as f:
        json.dump(
            {
                "wkt": wgs84_geom.shp.wkt,
                "date": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
            },
            f,
        )

    # Write the GeoJSON feature with category prediction.
    geojson_fname = window_path / OUTPUT_GEOJSON_SUFFIX
    geojson_fname.parent.mkdir(parents=True, exist_ok=True)
    with geojson_fname.open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "properties": {},
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            # Arbitrary geometry since it is classification task.
                            "type": "Point",
                            "coordinates": [0, 0],
                        },
                        "properties": {
                            "new_label": category,
                        },
                    }
                ],
            },
            f,
        )

    # Write list of good windows.
    # This is what index_windows would create.
    with (ds_path / WINDOWS_FNAME).open("w") as f:
        json.dump(["default"], f)

    # Run the make_tiles pipeline.
    make_tiles(MakeTilesArgs(ds_root=str(ds_path), tile_path=str(tile_path)))

    # Decode the resulting vector tile.
    mvt_fname = tile_path / "dataset" / str(ZOOM_LEVEL) / "0" / "0.pbf"
    assert mvt_fname.exists()
    with mvt_fname.open("rb") as f:
        mvt_data = mapbox_vector_tile.decode(f.read())

    # The mapbox vector tile maps from mvt_layer_name -> FeatureCollection.
    feat = mvt_data[category]["features"][0]
    polygon_coordinates = feat["geometry"]["coordinates"]
    # Polygon is a list of rings starting with exterior and then rest are interior.
    # In our case we should just have the exterior ring.
    assert len(polygon_coordinates) == 1
    exterior_coordinates = polygon_coordinates[0]
    # At zoom level 0, there is only one tile.
    # The coordinates are based on 4096x4096 grid over the tile.
    # However the top-left is (0, 0) while we just scale the projection coordinates so
    # our top-left is (-2048, -2048) at zoom 0, that means we need to apply an offset.
    exterior_coordinates = [
        (coord[0] - 2048, coord[1] - 2048) for coord in exterior_coordinates
    ]
    assert set(exterior_coordinates) == {
        (0, 0),
        (0, 5),
        (5, 0),
        (5, 5),
    }
