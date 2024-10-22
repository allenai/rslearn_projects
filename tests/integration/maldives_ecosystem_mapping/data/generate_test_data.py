#!/usr/bin/env python3

import json

import numpy as np
import rasterio
import shapely
from affine import Affine
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils import Projection, STGeometry

tif_size = 32


def create_tif(fname: str) -> None:
    profile = dict(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=3,
        crs=CRS.from_epsg(3857),
        transform=Affine(1, 0, 0, 0, -1, 0),
        height=tif_size,
        width=tif_size,
    )
    array = np.zeros((3, tif_size, tif_size), dtype=np.uint8)
    with rasterio.open(fname, "w", **profile) as src:
        src.write(array)


def create_json(fname: str) -> None:
    src_vertices = [
        (0, 0),
        (tif_size, 0),
        (tif_size, tif_size),
        (0, tif_size),
    ]
    src_projection = Projection(CRS.from_epsg(3857), 1, -1)
    dst_projection = WGS84_PROJECTION
    dst_vertices = []
    for vertex in src_vertices:
        src_geom = STGeometry(src_projection, shapely.Point(vertex[0], vertex[1]), None)
        dst_geom = src_geom.to_projection(dst_projection)
        dst_vertices.append(
            {
                "x": dst_geom.shp.x,
                "y": dst_geom.shp.y,
            }
        )

    json_data = {
        "mapping_area": [
            {
                "boundingPoly": [
                    {
                        "normalizedVertices": dst_vertices,
                    }
                ],
            }
        ],
        "annotations": [
            {
                "categories": [
                    {"name": "FM_1_3_INTERMITTENTLY_CLOSED_AND_OPEN_LAKES_AND_LAGOONS"}
                ],
                "boundingPoly": [
                    {
                        "normalizedVertices": dst_vertices,
                    }
                ],
            }
        ],
    }
    with open(fname, "w") as f:
        json.dump(json_data, f)


def create_islands_file(fname):
    json_data = {
        "type": "FeatureCollection",
        "crs": {
            "properties": {
                "name": "EPSG:3857",
            },
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "atoll": "x",
                    "islandName": "x",
                    "FCODE": "x",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0, 0],
                            [0, tif_size],
                            [tif_size, tif_size],
                            [tif_size, 0],
                        ]
                    ],
                },
            }
        ],
    }
    with open(fname, "w") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    create_tif("fake_2024-01-01-00-00.tif")
    create_json("fake_2024-01-01-00-00_labels.json")
    create_islands_file("islands.json")
