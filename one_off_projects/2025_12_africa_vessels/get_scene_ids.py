"""Get Sentinel-2 scene IDs that we should run vessel detection model on."""

import argparse
import json
import multiprocessing
from datetime import datetime, timezone, UTC

import shapely
import tqdm
from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2, Sentinel2Item
from rslearn.utils.geometry import STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat, GeojsonCoordinateMode
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from upath import UPath


def split_aoi(aoi: STGeometry, size: int = 1) -> list[STGeometry]:
    """Split up a big AOI into smaller geometries.

    Args:
        aoi: the AOI to split up.
        size: the size for sub-tiles to create within the bounds of the AOI.

    Returns:
        list of sub-tiles.
    """
    # We assume the tile has integer lon/lat coordinates and are 5x5 degrees.
    bounds = tuple(int(v) for v in aoi.shp.bounds)
    assert (bounds[2] - bounds[0]) == 5
    assert (bounds[3] - bounds[1]) == 5

    # Only size=1 really makes sense here since 5 has no larger factors besides 5.
    num_x_tiles = (bounds[2] - bounds[0]) // size
    num_y_tiles = (bounds[3] - bounds[1]) // size

    geoms: list[STGeometry] = []
    for col in range(num_x_tiles):
        for row in range(num_y_tiles):
            x_start = bounds[0] + col
            y_start = bounds[1] + row
            geom = STGeometry(WGS84_PROJECTION, shapely.box(x_start, y_start, x_start + size, y_start + size), aoi.time_range)
            geoms.append(geom)

    return geoms


def get_items(geom: STGeometry, cache_path: UPath) -> list[Sentinel2Item]:
    """Get the items matching the given geometry."""
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=100000)
    sentinel2 = Sentinel2(
        index_cache_dir=cache_path, use_rtree_index=False, use_bigquery=True
    )
    item_groups = sentinel2.get_items([geom], query_config)[0]
    items = []
    for group in item_groups:
        if len(group) != 1:
            raise ValueError("expected each item group to have one item with INTERSECTS space mode")
        items.append(group[0])
    return items


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Get Sentinel-2 scene IDs",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to cache stuff",
        required=True,
    )
    parser.add_argument(
        "--geojson",
        type=str,
        help="GeoJSON filename containing the area of interest",
        required=True,
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        help="Filename to write scene IDs",
        required=True,
    )
    parser.add_argument(
        "--geom_fname",
        type=str,
        help="Filename to write scene geometries",
        default=None,
    )
    args = parser.parse_args()

    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    features = vector_format.decode_from_file(UPath(args.geojson))
    assert len(features) == 1
    feat = features[0]

    geom = STGeometry(
        feat.geometry.projection,
        feat.geometry.shp,
        (
            datetime(2016, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )

    # Split up the AOI.
    geoms = split_aoi(geom)
    print(f"Got {len(geoms)} sub-tiles")

    # Process the AOIs in parallel.
    scene_ids = set()
    features: list[Feature] = []
    p = multiprocessing.Pool(64)
    outputs = star_imap_unordered(p, get_items, [dict(
        geom=geom,
        cache_path=UPath(args.cache_path)
    ) for geom in geoms])
    for item_list in tqdm.tqdm(outputs, total=len(geoms)):
        for item in item_list:
            if item.name in scene_ids:
                continue
            scene_ids.add(item.name)
            feat = Feature(item.geometry, {
                "scene_id": item.name,
            })
            features.append(feat)

    print(f"Got {len(scene_ids)} scene IDs after de-duplication")

    with open(args.out_fname, "w") as f:
        json.dump(list(scene_ids), f)

    if args.geom_fname:
        vector_format.encode_to_file(UPath(args.geom_fname), features)
