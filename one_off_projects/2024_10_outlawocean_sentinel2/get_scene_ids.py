"""Get Sentinel-2 scene IDs that we should run vessel detection model on."""

import argparse
import json
from datetime import datetime, timezone

import shapely
from rslearn.config import LayerType, QueryConfig, RasterLayerConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.utils import STGeometry
from upath import UPath

AOI = [
    [59.9913, -11.25],
    [61.8333, -11.25],
    [61.8333, -8.3333],
    [59.9913, -8.3333],
    [59.9913, -11.25],
]


if __name__ == "__main__":
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
        "--out_fname",
        type=str,
        help="JSON file to write the scene IDs",
        required=True,
    )
    args = parser.parse_args()

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=100000)
    layer_config = RasterLayerConfig(LayerType.RASTER, [])
    sentinel2 = Sentinel2(
        layer_config, index_cache_dir=UPath(args.cache_path), use_rtree_index=False
    )

    shp = shapely.Polygon(AOI)
    projection = WGS84_PROJECTION
    geom = STGeometry(
        projection,
        shp,
        (
            datetime(2015, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
        ),
    )
    item_groups = sentinel2.get_items([geom], query_config)[0]
    scene_ids = []
    for group in item_groups:
        assert len(group) == 1
        scene_ids.append(group[0].name)
    with open(args.out_fname, "w") as f:
        json.dump(scene_ids, f)
