"""Get Sentinel-2 scene IDs that we should run vessel detection model on."""

import argparse
import json
from datetime import datetime

import shapely
from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2
from rslearn.utils import STGeometry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get Sentinel-2 scene IDs matching an AOI and time range",
    )
    parser.add_argument(
        "--sx",
        type=float,
        help="Start longitude",
        required=True,
    )
    parser.add_argument(
        "--sy",
        type=float,
        help="Start latitude",
        required=True,
    )
    parser.add_argument(
        "--ex",
        type=float,
        help="End longitude",
        required=True,
    )
    parser.add_argument(
        "--ey",
        type=float,
        help="End latitude",
        required=True,
    )
    parser.add_argument(
        "--start_time",
        type=str,
        help="Start time",
        required=True,
    )
    parser.add_argument(
        "--end_time",
        type=str,
        help="End time",
        required=True,
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
    sentinel2 = Sentinel2(index_cache_dir=args.cache_path, use_rtree_index=False)

    shp = shapely.box(args.sx, args.sy, args.ex, args.ey)
    projection = WGS84_PROJECTION
    geom = STGeometry(
        projection,
        shp,
        (
            datetime.fromisoformat(args.start_time),
            datetime.fromisoformat(args.end_time),
        ),
    )
    item_groups = sentinel2.get_items([geom], query_config)[0]
    scene_ids = []
    for group in item_groups:
        assert len(group) == 1
        scene_ids.append(group[0].name)
    with open(args.out_fname, "w") as f:
        json.dump(scene_ids, f)
