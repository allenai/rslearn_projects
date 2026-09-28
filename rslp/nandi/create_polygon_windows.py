"""Create grid windows covering the Nandi label polygons.

The old pipeline made one window per labelled *point*, centred on that point, which
produced ~19K near-duplicate windows that each supervised a single pixel and taught the
model that the answer always sits at the window centre.

This script instead lays a fixed grid over the label footprint and creates one window
per occupied cell. Every polygon intersecting a window is written to that window's
`label` layer, so a window carries every label it covers rather than just one. At a
16 px grid that is ~950 windows carrying ~24 labelled pixels each: the same supervision
as before from ~24x fewer forward passes, with the labels spread across the window
instead of pinned to the centre.

Splits are assigned by hashing a coarse spatial block rather than a polygon ID. Grid
windows can straddle a polygon, so a polygon-level hash would leak a polygon across
splits; blocks much larger than a window keep every window touching a polygon on the
same side of the split.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

import affine
import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
import shapely.strtree
import tqdm
from rasterio.features import rasterize
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.nandi.classes import CLASS_NAMES

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"

# The window time range is the centre month of the season. The Sentinel-2 layer in the
# dataset config expands it with time_offset=-180d and duration=366d to pull the 12
# monthly mosaics spanning roughly 2022-09 to 2023-09.
START_TIME = datetime(2023, 3, 1, tzinfo=timezone.utc)
END_TIME = datetime(2023, 3, 31, tzinfo=timezone.utc)


def assign_split(block_col: int, block_row: int) -> str:
    """Assign a train/val/test split to a spatial block.

    Args:
        block_col: the block column index.
        block_row: the block row index.

    Returns:
        one of "train", "val" or "test".
    """
    digest = hashlib.sha256(f"{block_col}_{block_row}".encode()).hexdigest()[0]
    if digest in ("0", "1"):
        return "val"
    if digest in ("2", "3"):
        return "test"
    return "train"


def load_label_polygons(
    labels_path: UPath, projection: Projection
) -> list[tuple[shapely.Polygon, dict]]:
    """Load the label GeoJSON and reproject it into window pixel coordinates.

    Args:
        labels_path: the unified label GeoJSON from prepare_label_polygons.py.
        projection: the target projection.

    Returns:
        a list of (polygon in projection coordinates, properties) tuples.
    """
    with labels_path.open() as f:
        feature_collection = json.load(f)

    polygons = []
    for feature in feature_collection["features"]:
        shp = shapely.geometry.shape(feature["geometry"])
        geometry = STGeometry(WGS84_PROJECTION, shp, None).to_projection(projection)
        if geometry.shp.is_empty:
            continue
        polygons.append((geometry.shp, feature["properties"]))
    return polygons


def compute_utm_projection(labels_path: UPath) -> Projection:
    """Pick a single UTM projection for the whole AOI.

    Nandi County sits inside one UTM zone, so using one projection for every window
    keeps the grid globally consistent -- which is what makes block-level splits and
    seamless inference tiling possible.

    Args:
        labels_path: the unified label GeoJSON.

    Returns:
        the UTM projection at the window resolution.
    """
    with labels_path.open() as f:
        feature_collection = json.load(f)
    centroid = shapely.union_all(
        [shapely.geometry.shape(f["geometry"]) for f in feature_collection["features"]]
    ).centroid
    crs = get_utm_ups_crs(centroid.x, centroid.y)
    return Projection(crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)


def _covers_any_pixel_center(features: list[Feature], bounds: tuple) -> bool:
    """Check whether any polygon covers at least one pixel centre in the window.

    Args:
        features: the clipped label features, in window projection coordinates.
        bounds: the window bounds.

    Returns:
        whether rasterizing these features would label at least one pixel.
    """
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    shapes = []
    for feature in features:
        geometry = shapely.affinity.translate(
            feature.geometry.shp, xoff=-bounds[0], yoff=-bounds[1]
        )
        if not geometry.is_empty:
            shapes.append((geometry, 1))
    if not shapes:
        return False
    burned = rasterize(
        shapes,
        out_shape=(height, width),
        transform=affine.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    return bool(burned.any())


def create_windows(
    labels_path: UPath,
    ds_path: UPath,
    group: str,
    window_size: int,
    grid_size: int,
    block_size_m: float,
) -> None:
    """Create one window per occupied grid cell and write its label polygons.

    Args:
        labels_path: the unified label GeoJSON.
        ds_path: the rslearn dataset to create windows in.
        group: the window group name.
        window_size: the window edge length in pixels.
        grid_size: the grid step in pixels. Cells are expanded symmetrically to
            window_size, so window_size > grid_size gives every random crop of the
            window a guaranteed overlap with the labelled cell.
        block_size_m: the spatial block edge length, in metres, used for splitting.
    """
    projection = compute_utm_projection(labels_path)
    print(f"Window projection: {projection.crs.to_string()} @ {WINDOW_RESOLUTION} m")

    polygons = load_label_polygons(labels_path, projection)
    print(f"Loaded {len(polygons)} label polygons")

    tree = shapely.strtree.STRtree([polygon for polygon, _ in polygons])

    # Collect the grid cells any polygon touches.
    cells: set[tuple[int, int]] = set()
    for polygon, _ in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        for col in range(int(min_x // grid_size), int(max_x // grid_size) + 1):
            for row in range(int(min_y // grid_size), int(max_y // grid_size) + 1):
                cells.add((col, row))
    print(f"{len(cells)} occupied grid cells at grid_size={grid_size}")

    dataset = Dataset(ds_path)
    pad = (window_size - grid_size) // 2
    block_size_px = block_size_m / WINDOW_RESOLUTION

    split_counts: Counter[str] = Counter()
    class_pixel_counts: Counter[str] = Counter()
    empty_windows = 0

    for col, row in tqdm.tqdm(sorted(cells), desc="creating windows"):
        bounds = (
            col * grid_size - pad,
            row * grid_size - pad,
            (col + 1) * grid_size + pad,
            (row + 1) * grid_size + pad,
        )
        window_box = shapely.box(*bounds)
        # Buffer the clip box so a polygon edge landing exactly on the window border
        # cannot drop a pixel whose centre is inside the window.
        clip_box = shapely.box(
            bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2
        )

        features = []
        categories: Counter[str] = Counter()
        for idx in tree.query(window_box):
            polygon, properties = polygons[idx]
            clipped = polygon.intersection(clip_box)
            if clipped.is_empty:
                continue
            if not clipped.is_valid:
                clipped = shapely.make_valid(clipped)
            if clipped.is_empty:
                continue
            features.append(
                Feature(
                    STGeometry(projection, clipped, (START_TIME, END_TIME)),
                    dict(properties),
                )
            )
            categories[properties["category"]] += 1
            class_pixel_counts[properties["category"]] += int(
                polygon.intersection(window_box).area
            )

        if not features:
            empty_windows += 1
            continue

        # A polygon can touch a grid cell with its bounding box while covering none of
        # that cell's pixel centres -- the Nandi polygons are small (median ~7 px
        # across), so this happens for about a quarter of the cells a naive bbox sweep
        # produces. Those windows would rasterize to all-ignore and contribute no
        # gradient, so skip them rather than pay a forward pass for nothing.
        if not _covers_any_pixel_center(features, bounds):
            empty_windows += 1
            continue

        split = assign_split(
            int(bounds[0] // block_size_px), int(bounds[1] // block_size_px)
        )
        split_counts[split] += 1

        window = Window(
            storage=dataset.storage,
            group=group,
            name=f"{bounds[0]}_{bounds[1]}",
            projection=projection,
            bounds=bounds,
            time_range=(START_TIME, END_TIME),
            options={
                "split": split,
                "block": f"{int(bounds[0] // block_size_px)}_{int(bounds[1] // block_size_px)}",
                "num_polygons": len(features),
                "categories": sorted(categories),
            },
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()

        with window.data.open_layer_writer(LABEL_LAYER) as writer:
            writer.write_vector(GeojsonVectorFormat(), features)
        window.mark_layer_completed(LABEL_LAYER)

    print(f"\nsplits: {dict(split_counts)}  (skipped {empty_windows} empty cells)")
    print(f"\n{'class':<12} {'~pixels':>9}")
    print("-" * 22)
    for name in CLASS_NAMES:
        print(f"{name:<12} {class_pixel_counts.get(name, 0):>9}")


def main() -> None:
    """Create the polygon grid windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Unified label GeoJSON from prepare_label_polygons.py",
    )
    parser.add_argument("--ds_path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--group", type=str, default="polygon_grid", help="Window group name"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Window edge length in pixels. 16 matches the linear probe setup "
        "(patch_size 1, window_size 16), where window == crop so embeddings cache",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid step in pixels (defaults to window_size). Set smaller than "
        "window_size to leave room for random cropping during fine-tuning",
    )
    parser.add_argument(
        "--block_size_m",
        type=float,
        default=2000.0,
        help="Spatial block edge length in metres used to assign train/val/test",
    )
    args = parser.parse_args()

    create_windows(
        UPath(args.labels_path),
        UPath(args.ds_path),
        args.group,
        args.window_size,
        args.grid_size if args.grid_size is not None else args.window_size,
        args.block_size_m,
    )


if __name__ == "__main__":
    main()
