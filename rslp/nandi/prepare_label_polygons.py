"""Build one unified label-polygon GeoJSON for Nandi crop type mapping.

The old pipeline exploded every polygon into ~19K 10 m points and created one window
per point. This script keeps the polygons intact and merges every label source into a
single GeoJSON, which `create_polygon_windows.py` then turns into windows.

Sources:

1. The CGIAR ground truth polygons (shapefile).
2. Studio annotation exports (GeoJSON), used to top up under-represented classes.
3. ESA WorldCover blobs, used for classes the field survey never collected. Shrubland
   is the important one: it is 27.6% of Nandi County and had no class at all, so every
   shrubland pixel was previously forced into Grassland, Trees or a crop class.

WorldCover blobs are eroded, size-filtered, and excluded near ground truth, so the weak
labels never sit next to a surveyed polygon where they could contradict it.
"""

import argparse
import json
import math
import random
from typing import Any

import fiona
import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
from pyproj import CRS, Transformer
from rslearn.utils import get_utm_ups_crs
from upath import UPath

from rslp.nandi.classes import (
    CLASS_NAMES,
    SOURCE_PRIORITY,
    WORLDCOVER_CLASS_MAP,
    WORLDCOVER_EXTRACTION_OVERRIDES,
    normalize_category,
)

# WorldCover pixels are ~9.3 m in Nandi; we treat the grid as 10 m throughout.
PIXEL_AREA_M2 = 100.0


def _category_from_properties(properties: dict[str, Any]) -> str | None:
    """Pull a category out of a GeoJSON feature's properties.

    Handles both plain `category`/`Category` properties and the Studio export format,
    where the label lives in a `metadata_values` entry named `tag_name`.

    Args:
        properties: the feature properties.

    Returns:
        the normalized class name, or None if there is none we model.
    """
    for key in ("category", "Category", "class", "label"):
        if properties.get(key):
            name = normalize_category(str(properties[key]))
            if name is not None:
                return name

    for entry in properties.get("metadata_values", []) or []:
        if entry.get("name") == "tag_name" and entry.get("tag_name"):
            name = normalize_category(str(entry["tag_name"]))
            if name is not None:
                return name

    return None


def _iter_polygons(geometry: Any) -> list[shapely.Polygon]:
    """Flatten a geometry into its constituent polygons.

    Args:
        geometry: a shapely geometry.

    Returns:
        the list of polygons, empty if the geometry holds none.
    """
    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    if isinstance(geometry, shapely.Polygon):
        return [geometry] if not geometry.is_empty else []
    if hasattr(geometry, "geoms"):
        polygons = []
        for part in geometry.geoms:
            polygons.extend(_iter_polygons(part))
        return polygons
    return []


def read_vector_source(
    path: UPath, source: str, id_field: str | None
) -> list[dict[str, Any]]:
    """Read labelled polygons from a shapefile or GeoJSON.

    Args:
        path: the vector file to read.
        source: the source tag to record on each output feature.
        id_field: optional property to use as the source ID.

    Returns:
        a list of records with `geometry` (WGS84 shapely), `category`, `source` and
            `source_id`.
    """
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    with fiona.open(str(path)) as src:
        src_crs = CRS.from_wkt(src.crs_wkt)
        to_wgs84 = None
        if src_crs != CRS.from_epsg(4326):
            to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

        for idx, feature in enumerate(src):
            properties = dict(feature.properties)
            category = _category_from_properties(properties)
            if category is None:
                raw = str(properties.get("Category") or properties.get("category"))
                skipped[raw] = skipped.get(raw, 0) + 1
                continue

            geometry = shapely.geometry.shape(feature.geometry)
            if to_wgs84 is not None:
                geometry = shapely.ops.transform(to_wgs84.transform, geometry)

            source_id = str(properties.get(id_field, idx)) if id_field else str(idx)
            for part_idx, polygon in enumerate(_iter_polygons(geometry)):
                records.append(
                    dict(
                        geometry=polygon,
                        category=category,
                        source=source,
                        source_id=f"{source_id}_{part_idx}",
                    )
                )

    if skipped:
        print(f"  {path.name}: skipped unmapped categories {skipped}")
    return records


def _erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a boolean mask with a 4-connected structuring element.

    Done in numpy so the pipeline does not need scipy. Eroding before polygonizing
    both drops the mixed pixels on every blob edge and collapses the speckle that
    would otherwise produce hundreds of thousands of tiny polygons.

    Args:
        mask: the boolean mask to erode.
        iterations: how many times to erode.

    Returns:
        the eroded mask.
    """
    for _ in range(iterations):
        out = mask.copy()
        out[1:, :] &= mask[:-1, :]
        out[:-1, :] &= mask[1:, :]
        out[:, 1:] &= mask[:, :-1]
        out[:, :-1] &= mask[:, 1:]
        out[0, :] = False
        out[-1, :] = False
        out[:, 0] = False
        out[:, -1] = False
        mask = out
    return mask


def _tile_polygon(polygon: shapely.Polygon, tile_m: float) -> list[shapely.Polygon]:
    """Cut a polygon into tile-aligned pieces.

    Used to enforce a pixel budget on a class without throwing away whole polygons.
    Because every cut runs through the polygon's interior, the resulting pieces contain
    only interior pixels -- subsampling this way never introduces mixed edge pixels.

    Args:
        polygon: the polygon to cut, in a projected (metre) CRS.
        tile_m: the tile edge length in metres.

    Returns:
        the list of polygon pieces.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    pieces: list[shapely.Polygon] = []
    x = math.floor(min_x / tile_m) * tile_m
    while x < max_x:
        y = math.floor(min_y / tile_m) * tile_m
        while y < max_y:
            pieces.extend(
                _iter_polygons(
                    polygon.intersection(shapely.box(x, y, x + tile_m, y + tile_m))
                )
            )
            y += tile_m
        x += tile_m
    return pieces


def enforce_pixel_budget(
    records: list[dict[str, Any]],
    utm_crs: CRS,
    budget_pixels: int,
    tile_m: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Cap the pixel count of each (class, source) group.

    Without this a handful of very large polygons dominate training: the 14 Studio
    forest polygons alone cover ~73K pixels, against ~3K for each surveyed crop class.

    Args:
        records: the label records to trim.
        utm_crs: the UTM CRS to measure areas in.
        budget_pixels: the maximum pixels to keep per (class, source) group.
        tile_m: tile size used to cut oversized polygons.
        rng: random source for sampling.

    Returns:
        the trimmed records.
    """
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["category"], record["source"]), []).append(record)

    budget_area = budget_pixels * PIXEL_AREA_M2
    kept: list[dict[str, Any]] = []

    for (category, source), group in groups.items():
        utm_geometries = [
            shapely.ops.transform(to_utm.transform, r["geometry"]) for r in group
        ]
        total = sum(g.area for g in utm_geometries)
        if total <= budget_area:
            kept.extend(group)
            continue

        pieces: list[tuple[str, int, shapely.Polygon]] = []
        for record, geometry in zip(group, utm_geometries):
            for piece_idx, piece in enumerate(_tile_polygon(geometry, tile_m)):
                pieces.append((record["source_id"], piece_idx, piece))
        rng.shuffle(pieces)

        taken = 0.0
        sampled = 0
        for source_id, piece_idx, piece in pieces:
            if taken >= budget_area:
                break
            kept.append(
                dict(
                    geometry=shapely.ops.transform(to_wgs84.transform, piece),
                    category=category,
                    source=source,
                    source_id=f"{source_id}_t{piece_idx}",
                )
            )
            taken += piece.area
            sampled += 1

        print(
            f"  {category}/{source}: {total / PIXEL_AREA_M2:.0f} px over budget, "
            f"kept {sampled} tiles covering ~{taken / PIXEL_AREA_M2:.0f} px"
        )

    return kept


def read_worldcover_blobs(
    tif_path: UPath,
    exclusion_geometry: shapely.geometry.base.BaseGeometry | None,
    utm_crs: CRS,
    min_blob_pixels: int,
    erosion_iterations: int,
) -> list[dict[str, Any]]:
    """Extract weak label polygons from an ESA WorldCover raster.

    Args:
        tif_path: the WorldCover GeoTIFF covering the AOI.
        exclusion_geometry: blobs intersecting this (already-buffered) geometry, in
            UTM, are dropped so weak labels never touch surveyed ground truth.
        utm_crs: the UTM CRS to measure areas and apply the exclusion in.
        min_blob_pixels: drop blobs smaller than this after erosion. Overridden
            per class by WORLDCOVER_EXTRACTION_OVERRIDES.
        erosion_iterations: how many pixels to erode off each blob edge. Overridden
            per class by WORLDCOVER_EXTRACTION_OVERRIDES.

    Returns:
        a list of records in the same shape as `read_vector_source`.
    """
    records: list[dict[str, Any]] = []

    with rasterio.open(str(tif_path)) as src:
        band = src.read(1)
        raster_crs = CRS.from_wkt(src.crs.to_wkt())
        transform = src.transform

    to_utm = Transformer.from_crs(raster_crs, utm_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    for code, category in sorted(WORLDCOVER_CLASS_MAP.items()):
        overrides = WORLDCOVER_EXTRACTION_OVERRIDES.get(code, {})
        class_erosion = overrides.get("erosion_iterations", erosion_iterations)
        class_min_blob = overrides.get("min_blob_pixels", min_blob_pixels)
        mask = _erode_mask(band == code, class_erosion)
        if not mask.any():
            print(f"  WorldCover {code} ({category}): nothing left after erosion")
            continue

        kept = 0
        total_area = 0.0
        for geom_dict, _ in rasterio.features.shapes(
            mask.astype(np.uint8), mask=mask, transform=transform
        ):
            polygon = shapely.geometry.shape(geom_dict)
            utm_polygon = shapely.ops.transform(to_utm.transform, polygon)
            if utm_polygon.area < class_min_blob * PIXEL_AREA_M2:
                continue
            if exclusion_geometry is not None and utm_polygon.intersects(
                exclusion_geometry
            ):
                continue
            records.append(
                dict(
                    geometry=shapely.ops.transform(to_wgs84.transform, polygon),
                    category=category,
                    source="worldcover",
                    source_id=f"wc{code}_{kept}",
                )
            )
            kept += 1
            total_area += utm_polygon.area

        print(
            f"  WorldCover {code} ({category}): {kept} eligible blobs covering "
            f"~{total_area / PIXEL_AREA_M2:.0f} px before budgeting"
        )

    return records


def summarize(records: list[dict[str, Any]], utm_crs: CRS) -> None:
    """Print a per-class polygon and pixel count summary.

    Args:
        records: the label records.
        utm_crs: the UTM CRS to measure pixel areas in.
    """
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    stats: dict[tuple[str, str], list[float]] = {}
    for record in records:
        area = shapely.ops.transform(to_utm.transform, record["geometry"]).area
        stats.setdefault((record["category"], record["source"]), []).append(area)

    print(f"\n{'class':<12} {'source':<12} {'polygons':>9} {'~pixels':>9}")
    print("-" * 45)
    for name in CLASS_NAMES:
        for source in sorted(SOURCE_PRIORITY, key=lambda s: -SOURCE_PRIORITY[s]):
            areas = stats.get((name, source))
            if not areas:
                continue
            print(
                f"{name:<12} {source:<12} {len(areas):>9} "
                f"{sum(areas) / PIXEL_AREA_M2:>9.0f}"
            )


def main() -> None:
    """Build the unified label GeoJSON from all configured sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt_shapefile",
        type=str,
        required=True,
        help="CGIAR ground truth polygon shapefile",
    )
    parser.add_argument(
        "--studio_geojson",
        type=str,
        action="append",
        default=[],
        help="Studio annotation GeoJSON export (repeatable)",
    )
    parser.add_argument(
        "--worldcover_tif",
        type=str,
        default=None,
        help="ESA WorldCover GeoTIFF covering the AOI",
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Output GeoJSON path"
    )
    parser.add_argument(
        "--gt_id_field",
        type=str,
        default="unique_id",
        help="Property on the ground truth to use as the polygon ID",
    )
    parser.add_argument(
        "--max_pixels_per_class_source",
        type=int,
        default=6000,
        help="Pixel budget per (class, source) group. The default sits above every "
        "surveyed crop class (max ~4.8K px) so field data is never discarded, while "
        "capping the large Studio forest and WorldCover polygons",
    )
    parser.add_argument(
        "--budget_tile_m",
        type=float,
        default=160.0,
        help="Tile size used to cut oversized polygons down to the pixel budget",
    )
    parser.add_argument(
        "--worldcover_min_blob_pixels",
        type=int,
        default=16,
        help="Drop WorldCover blobs smaller than this after erosion",
    )
    parser.add_argument(
        "--worldcover_erosion_iterations",
        type=int,
        default=2,
        help="Pixels to erode off each WorldCover blob edge",
    )
    parser.add_argument(
        "--worldcover_exclusion_m",
        type=float,
        default=100.0,
        help="Drop WorldCover blobs within this distance of any surveyed polygon",
    )
    parser.add_argument("--seed", type=int, default=42, help="Blob sampling seed")
    args = parser.parse_args()

    print("Reading ground truth polygons...")
    records = read_vector_source(
        UPath(args.gt_shapefile), "groundtruth", args.gt_id_field
    )
    print(f"  {len(records)} polygons")

    for path in args.studio_geojson:
        print(f"Reading Studio annotations from {path}...")
        studio = read_vector_source(UPath(path), "studio", "id")
        print(f"  {len(studio)} polygons")
        records.extend(studio)

    centroid = shapely.union_all([r["geometry"] for r in records]).centroid
    utm_crs = CRS.from_wkt(get_utm_ups_crs(centroid.x, centroid.y).to_wkt())
    print(f"AOI UTM CRS: {utm_crs.to_string()}")

    if args.worldcover_tif:
        print("Extracting WorldCover blobs...")
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        exclusion = shapely.union_all(
            [shapely.ops.transform(to_utm.transform, r["geometry"]) for r in records]
        ).buffer(args.worldcover_exclusion_m)
        records.extend(
            read_worldcover_blobs(
                UPath(args.worldcover_tif),
                exclusion,
                utm_crs,
                args.worldcover_min_blob_pixels,
                args.worldcover_erosion_iterations,
            )
        )

    print("Enforcing per-class pixel budget...")
    records = enforce_pixel_budget(
        records,
        utm_crs,
        args.max_pixels_per_class_source,
        args.budget_tile_m,
        random.Random(args.seed),
    )

    summarize(records, utm_crs)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "category": record["category"],
                    "source": record["source"],
                    "source_id": record["source_id"],
                    "priority": SOURCE_PRIORITY[record["source"]],
                },
                "geometry": json.loads(shapely.to_geojson(record["geometry"])),
            }
            for record in records
        ],
    }

    out_path = UPath(args.out_path)
    with out_path.open("w") as f:
        json.dump(feature_collection, f)
    print(f"\nWrote {len(records)} polygons to {out_path}")


if __name__ == "__main__":
    main()
