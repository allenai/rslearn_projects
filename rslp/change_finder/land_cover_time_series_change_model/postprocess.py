"""Convert prediction rasters to GeoJSON change polygons.

Reads the ``output_change`` layer from each prediction window, thresholds
the binary-change probability band, extracts connected components, and
writes a GeoJSON FeatureCollection with one polygon per component.

Usage::

    python -m rslp.change_finder.land_cover_time_series_change_model.postprocess \
        --dataset_path /path/to/predict_dataset \
        --output geojson_out.geojson \
        --threshold 128
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.raster_format import (
    get_bandset_dirname,
    get_raster_projection_and_bounds,
)
from scipy import ndimage
from upath import UPath

BINARY_CHANGE_BAND = 2  # band index for "change" probability
SRC_BAND_OFFSET = 3  # bands 3..15
DST_BAND_OFFSET = 16  # bands 16..28
NUM_LC_CLASSES = 13

LC_CLASS_NAMES = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]

OUTPUT_LAYER = "output_change"
OUTPUT_BANDS = [
    "binary_nodata",
    "binary_no_change",
    "binary_change",
    "src_nodata",
    "src_bare",
    "src_burnt",
    "src_crops",
    "src_fallow",
    "src_grassland",
    "src_lichen",
    "src_shrub",
    "src_snow",
    "src_tree",
    "src_urban",
    "src_water",
    "src_wetland",
    "dst_nodata",
    "dst_bare",
    "dst_burnt",
    "dst_crops",
    "dst_fallow",
    "dst_grassland",
    "dst_lichen",
    "dst_shrub",
    "dst_snow",
    "dst_tree",
    "dst_urban",
    "dst_water",
    "dst_wetland",
]


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output_change geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _dominant_class(probs: np.ndarray, mask: np.ndarray) -> tuple[int, str]:
    """Return the (index, name) of the class with the highest average prob over mask pixels.

    Args:
        probs: (NUM_LC_CLASSES, H, W) uint8 probability array.
        mask: (H, W) bool array selecting the component pixels.

    Skips class 0 (nodata).
    """
    mean_probs = probs[1:, mask].mean(axis=1)
    idx = int(mean_probs.argmax()) + 1  # +1 to skip nodata
    return idx, LC_CLASS_NAMES[idx]


def process_window(
    window_dir: UPath,
    threshold: int,
) -> list[dict]:
    """Process one prediction window and return GeoJSON-ready feature dicts."""
    tif_path = _get_geotiff_path(window_dir)
    if tif_path is None:
        return []

    with rasterio.open(tif_path) as src:
        arr = src.read()
        projection, bounds = get_raster_projection_and_bounds(src)

    col0, row0 = bounds[0], bounds[1]

    change_score = arr[BINARY_CHANGE_BAND]
    binary_mask = change_score >= threshold

    if not binary_mask.any():
        return []

    labels, num_components = ndimage.label(binary_mask)
    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES]

    features = []
    for comp_id in range(1, num_components + 1):
        comp_mask = labels == comp_id
        num_pixels = int(comp_mask.sum())
        avg_score = float(change_score[comp_mask].mean())

        src_idx, src_name = _dominant_class(src_probs, comp_mask)
        dst_idx, dst_name = _dominant_class(dst_probs, comp_mask)

        # Vectorize the component mask into polygon(s) in pixel coordinates.
        shapes = list(
            rasterio.features.shapes(
                comp_mask.astype(np.uint8),
                mask=comp_mask,
                connectivity=8,
            )
        )
        if not shapes:
            continue

        # Merge all sub-polygons and shift to dataset pixel coords.
        polys = []
        for geom, _ in shapes:
            shp = shapely.geometry.shape(geom)
            shp = shapely.affinity.translate(shp, xoff=col0, yoff=row0)
            polys.append(shp)

        merged = shapely.ops.unary_union(polys)

        # Convert to WGS84 lon/lat.
        geom_proj = STGeometry(projection, merged, None)
        geom_wgs84 = geom_proj.to_projection(WGS84_PROJECTION)

        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom_wgs84.shp),
                "properties": {
                    "num_pixels": num_pixels,
                    "avg_score": round(avg_score, 2),
                    "src_class": src_name,
                    "src_class_idx": src_idx,
                    "dst_class": dst_name,
                    "dst_class_idx": dst_idx,
                },
            }
        )

    return features


def main() -> None:
    """Main entrypoint for land cover change postprocess script."""
    parser = argparse.ArgumentParser(
        description="Convert prediction rasters to GeoJSON change polygons."
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Root of the prediction dataset."
    )
    parser.add_argument("--output", required=True, help="Output GeoJSON file path.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binary change probability threshold (0-255).",
    )
    args = parser.parse_args()

    ds_root = UPath(args.dataset_path)
    predict_dir = ds_root / "windows" / "predict"

    all_features: list[dict] = []
    for window_dir in sorted(predict_dir.iterdir()):
        if not window_dir.is_dir():
            continue
        feats = process_window(window_dir, args.threshold)
        print(f"{window_dir.name}: {len(feats)} components")
        all_features.extend(feats)

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    out_path = UPath(args.output)
    with out_path.open("w") as f:
        json.dump(geojson, f)

    print(f"Wrote {len(all_features)} features to {args.output}")


if __name__ == "__main__":
    main()
