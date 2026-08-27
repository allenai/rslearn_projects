"""Convert v2 LCC prediction rasters to GeoJSON change polygons.

Polygonization operates on the compact uint8 summary representation (see
SUMMARY_BANDS / summary_window_array):

1. A pixel is a change pixel if the argmax pre or post change category is a
   real category (not "none"; the pre head is the merged pre+same head).
2. Vectorize connected components of pixels sharing the same (pre category,
   post category) combination to WGS-84 polygons, using a single
   rasterio.features.shapes pass over the raster.
3. Per polygon, sample the source/destination land cover classes, the change
   start/end months, and the scores at a representative interior point.

Two input modes:

- ``--dataset_path``: read the full ``output_change`` layer from each
  prediction window of an rslearn dataset (converted to the summary
  representation on the fly).
- ``--summary_path``: read the merged ``*_summary.tif`` tiles written by the
  prediction pipeline (write_summary_raster).

Two output modes (at least one must be set):

- ``--out_dir``: write one GeoJSON per input raster from within each worker
  job (``<tile>_summary.geojson`` per summary tile, ``<window>.geojson`` per
  window), so features are not retained across jobs.
- ``--output``: additionally (or instead) merge all features into a single
  GeoJSON file; this retains every feature in memory until the end.

Usage::

    python -m rslp.change_finder_v2.lcc_model.postprocess \
        --dataset_path /path/to/predict_dataset \
        --out_dir geojson_out/

    python -m rslp.change_finder_v2.lcc_model.postprocess \
        --summary_path /path/to/tile_outputs \
        --output geojson_out.geojson
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from datetime import datetime, timezone

import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.geometry
import tqdm
from rslearn.utils.geometry import (
    WGS84_PROJECTION,
    PixelBounds,
    Projection,
    STGeometry,
)
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from rslearn.utils.raster_format import (
    get_bandset_dirname,
    get_raster_projection_and_bounds,
)
from upath import UPath

from .timestamp_encoding import TIMESTAMP_EPOCH

BINARY_CHANGE_BAND = 2
SRC_BAND_OFFSET = 3
DST_BAND_OFFSET = 16
# The timestamp section is two bands holding the predicted pre/post change dates
# encoded as integer days since TIMESTAMP_EPOCH (see timestamp_encoding.py).
TS_PRE_DAYS_BAND = 29
TS_POST_DAYS_BAND = 30
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

# Change-category output bands, in the same order as the model output and the
# output_change layer in config.json (class layout [nodata, none, <options...>]).
# The pre_change head is the merged pre+same head: the former same_change
# categories are appended after the pre categories (see
# SinglePassMultiTask merge_same_into_pre).
PRE_CHANGE_BANDS = [
    "pre_change_nodata",
    "pre_change_none",
    "pre_change_deforestation",
    "pre_change_urban_erosion",
    "pre_change_wetland_loss",
    "pre_change_water_contract",
    "pre_change_removed_crop_structure",
    "pre_change_agricultural_activity",
    "pre_change_wildfire",
    "pre_change_ice_motion",
    "pre_change_flooding",
]
POST_CHANGE_BANDS = [
    "post_change_nodata",
    "post_change_none",
    "post_change_vegetation_growth",
    "post_change_new_building",
    "post_change_new_road",
    "post_change_new_infrastructure",
    "post_change_new_crop_field",
    "post_change_new_aquafarm",
    "post_change_site_clearing",
    "post_change_water_expand",
    "post_change_mining",
    "post_change_new_crop_structure",
    "post_change_selective_logging",
    "post_change_landslide",
    "post_change_settlement",
]

# Band offsets of the two change-category sections within the output raster.
PRE_CHANGE_BAND_OFFSET = TS_POST_DAYS_BAND + 1
POST_CHANGE_BAND_OFFSET = PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)

# Plain category names (class layout [nodata, none, <options...>]).
PRE_CHANGE_CATEGORY_NAMES = [b.removeprefix("pre_change_") for b in PRE_CHANGE_BANDS]
POST_CHANGE_CATEGORY_NAMES = [b.removeprefix("post_change_") for b in POST_CHANGE_BANDS]

# Class index within the merged pre+same head where the former same_change
# categories start (they are appended after the original pre categories).
MERGED_SAME_CATEGORY_START = PRE_CHANGE_CATEGORY_NAMES.index("agricultural_activity")

OUTPUT_LAYER = "output_change"
OUTPUT_BANDS = [
    "binary_nodata",
    "binary_no_change",
    "binary_change",
    *(
        f"src_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    *(
        f"dst_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    "ts_pre_days",
    "ts_post_days",
    *PRE_CHANGE_BANDS,
    *POST_CHANGE_BANDS,
]

# Compact uint8 summary raster layout (see summary_window_array). The class
# bands hold the per-head argmax class index (0 = no prediction; argmax skips
# the nodata class), the score bands hold the probability (0-255) of the
# argmax class, and the month bands hold 0 for no prediction or 1 + whole
# calendar months since TIMESTAMP_EPOCH (Jan 2015; 255 reaches March 2036).
# This layout matches the olmoearth_lcc_viewer COGs exactly, so its
# make_cogs.py converts summary rasters without any band changes.
SUMMARY_BANDS = [
    "binary_change",
    "pre_class",
    "post_class",
    "src_class",
    "dst_class",
    "pre_score",
    "post_score",
    "ts_pre_month",
    "ts_post_month",
]

# Indices into SUMMARY_BANDS.
SUMMARY_BINARY_CHANGE_BAND = SUMMARY_BANDS.index("binary_change")
SUMMARY_PRE_CLASS_BAND = SUMMARY_BANDS.index("pre_class")
SUMMARY_POST_CLASS_BAND = SUMMARY_BANDS.index("post_class")
SUMMARY_SRC_CLASS_BAND = SUMMARY_BANDS.index("src_class")
SUMMARY_DST_CLASS_BAND = SUMMARY_BANDS.index("dst_class")
SUMMARY_PRE_SCORE_BAND = SUMMARY_BANDS.index("pre_score")
SUMMARY_POST_SCORE_BAND = SUMMARY_BANDS.index("post_score")
SUMMARY_TS_PRE_MONTH_BAND = SUMMARY_BANDS.index("ts_pre_month")
SUMMARY_TS_POST_MONTH_BAND = SUMMARY_BANDS.index("ts_post_month")

# Class index of the "none" category in both category heads (index 0 is
# nodata, which the summary argmax skips; 0 in the summary class bands instead
# means no prediction).
NONE_CATEGORY_IDX = 1


def _days_to_month_values(days: np.ndarray) -> np.ndarray:
    """Convert day-since-epoch values to the summary month encoding.

    The result is 1 + whole calendar months between TIMESTAMP_EPOCH and the
    day's date, clipped to the uint8 range.
    """
    epoch_day = np.datetime64(TIMESTAMP_EPOCH.date().isoformat())
    dates = epoch_day + days.astype("timedelta64[D]")
    months = (dates.astype("datetime64[M]") - epoch_day.astype("datetime64[M]")).astype(
        np.int64
    ) + 1
    return np.clip(months, 1, 255).astype(np.uint8)


def _month_value_to_date(month_value: int) -> datetime:
    """Decode a summary month value back to the first day of that month (UTC).

    Inverse of _days_to_month_values: month_value is 1 + whole calendar months
    since TIMESTAMP_EPOCH.
    """
    months_since_epoch = int(month_value) - 1
    year = TIMESTAMP_EPOCH.year + months_since_epoch // 12
    month = 1 + months_since_epoch % 12
    return datetime(year, month, 1, tzinfo=timezone.utc)


def summary_window_array(arr: np.ndarray) -> np.ndarray:
    """Derive the uint8 summary bands from a full output window array.

    Args:
        arr: (len(OUTPUT_BANDS), H, W) array with probabilities scaled 0-255.

    Returns:
        (len(SUMMARY_BANDS), H, W) uint8 array.
    """
    # Pixels with no prediction were left all-zero; use the binary head to
    # detect them.
    valid = arr[0 : BINARY_CHANGE_BAND + 1].max(axis=0) > 0

    head_probs = {
        "pre": arr[
            PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)
        ],
        "post": arr[
            POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET + len(POST_CHANGE_BANDS)
        ],
        "src": arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES],
        "dst": arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES],
    }
    classes: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for head, probs in head_probs.items():
        # Argmax excluding the nodata class (index 0); 0 is reserved for
        # pixels with no prediction.
        argmax = probs[1:].argmax(axis=0) + 1
        scores[head] = np.clip(
            np.take_along_axis(probs, argmax[None], axis=0)[0], 0, 255
        ).astype(np.uint8)
        classes[head] = np.where(valid, argmax, 0).astype(np.uint8)

    months = {}
    for head, band in [("pre", TS_PRE_DAYS_BAND), ("post", TS_POST_DAYS_BAND)]:
        months[head] = np.where(valid, _days_to_month_values(arr[band]), 0).astype(
            np.uint8
        )

    return np.stack(
        [
            np.clip(arr[BINARY_CHANGE_BAND], 0, 255).astype(np.uint8),
            classes["pre"],
            classes["post"],
            classes["src"],
            classes["dst"],
            scores["pre"],
            scores["post"],
            months["pre"],
            months["post"],
        ]
    )


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output_change geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _shape_to_feature(
    pixel_poly: shapely.geometry.base.BaseGeometry,
    summary: np.ndarray,
    pre_idx: int,
    post_idx: int,
    projection: Projection,
    bounds: PixelBounds,
) -> dict:
    """Build a GeoJSON feature dict for one vectorized connected component.

    Per-polygon properties are sampled at a representative interior point of
    the polygon (guaranteed to be inside it, unlike the centroid), so they
    reflect one change pixel of the component rather than an aggregate.

    Args:
        pixel_poly: the component's polygon in raster pixel coordinates
            (before translating by the bounds offset).
        summary: (len(SUMMARY_BANDS), H, W) uint8 summary array.
        pre_idx: the component's pre change category class index.
        post_idx: the component's post change category class index.
        projection: the projection of the raster.
        bounds: the pixel bounds of the raster within the projection.
    """
    # The polygon follows pixel edges, so its area is its pixel count (holes
    # are already excluded).
    num_pixels = int(round(pixel_poly.area))
    col0, row0 = bounds[0], bounds[1]

    # Area from the projection's per-pixel resolution (10 m pixels = 0.01 ha).
    pixel_area_hectares = abs(projection.x_resolution * projection.y_resolution) / 10000
    area_hectares = round(num_pixels * pixel_area_hectares, 2)

    # Sample the summary bands at a representative interior point.
    point = pixel_poly.representative_point()
    height, width = summary.shape[1], summary.shape[2]
    row = min(max(int(point.y), 0), height - 1)
    col = min(max(int(point.x), 0), width - 1)
    pixel = summary[:, row, col]

    binary_change_score = float(pixel[SUMMARY_BINARY_CHANGE_BAND])

    # Average the argmax score of each head that predicts a real category
    # (usually one head; sometimes both, like deforestation+mining).
    head_scores = []
    if pre_idx != NONE_CATEGORY_IDX:
        head_scores.append(float(pixel[SUMMARY_PRE_SCORE_BAND]))
    if post_idx != NONE_CATEGORY_IDX:
        head_scores.append(float(pixel[SUMMARY_POST_SCORE_BAND]))
    category_change_score = float(np.mean(head_scores))

    src_id = int(pixel[SUMMARY_SRC_CLASS_BAND])
    dst_id = int(pixel[SUMMARY_DST_CLASS_BAND])

    # Change start/end months, decoded to the first day of the month.
    pre_month = int(pixel[SUMMARY_TS_PRE_MONTH_BAND])
    post_month = int(pixel[SUMMARY_TS_POST_MONTH_BAND])

    shp = shapely.affinity.translate(pixel_poly, xoff=col0, yoff=row0)
    geom_wgs84 = STGeometry(projection, shp, None).to_projection(WGS84_PROJECTION)

    props: dict = {
        "num_pixels": num_pixels,
        "area_hectares": area_hectares,
        "binary_change_score": round(binary_change_score, 2),
        "category_change_score": round(category_change_score, 2),
        # The component's defining change categories. The pre head is the
        # merged pre+same head, so pre_change_category may also be one of the
        # former same_change categories.
        "pre_change_category": PRE_CHANGE_CATEGORY_NAMES[pre_idx],
        "post_change_category": POST_CHANGE_CATEGORY_NAMES[post_idx],
        "src_class": LC_CLASS_NAMES[src_id],
        "src_class_idx": src_id,
        "dst_class": LC_CLASS_NAMES[dst_id],
        "dst_class_idx": dst_id,
        "pre_change_date": _month_value_to_date(pre_month).isoformat(),
        "post_change_date": _month_value_to_date(post_month).isoformat(),
    }

    return {
        "type": "Feature",
        "geometry": shapely.geometry.mapping(geom_wgs84.shp),
        "properties": props,
    }


def features_from_summary(
    summary: np.ndarray,
    projection: Projection,
    bounds: PixelBounds,
    min_pixels: int,
) -> list[dict]:
    """Polygonize a summary array into GeoJSON feature dicts (WGS84).

    Change pixels are those where at least one category head predicts a real
    category (not "none"). Connected components group change pixels sharing
    the same (pre category, post category) combination; they are all
    vectorized in a single rasterio.features.shapes pass over the raster.

    Args:
        summary: (len(SUMMARY_BANDS), H, W) uint8 summary array (see
            summary_window_array).
        projection: the projection of the raster.
        bounds: the pixel bounds of the raster within the projection.
        min_pixels: minimum connected-component size to keep.

    Returns:
        list of GeoJSON feature dicts with WGS84 geometries.
    """
    pre_class = summary[SUMMARY_PRE_CLASS_BAND]
    post_class = summary[SUMMARY_POST_CLASS_BAND]

    # Change pixels: a valid prediction (class 0 means no prediction) where at
    # least one head predicts a real category.
    change_mask = (pre_class > NONE_CATEGORY_IDX) | (post_class > NONE_CATEGORY_IDX)
    change_mask &= (pre_class > 0) & (post_class > 0)
    if not change_mask.any():
        return []

    # Encode each (pre, post) category combination as a unique integer.
    # shapes() then yields one polygon per connected region of equal value
    # within the mask, i.e. one polygon per component.
    combo_labels = pre_class.astype(np.uint16) * len(
        POST_CHANGE_CATEGORY_NAMES
    ) + post_class.astype(np.uint16)

    features: list[dict] = []
    for geom, combo_val in rasterio.features.shapes(combo_labels, mask=change_mask):
        pixel_poly = shapely.geometry.shape(geom)
        if pixel_poly.area < min_pixels:
            continue

        pre_idx = int(combo_val) // len(POST_CHANGE_CATEGORY_NAMES)
        post_idx = int(combo_val) % len(POST_CHANGE_CATEGORY_NAMES)
        features.append(
            _shape_to_feature(
                pixel_poly,
                summary,
                pre_idx,
                post_idx,
                projection,
                bounds,
            )
        )

    return features


def _write_feature_collection(features: list[dict], out_path: UPath) -> None:
    """Write a GeoJSON FeatureCollection to out_path."""
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    with out_path.open("w") as f:
        json.dump(geojson, f)


def process_window(
    window_dir: UPath,
    min_pixels: int,
    out_path: UPath | None = None,
    return_features: bool = True,
) -> tuple[int, list[dict]]:
    """Process one prediction window into GeoJSON features.

    If out_path is set, the features are written there as a FeatureCollection
    (windows with no output geotiff write nothing). Returns the feature count
    and the features themselves (empty list if return_features is False).
    """
    tif_path = _get_geotiff_path(window_dir)
    if tif_path is None:
        return 0, []

    with rasterio.open(tif_path) as src:
        arr = src.read()
        projection, bounds = get_raster_projection_and_bounds(src)

    features = features_from_summary(
        summary_window_array(arr), projection, bounds, min_pixels
    )
    if out_path is not None:
        _write_feature_collection(features, out_path)
    return len(features), features if return_features else []


def process_summary_tif(
    tif_path: UPath,
    min_pixels: int,
    out_path: UPath | None = None,
    return_features: bool = True,
) -> tuple[int, list[dict]]:
    """Polygonize one merged summary GeoTIFF into GeoJSON features.

    If out_path is set, the features are written there as a FeatureCollection.
    Returns the feature count and the features themselves (empty list if
    return_features is False).
    """
    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            summary = src.read()
            projection, bounds = get_raster_projection_and_bounds(src)

    if summary.shape[0] != len(SUMMARY_BANDS):
        raise ValueError(
            f"{tif_path} has {summary.shape[0]} bands, expected "
            f"{len(SUMMARY_BANDS)} summary bands"
        )

    features = features_from_summary(summary, projection, bounds, min_pixels)
    if out_path is not None:
        _write_feature_collection(features, out_path)
    return len(features), features if return_features else []


def collect_features(
    dataset_path: str,
    min_pixels: int = 10,
    workers: int = 32,
    out_dir: UPath | None = None,
    return_features: bool = True,
) -> tuple[int, list[dict]]:
    """Scan all predict windows and polygonize them to GeoJSON features (WGS84).

    If out_dir is set, each window's features are written to
    ``<out_dir>/<window_name>.geojson`` from within the worker job. Returns
    the total feature count and the merged features (empty list if
    return_features is False). The count is 0 if the predict group has no
    windows.
    """
    ds_root = UPath(dataset_path)
    predict_dir = ds_root / "windows" / "predict"

    if not predict_dir.exists():
        return 0, []

    kwargs_list = [
        dict(
            window_dir=window_dir,
            min_pixels=min_pixels,
            out_path=(
                out_dir / f"{window_dir.name}.geojson" if out_dir is not None else None
            ),
            return_features=return_features,
        )
        for window_dir in sorted(predict_dir.iterdir())
        if window_dir.is_dir()
    ]

    total_count = 0
    all_features: list[dict] = []
    with make_pool_and_star_imap_unordered(
        workers, process_window, kwargs_list
    ) as results:
        for num_features, features in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Processing"
        ):
            total_count += num_features
            all_features.extend(features)
    return total_count, all_features


def collect_features_from_summaries(
    summary_path: str,
    min_pixels: int = 10,
    workers: int = 4,
    out_dir: UPath | None = None,
    return_features: bool = True,
) -> tuple[int, list[dict]]:
    """Polygonize all ``*_summary.tif`` tiles under a directory.

    These are the merged summary rasters written by the prediction pipeline
    (write_summary_raster). Each tile is processed whole, so components are
    not split at window boundaries; a full 32768x32768 tile is ~10 GB in
    memory, so keep workers low.

    If out_dir is set, each tile's features are written to
    ``<out_dir>/<tile>_summary.geojson`` from within the worker job. Returns
    the total feature count and the merged features (empty list if
    return_features is False).
    """
    root = UPath(summary_path)
    tif_paths = sorted(root.glob("*_summary.tif"))
    if not tif_paths:
        raise ValueError(f"no *_summary.tif files found under {summary_path}")

    kwargs_list = [
        dict(
            tif_path=tif_path,
            min_pixels=min_pixels,
            out_path=(
                out_dir / tif_path.name.replace(".tif", ".geojson")
                if out_dir is not None
                else None
            ),
            return_features=return_features,
        )
        for tif_path in tif_paths
    ]

    total_count = 0
    all_features: list[dict] = []
    with make_pool_and_star_imap_unordered(
        workers, process_summary_tif, kwargs_list
    ) as results:
        for num_features, features in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Processing"
        ):
            total_count += num_features
            all_features.extend(features)
    return total_count, all_features


def create_geojson(
    output: str | None = None,
    out_dir: str | None = None,
    dataset_path: str | None = None,
    summary_path: str | None = None,
    min_pixels: int = 10,
    workers: int | None = None,
) -> None:
    """Polygonize predictions and write GeoJSON FeatureCollections.

    Exactly one of dataset_path (rslearn predict dataset) or summary_path
    (directory of merged ``*_summary.tif`` tiles) must be provided.

    At least one of out_dir (one GeoJSON per input raster, written from
    within each worker job) or output (a single merged GeoJSON, which
    requires retaining all features in memory) must be provided.
    """
    if (dataset_path is None) == (summary_path is None):
        raise ValueError("provide exactly one of dataset_path or summary_path")
    if output is None and out_dir is None:
        raise ValueError("provide at least one of output or out_dir")

    out_dir_path: UPath | None = None
    if out_dir is not None:
        out_dir_path = UPath(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

    return_features = output is not None

    if dataset_path is not None:
        total_count, all_features = collect_features(
            dataset_path=dataset_path,
            min_pixels=min_pixels,
            workers=32 if workers is None else workers,
            out_dir=out_dir_path,
            return_features=return_features,
        )
    else:
        assert summary_path is not None
        total_count, all_features = collect_features_from_summaries(
            summary_path=summary_path,
            min_pixels=min_pixels,
            workers=4 if workers is None else workers,
            out_dir=out_dir_path,
            return_features=return_features,
        )

    if out_dir is not None:
        print(f"Wrote per-raster GeoJSONs ({total_count} features) to {out_dir}")
    if output is not None:
        _write_feature_collection(all_features, UPath(output))
        print(f"Wrote {total_count} features to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Convert v2 LCC prediction rasters to GeoJSON change polygons."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset_path", help="Root of the prediction dataset.")
    input_group.add_argument(
        "--summary_path",
        help="Directory containing merged *_summary.tif tiles from the "
        "prediction pipeline (write_summary_raster).",
    )
    parser.add_argument(
        "--out_dir",
        help="Directory to write one GeoJSON per input raster (per summary "
        "tile for --summary_path, per window for --dataset_path).",
    )
    parser.add_argument(
        "--output",
        help="Optional single merged GeoJSON file; retains all features in "
        "memory across per-raster jobs.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=10,
        help="Minimum pixels for a connected component to be included.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default 32 for --dataset_path, 4 for "
        "--summary_path since each tile is large in memory).",
    )
    args = parser.parse_args()

    if args.output is None and args.out_dir is None:
        parser.error("at least one of --out_dir and --output is required")

    create_geojson(
        output=args.output,
        out_dir=args.out_dir,
        dataset_path=args.dataset_path,
        summary_path=args.summary_path,
        min_pixels=args.min_pixels,
        workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
