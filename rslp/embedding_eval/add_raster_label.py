"""Compute label_raster layer from label (vector) layer.

This way the label can be used in SegmentationTask / PerPixelRegressionTask. The label
will only be at the center of the raster though.

Reads the vector label layer (GeoJSON with single feature and "label" property),
maps the label string to a class index, and writes a single-band raster with
that value at the point location and nodata elsewhere. Uses window projection
and bounds for raster extent.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys

import numpy as np
from rslearn.config import LayerType
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from tqdm import tqdm
from upath import UPath

LABEL_PROPERTY = "label"
NODATA = 255


def _discover_label_from_geojson(window: Window, label_layer: str) -> str | None:
    """Worker: read one geojson and return label value."""
    layer_dir = window.get_layer_dir(label_layer)
    features = GeojsonVectorFormat().decode_vector(
        layer_dir, window.projection, window.bounds
    )
    if not features:
        return None
    return features[0].properties.get(LABEL_PROPERTY)


def add_raster_label_for_window(
    dataset: Dataset,
    window: Window,
    label_layer: str,
    label_raster_layer: str,
    class_names: list[str],
) -> None:
    """Compute label_raster from label for one window."""
    label_dir = window.get_layer_dir(label_layer)
    features = GeojsonVectorFormat().decode_vector(
        label_dir, window.projection, window.bounds
    )
    if len(features) != 1:
        raise ValueError(
            f"expected one feature but got {len(features)} in window {window.name}"
        )

    feat = features[0]
    class_idx = class_names.index(feat.properties[LABEL_PROPERTY])

    # Create raster label that is all NODATA except at center pixel, where we use the
    # class ID.
    height = window.bounds[3] - window.bounds[1]
    width = window.bounds[2] - window.bounds[0]
    arr = np.full((1, height, width), NODATA, dtype=np.uint8)
    arr[0, height // 2, width // 2] = class_idx

    out_dir = window.get_raster_dir(label_raster_layer, ["label"])
    raster_format = (
        dataset.layers[label_raster_layer].band_sets[0].instantiate_raster_format()
    )
    raster_format.encode_raster(
        out_dir, window.projection, window.bounds, arr, nodata_val=NODATA
    )
    window.mark_layer_completed(label_raster_layer)


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compute label_raster from label (vector) layer for each window"
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--label-layer",
        type=str,
        default="label",
        help="Name of the vector label layer",
    )
    parser.add_argument(
        "--label-raster-layer",
        type=str,
        default="label_raster",
        help="Name of the output raster label layer",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes",
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path).resolve()
    dataset = Dataset(ds_path)
    if args.label_raster_layer not in dataset.layers:
        print(
            f"Config must define layer '{args.label_raster_layer}' (type raster, band 'label')",
            file=sys.stderr,
        )
        sys.exit(1)

    label_layer_cfg = dataset.layers.get(args.label_layer)
    if label_layer_cfg is None:
        raise ValueError(f"Label layer '{args.label_layer}' not in config")
    if label_layer_cfg.type != LayerType.VECTOR:
        raise ValueError(f"Label layer '{args.label_layer}' must be type vector")
    class_names = label_layer_cfg.class_names
    windows = dataset.storage.get_windows()
    ctx = multiprocessing.get_context("forkserver")

    # If no class_names in config, discover from all label files (order = first seen)
    if class_names is None:
        discovery_kwargs_list = [
            {"window": w, "label_layer": args.label_layer} for w in windows
        ]
        with ctx.Pool(args.workers) as pool:
            results = list(
                tqdm(
                    star_imap_unordered(
                        pool, _discover_label_from_geojson, discovery_kwargs_list
                    ),
                    total=len(discovery_kwargs_list),
                    desc="Discovering class names",
                )
            )
        seen: list[str] = []
        for val in results:
            if val is not None and val not in seen:
                seen.append(val)
        class_names = seen
        if not class_names:
            print(
                "No label values found; ensure label layer is materialized.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Discovered class_names: {class_names}", file=sys.stderr)

    add_kwargs_list = [
        {
            "dataset": dataset,
            "window": w,
            "label_layer": args.label_layer,
            "label_raster_layer": args.label_raster_layer,
            "class_names": class_names,
        }
        for w in windows
    ]
    with ctx.Pool(args.workers) as pool:
        list(
            tqdm(
                star_imap_unordered(pool, add_raster_label_for_window, add_kwargs_list),
                total=len(add_kwargs_list),
                desc="Adding raster label",
            )
        )


if __name__ == "__main__":
    main()
