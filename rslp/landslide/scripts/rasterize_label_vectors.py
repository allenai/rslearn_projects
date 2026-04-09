"""Rasterize vector label GeoJSON into label_raster GeoTIFFs per rslearn window.

The create_{dataset_name}_windows.py script creates the vector label GeoJSON files, so this
script is used to rasterize them into GeoTIFFs using ``rasterio.features.rasterize``.
"""

from __future__ import annotations

import argparse
import multiprocessing
from typing import Any

import affine
import numpy as np
import shapely
import tqdm
from rasterio.features import rasterize
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

DEFAULT_VECTOR_LAYER = "label"
DEFAULT_RASTER_LAYER = "label_raster"
DEFAULT_BAND = "label"

# Testing: restrict to these window ``name`` values.
# _TEST_ONLY_WINDOW_NAMES: frozenset[str] | None = frozenset(
#     {
#         "63628_positive_43.9371_12.2082_2023",
#         "21976_positive_44.3620_11.4126_2023",
#     }
# )
_TEST_ONLY_WINDOW_NAMES: frozenset[str] | None = None


def _default_draw_class_order(class_names: list[str]) -> list[int]:
    """Return class indices bottom-to-top for rasterize (later = on top).

    Non-positive classes are drawn first; **positive** landslide-like classes are drawn
    last so they are not covered by buffers. Names that contain ``landslide`` only as
    part of ``no_landslide`` (or similar) are **not** treated as positive — otherwise
    ``no_landslide`` would paint over ``no_data``, which matches neither the vector
    vis order nor the intended semantics.
    """
    indices = list(range(len(class_names)))
    positive_landslide = [
        i
        for i, name in enumerate(class_names)
        if "landslide" in name.lower() and "no_landslide" not in name.lower()
    ]
    if not positive_landslide:
        return indices
    rest = [i for i in indices if i not in positive_landslide]
    return rest + positive_landslide


def _property_to_class_index(
    raw: Any,
    name_to_index: dict[str, int],
    num_classes: int,
) -> int | None:
    """Map a feature property value to a raster class index, or None to skip."""
    if isinstance(raw, bool | np.bool_):
        return int(raw)

    if isinstance(raw, int | np.integer):
        v = int(raw)
        if 0 <= v < num_classes:
            return v
        return None

    if isinstance(raw, str):
        if raw in name_to_index:
            return name_to_index[raw]
        # Allow numeric strings
        if raw.isdigit():
            v = int(raw)
            if 0 <= v < num_classes:
                return v
        return None

    return None


def rasterize_window(
    window: Window,
    *,
    vector_layer: str,
    raster_layer: str,
    bands: list[str],
    class_property: str,
    class_names: list[str],
    draw_order: list[int] | None,
    all_touched: bool,
    default_fill_index: int,
) -> None:
    """Rasterize ``vector_layer`` GeoJSON into ``raster_layer`` for one window."""
    min_x, min_y, max_x, max_y = window.bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid window bounds {window.bounds} for {window.name}")

    label_dir = window.get_layer_dir(vector_layer)
    geojson_file = label_dir / GeojsonVectorFormat.fname
    if not geojson_file.exists():
        return

    features = GeojsonVectorFormat().decode_vector(
        label_dir, window.projection, window.bounds
    )

    name_to_index = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    order = (
        draw_order if draw_order is not None else _default_draw_class_order(class_names)
    )
    draw_rank = {class_idx: rank for rank, class_idx in enumerate(order)}

    clip = shapely.box(0, 0, width, height)

    shapes: list[tuple[Any, int]] = []
    for feat in features:
        idx = _property_to_class_index(
            feat.properties.get(class_property),
            name_to_index,
            num_classes,
        )
        if idx is None:
            continue

        geom = feat.geometry.shp
        if geom is None or geom.is_empty:
            continue

        local = shapely.affinity.translate(geom, xoff=-min_x, yoff=-min_y)
        local = local.intersection(clip)
        if local.is_empty:
            continue
        if not local.is_valid:
            local = shapely.make_valid(local)
        if local.is_empty:
            continue

        shapes.append((local, idx))

    shapes.sort(key=lambda t: draw_rank.get(t[1], 0))

    transform = affine.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    if not shapes:
        data = np.full((height, width), default_fill_index, dtype=np.uint8)
    else:
        data = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=default_fill_index,
            all_touched=all_touched,
            dtype=np.uint8,
        )

    raster = np.expand_dims(data, axis=0).astype(np.uint8, copy=False)
    raster_dir = window.get_raster_dir(raster_layer, bands)
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=raster),
    )
    window.mark_layer_completed(raster_layer)


def _worker_init(
    vector_layer: str,
    raster_layer: str,
    bands: list[str],
    class_property: str,
    class_names: list[str],
    draw_order: list[int] | None,
    all_touched: bool,
    default_fill_index: int,
) -> None:
    global _WORKER_KWARGS
    _WORKER_KWARGS = {
        "vector_layer": vector_layer,
        "raster_layer": raster_layer,
        "bands": bands,
        "class_property": class_property,
        "class_names": class_names,
        "draw_order": draw_order,
        "all_touched": all_touched,
        "default_fill_index": default_fill_index,
    }


def _process_window(window: Window) -> None:
    try:
        rasterize_window(window, **_WORKER_KWARGS)
    except Exception as e:
        print(f"Error rasterizing {window.group}/{window.name}: {e}")


_WORKER_KWARGS: dict[str, Any] = {}


def main() -> None:
    """CLI entry: rasterize vector labels to raster layers for all windows."""
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize per-window vector labels to label_raster using "
            "rasterio.features.rasterize"
        )
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Dataset root (contains config.json and windows/)",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--vector_layer",
        type=str,
        default=DEFAULT_VECTOR_LAYER,
        help="Vector layer name (GeoJSON under layers/<name>/data.geojson)",
    )
    parser.add_argument(
        "--raster_layer",
        type=str,
        default=DEFAULT_RASTER_LAYER,
        help="Output raster layer name",
    )
    parser.add_argument(
        "--band",
        type=str,
        default=DEFAULT_BAND,
        help="Band name for label_raster (must match config band set)",
    )
    parser.add_argument(
        "--class_property",
        type=str,
        default=None,
        help="GeoJSON property for class (default: label layer class_property_name)",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Comma-separated class names (default: from config label layer)",
    )
    parser.add_argument(
        "--rasterize_order",
        type=str,
        default=None,
        help=(
            "Comma-separated class names: bottom to top (last wins). "
            "Default: auto (landslide-named classes last)"
        ),
    )
    parser.add_argument(
        "--no_all_touched",
        action="store_true",
        help="Use center-pixel rule instead of all_touched",
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    dataset = Dataset(ds_path)

    if args.vector_layer not in dataset.layers:
        raise KeyError(f"No layer {args.vector_layer!r} in dataset config")

    label_cfg = dataset.layers[args.vector_layer]
    class_property = args.class_property or label_cfg.class_property_name or "label"

    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",") if s.strip()]
    elif label_cfg.class_names:
        class_names = list(label_cfg.class_names)
    else:
        raise ValueError(
            "No class_names in config for vector layer; pass --class_names"
        )

    draw_order: list[int] | None
    if args.rasterize_order:
        ordered_names = [
            s.strip() for s in args.rasterize_order.split(",") if s.strip()
        ]
        name_to_i = {n: i for i, n in enumerate(class_names)}
        draw_order = []
        for n in ordered_names:
            if n not in name_to_i:
                raise ValueError(
                    f"Unknown class {n!r} in --rasterize_order; "
                    f"expected one of {list(class_names)}"
                )
            draw_order.append(name_to_i[n])
    else:
        draw_order = None

    default_fill = 0

    multiprocessing.set_start_method("forkserver")
    windows = dataset.load_windows(workers=args.workers, show_progress=True)

    if _TEST_ONLY_WINDOW_NAMES is not None:
        before = len(windows)
        windows = [w for w in windows if w.name in _TEST_ONLY_WINDOW_NAMES]
        found_names = {w.name for w in windows}
        missing = _TEST_ONLY_WINDOW_NAMES - found_names
        print(
            f"TEST filter: {before} -> {len(windows)} windows "
            f"(allowed={sorted(_TEST_ONLY_WINDOW_NAMES)})"
        )
        if missing:
            print(f"WARNING: no loaded window matched name(s): {sorted(missing)}")

    print(f"Windows: {len(windows)}")
    print(f"class_property={class_property!r}, class_names={class_names}")

    p = multiprocessing.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(
            args.vector_layer,
            args.raster_layer,
            [args.band],
            class_property,
            class_names,
            draw_order,
            not args.no_all_touched,
            default_fill,
        ),
    )
    for _ in tqdm.tqdm(
        p.imap_unordered(_process_window, windows),
        total=len(windows),
    ):
        pass
    p.close()
    p.join()
    print("Done.")


if __name__ == "__main__":
    main()
