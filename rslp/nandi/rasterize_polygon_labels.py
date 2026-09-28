"""Burn the per-window label polygons into the `label_raster` layer.

Polygons are rasterized with ``all_touched=False``, so a pixel is labelled only when
its centre falls inside a polygon. Every other pixel -- including the boundary ring
that ``all_touched=True`` would have claimed -- is written as IGNORE_VALUE and masked
out of both the loss and the metrics by SegmentationTask's ``nodata_value``.

That ring is not a rounding detail: across the Nandi ground truth it is 11,381 pixels,
33% of everything ``all_touched=True`` would label. Those are mixed pixels straddling a
field edge, and letting them carry a hard class label is a good way to teach the model
the wrong spectra for small fields.

Where polygons overlap, the draw order is (source priority ascending, area descending),
so ground truth is drawn over Studio annotations and both over WorldCover, and a small
polygon is drawn over a large one that contains it.
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import affine
import numpy as np
import shapely
import shapely.affinity
import tqdm
from rasterio.features import rasterize
from rslearn.dataset import Dataset, Window
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.nandi.classes import CLASS_IDS, CLASS_NAMES, IGNORE_VALUE

LABEL_LAYER = "label"
RASTER_LAYER = "label_raster"
BAND_NAME = "class"

_DATASET: Dataset | None = None


def _worker_init(ds_path: str) -> None:
    """Open the dataset once per worker process.

    Args:
        ds_path: the dataset path.
    """
    global _DATASET
    _DATASET = Dataset(UPath(ds_path))


def rasterize_window(window: Window) -> Counter[int]:
    """Rasterize one window's label polygons into its label_raster layer.

    Args:
        window: the window to rasterize.

    Returns:
        a Counter of class ID to pixel count for this window.
    """
    assert _DATASET is not None
    window._data = _DATASET.window_data_storage_factory.create(window)

    min_x, min_y, max_x, max_y = window.bounds
    width = max_x - min_x
    height = max_y - min_y

    features = window.data.read_vector(LABEL_LAYER, GeojsonVectorFormat())

    clip = shapely.box(0, 0, width, height)
    shapes: list[tuple[Any, int, float]] = []
    for feature in features:
        class_id = CLASS_IDS.get(feature.properties["category"])
        if class_id is None:
            continue
        geometry = shapely.affinity.translate(
            feature.geometry.shp, xoff=-min_x, yoff=-min_y
        ).intersection(clip)
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
        if geometry.is_empty:
            continue
        shapes.append(
            (geometry, class_id, float(feature.properties.get("priority", 0)))
        )

    # Lower priority first and larger area first, so the most specific, most trusted
    # polygon is drawn last and therefore wins.
    shapes.sort(key=lambda item: (item[2], -item[0].area))

    if shapes:
        data = rasterize(
            [(geometry, class_id) for geometry, class_id, _ in shapes],
            out_shape=(height, width),
            transform=affine.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            fill=IGNORE_VALUE,
            all_touched=False,
            dtype=np.uint8,
        )
    else:
        data = np.full((height, width), IGNORE_VALUE, dtype=np.uint8)

    with window.data.open_layer_writer(RASTER_LAYER) as writer:
        writer.write_raster(
            [BAND_NAME],
            GeotiffRasterFormat(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=data[None, :, :]),
        )
    window.mark_layer_completed(RASTER_LAYER)

    values, counts = np.unique(data, return_counts=True)
    return Counter({int(v): int(c) for v, c in zip(values, counts)})


def main() -> None:
    """Rasterize label polygons for every window in the group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--group", type=str, action="append", default=None, help="Window group"
    )
    parser.add_argument("--workers", type=int, default=64, help="Worker processes")
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(
        groups=args.group, workers=args.workers, show_progress=True
    )
    print(f"Rasterizing {len(windows)} windows")

    totals: Counter[int] = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_worker_init, initargs=(args.ds_path,)
    ) as pool:
        for counts in tqdm.tqdm(
            pool.imap_unordered(rasterize_window, windows), total=len(windows)
        ):
            totals.update(counts)

    labelled = sum(count for value, count in totals.items() if value != IGNORE_VALUE)
    total = sum(totals.values())
    print(f"\n{'class':<12} {'pixels':>9} {'% of labelled':>14}")
    print("-" * 38)
    for class_id, name in enumerate(CLASS_NAMES):
        count = totals.get(class_id, 0)
        share = 100 * count / labelled if labelled else 0
        print(f"{name:<12} {count:>9} {share:>13.1f}%")
    print("-" * 38)
    print(f"{'labelled':<12} {labelled:>9} {100 * labelled / total:>13.1f}% of pixels")
    print(f"{'ignored':<12} {totals.get(IGNORE_VALUE, 0):>9}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
