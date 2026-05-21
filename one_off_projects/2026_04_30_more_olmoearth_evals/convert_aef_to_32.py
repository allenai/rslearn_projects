"""Crop AEF eval datasets from 256x256 down to 32x32 around the labeled center.

For each dataset under ``alphaearth_supplemental_evaluations/<dataset>/`` we copy
the windows into ``olmoearth_evals/<dataset>/`` keeping
only the Sentinel-2 raster groups, the ``label_raster`` raster layer, and the
``label`` vector layer. Sentinel-1 and the GSE/AlphaEarth embedding layers are
dropped. Each kept GeoTIFF is read at the new centered 32x32 bounds via
``GeotiffRasterFormat`` (which uses a WarpedVRT under the hood) and rewritten
in place with the corresponding pixel transform; ``metadata.json`` is updated
so the window bounds match the new size.

A stripped-down ``aef_config.json`` (only ``sentinel2``, ``label_raster``,
``label``) is copied into each destination dataset's ``config.json``.

Datasets are processed sequentially; window processing within a dataset uses a
pool of 128 worker processes.
"""

import json
import multiprocessing as mp
import shutil
from pathlib import Path

import tqdm
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

SRC_BASE = Path(
    "/weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations"
)
DST_BASE = Path("/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals")
# Stripped-down rslearn dataset config that only declares the layers we keep
# (sentinel2, label_raster, label). Copied into each destination dataset.
DST_CONFIG_SRC = Path(__file__).parent / "aef_config.json"

# AEF datasets currently referenced by data/olmoearth_evals/tasks/*.yaml.
DATASETS = [
    "africa_crop_mask",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lu",
    "us_trees",
]

CROP_SIZE = 32
NUM_PROC = 128

# Layer (sub)directory names to keep. Sentinel2 has additional item groups
# named ``sentinel2.1`` ... ``sentinel2.11``; we match by prefix on the part
# before the dot.
KEEP_RASTER_LAYER_BASES = {"sentinel2", "label_raster"}
KEEP_VECTOR_LAYER_BASES = {"label"}

GEOTIFF_FORMAT = GeotiffRasterFormat()


def _layer_base(layer_dirname: str) -> str:
    return layer_dirname.split(".", 1)[0]


def process_window(args: tuple[Path, Path]) -> None:
    """Convert one source window directory into the cropped destination layout."""
    src_window_dir, dst_window_dir = args

    # metadata.json is written last, so its presence means the window is done.
    if (dst_window_dir / "metadata.json").exists():
        return

    src_metadata_path = src_window_dir / "metadata.json"
    if not src_metadata_path.exists():
        return

    with src_metadata_path.open() as f:
        metadata = json.load(f)

    bounds = metadata["bounds"]
    cx = (bounds[0] + bounds[2]) // 2
    cy = (bounds[1] + bounds[3]) // 2
    new_bounds: tuple[int, int, int, int] = (
        cx - CROP_SIZE // 2,
        cy - CROP_SIZE // 2,
        cx + CROP_SIZE // 2,
        cy + CROP_SIZE // 2,
    )
    projection = Projection.deserialize(metadata["projection"])

    dst_window_dir.mkdir(parents=True, exist_ok=True)

    src_layers_dir = src_window_dir / "layers"
    if src_layers_dir.exists():
        dst_layers_dir = dst_window_dir / "layers"
        for layer_dir in src_layers_dir.iterdir():
            if not layer_dir.is_dir():
                continue
            base = _layer_base(layer_dir.name)

            if base in KEEP_VECTOR_LAYER_BASES:
                dst_layer_dir = dst_layers_dir / layer_dir.name
                if dst_layer_dir.exists():
                    shutil.rmtree(dst_layer_dir)
                shutil.copytree(layer_dir, dst_layer_dir)
                continue

            if base not in KEEP_RASTER_LAYER_BASES:
                continue

            dst_layer_dir = dst_layers_dir / layer_dir.name
            dst_layer_dir.mkdir(parents=True, exist_ok=True)
            for child in layer_dir.iterdir():
                if child.is_file():
                    # E.g. ``completed`` marker.
                    shutil.copy2(child, dst_layer_dir / child.name)
                    continue
                # band-set sub-directory containing geotiff.tif (+ optional
                # metadata.json sidecar with timestamps).
                src_band_dir = UPath(str(child))
                dst_band_dir = UPath(str(dst_layer_dir / child.name))
                raster = GEOTIFF_FORMAT.decode_raster(
                    src_band_dir, projection, new_bounds
                )
                GEOTIFF_FORMAT.encode_raster(
                    dst_band_dir, projection, new_bounds, raster
                )

    src_items_path = src_window_dir / "items.json"
    if src_items_path.exists():
        shutil.copy2(src_items_path, dst_window_dir / "items.json")

    new_metadata = dict(metadata)
    new_metadata["bounds"] = list(new_bounds)
    with open_atomic(UPath(str(dst_window_dir / "metadata.json")), "w") as f:
        json.dump(new_metadata, f)


def process_dataset(dataset_name: str) -> None:
    """Process all windows in one AEF dataset, parallelized across processes."""
    src_ds = SRC_BASE / dataset_name
    dst_ds = DST_BASE / dataset_name
    print(f"processing dataset {dataset_name}")
    if not src_ds.exists():
        print(f"  src does not exist: {src_ds}, skipping")
        return

    dst_ds.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DST_CONFIG_SRC, dst_ds / "config.json")

    src_windows_root = src_ds / "windows"
    if not src_windows_root.exists():
        print(f"  no windows dir: {src_windows_root}, skipping")
        return

    jobs: list[tuple[Path, Path]] = []
    for group_dir in sorted(src_windows_root.iterdir()):
        if not group_dir.is_dir():
            continue
        for window_dir in sorted(group_dir.iterdir()):
            if not window_dir.is_dir():
                continue
            dst_window_dir = dst_ds / "windows" / group_dir.name / window_dir.name
            jobs.append((window_dir, dst_window_dir))

    print(f"  {len(jobs)} windows")
    if not jobs:
        return

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(process_window, jobs),
            total=len(jobs),
            desc=dataset_name,
        ):
            pass


def main() -> None:
    for ds in DATASETS:
        process_dataset(ds)


if __name__ == "__main__":
    main()
