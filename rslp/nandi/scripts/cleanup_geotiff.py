"""Final clean-up for a GeoTIFF using multiclean.clean_array.

Example:
    python clean_geotiff.py \
        --input /weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_2018_v4_1.tif \
        --output /weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_2018_v4_1_cleaned.tif \
        --smooth-edge-size 1 --min-island-size 4 --connectivity 8 --max-workers 4
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from multiclean import clean_array


def clean_geotiff(
    input_tif: Path,
    output_tif: Path,
    band: int = 1,
    smooth_edge_size: int = 1,
    min_island_size: int = 4,
    connectivity: int = 8,
    max_workers: int = 4,
    fill_nan: bool = False,
) -> None:
    """Clean a GeoTIFF with multiclean."""
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        arr = src.read(band)

    cleaned = clean_array(
        arr,
        class_values=np.unique(arr).tolist(),
        smooth_edge_size=smooth_edge_size,
        min_island_size=min_island_size,
        connectivity=connectivity,
        max_workers=max_workers,
        fill_nan=fill_nan,
    )

    # write single band back with original profile
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(cleaned.astype(profile["dtype"]), band)

    print(f"Cleaned GeoTIFF saved to {output_tif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a GeoTIFF with multiclean.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument("--smooth-edge-size", type=int, default=1)
    parser.add_argument("--min-island-size", type=int, default=4)
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--fill-nan", action="store_true")
    args = parser.parse_args()

    clean_geotiff(
        input_tif=args.input,
        output_tif=args.output,
        band=args.band,
        smooth_edge_size=args.smooth_edge_size,
        min_island_size=args.min_island_size,
        connectivity=args.connectivity,
        max_workers=args.max_workers,
        fill_nan=args.fill_nan,
    )
