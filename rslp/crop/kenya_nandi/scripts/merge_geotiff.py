"""This script merges multiple GeoTIFF files into a single mosaic GeoTIFF file."""

from pathlib import Path

import rasterio
from rasterio.merge import merge

# Base directory containing subfolders
base_dir = Path(
    "/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616/windows/nandi_county"
)
output_dir = Path("/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type")

# Find all immediate subdirectories under base_dir
subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

# Construct paths to geotiff.tif under the known structure
tif_files = []
for subdir in subdirs:
    tif_path = subdir / "layers/prediction_v3/output/geotiff.tif"
    if tif_path.exists():
        tif_files.append(tif_path)
    else:
        print(f"Warning: {tif_path} not found.")

print(f"Found {len(tif_files)} GeoTIFF files.")

# Open all rasters
src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

# Mosaic
mosaic, out_trans = merge(src_files_to_mosaic)

# Use metadata from first image as template
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw",
    }
)

# Output path
output_path = output_dir / "mosaic_output_v3.tif"

# Write mosaic
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f"Mosaic written to: {output_path}")
