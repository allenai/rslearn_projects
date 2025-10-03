# Final clean up on the geotiff files that we generated.

import numpy as np
import rasterio
from rasterio.enums import Resampling
from multiclean import clean_array

# Input / output paths
input_tif = "/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_2018_v4_1.tif"
output_tif = "/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_2018_v4_1_cleaned.tif"

# Open the input GeoTIFF
with rasterio.open(input_tif) as src:
    profile = src.profile.copy()
    array = src.read(1)  # read first band, adjust if multi-band

# Clean the array
cleaned = clean_array(
    array,
    class_values=np.unique(array).tolist(),  # use all classes found in the tif
    smooth_edge_size=1,
    min_island_size=4,
    connectivity=8,
    max_workers=4,
    fill_nan=False
)

# Save to a new GeoTIFF
with rasterio.open(output_tif, "w", **profile) as dst:
    dst.write(cleaned.astype(profile["dtype"]), 1)

print(f"Cleaned GeoTIFF saved to {output_tif}")