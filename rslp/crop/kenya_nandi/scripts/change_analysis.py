import rasterio
import numpy as np

mapA_path = "/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_2018_v4.tif"
mapB_path = "/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_output_v4.tif"
out_path = "/weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/changed_to_coffee_v4.tif"

# Report for the Sugarcane as well

# Read both GeoTIFFs
with rasterio.open(mapA_path) as srcA, rasterio.open(mapB_path) as srcB:
    A = srcA.read(1)
    B = srcB.read(1)
    profile = srcB.profile  # keep georeferencing from B

# Compute mask: values that changed to Coffee (1)
changed_to_coffee = ((A != 1) & (B == 1)).astype(np.uint8)

# Update profile to match output dtype
profile.update(dtype=rasterio.uint8, count=1)

# Write result as GeoTIFF
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(changed_to_coffee, 1)

print(f"GeoTIFF exported: {out_path}")