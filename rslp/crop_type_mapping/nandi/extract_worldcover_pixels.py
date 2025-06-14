"""Extract water/built-up pixels from WorldCover."""

import numpy as np
import pandas as pd
import rasterio
from upath import UPath

# The WorldCovergeotiff covers most of the Kenya/Nandi county
tiff_path = "gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/WorldCover/NandiCounty_worldcover.tif"
output_path = "gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiWorldCoverPoints.csv"
output_sampled_path = "gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiWorldCoverPoints_sampled.csv"

with rasterio.open(tiff_path) as src:
    band = src.read(1)  # Read the first band
    transform = src.transform  # Get the affine transform

    # Find where the value is either 50 (Built-up) or 80 (Water)
    mask = (band == 50) | (band == 80)
    row_indices, col_indices = np.where(mask)
    # print how many with value 50 and 80
    print(
        f"Found {len(row_indices[band[row_indices, col_indices] == 50])} pixels with value 50"
    )  # 134416
    print(
        f"Found {len(row_indices[band[row_indices, col_indices] == 80])} pixels with value 80"
    )  # 2982

    # Filter out points outside the boundaries
    with UPath(output_path).open("w") as f:
        # Write the header
        f.write("longitude,latitude,value\n")
        for row, col in zip(row_indices, col_indices):
            lon, lat = rasterio.transform.xy(transform, row, col)
            f.write(f"{lon},{lat},{band[row, col]}\n")
            print(f"Wrote {lon},{lat},{band[row, col]}")

    # Select 1K points for each category
    with UPath(output_path).open("r") as f:
        df = pd.read_csv(f)
    df_50 = df[df["value"] == 50].sample(1000, random_state=42)
    df_80 = df[df["value"] == 80].sample(1000, random_state=42)
    df_sampled = pd.concat([df_50, df_80]).reset_index(drop=True)
    with UPath(output_sampled_path).open("w") as f:
        df_sampled.to_csv(f, index=True)
