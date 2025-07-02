"""Turn shapefiles into csvs.

It's easier for google cloud if the files are in a single csv instead of
in a shapefile, so we process it into csvs.
"""

import geopandas
import pandas as pd
from upath import UPath


def process_files(shapefile_path: UPath) -> pd.DataFrame:
    """Create windows for crop type mapping.

    Args:
        shapefile_path: path to the shapefile
    """
    df = geopandas.read_file(shapefile_path)
    is_crop = 1
    if "non" in shapefile_path.name.lower():
        is_crop = 0

    df["is_crop"] = is_crop

    df["longitude"] = df.geometry.centroid.x
    df["latitude"] = df.geometry.centroid.y

    df["org_file"] = shapefile_path.name
    df.reset_index()
    df["unique_id"] = df.apply(lambda x: f"{x.name}-{x.org_file}", axis=1)

    return df[["is_crop", "latitude", "longitude", "org_file", "unique_id"]]


if __name__ == "__main__":
    for filename in ["crop_merged_v2", "noncrop_merged_v2"]:
        df = process_files(UPath(filename))
        df.to_csv(f"{UPath(filename).stem}.csv")
