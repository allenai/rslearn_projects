"""This script processes the WorldCereal dataset to create global labels."""

import os

import geopandas as gpd
import pandas as pd


def process_worldcereal_data(data_dir: str, output_csv: str) -> None:
    """Process WorldCereal data from geoparquet files and save to CSV.

    Args:
        data_dir (str): Directory containing geoparquet files.
        output_csv (str): Path to save the output CSV file.
    """
    all_items = []

    for file in os.listdir(data_dir):
        gdf = gpd.read_parquet(os.path.join(data_dir, file))
        print(f"{file}: {len(gdf)}")

        items = []
        for index, row in gdf.iterrows():
            item = {}
            item["sample_id"] = row["sample_id"]
            item["longitude"], item["latitude"] = (
                row["geometry"].centroid.x,
                row["geometry"].centroid.y,
            )
            item["valid_time"] = row["valid_time"]
            item["ewoc_code"] = row["ewoc_code"]
            # item["sampling_ewoc_code"] = row["sampling_ewoc_code"]
            item["h3_l3_cell"] = row["h3_l3_cell"]
            item["quality_score_lc"] = row["quality_score_lc"]
            item["quality_score_ct"] = row["quality_score_ct"]
            # item["extract"] = row["extract"]
            # item["irrigation_status"] = row["irrigation_status"]
            items.append(item)

        all_items.extend(items)

    df = pd.DataFrame(all_items)
    df.to_csv(output_csv, index=True)


# Save the WorldCereal data to a csv file
WC_DATA_DIR = "/weka/dfive-default/yawenz/datasets/WorldCereal/geoparquets"
process_worldcereal_data(WC_DATA_DIR, "worldcereal_points.csv")
