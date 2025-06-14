"""This script processes the WorldCereal dataset to create global labels."""


import os
import geopandas as gpd
import pandas as pd

WC_DATA_DIR = "/weka/dfive-default/yawenz/rslearn_projects/rslp/crop_type_mapping/geoparquets"

all_items = []

# for each file in the directory, read the file, and print the number of features
for file in os.listdir(WC_DATA_DIR):
    gdf = gpd.gpd.read_parquet(os.path.join(WC_DATA_DIR, file))
    print(f'{file}: {len(gdf)}')
    
    items = []
    for index, row in gdf.iterrows():
        item = {}
        item["sample_id"] = row["sample_id"]
        item["longitude"], item["latitude"] = row["geometry"].centroid.x, row["geometry"].centroid.y
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

# convert all_items to a pandas dataframe
df = pd.DataFrame(all_items)

# save the dataframe to a csv file
df.to_csv("worldcereal_points.csv", index=True)



        

        
