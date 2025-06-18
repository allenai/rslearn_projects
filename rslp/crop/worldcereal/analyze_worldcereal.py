"""This script is used to analyze the WorldCereal dataset."""

import os

import pandas as pd

# Load the WorldCereal dataset
csv_dir = "/weka/dfive-default/yawenz/datasets/WorldCereal/csv"
df = pd.read_csv(os.path.join(csv_dir, "worldcereal_points.csv"))

print(df["ewoc_code"].nunique())  # 549
print(df["h3_l3_cell"].nunique())  # 1378

# Each h3_l3_cell sample 100 points, if less then 100 points, sample what is available
df_filtered = (
    df.groupby("h3_l3_cell")
    .apply(lambda x: x.sample(min(100, len(x))))
    .reset_index(drop=True)
)
print(df_filtered.shape)  # (1378, 10)

# Save the filtered dataframe to a csv file
df_filtered.to_csv(os.path.join(csv_dir, "worldcereal_points_filtered.csv"))

# Load the filtered dataframe
df = pd.read_csv(os.path.join(csv_dir, "worldcereal_points_filtered.csv"))
print(df.shape)  # (66000, 10)

# Get the specific classes from the ewoc_code column
df["level_1"] = df["ewoc_code"].apply(lambda x: str(x)[0:2])
df["level_2"] = df["ewoc_code"].apply(lambda x: str(x)[2:4])
df["level_3"] = df["ewoc_code"].apply(lambda x: str(x)[4:6])

level_1_lookup = {
    "00": "unknown",
    "10": "cropland_unclassified",
    "11": "temporary_crops",
    "12": "permanent_crops",
    "14": "mixed_cropland",
    "15": "greenhouse_foil_film_indoor",
    "16": "non_cropland_incl_perennial",
    "17": "non_cropland_excl_perennial",
    "20": "non_cropland_herbaceous",
    "25": "non_cropland_mixed",
    "30": "shrubland",
    "40": "trees_unspecified",
    "41": "trees_broadleaved",
    "42": "trees_coniferous",
    "43": "trees_mixed",
    "50": "bare_sparsely_vegetated",
    "60": "built_up",
    "70": "open_water",
}

# There is no "unknown" in the level_1 column
print(
    df["level_1"].unique()
)  # ['20' '25' '12' '43' '11' '42' '41' '40' '14' '16' '30' '50' '60' '70' '15' '10']

# level_1
# 10      497
# 11    31601
# 12     3613
# 14       85
# 15       38
# 16      275
# 20    19860
# 25     1544
# 30      580
# 40      344
# 41     2881
# 42     2178
# 43     1724
# 50      193
# 60      458
# 70      129

# Cropland: 35,834
# Non-cropland: 30,166

df["level_123"] = df["ewoc_code"].apply(lambda x: str(x)[0:6])
df.to_csv(os.path.join(csv_dir, "worldcereal_points_filtered_level_123.csv"))
print(df["level_123"].nunique())  # 91 classes
df_level_123 = df.groupby(["level_123", "h3_l3_cell"]).size().reset_index()
df_level_123.to_csv(os.path.join(csv_dir, "worldcereal_level_123.csv"))
