"""Create samples for the Tolbi project."""

import os
import argparse
import numpy as np
import pandas as pd
import json
from shapely.geometry import shape, Point
from shapely.ops import unary_union


# Label reference years
POSTIVE_LABEL_YEAR = 2024
NEGATIVE_LABEL_YEAR = 2015

# Label mapping
POSITIVE_LABEL_MAPPING = {
    "Cocoa": "cacao",
    "Cacao": "cacao",
    "palm": "palmoil",
    "PalmOil": "palmoil",
    "Rubber": "rubber",
}
# Only keep tree, shrub, and others
NEGATIVE_LABEL_MAPPING = {
    "bare": "others",
    "burnt": "others",
    "crops": "others",
    "fallow/shifting cultivation": "others",
    "grassland": "others",
    "Not sure": "unknown",  # Should not use this
    "urban/built-up": "others",
    "water": "others",
    "wetland (herbaceous)": "others",
    "tree": "tree",
    "shrub": "shrub",
}

def get_points_within_polygon(
    geojson_data: dict,
) -> list[tuple[float, float, str, int]]:
    """Get all points within the polygons of each feature, with per-feature
    class_name and id (no fallback)."""
    grid_points = []
    for feature in geojson_data["features"]:
        geom = shape(feature["geometry"])
        class_name = feature["properties"]["crop"]
        class_name = POSITIVE_LABEL_MAPPING.get(class_name, class_name)
        minx, miny, maxx, maxy = geom.bounds
        # Define grid size in degrees, about 10 meters at the equator
        grid_size = 0.0001
        lons = np.arange(minx, maxx, grid_size)
        lats = np.arange(miny, maxy, grid_size)
        for lon in lons:
            for lat in lats:
                point = Point(lon, lat)
                if geom.contains(point):
                    grid_points.append((lon, lat, class_name, POSTIVE_LABEL_YEAR))
    return grid_points


def create_positive_samples(
    geojson_dir: str, 
    output_path: str,
) -> pd.DataFrame:
    """Create positive samples from the geojsons."""
    positive_samples = []
    for geojson_file in os.listdir(geojson_dir):
        if geojson_file.endswith(".geojson"):
            geojson_path = os.path.join(geojson_dir, geojson_file)
            with open(geojson_path, "r") as f:
                geojson = json.load(f)
                grid_points = get_points_within_polygon(geojson)
                positive_samples.extend(grid_points)
    print(f"Created {len(positive_samples)} positive samples")
    # Remove duplicates (overlapped polygons)
    positive_samples_df = pd.DataFrame(
        positive_samples, columns=["longitude", "latitude", "class_name", "reference_year"]
    ).drop_duplicates()
    print(positive_samples_df.groupby("class_name").size())
    positive_samples_df.to_csv(output_path, index=False)
    return positive_samples_df


def create_negative_samples(
    worldcover_path: str, 
    output_path: str,
) -> pd.DataFrame:
    """Create negative samples from the worldcover."""
    negative_samples_df = pd.read_csv(worldcover_path)
    negative_samples_df["class_name"] = negative_samples_df["class_name"].map(NEGATIVE_LABEL_MAPPING)
    # Remove unknown samples
    negative_samples_df = negative_samples_df[negative_samples_df["class_name"] != "unknown"]
    negative_samples_df = negative_samples_df[["longitude", "latitude", "class_name", "reference_year"]]
    print(f"Created {len(negative_samples_df)} negative samples")
    negative_samples_df.to_csv(output_path, index=False)
    return negative_samples_df


def combine_samples(
    positive_samples_df: pd.DataFrame,
    negative_samples_df: pd.DataFrame,
    sample_size: int,
    output_path: str,
) -> pd.DataFrame:
    """Combine the positive and negative samples."""
    dfs = []
    df = pd.concat([positive_samples_df, negative_samples_df])
    # Same number of samples per class
    for class_name in df["class_name"].unique():
        df_class = df[df["class_name"] == class_name]
        df_class_sampled = df_class.sample(sample_size, random_state=42)
        dfs.append(df_class_sampled)
    df_sampled = pd.concat(dfs).reset_index(drop=True)
    print(f"Created {len(df_sampled)} samples")
    print(df_sampled.groupby("class_name").size())
    df_sampled.to_csv(output_path, index=False)
    return df_sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos_geojson_dir", 
        type=str, 
        required=True,
        help="Directory containing positive sample geojson files."
    )
    parser.add_argument(
        "--pos_output", 
        type=str, 
        required=True,
        help="Output CSV path for positive samples."
    )
    parser.add_argument(
        "--neg_input", 
        type=str, 
        required=True,
        help="Input CSV path for negative samples (e.g., filtered WorldCover for region)."
    )
    parser.add_argument(
        "--neg_output", 
        type=str, 
        required=True,
        help="Output CSV path for negative samples."
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        required=True,
        help="Number of samples per class to include in the combined output."
    )
    parser.add_argument(
        "--combined_output", 
        type=str, 
        required=True,
        help="Output CSV path for combined positive and negative samples."
    )
    args = parser.parse_args()

    positive_samples_df = create_positive_samples(args.pos_geojson_dir, args.pos_output)
    negative_samples_df = create_negative_samples(args.neg_input, args.neg_output)
    combined_samples_df = combine_samples(positive_samples_df, negative_samples_df, args.sample_size, args.combined_output)


