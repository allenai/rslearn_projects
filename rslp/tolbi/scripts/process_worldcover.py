"""Process WorldCover labels and extract negative samples for the Tolbi
project."""

import argparse
import pandas as pd


# WorldCover labels reference year is 2015
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


def process_worldcover(
    worldcover_path: str,
    output_path: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> None:
    """Process WorldCover labels and extract points."""
    df = pd.read_csv(worldcover_path)
    df["latitude"] = df["center_y"]
    df["longitude"] = df["center_x"]
    df = df[
        df["latitude"].between(min_lat, max_lat)
        & df["longitude"].between(min_lon, max_lon)
    ]
    # Add timestamps
    df["start_time"] = START_DATE
    df["end_time"] = END_DATE
    # Add tags & ids
    df["task_name"] = df["sampleid"]
    df["tag_name"] = df["class_name"]
    print(df["tag_name"].unique())
    df["id"] = df.index
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process WorldCover labels and extract points for Tolbi."
    )
    parser.add_argument(
        "--input", 
        type=str,
        required=True,
        help="Path to WorldCover CSV input file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file for Ivory Coast region.",
    )
    parser.add_argument(
        "--min-lat",
        required=True,
        type=float,
        help="Minimum latitude for filtering points.",
    )
    parser.add_argument(
        "--max-lat",
        required=True,
        type=float,
        help="Maximum latitude for filtering points.",
    )
    parser.add_argument(
        "--min-lon",
        required=True,
        type=float,
        help="Minimum longitude for filtering points.",
    )
    parser.add_argument(
        "--max-lon",
        required=True,
        type=float,
        help="Maximum longitude for filtering points.",
    )

    args = parser.parse_args()

    process_worldcover(
        args.input,
        args.output,
        args.min_lat,
        args.max_lat,
        args.min_lon,
        args.max_lon,
    )
