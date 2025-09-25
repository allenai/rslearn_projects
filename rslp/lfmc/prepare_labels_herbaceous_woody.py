"""Prepare the LFMC labels for the herbaceous and woody model."""

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dateutil import parser as dateutil_parser
from tqdm import tqdm

from rslp.lfmc.constants import CONUS_STATES, Column

SHEET_NAME = "LFMC data"

EXCEL_COLUMN_MAP = {
    "Sorting ID": Column.SORTING_ID,
    "Contact": Column.CONTACT,
    "Site name": Column.SITE_NAME,
    "Country": Column.COUNTRY,
    "State/Region": Column.STATE_REGION,
    "Latitude (WGS84, EPSG:4326)": Column.LATITUDE,
    "Longitude (WGS84, EPSG:4326)": Column.LONGITUDE,
    "Sampling date (YYYYMMDD)": Column.SAMPLING_DATE,
    "Protocol": Column.PROTOCOL,
    "LFMC value (%)": Column.LFMC_VALUE,
    "Species collected": Column.SPECIES_COLLECTED,
    "Species functional type": Column.SPECIES_FUNCTIONAL_TYPE,
}

TASK_NAME_COLUMN = "task_name"
START_TIME_COLUMN = "start_time"
END_TIME_COLUMN = "end_time"

# Note: underscores are not yet supported in attribute names because of an rslearn restriction.
LFMC_VALUE_HERBACEOUS_COLUMN = "herbaceousvalue"
LFMC_VALUE_WOODY_COLUMN = "woodyvalue"
FUEL_TYPE_COLUMN = "fueltype"

HERBACEOUS_FUNCTIONAL_TYPES = ["forb", "grass"]
WOODY_FUNCTIONAL_TYPES = ["large shrub", "shrub", "small tree", "subshrub", "tree"]

INPUT_EXCEL_URL = "https://springernature.figshare.com/ndownloader/files/45049786"


def parse_bounding_box(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse bounding box string into coordinates.

    Args:
        bbox_str: Bounding box in format "min_lon,min_lat,max_lon,max_lat"

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)

    Raises:
        ValueError: If the bounding box format is invalid
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(",")]
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")

        min_lon, min_lat, max_lon, max_lat = coords

        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")

        return min_lon, min_lat, max_lon, max_lat
    except ValueError as e:
        if "could not convert string to float" in str(e):
            raise ValueError("All coordinates must be valid numbers") from e
        raise


def download_excel(output_path: Path) -> None:
    """Download the Excel file.

    Args:
        output_path: path to write the Excel file
    """
    response = requests.get(INPUT_EXCEL_URL, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as progress_bar:
        with open(output_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


def create_csv(
    input_excel_path: Path,
    output_csv_path: Path,
    start_date: datetime,
    country_filter: str | None,
    state_region_filter: list[str] | None,
    bounding_box: tuple[float, float, float, float] | None,
) -> None:
    """Create the CSV file.

    Args:
        input_excel_path: path to the Excel file
        output_csv_path: path to the CSV file
        start_date: start date
        country_filter: country filter
        state_region_filter: state region filter
        bounding_box: bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    print("Reading the Excel file")
    df = pd.read_excel(
        input_excel_path, sheet_name=SHEET_NAME, usecols=list(EXCEL_COLUMN_MAP.keys())
    )
    df = df.rename(columns=EXCEL_COLUMN_MAP)
    print(f"Initial number of samples: {len(df)}")

    # Calculate 99.9% percentile and clip LFMC values
    percentile_99_9 = round(df[Column.LFMC_VALUE].quantile(0.999))
    print(
        f"99.9% percentile of LFMC values: {percentile_99_9:.2f} (rounded to: {percentile_99_9})"
    )
    df[Column.LFMC_VALUE] = df[Column.LFMC_VALUE].clip(lower=0, upper=percentile_99_9)
    print(f"Clipped LFMC values to range [0, {percentile_99_9}]")

    df = df[df[Column.SAMPLING_DATE] >= start_date]
    print(f"After filtering by date: {len(df)} samples")

    if country_filter is not None:
        df = df[df[Column.COUNTRY] == country_filter]
        print(f"After filtering by country: {len(df)} samples")

    if state_region_filter is not None:
        df = df[df[Column.STATE_REGION].isin(state_region_filter)]
        print(f"After filtering by state/region: {len(df)} samples")

    if bounding_box is not None:
        min_lon, min_lat, max_lon, max_lat = bounding_box
        df = df[
            (df[Column.LONGITUDE] >= min_lon)
            & (df[Column.LONGITUDE] <= max_lon)
            & (df[Column.LATITUDE] >= min_lat)
            & (df[Column.LATITUDE] <= max_lat)
        ]
        print(
            f"After filtering by bounding box [{min_lon}, {min_lat}, {max_lon}, {max_lat}]: {len(df)} samples"
        )

    # Filter out rows with NaN LFMC values
    initial_count = len(df)
    df = df.dropna(subset=[Column.LFMC_VALUE])
    final_count = len(df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with NaN LFMC values")
    print(f"After filtering NaN LFMC values: {len(df)} samples")
    print(df[Column.LFMC_VALUE].describe())

    # Show unique species functional types before grouping
    unique_functional_types = df[Column.SPECIES_FUNCTIONAL_TYPE].unique()
    print(
        f"Unique species functional types ({len(unique_functional_types)}): {list(unique_functional_types)}"
    )

    # Add herbaceous and woody boolean columns (case insensitive)
    df["herbaceous"] = (
        df[Column.SPECIES_FUNCTIONAL_TYPE].str.lower().isin(["forb", "grass"])
    )
    df["woody"] = df[Column.SPECIES_FUNCTIONAL_TYPE].str.lower().isin(["shrub", "tree"])

    # Group by location and date, then calculate separate averages for herbaceous and woody
    def calculate_lfmc_averages(grouped_df: pd.DataFrame) -> pd.Series:
        herbaceous_samples = grouped_df[grouped_df["herbaceous"]]
        woody_samples = grouped_df[grouped_df["woody"]]

        result = {
            LFMC_VALUE_HERBACEOUS_COLUMN: herbaceous_samples[Column.LFMC_VALUE].mean()
            if len(herbaceous_samples) > 0
            else pd.NA,
            LFMC_VALUE_WOODY_COLUMN: woody_samples[Column.LFMC_VALUE].mean()
            if len(woody_samples) > 0
            else pd.NA,
            Column.SITE_NAME: grouped_df[Column.SITE_NAME].iloc[0],
            Column.STATE_REGION: grouped_df[Column.STATE_REGION].iloc[0],
            Column.COUNTRY: grouped_df[Column.COUNTRY].iloc[0],
        }

        # Determine class based on what types of samples are present
        has_herbaceous = len(herbaceous_samples) > 0
        has_woody = len(woody_samples) > 0

        if has_herbaceous and has_woody:
            result[FUEL_TYPE_COLUMN] = "herbaceous_woody"
        elif has_herbaceous:
            result[FUEL_TYPE_COLUMN] = "herbaceous"
        elif has_woody:
            result[FUEL_TYPE_COLUMN] = "woody"
        else:
            result[FUEL_TYPE_COLUMN] = "other"

        return pd.Series(result)

    grouped_df = (
        df.groupby(
            [
                Column.LATITUDE,
                Column.LONGITUDE,
                Column.SAMPLING_DATE,
            ]
        )
        .apply(calculate_lfmc_averages, include_groups=False)
        .reset_index()
    )

    # Create unique task names by combining site name with count suffix
    site_counts = grouped_df.groupby(Column.SITE_NAME).cumcount() + 1
    grouped_df[TASK_NAME_COLUMN] = (
        grouped_df[Column.SITE_NAME].astype(str)
        + "_"
        + site_counts.astype(str).str.zfill(5)
    )
    grouped_df[START_TIME_COLUMN] = pd.to_datetime(
        grouped_df[Column.SAMPLING_DATE], errors="raise"
    )
    grouped_df[END_TIME_COLUMN] = pd.to_datetime(
        grouped_df[Column.SAMPLING_DATE], errors="raise"
    )

    print(f"Number of tasks: {grouped_df[TASK_NAME_COLUMN].nunique()}")
    print(f"Number of samples: {len(grouped_df)}")
    print(f"Min start time: {grouped_df[START_TIME_COLUMN].min()}")
    print(f"Max end time: {grouped_df[END_TIME_COLUMN].max()}")

    # Output class distribution
    print("\nClass distribution:")
    class_counts = grouped_df[FUEL_TYPE_COLUMN].value_counts()
    total_samples = len(grouped_df)
    for class_name, count in class_counts.sort_values(ascending=False).items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    # Output LFMC statistics
    print("\nLFMC statistics:")
    herbaceous_valid = grouped_df[LFMC_VALUE_HERBACEOUS_COLUMN].notna().sum()
    woody_valid = grouped_df[LFMC_VALUE_WOODY_COLUMN].notna().sum()
    print(
        f"  Locations with herbaceous LFMC: {herbaceous_valid} ({(herbaceous_valid / total_samples) * 100:.1f}%)"
    )
    print(
        f"  Locations with woody LFMC: {woody_valid} ({(woody_valid / total_samples) * 100:.1f}%)"
    )

    if herbaceous_valid > 0:
        herbaceous_mean = grouped_df[LFMC_VALUE_HERBACEOUS_COLUMN].mean()
        herbaceous_std = grouped_df[LFMC_VALUE_HERBACEOUS_COLUMN].std()
        print(f"  Herbaceous LFMC mean: {herbaceous_mean:.2f} ± {herbaceous_std:.2f}")

    if woody_valid > 0:
        woody_mean = grouped_df[LFMC_VALUE_WOODY_COLUMN].mean()
        woody_std = grouped_df[LFMC_VALUE_WOODY_COLUMN].std()
        print(f"  Woody LFMC mean: {woody_mean:.2f} ± {woody_std:.2f}")

    grouped_df.to_csv(output_csv_path, index=False)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path.cwd() / "lfmc-labels-herbaceous-woody.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--start_date",
        type=dateutil_parser.parse,
        default=datetime(2015, 7, 5),  # Earliest date with all modalities available
        help="Start date to filter the data",
    )
    parser.add_argument(
        "--preset",
        choices=["conus", "global"],
        default="global",
        help="Preset for the country and state/region filter",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box in format 'min_lon,min_lat,max_lon,max_lat' (e.g., '-125,32,-114,42')",
    )
    args = parser.parse_args()

    if args.preset == "global":
        country_filter = None
        state_region_filter = None
    elif args.preset == "conus":
        country_filter = "USA"
        state_region_filter = CONUS_STATES
    else:
        raise ValueError(f"Invalid preset: {args.preset}")

    # Parse bounding box if provided
    bounding_box = None
    if args.bbox is not None:
        bounding_box = parse_bounding_box(args.bbox)

    csv_path = args.csv_path.expanduser()
    with tempfile.TemporaryDirectory() as temp_dir:
        excel_path = Path(temp_dir) / "lfmc.xlsx"
        download_excel(excel_path)
        create_csv(
            excel_path,
            csv_path,
            args.start_date,
            country_filter,
            state_region_filter,
            bounding_box,
        )


if __name__ == "__main__":
    main()
