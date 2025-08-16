"""Prepare the LFMC labels."""

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

COLUMNS = {
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

INPUT_EXCEL_URL = "https://springernature.figshare.com/ndownloader/files/45049786"


def download_excel(output_path: Path) -> None:
    """Download the Excel file.

    Args:
        output_path: path to write the Excel file
    """
    response = requests.get(INPUT_EXCEL_URL, stream=True)
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
) -> None:
    """Create the CSV file.

    Args:
        input_excel_path: path to the Excel file
        output_csv_path: path to the CSV file
        start_date: start date
    """
    print("Reading the Excel file")
    df = pd.read_excel(
        input_excel_path, sheet_name=SHEET_NAME, usecols=list(COLUMNS.keys())
    )
    df = df.rename(columns=COLUMNS)
    print(f"Initial number of samples: {len(df)}")

    df = df[df[Column.SAMPLING_DATE] >= start_date]
    print(f"After filtering by date, number of samples: {len(df)}")

    df = df[
        (df[Column.COUNTRY] == "USA") & (df[Column.STATE_REGION].isin(CONUS_STATES))
    ]
    print(f"After filtering by location, number of samples: {len(df)}")

    # From the Globe-LFMC-2.0 paper:
    # "For remote sensing applications, it is recommended to average the LFMC measurements taken on
    # the same date and located within the same pixel of the product employed in the study. The
    # choice of which functional type to include in the average can be guided by the land cover type
    # of that pixel. For example, in open canopy forests, both trees and shrubs (or grass) could be
    # included."
    grouped_df = df.groupby(
        [
            Column.LATITUDE,
            Column.LONGITUDE,
            Column.SAMPLING_DATE,
        ],
        as_index=False,
    ).agg(
        {
            Column.SITE_NAME: "first",
            Column.SORTING_ID: "first",
            Column.LFMC_VALUE: "mean",
            Column.STATE_REGION: "first",
            Column.COUNTRY: "first",
        }
    )
    print(f"After grouping, number of samples: {len(grouped_df)}")

    grouped_df.to_csv(output_csv_path, index=False)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path.cwd() / "lfmc-labels.csv",
    )
    parser.add_argument(
        "--start_date",
        type=dateutil_parser.parse,
        default=datetime(2017, 1, 1),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        excel_path = Path(temp_dir) / "lfmc.xlsx"
        download_excel(excel_path)
        create_csv(excel_path, args.csv_path, args.start_date)


if __name__ == "__main__":
    main()
