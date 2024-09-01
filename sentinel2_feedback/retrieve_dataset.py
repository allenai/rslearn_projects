import argparse
import csv
import json
import logging
import os
import sys

import requests
from pydantic import BaseModel

SKYLIGHT_GRAPHQL_API = os.getenv(
    "SKYLIGHT_GRAPHQL_API", "https://api-int.skylight.earth/graphql"
)

logger = logging.getLogger(__name__)


class ArgsModel(BaseModel):
    token: str
    feedback_csv: str
    chips_dir: str
    output_csv: str


def query_event_by_id(
    args: ArgsModel, session: requests.Session, event_id: str
) -> dict:
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }
    query = {
        "query": """
        query Event($eventId: ID!) {
            event(eventId: $eventId) {
                event_id
                event_type
                event_details {
                    image_url
                }
                start {
                    time
                    point { lat lon }
                }
            }
        }
        """,
        "variables": {
            "eventId": event_id,
        },
    }

    response = session.post(
        SKYLIGHT_GRAPHQL_API,
        headers=headers,
        data=json.dumps(query),
        timeout=5,
    )
    try:
        response.raise_for_status()
        if "errors" in response.json():
            raise requests.exceptions.HTTPError(response.json()["errors"])
    except requests.exceptions.HTTPError as e:
        logger.error(response.text)
        raise e
    return response.json()["data"]["event"]


def download_chip(args: ArgsModel, event_data: dict) -> str:
    event_id = event_data["event_id"]
    chip_url = event_data["event_details"]["image_url"]
    response = requests.get(chip_url, stream=True)
    response.raise_for_status()

    output_path = os.path.join(args.chips_dir, f"{event_id}.png")
    with open(output_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)

    return output_path


def process_events(args: ArgsModel, session: requests.Session):
    with open(args.feedback_csv, mode="r") as file:
        reader = csv.DictReader(file)
        with open(args.output_csv, mode="w", newline="") as output_file:
            fieldnames = ["event_id", "label", "lat", "lon", "chip_path", "time"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                event_id = row["event_id"]
                print(f"Processing event {event_id}")

                # Query event by event_id
                try:
                    event_data = query_event_by_id(args, session, event_id)
                    if not event_data:
                        raise Exception(f"No data found for event {event_id}")

                    # Download the chip and get the local path
                    chip_path = download_chip(args, event_data)

                    # Extract label and coordinates
                    label = row["value"]
                    point = event_data["start"]["point"]

                    # Write to the output CSV
                    writer.writerow(
                        {
                            "event_id": event_id,
                            "label": label,
                            "lat": point["lat"],
                            "lon": point["lon"],
                            "chip_path": chip_path,
                            "time": event_data["start"]["time"],
                        }
                    )
                except Exception as e:
                    print(f"Failed to process event {event_id}: {e}")
                    raise e


if __name__ == "__main__":
    session = requests.Session()
    parser = argparse.ArgumentParser(
        description="Retrieves chips for events from the Skylight API."
    )
    parser.add_argument(
        "--token", type=str, required=True, help="Authorization token for the API."
    )
    parser.add_argument(
        "--feedback_csv",
        type=str,
        required=True,
        help="CSV file containing event eo_sentinel2 event ids and feedback labels.",
    )
    parser.add_argument(
        "--chips_dir",
        type=str,
        required=True,
        help="Directory where to store the chips.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file to store the dataset information.",
    )
    parsed_args = parser.parse_args()
    args = ArgsModel(**vars(parsed_args))  # convert parsed args to pydantic model

    os.makedirs(args.chips_dir, exist_ok=True)
    process_events(args, session)
