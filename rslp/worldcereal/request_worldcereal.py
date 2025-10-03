"""This script requests the WorldCereal collections data."""

import pandas as pd
import requests


def request_worldcereal_data(api_url: str) -> None:
    """Fetch and process WorldCereal collections data.

    Args:
        api_url (str): The URL of the WorldCereal API.
    """
    timeout = 10  # seconds

    # Initial request to get collections
    collection_response = requests.get(
        f"{api_url}/collections?SkipCount=0&MaxResultCount=10", timeout=timeout
    )
    res = collection_response.json()

    total = res["totalCount"]
    items = res["items"]

    print(f"Total Public Collections: {total}")
    print(f"Current Response Collection Count: {len(items)}")

    accumulated_count = len(items)

    # Paginate through all collections
    while accumulated_count < total:
        req_url = (
            f"{api_url}/collections?SkipCount={accumulated_count}&MaxResultCount=10"
        )
        collection_response = requests.get(req_url, timeout=timeout)
        res2 = collection_response.json()
        print(res2)
        total = res2["totalCount"]
        items += res2["items"]
        accumulated_count += len(res2["items"])
        print(f"Accumulated Collections Count: {accumulated_count}")

    print(items[0])

    df = pd.DataFrame(items)
    df.to_csv("worldcereal_collections.csv", index=False)

    # Filter rows with high confidence
    df_filtered = df[df["confidenceLandCover"] >= 90]
    df_filtered = df_filtered[df_filtered["confidenceCropType"] >= 90]

    print(f"Total filtered rows: {len(df_filtered)}")  # 89 out of 140 collections

    for index, row in df_filtered.iterrows():
        collection_id = row["collectionId"]
        print(f"Processing collectionId: {collection_id}")

        # Fetch metadata for each collection
        metadata_url = f"{api_url}/collections/{collection_id}/metadata/items"
        metadata = requests.get(metadata_url, timeout=timeout)
        metadata = metadata.json()

        # Download files ending with .geoparquet
        for item in metadata:
            if (
                isinstance(item, dict)
                and "value" in item
                and isinstance(item["value"], str)
                and item["value"].endswith(".geoparquet")
            ):
                print(item["value"])
                file_url = item["value"]
                file_name = file_url.split("/")[-1]
                response = requests.get(file_url, timeout=timeout)
                with open(file_name, "wb") as file:
                    file.write(response.content)


request_worldcereal_data("https://ewoc-rdm-api.iiasa.ac.at")
