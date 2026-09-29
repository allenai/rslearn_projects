"""Small helpers for the Studio API endpoints used by the project scripts.

The task/annotation/labelset write endpoints are not in the public OpenAPI schema;
the request bodies here were verified against the API on 2026-09-28 (labelsets and
metadata fields are attached to the project via project_settings_id).
"""

import os
import time
from typing import Any

import requests

BASE_URL = "https://olmoearth.allenai.org/api/v1"
TIMEOUT = 60
MAX_RETRIES = 3

# One color per forest loss driver class, shared by the category and validate
# labelsets so the same class looks the same in both.
CLASS_COLORS = {
    "agriculture": "#1f77b4",
    "mining": "#ff7f0e",
    "airstrip": "#2ca02c",
    "road": "#d62728",
    "logging": "#9467bd",
    "burned": "#8c564b",
    "landslide": "#e377c2",
    "hurricane": "#7f7f7f",
    "river": "#17becf",
    "none": "#bcbd22",
}
COUNTRY_COLORS = {
    "bo": "#1f77b4",
    "br": "#2ca02c",
    "co": "#ff7f0e",
    "ec": "#9467bd",
    "pe": "#d62728",
}


class Studio:
    """Thin wrapper around a requests session with the Studio API key."""

    def __init__(self) -> None:
        """Create the session using STUDIO_API_KEY."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {os.environ['STUDIO_API_KEY']}",
                "Accept": "application/json",
            }
        )

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Issue a request, retrying server errors, and return the JSON body."""
        kwargs.setdefault("timeout", TIMEOUT)
        for attempt in range(MAX_RETRIES):
            resp = self.session.request(method, BASE_URL + path, **kwargs)
            if resp.status_code < 500 or attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)
        if resp.status_code != 200:
            raise ValueError(f"{method} {path} failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Get the project record, including its settings."""
        return self.request("GET", f"/projects/{project_id}")["records"][0]

    def search_all(self, kind: str, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Page through a */search endpoint."""
        records: list[dict[str, Any]] = []
        while True:
            page = self.request(
                "POST",
                f"/{kind}/search",
                json={**query, "limit": 1000, "offset": len(records)},
            )["records"]
            if not page:
                return records
            records.extend(page)

    def ensure_labelset_field(
        self, project_id: str, field_name: str, label_colors: dict[str, str]
    ) -> tuple[str, dict[str, str]]:
        """Ensure a labelset metadata field with these labels exists in the project.

        Returns:
            (metadata_field_id, {label_name: label_id})
        """
        settings = self.get_project(project_id)["settings"]
        settings_id = settings["id"]
        field = next(
            (
                f
                for f in settings["annotation_metadata_fields"]
                if f["name"] == field_name
            ),
            None,
        )
        if field is None:
            labelset = next(
                (ls for ls in settings["labelsets"] if ls["name"] == field_name), None
            )
            if labelset is None:
                print(f"creating labelset {field_name}")
                labelset = self.request(
                    "POST",
                    "/labelsets",
                    json={
                        "name": field_name,
                        "display_name": field_name,
                        "project_settings_id": settings_id,
                    },
                )["records"][0]
            labelset_id = labelset["id"]
        else:
            if field["data_type"] != "labelset":
                raise ValueError(f"field {field_name} exists but is not a labelset")
            labelset_id = field["labelset_id"]

        label_ids = {
            label["name"]: label["id"]
            for label in settings["labels"]
            if label["labelset_id"] == labelset_id
        }
        missing = [name for name in label_colors if name not in label_ids]
        if missing:
            print(f"creating labels {missing} in {field_name}")
            records = self.request(
                "POST",
                "/labels",
                json=[
                    # Studio shows display_name in the annotation UI, so it must be
                    # set or the options appear as blank color swatches.
                    {
                        "name": name,
                        "display_name": name,
                        "color": label_colors[name],
                        "labelset_id": labelset_id,
                    }
                    for name in missing
                ],
            )["records"]
            for record in records:
                label_ids[record["name"]] = record["id"]

        if field is None:
            print(f"creating metadata field {field_name}")
            field = self.request(
                "POST",
                "/annotation_metadata_fields",
                json={
                    "name": field_name,
                    "display_name": field_name,
                    "data_type": "labelset",
                    "project_settings_id": settings_id,
                    "labelset_id": labelset_id,
                    "required": False,
                },
            )["records"][0]
        return field["id"], label_ids
