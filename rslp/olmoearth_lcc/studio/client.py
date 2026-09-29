"""Minimal OlmoEarth Studio API client used by the upload and download scripts."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import requests

DEFAULT_BASE_URL = "https://olmoearth.allenai.org/api/v1"
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
# Studio search endpoints cap limit (and offset) at 10000.
MAX_PAGE_SIZE = 10000


class StudioClient:
    """Authenticated requests against the Studio API, with retries on 5xx."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        """Create a client.

        Args:
            base_url: API root including /api/v1. Defaults to $STUDIO_API_URL, else
                production.
            api_key: Studio API key. Defaults to $STUDIO_API_KEY.
        """
        self.base_url = (
            base_url or os.environ.get("STUDIO_API_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        key = api_key or os.environ["STUDIO_API_KEY"]
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send a request and return the decoded JSON body, raising on errors."""
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        url = f"{self.base_url}{path}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.ConnectionError:
                if attempt == MAX_RETRIES - 1:
                    raise
            else:
                if resp.status_code < 500 or attempt == MAX_RETRIES - 1:
                    break
            time.sleep(RETRY_BACKOFF * (2**attempt))
        if not resp.ok:
            raise requests.HTTPError(
                f"{method} {path} failed with {resp.status_code}: {resp.text}",
                response=resp,
            )
        return resp.json()

    def create(self, path: str, body: Any) -> dict[str, Any]:
        """POST a create request and return the single created record."""
        return self.request("POST", path, json=body)["records"][0]

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Fetch a project, including its settings (fields, labelsets, labels)."""
        records = self.request("GET", f"/projects/{project_id}")["records"]
        if len(records) != 1:
            raise ValueError(f"expected one project for {project_id}, got {records}")
        return records[0]

    def search_all(
        self, path: str, query: dict[str, Any], page_size: int = 5000
    ) -> Iterator[dict[str, Any]]:
        """Yield every record matching query, oldest first.

        Pages with a creation_time cursor rather than offsets, since Studio caps
        offset at 10000.
        """
        seen: set[str] = set()
        cursor: str | None = None
        limit = min(page_size, MAX_PAGE_SIZE)
        while True:
            body = {
                **query,
                "sort_by": "creation_time",
                "sort_direction": "asc",
                "limit": limit,
            }
            if cursor is not None:
                body["creation_time"] = {"gte": cursor}
            records = self.request("POST", path, json=body)["records"]
            new = [record for record in records if record["id"] not in seen]
            for record in new:
                seen.add(record["id"])
                yield record
            if len(records) < limit:
                return
            if not new:
                raise RuntimeError(
                    f"more than {limit} records share creation_time {cursor}"
                )
            cursor = records[-1]["creation_time"]
