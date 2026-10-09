"""Client for the OlmoEarth Studio API."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

BASE_URL = "https://olmoearth.allenai.org/api/v1"


class StudioClient:
    """Minimal client for Studio projects, tasks, annotations, and labels."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        page_size: int = 1000,
        timeout: float = 30,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Studio bearer token. Defaults to ``STUDIO_API_KEY``.
            base_url: Studio API v1 base URL.
            session: Optional requests session, primarily for tests.
            page_size: Number of records requested per search page.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of attempts per request. Connection errors
                and 5xx responses are retried with exponential backoff.
            retry_backoff: Base delay in seconds between attempts; attempt ``i``
                waits ``retry_backoff * 2**i``.
        """
        self.api_key = api_key or os.environ["STUDIO_API_KEY"]
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.page_size = page_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @property
    def headers(self) -> dict[str, str]:
        """Return authorization headers for Studio requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Send a request, retrying connection errors and 5xx responses.

        Args:
            method: HTTP method.
            path: Path relative to the base URL, e.g. ``tasks/search``.
            kwargs: Additional arguments passed to ``requests.Session.request``.

        Returns:
            The successful response.

        Raises:
            requests.HTTPError: if the final response has a 4xx or 5xx status. The
                message includes the response body.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method, url, headers=self.headers, timeout=self.timeout, **kwargs
                )
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
                continue
            if response.status_code < 500 or attempt == self.max_retries - 1:
                break
            time.sleep(self.retry_backoff * (2**attempt))

        if response.status_code >= 400:
            raise requests.HTTPError(
                f"Studio API {method} {path} returned {response.status_code}: "
                f"{response.text}",
                response=response,
            )
        return response

    def _first_record(self, response: requests.Response) -> dict[str, Any]:
        return response.json()["records"][0]

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Fetch one project definition, including its annotation template."""
        records = self._request("GET", f"projects/{project_id}").json()["records"]
        if len(records) != 1:
            raise ValueError(
                f"expected one Studio project for {project_id}, got {len(records)}"
            )
        return records[0]

    def search_all(
        self,
        resource: str,
        project_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all records for a project from a paginated search endpoint.

        Args:
            resource: The resource to search, e.g. ``tasks`` or ``annotations``.
            project_id: The Studio project ID.
            filters: Additional search filters, e.g.
                ``{"status": {"inc": ["reviewed"]}}``.

        Returns:
            All matching records.
        """
        offset = 0
        records: list[dict[str, Any]] = []
        while True:
            body = {
                **(filters or {}),
                "project_id": {"eq": project_id},
                "limit": self.page_size,
                "offset": offset,
            }
            page = self._request("POST", f"{resource}/search", json=body).json()[
                "records"
            ]
            if not page:
                break
            records.extend(page)
            offset += len(page)
        return records

    def get_tasks(
        self, project_id: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all tasks in a project matching the optional filters."""
        return self.search_all("tasks", project_id, filters)

    def get_annotations(
        self, project_id: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all annotations in a project matching the optional filters."""
        return self.search_all("annotations", project_id, filters)

    def get_project_data(self, project_id: str) -> dict[str, Any]:
        """Fetch the project definition and all records needed for inventory."""
        return {
            "project": self.get_project(project_id),
            "tasks": self.get_tasks(project_id),
            "annotations": self.get_annotations(project_id),
        }

    def create_task(
        self,
        project_id: str,
        name: str,
        geom_wkt: str,
        start_time: str,
        end_time: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Create a task and return its ID."""
        body: dict[str, Any] = {
            "name": name,
            "project_id": project_id,
            "geom": geom_wkt,
            "start_time": start_time,
        }
        if end_time is not None:
            body["end_time"] = end_time
        if attributes is not None:
            body["attributes"] = attributes
        return self._first_record(self._request("POST", "tasks", json=body))["id"]

    def update_task(self, task_id: str, body: dict[str, Any]) -> None:
        """Update a task with the given fields."""
        self._request("PUT", f"tasks/{task_id}", json=body)

    def delete_task(self, task_id: str) -> None:
        """Delete a task."""
        self._request("DELETE", f"tasks/{task_id}")

    def create_annotation(
        self, task_id: str, geom_wkt: str, status: str = "pending"
    ) -> str:
        """Create an annotation with no metadata values and return its ID."""
        body = {"status": status, "geom": geom_wkt, "task_id": task_id}
        return self._first_record(self._request("POST", "annotations", json=body))["id"]

    def upload_image(
        self,
        task_id: str,
        source: str,
        start_time: str,
        end_time: str,
        image_bytes: bytes,
        filename: str,
        attributes: dict[str, Any] | None = None,
        content_type: str = "image/tiff",
    ) -> str:
        """Upload an image attached to a task and return its ID."""
        data = {
            "task_id": task_id,
            "source": source,
            "start_time": start_time,
            "end_time": end_time,
            "attributes": json.dumps(attributes or {}),
        }
        files = {"image_file": (filename, image_bytes, content_type)}
        response = self._request("POST", "images/upload", data=data, files=files)
        return self._first_record(response)["id"]

    def create_labelset(self, name: str, template_id: str) -> dict[str, Any]:
        """Create a labelset in a project template and return its record."""
        body = {"name": name, "template_id": template_id}
        return self._first_record(self._request("POST", "labelsets", json=body))

    def create_annotation_metadata_field(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create an annotation metadata field and return its record."""
        return self._first_record(
            self._request("POST", "annotation_metadata_fields", json=body)
        )

    def create_labels(self, labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Create labels in batch and return their records."""
        return self._request("POST", "labels", json=labels).json()["records"]
