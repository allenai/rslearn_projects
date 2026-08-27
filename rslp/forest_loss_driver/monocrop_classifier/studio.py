"""Read-only access to monocrop annotations in OlmoEarth Studio."""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = "https://olmoearth.allenai.org/api/v1"
DEFAULT_PROJECT_IDS = (
    "30b9bbb2-1ac9-4cf6-baa5-1b77ce79881c",
    "dd3e9ecb-4060-4a9b-af71-05b4bf8ad747",
    "8188a029-fd50-4670-a6f2-243afc3e1b83",
)


class StudioClient:
    """Minimal read-only client for Studio projects, tasks, and annotations."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        page_size: int = 1000,
        timeout: float = 30,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: Studio bearer token. Defaults to ``STUDIO_API_KEY``.
            base_url: Studio API v1 base URL.
            session: Optional requests session, primarily for tests.
            page_size: Number of records requested per search page.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.environ["STUDIO_API_KEY"]
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.page_size = page_size
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        """Return authorization headers for Studio requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def get_project(self, project_id: str) -> dict[str, Any]:
        """Fetch one project definition, including its annotation template."""
        response = self.session.get(
            f"{self.base_url}/projects/{project_id}",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        records = response.json()["records"]
        if len(records) != 1:
            raise ValueError(
                f"expected one Studio project for {project_id}, got {len(records)}"
            )
        return records[0]

    def search_all(self, resource: str, project_id: str) -> list[dict[str, Any]]:
        """Fetch all records for a project from a paginated search endpoint."""
        offset = 0
        records: list[dict[str, Any]] = []
        while True:
            response = self.session.post(
                f"{self.base_url}/{resource}/search",
                json={
                    "project_id": {"eq": project_id},
                    "limit": self.page_size,
                    "offset": offset,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            page = response.json()["records"]
            if not page:
                break
            records.extend(page)
            offset += len(page)
        return records

    def get_tasks(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch all tasks in a project."""
        return self.search_all("tasks", project_id)

    def get_annotations(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch all annotations in a project."""
        return self.search_all("annotations", project_id)

    def get_project_data(self, project_id: str) -> dict[str, Any]:
        """Fetch the project definition and all records needed for inventory."""
        return {
            "project": self.get_project(project_id),
            "tasks": self.get_tasks(project_id),
            "annotations": self.get_annotations(project_id),
        }
