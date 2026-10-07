from typing import Any

import pytest
import requests

from rslp.utils.studio import StudioClient


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self.payload = payload
        self.text = text

    def json(self) -> Any:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(session: FakeSession) -> StudioClient:
    return StudioClient(
        api_key="key",
        base_url="https://studio.test/api/v1/",
        session=session,  # type: ignore[arg-type]
        page_size=2,
        retry_backoff=0,
    )


def test_search_all_paginates_and_merges_filters() -> None:
    a = {"id": "a", "creation_time": "t1"}
    b = {"id": "b", "creation_time": "t2"}
    c = {"id": "c", "creation_time": "t2"}
    d = {"id": "d", "creation_time": "t3"}
    session = FakeSession(
        [
            FakeResponse(200, {"records": [a, b]}),
            # The cursor is inclusive, so b is returned again.
            FakeResponse(200, {"records": [b, c]}),
            FakeResponse(200, {"records": [c, d]}),
            FakeResponse(200, {"records": [d]}),
        ]
    )
    client = make_client(session)

    tasks = client.get_tasks("proj", filters={"status": {"inc": ["reviewed"]}})

    assert [task["id"] for task in tasks] == ["a", "b", "c", "d"]
    assert [call["json"].get("creation_time") for call in session.calls] == [
        None,
        {"gte": "t2"},
        {"gte": "t2"},
        {"gte": "t3"},
    ]
    first = session.calls[0]
    assert first["method"] == "POST"
    assert first["url"] == "https://studio.test/api/v1/tasks/search"
    assert first["headers"]["Authorization"] == "Bearer key"
    assert first["json"] == {
        "status": {"inc": ["reviewed"]},
        "project_id": {"eq": "proj"},
        "sort_by": "creation_time",
        "sort_direction": "asc",
        "limit": 2,
    }


def test_search_all_raises_when_cursor_cannot_advance() -> None:
    b = {"id": "b", "creation_time": "t2"}
    c = {"id": "c", "creation_time": "t2"}
    session = FakeSession(
        [
            FakeResponse(200, {"records": [b, c]}),
            FakeResponse(200, {"records": [b, c]}),
        ]
    )
    client = make_client(session)

    # b and c share creation_time t2 and fill a whole page, so paging cannot advance.
    with pytest.raises(RuntimeError, match="share creation_time t2"):
        client.get_tasks("proj")


def test_search_all_rejects_paging_filters() -> None:
    client = make_client(FakeSession([]))

    with pytest.raises(ValueError, match="creation_time"):
        client.get_tasks("proj", filters={"creation_time": {"gte": "t"}})


def test_retries_server_errors_and_connection_errors() -> None:
    session = FakeSession(
        [
            FakeResponse(503, text="unavailable"),
            requests.ConnectionError("reset"),
            FakeResponse(200, {"records": [{"id": "p", "template": {}}]}),
        ]
    )
    client = make_client(session)

    assert client.get_project("p")["id"] == "p"
    assert len(session.calls) == 3


def test_raises_after_last_retry() -> None:
    session = FakeSession([FakeResponse(500, text="boom")] * 3)
    client = make_client(session)

    with pytest.raises(requests.HTTPError, match="boom"):
        client.delete_task("t")
    assert len(session.calls) == 3


def test_client_error_is_not_retried_and_includes_body() -> None:
    session = FakeSession([FakeResponse(400, text="bad geom")])
    client = make_client(session)

    with pytest.raises(requests.HTTPError, match="returned 400: bad geom"):
        client.create_annotation("t", "POINT (0 0)")
    assert len(session.calls) == 1


def test_create_task_returns_id_and_omits_unset_fields() -> None:
    session = FakeSession([FakeResponse(200, {"records": [{"id": "task-1"}]})])
    client = make_client(session)

    task_id = client.create_task("proj", "name", "POINT (0 0)", "2024-01-01T00:00:00")

    assert task_id == "task-1"
    assert session.calls[0]["json"] == {
        "name": "name",
        "project_id": "proj",
        "geom": "POINT (0 0)",
        "start_time": "2024-01-01T00:00:00",
    }


def test_upload_image_sends_multipart() -> None:
    session = FakeSession([FakeResponse(200, {"records": [{"id": "img-1"}]})])
    client = make_client(session)

    image_id = client.upload_image(
        "t", "sentinel2", "s", "e", b"tiff", "a.tif", attributes={"group_idx": 1}
    )

    assert image_id == "img-1"
    call = session.calls[0]
    assert call["url"] == "https://studio.test/api/v1/images/upload"
    assert call["data"]["attributes"] == '{"group_idx": 1}'
    assert call["files"] == {"image_file": ("a.tif", b"tiff", "image/tiff")}
    assert "json" not in call
