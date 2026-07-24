from typing import Any

from rslp.forest_loss_driver.monocrop_classifier.studio import StudioClient


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append({"url": url, **kwargs})
        offset = kwargs["json"]["offset"]
        pages = {
            0: [{"id": "a"}, {"id": "b"}],
            2: [{"id": "c"}],
            3: [],
        }
        return FakeResponse({"records": pages[offset]})


def test_search_all_paginates_until_empty() -> None:
    session = FakeSession()
    client = StudioClient(
        "secret",
        base_url="https://studio.example/api/v1/",
        session=session,  # type: ignore[arg-type]
        page_size=2,
    )

    records = client.search_all("annotations", "project-id")

    assert [record["id"] for record in records] == ["a", "b", "c"]
    assert [call["json"]["offset"] for call in session.posts] == [0, 2, 3]
    assert all(call["json"]["limit"] == 2 for call in session.posts)
    assert all(
        call["json"]["project_id"] == {"eq": "project-id"} for call in session.posts
    )
