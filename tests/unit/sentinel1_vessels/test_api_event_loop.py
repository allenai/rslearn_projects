"""Unit tests for the Sentinel-1 API request handling."""

import inspect

from rslp.sentinel1_vessels import api_main


def test_detections_endpoint_is_not_async() -> None:
    """An async handler would run the prediction on the event loop.

    That blocks every other request for the length of a run, including the health probe,
    until Kubernetes gives up on the pod and sends SIGTERM. FastAPI runs a sync handler
    in a worker thread instead.

    Asserted directly rather than by driving the app, because TestClient gives each
    request its own event loop and so cannot reproduce the blocking.
    """
    assert not inspect.iscoroutinefunction(api_main.get_detections)
