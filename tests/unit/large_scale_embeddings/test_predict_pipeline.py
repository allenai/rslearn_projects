"""Unit tests for rslp.large_scale_embeddings.predict_pipeline."""

import pathlib
from datetime import UTC, datetime

from rslearn.dataset.window import WindowLayerData
from upath import UPath

from rslp.large_scale_embeddings.predict_pipeline import (
    _collect_provenance,
    get_provenance_fname,
)

MARKER_NAME = "EPSG:32628_45056_-720896.json"


def test_get_provenance_fname_is_marker_sibling() -> None:
    """Provenance lands beside the marker directory, under the same basename."""
    marker_fname = UPath(f"gs://bucket/archive/completed_2025/{MARKER_NAME}")
    assert get_provenance_fname(marker_fname) == UPath(
        f"gs://bucket/archive/provenance_2025/{MARKER_NAME}"
    )


def test_get_provenance_fname_unrecognized_directory() -> None:
    """A marker directory not named completed_* still gets a distinct sibling."""
    marker_fname = UPath(f"gs://bucket/archive/markers/{MARKER_NAME}")
    assert get_provenance_fname(marker_fname) == UPath(
        f"gs://bucket/archive/markers_provenance/{MARKER_NAME}"
    )


class FakeWindow:
    """A window that reports fixed layer datas."""

    def __init__(self, name: str, layer_datas: dict[str, WindowLayerData]) -> None:
        """Initialize a new FakeWindow."""
        self.name = name
        self._layer_datas = layer_datas

    def load_layer_datas(self) -> dict[str, WindowLayerData]:
        """Return the fixed layer datas."""
        return self._layer_datas


def test_collect_provenance_records_items_and_periods() -> None:
    """Each mosaic's source items and requested period are recorded per layer."""
    period = (datetime(2025, 1, 6, tzinfo=UTC), datetime(2025, 2, 5, tzinfo=UTC))
    window = FakeWindow(
        "22_-176",
        {
            "sentinel2_l2a": WindowLayerData(
                layer_name="sentinel2_l2a",
                serialized_item_groups=[[{"name": "S2A_scene", "cloud_cover": 1.5}]],
                group_time_ranges=[period],
            ),
            # The output layer is this pipeline's own product, not a source.
            "output": WindowLayerData(
                layer_name="output",
                serialized_item_groups=[[{"name": "irrelevant"}]],
            ),
        },
    )

    provenance = _collect_provenance([window])

    assert set(provenance) == {"22_-176"}
    assert set(provenance["22_-176"]) == {"sentinel2_l2a"}
    assert provenance["22_-176"]["sentinel2_l2a"] == [
        {
            "time_range": ["2025-01-06T00:00:00+00:00", "2025-02-05T00:00:00+00:00"],
            "items": [{"name": "S2A_scene", "cloud_cover": 1.5}],
        }
    ]


def test_collect_provenance_without_group_time_ranges() -> None:
    """Layers prepared without per-group periods still record their items."""
    window = FakeWindow(
        "0_0",
        {
            "landsat": WindowLayerData(
                layer_name="landsat",
                serialized_item_groups=[[{"name": "LC09_scene"}]],
            )
        },
    )

    provenance = _collect_provenance([window])

    assert provenance["0_0"]["landsat"] == [
        {"time_range": None, "items": [{"name": "LC09_scene"}]}
    ]


def test_collect_provenance_window_without_items(tmp_path: pathlib.Path) -> None:
    """A window whose prepare found nothing contributes an empty entry, not an error."""
    assert _collect_provenance([FakeWindow("1_1", {})]) == {"1_1": {}}
