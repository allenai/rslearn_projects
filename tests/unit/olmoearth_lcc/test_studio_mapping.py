import copy
import itertools
from typing import Any

import pytest
import shapely
import shapely.wkt

from rslp.olmoearth_lcc.studio.mapping import (
    entry_to_task_payload,
    normalize_time,
    point_to_annotation_payload,
    task_and_annotations_to_entry,
    task_index,
    task_name,
    unmapped_point_keys,
    window_polygon,
)
from rslp.olmoearth_lcc.studio.schema import (
    ALL_FIELDS,
    NEGATIVE,
    POSITIVE,
    POST_CHANGE_CATEGORIES,
    ensure_template,
    is_labelset_field,
    labelset_owner,
    required_labels,
)

PROJECTION = {"crs": "EPSG:32651", "x_resolution": 10, "y_resolution": -10}
BOUNDS = [28000, -165000, 28128, -164872]


def make_entry(**overrides: Any) -> dict[str, Any]:
    entry = {
        "projection": PROJECTION,
        "bounds": BOUNDS,
        "window_name": "example_window",
        "group": "default",
        "time_range": ["2017-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"],
        "description": "New subdivision replacing forest",
        "anchor_date": "2021-06-01",
        "positive_points": [
            {
                "lon": 121.50012345678901,
                "lat": 14.600987654321,
                "pre_change": "2020-01-15",
                "first_date_change_noticeable": "2020-04-27",
                "post_change": "2020-07-15",
                "stop_date": "2021-06-01",
                "pre_category": "tree",
                "post_category": "urban/built-up",
                "post_change_category": "new_building",
            },
            {"lon": 121.502, "lat": 14.602, "pre_category": "agriculture"},
        ],
        "negative_points": [{"lon": 121.51, "lat": 14.61}],
    }
    entry.update(overrides)
    return entry


class FakeStudio:
    """Turns write payloads into the records Studio's search endpoints return."""

    def __init__(self, entries: list[dict[str, Any]]):
        self.field_ids = {name: f"field-{name}" for name in ALL_FIELDS}
        labelset_ids = {
            name: {label: f"label-{name}-{label}" for label in labels}
            for name, labels in required_labels(entries).items()
        }
        # Fields that share a labelset get the same label ids.
        self.label_ids = {
            name: labelset_ids[labelset_owner(name)]
            for name in ALL_FIELDS
            if is_labelset_field(name)
        }
        self.field_names = {v: k for k, v in self.field_ids.items()}
        self.label_names = {
            label_id: label
            for labels in self.label_ids.values()
            for label, label_id in labels.items()
        }
        self._clock = itertools.count()

    def _creation_time(self) -> str:
        return f"2026-09-28T00:00:00.{next(self._clock):06d}+00:00"

    def task_record(self, index: int, entry: dict[str, Any]) -> dict[str, Any]:
        payload = entry_to_task_payload(index, entry)
        return {
            "id": f"task-{index}",
            "name": payload["name"],
            "attributes": payload["attributes"],
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "creation_time": self._creation_time(),
            "geom_wkt": payload["geom"],
        }

    def annotation_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = []
        for value in payload["metadata_values"]:
            name = self.field_names[value["metadata_field_id"]]
            label_id = value.get("label_id")
            values.append(
                {
                    "metadata_field_id": value["metadata_field_id"],
                    "name": name,
                    "value": value["value"],
                    "label_id": label_id,
                    "label_name": self.label_names.get(label_id) if label_id else None,
                }
            )
        creation_time = self._creation_time()
        return {
            "id": f"annotation-{creation_time}",
            "task_id": payload["task_id"],
            "geom_wkt": shapely.wkt.loads(payload["geom"]).wkt,
            "metadata_values": values,
            "creation_time": creation_time,
        }

    def round_trip(self, index: int, entry: dict[str, Any]) -> dict[str, Any]:
        task = self.task_record(index, entry)
        annotations = []
        for kind, key in ((POSITIVE, "positive_points"), (NEGATIVE, "negative_points")):
            for point in entry.get(key, []):
                payload = point_to_annotation_payload(
                    point,
                    kind,
                    self.field_ids,
                    self.label_ids,
                    task["id"],
                    entry.get("time_range"),
                )
                annotations.append(self.annotation_record(payload))
        return task_and_annotations_to_entry(
            task, annotations, self.field_names, self.label_names
        )


def assert_entries_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("positive_points", "negative_points"):
        assert len(actual[key]) == len(expected[key])
        for got, want in zip(actual[key], expected[key]):
            assert got["lon"] == pytest.approx(want["lon"], abs=1e-12)
            assert got["lat"] == pytest.approx(want["lat"], abs=1e-12)
            assert {k: v for k, v in got.items() if k not in ("lon", "lat")} == {
                k: v for k, v in want.items() if k not in ("lon", "lat")
            }
    rest = {k: v for k, v in actual.items() if not k.endswith("_points")}
    assert rest == {k: v for k, v in expected.items() if not k.endswith("_points")}


def test_round_trip() -> None:
    entry = make_entry(metadata={"source": "tiles.csv", "tile_year": 2023})
    studio = FakeStudio([entry])
    assert_entries_equal(studio.round_trip(3, entry), entry)


def test_round_trip_keeps_date_only_time_range_verbatim() -> None:
    entry = make_entry(time_range=["2020-10-09", "2026-10-09"])
    payload = entry_to_task_payload(0, entry)
    assert payload["start_time"] == "2020-10-09T00:00:00+00:00"
    assert payload["end_time"] == "2026-10-09T00:00:00+00:00"
    assert FakeStudio([entry]).round_trip(0, entry)["time_range"] == entry["time_range"]


def test_legacy_value_gets_a_label_and_survives() -> None:
    entry = make_entry(
        positive_points=[
            {
                "lon": 1.0,
                "lat": 2.0,
                "pre_category": "nodata",
                "fine_change_category": "repainted_roof",
            }
        ]
    )
    labels = required_labels([entry])
    assert labels["pre_category"][-1] == "nodata"
    assert "nodata" not in labels["post_category"]
    assert_entries_equal(FakeStudio([entry]).round_trip(0, entry), entry)


def test_missing_and_empty_fields_are_omitted() -> None:
    entry = make_entry(
        description="",
        positive_points=[
            {
                "lon": 1.0,
                "lat": 2.0,
                "pre_change": "2020-01-01",
                "first_date_change_noticeable": "",
                "same_change_category": "",
            }
        ],
    )
    studio = FakeStudio([entry])
    payload = point_to_annotation_payload(
        entry["positive_points"][0],
        POSITIVE,
        studio.field_ids,
        studio.label_ids,
        "task-0",
        None,
    )
    assert [v["metadata_field_id"] for v in payload["metadata_values"]] == [
        "field-point_type",
        "field-pre_change",
    ]
    assert "start_time" not in payload

    expected = copy.deepcopy(entry)
    del expected["description"]
    expected["positive_points"] = [{"lon": 1.0, "lat": 2.0, "pre_change": "2020-01-01"}]
    assert_entries_equal(studio.round_trip(0, entry), expected)


def test_entry_with_no_points_and_no_time_range() -> None:
    entry = make_entry(positive_points=[], negative_points=[])
    del entry["time_range"]
    payload = entry_to_task_payload(7, entry)
    assert "start_time" not in payload and "end_time" not in payload
    assert_entries_equal(FakeStudio([entry]).round_trip(7, entry), entry)


def test_negative_points_carry_only_point_type_and_terminal_state() -> None:
    studio = FakeStudio([])
    payload = point_to_annotation_payload(
        {"lon": 1.0, "lat": 2.0},
        NEGATIVE,
        studio.field_ids,
        studio.label_ids,
        "task-0",
        None,
    )
    assert payload["metadata_values"] == [
        {
            "metadata_field_id": "field-point_type",
            "value": None,
            "label_id": "label-point_type-negative",
        }
    ]
    assert unmapped_point_keys({"lon": 1, "lat": 2, "pre_change": "x"}, NEGATIVE) == [
        "pre_change"
    ]
    assert not unmapped_point_keys(
        {"lon": 1, "lat": 2, "terminal_state": "road"}, NEGATIVE
    )


def test_terminal_state_round_trips_on_both_kinds() -> None:
    entry = make_entry(
        positive_points=[
            {
                "lon": 1.0,
                "lat": 2.0,
                "post_change_category": "new_road",
                "terminal_state": "road",
            }
        ],
        negative_points=[
            {"lon": 3.0, "lat": 4.0, "terminal_state": "tree"},
            {"lon": 5.0, "lat": 6.0},
        ],
    )
    studio = FakeStudio([entry])
    assert_entries_equal(studio.round_trip(0, entry), entry)


def test_points_keep_creation_order_within_kind() -> None:
    entry = make_entry()
    studio = FakeStudio([entry])
    task = studio.task_record(0, entry)
    # A negative created first, then a positive added later in the lab.
    records = [
        studio.annotation_record(
            point_to_annotation_payload(
                p, kind, studio.field_ids, studio.label_ids, task["id"], None
            )
        )
        for kind, p in [
            (NEGATIVE, {"lon": 1.0, "lat": 1.0}),
            (POSITIVE, {"lon": 2.0, "lat": 2.0}),
            (NEGATIVE, {"lon": 3.0, "lat": 3.0}),
            (POSITIVE, {"lon": 4.0, "lat": 4.0}),
        ]
    ]
    result = task_and_annotations_to_entry(
        task, list(reversed(records)), studio.field_names, studio.label_names
    )
    assert [p["lon"] for p in result["positive_points"]] == [2.0, 4.0]
    assert [p["lon"] for p in result["negative_points"]] == [1.0, 3.0]


def test_task_name_encodes_json_index() -> None:
    assert task_name(42, "example_window") == "00042 example_window"
    assert task_index("00042 example_window") == 42
    assert sorted([task_name(i, "w") for i in (10, 2, 100)]) == [
        task_name(i, "w") for i in (2, 10, 100)
    ]


def test_window_polygon_has_four_reprojected_corners() -> None:
    polygon = window_polygon(PROJECTION, BOUNDS)
    assert len(polygon.exterior.coords) == 5
    # UTM 51N easting 280000..281280 m, northing 1648720..1650000 m.
    west, south, east, north = polygon.bounds
    assert 120.9 < west < east < 121.0
    assert 14.9 < south < north < 14.95
    # UTM grid north is rotated from true north, so the corners are not a bbox.
    assert not polygon.equals(shapely.box(*polygon.bounds))


def test_normalize_time_accepts_dates_and_offsets() -> None:
    assert normalize_time("2020-10-09") == "2020-10-09T00:00:00+00:00"
    assert normalize_time("2017-01-01T00:00:00+00:00") == "2017-01-01T00:00:00+00:00"


def test_every_field_is_labelset_or_date() -> None:
    assert [name for name in ALL_FIELDS if not is_labelset_field(name)] == [
        "pre_change",
        "first_date_change_noticeable",
        "transition_date",
        "transition_stop_date",
        "post_change",
        "stop_date",
    ]


def test_transition_fields_round_trip() -> None:
    point = {
        "lon": 1.0,
        "lat": 2.0,
        "pre_change": "2020-01-15",
        "first_date_change_noticeable": "2020-02-01",
        "transition_date": "2020-04-01",
        "transition_stop_date": "2020-06-01",
        "post_change": "2020-09-01",
        "pre_category": "mangrove",
        "transition_category": "bare",
        "post_category": "agriculture",
        "pre_change_category": "mangrove_loss",
        "post_change_category": "new_agriculture",
        "transition_change_category": "site_clearing",
    }
    entry = make_entry(positive_points=[point])
    studio = FakeStudio([entry])
    # transition_change_category labels come from post_change_category's labelset.
    assert studio.label_ids["transition_change_category"]["site_clearing"] == (
        "label-post_change_category-site_clearing"
    )
    assert_entries_equal(studio.round_trip(0, entry), entry)
    assert not unmapped_point_keys(point, POSITIVE)


def test_shared_labelset_collects_unseen_values() -> None:
    entry = make_entry(
        positive_points=[
            {"lon": 1.0, "lat": 2.0, "transition_change_category": "legacy_value"}
        ]
    )
    labels = required_labels([entry])
    assert "transition_change_category" not in labels
    assert labels["post_change_category"][-1] == "legacy_value"
    assert_entries_equal(FakeStudio([entry]).round_trip(0, entry), entry)


class FakeStudioClient:
    """In-memory project settings for ensure_template."""

    def __init__(self, fields: list[dict[str, Any]], labelsets: list[dict[str, Any]]):
        self.settings: dict[str, Any] = {
            "id": "settings",
            "name": "settings",
            "label_geometry": "point",
            "annotation_metadata_fields": fields,
            "labelsets": labelsets,
        }
        self._ids = itertools.count()
        self.label_posts: list[tuple[str, list[str]]] = []

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids)}"

    def get_project(self, project_id: str) -> dict[str, Any]:
        return {"id": project_id, "settings": copy.deepcopy(self.settings)}

    def create(self, path: str, body: Any) -> dict[str, Any]:
        return self.request("POST", path, json=body)["records"][0]

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        body: Any = kwargs["json"]
        if path == "/labelsets":
            labelset = {
                "id": self._new_id("labelset"),
                "name": body["name"],
                "labels": [
                    {"id": self._new_id("label"), "name": label["name"]}
                    for label in body["labels"]
                ],
            }
            self.settings["labelsets"].append(labelset)
            return {"records": [labelset]}
        if path == "/annotation_metadata_fields":
            field = {**body, "id": self._new_id("field")}
            self.settings["annotation_metadata_fields"].append(field)
            return {"records": [field]}
        if path == "/labels":
            labelset_id = body[0]["labelset_id"]
            self.label_posts.append((labelset_id, [label["name"] for label in body]))
            (labelset,) = (
                ls for ls in self.settings["labelsets"] if ls["id"] == labelset_id
            )
            records = [
                {"id": self._new_id("label"), "name": label["name"]} for label in body
            ]
            labelset["labels"].extend(records)
            return {"records": records}
        raise AssertionError(f"unexpected request {method} {path}")


def test_ensure_template_shares_post_change_labelset() -> None:
    client = FakeStudioClient([], [])
    entry = make_entry(
        positive_points=[
            {"lon": 1.0, "lat": 2.0, "transition_change_category": "legacy_value"}
        ]
    )
    schema = ensure_template(client, "project", [entry])  # type: ignore[arg-type]
    fields = {
        field["name"]: field for field in client.settings["annotation_metadata_fields"]
    }
    assert set(fields) == set(ALL_FIELDS)
    assert (
        fields["transition_change_category"]["labelset_id"]
        == fields["post_change_category"]["labelset_id"]
    )
    assert (
        fields["transition_category"]["labelset_id"]
        != fields["post_category"]["labelset_id"]
    )
    assert "transition_change_category" not in {
        labelset["name"] for labelset in client.settings["labelsets"]
    }
    # The unseen value is part of the shared labelset from creation, so nothing is
    # added afterwards, and both fields resolve it.
    assert client.label_posts == []
    assert (
        schema.label_ids["transition_change_category"]["legacy_value"]
        == schema.label_ids["post_change_category"]["legacy_value"]
    )

    # Re-running with another unseen value adds it once, to the shared labelset.
    entry["positive_points"][0]["transition_change_category"] = "another_value"
    schema = ensure_template(client, "project", [entry])  # type: ignore[arg-type]
    assert client.label_posts == [
        (fields["post_change_category"]["labelset_id"], ["another_value"])
    ]
    assert "another_value" in schema.label_ids["transition_change_category"]
    assert set(POST_CHANGE_CATEGORIES) <= set(
        schema.label_ids["transition_change_category"]
    )
