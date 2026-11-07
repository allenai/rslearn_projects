"""The Studio project template for LCC annotations: field names and vocabularies.

The lab (olmoearth_studio/ui/labs/lcc-annotator/src/schema.ts) resolves the same
fields by name, and its FIELDS.md lists the labels each field offers, so keep them
in sync. The training classes these labels map onto are in
lcc_model/output_bands.py (mapped by lcc_model/prepare.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rslp.olmoearth_lcc.lcc_model.output_bands import LC_CLASS_NAMES
from rslp.utils.studio import BASE_URL, StudioClient

POINT_TYPE_FIELD = "point_type"
POSITIVE = "positive"
NEGATIVE = "negative"
POINT_TYPE_COLORS = {POSITIVE: "#2ca02c", NEGATIVE: "#d62728"}

DATE_FIELDS = (
    "pre_change",
    "first_date_change_noticeable",
    "transition_date",
    "post_change",
    "stop_date",
)

# The training land cover classes without nodata.
LAND_COVER_CATEGORIES = tuple(LC_CLASS_NAMES[1:])
PRE_CHANGE_CATEGORIES = (
    "deforestation",
    "mangrove_loss",
    "urban_erosion",
    "wetland_loss",
    "water_contract",
    "river_channel_migration",
    "river_contract",
    "reservoir_level_drop",
    "lake_level_drop",
    "water_contract_other",
    "removed_crop_structure",
)
POST_CHANGE_CATEGORIES = (
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_crop_field",
    "new_aquafarm",
    "site_clearing",
    "water_expand",
    "mining",
    "borrow_pit",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
    "coastal_erosion",
    "land_reclamation",
    "river_flooding",
    "flooding",
    "reservoir_level_rise",
    "lake_level_rise",
    "storm_damage",
    "vegetation_dieback",
    "agricultural_activity",
    "wildfire",
    "ice_motion",
    "river_channel_migration",
)
# Legacy: these are now post-change categories and the lab ignores this field, but
# it is kept so old values still round-trip.
SAME_CHANGE_CATEGORIES = (
    "agricultural_activity",
    "wildfire",
    "ice_motion",
    "flooding",
)
# Legacy: the lab only shows this field when a point already has a value.
FINE_CHANGE_CATEGORIES = (
    "new_solar_farm",
    "new_wind_turbine",
    "new_power_tower",
    "new_building",
    "new_road",
    "resurfaced_road",
    "repainted_roof",
    "tree_crops_harvested",
    "tree_crops_growth_gradual",
    "wetland_loss",
    "new_crop_field",
    "deforestation",
    "wildfire",
    "mining",
    "removed_building",
    "removed_road",
    "site_clearing",
    "crop_temporary_building_erected",
    "new_aquafarm",
    "new_offshore_infrastructure",
)

CATEGORY_FIELDS: dict[str, tuple[str, ...]] = {
    "pre_category": LAND_COVER_CATEGORIES,
    "transition_category": LAND_COVER_CATEGORIES,
    "post_category": LAND_COVER_CATEGORIES,
    "pre_change_category": PRE_CHANGE_CATEGORIES,
    "post_change_category": POST_CHANGE_CATEGORIES,
    "transition_change_category": POST_CHANGE_CATEGORIES,
    "same_change_category": SAME_CHANGE_CATEGORIES,
    "fine_change_category": FINE_CHANGE_CATEGORIES,
}

# Fields that use another field's labelset instead of their own: {field: owner}.
# The owner must come earlier in ALL_FIELDS.
SHARED_LABELSETS = {"transition_change_category": "post_change_category"}

# Positive point fields in v2 JSON order.
POSITIVE_POINT_FIELDS = DATE_FIELDS + tuple(CATEGORY_FIELDS)

FIELD_DISPLAY_NAMES = {
    POINT_TYPE_FIELD: "Point type",
    "pre_change": "Pre-change date",
    "first_date_change_noticeable": "First date change noticeable",
    "transition_date": "Transition date (optional)",
    "post_change": "Post-change date",
    "stop_date": "Stop date",
    "pre_category": "Pre land cover",
    "transition_category": "Transition category (optional)",
    "post_category": "Post land cover",
    "pre_change_category": "Pre-change category",
    "post_change_category": "Post-change category",
    "transition_change_category": "Transition change category (optional)",
    "same_change_category": "Same-change category (legacy)",
    "fine_change_category": "Fine change category (legacy)",
}

# All fields, in display order. point_type comes first so Studio's own map
# colours points by it.
ALL_FIELDS = (POINT_TYPE_FIELD,) + POSITIVE_POINT_FIELDS

LABEL_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)


def is_labelset_field(field_name: str) -> bool:
    """Whether the field is a labelset (else it is a YYYY-MM-DD text field)."""
    return field_name == POINT_TYPE_FIELD or field_name in CATEGORY_FIELDS


def field_vocabulary(field_name: str) -> tuple[str, ...]:
    """The built-in labels for a labelset field."""
    if field_name == POINT_TYPE_FIELD:
        return (POSITIVE, NEGATIVE)
    return CATEGORY_FIELDS[field_name]


def labelset_owner(field_name: str) -> str:
    """The field whose labelset a labelset field uses (usually itself)."""
    return SHARED_LABELSETS.get(field_name, field_name)


def label_color(field_name: str, label_name: str, index: int) -> str:
    """Colour for a new label: fixed for point types, else from the palette."""
    if field_name == POINT_TYPE_FIELD and label_name in POINT_TYPE_COLORS:
        return POINT_TYPE_COLORS[label_name]
    return LABEL_PALETTE[index % len(LABEL_PALETTE)]


def required_labels(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Labels each labelset needs: its vocabulary plus unseen values in entries.

    The result is keyed by the labelset's owner field (see labelset_owner), so
    fields that share a labelset contribute to one list. Values not in the
    vocabulary (e.g. legacy categories) are appended in the order they are first
    seen, so they survive a round trip.
    """
    labels = {
        name: list(field_vocabulary(name))
        for name in ALL_FIELDS
        if is_labelset_field(name) and labelset_owner(name) == name
    }
    for entry in entries:
        for point in entry.get("positive_points", []):
            for field_name in CATEGORY_FIELDS:
                value = point.get(field_name)
                owner_labels = labels[labelset_owner(field_name)]
                if value and value not in owner_labels:
                    owner_labels.append(value)
    return labels


@dataclass
class ProjectSchema:
    """IDs of the LCC fields and labels in one Studio project."""

    settings_id: str
    field_ids: dict[str, str]
    # {field name: {label name: label id}}
    label_ids: dict[str, dict[str, str]]

    @property
    def field_names(self) -> dict[str, str]:
        """{field id: field name}."""
        return {field_id: name for name, field_id in self.field_ids.items()}

    @property
    def label_names(self) -> dict[str, str]:
        """{label id: label name}, across all fields."""
        return {
            label_id: name
            for labels in self.label_ids.values()
            for name, label_id in labels.items()
        }


def project_settings(project: dict[str, Any]) -> dict[str, Any]:
    """The project's settings, or an error explaining why there are none."""
    settings = project.get("settings")
    if settings is None:
        if "template" in project:
            raise ValueError(
                "this Studio deployment predates the Template -> ProjectSettings "
                "rename; these scripts need a newer Studio"
            )
        raise ValueError(f"project {project['id']} has no settings")
    return settings


def resolve_schema(project: dict[str, Any]) -> ProjectSchema:
    """Look up the LCC fields and their labels by name.

    Fields the project lacks are left out; ensure_template creates them.
    """
    settings = project_settings(project)
    labelsets = {
        labelset["id"]: labelset for labelset in settings.get("labelsets") or []
    }
    field_ids: dict[str, str] = {}
    label_ids: dict[str, dict[str, str]] = {}
    # Fields on the same labelset share one {label name: label id} dict, so labels
    # added for one field are visible to the others.
    labels_by_labelset: dict[str, dict[str, str]] = {}
    for field in settings.get("annotation_metadata_fields") or []:
        name = field["name"]
        if name not in ALL_FIELDS:
            continue
        expected = "labelset" if is_labelset_field(name) else "text"
        if field["data_type"] != expected:
            raise ValueError(
                f"field {name!r} is {field['data_type']}, expected {expected}"
            )
        field_ids[name] = field["id"]
        if expected == "labelset":
            labelset_id = field["labelset_id"]
            if labelset_id not in labels_by_labelset:
                labelset = labelsets.get(labelset_id) or {}
                labels_by_labelset[labelset_id] = {
                    label["name"]: label["id"] for label in labelset.get("labels") or []
                }
            label_ids[name] = labels_by_labelset[labelset_id]
    return ProjectSchema(settings["id"], field_ids, label_ids)


def make_client(base_url: str | None = None) -> StudioClient:
    """Create the Studio client used by the upload and download scripts.

    Args:
        base_url: API root including /api/v1. Defaults to $STUDIO_API_URL, else
            production.

    Returns:
        the client, which reads the API key from $STUDIO_API_KEY.
    """
    return StudioClient(
        base_url=base_url or os.environ.get("STUDIO_API_URL") or BASE_URL,
        timeout=60,
        max_retries=6,
    )


def ensure_template(
    client: StudioClient, project_id: str, entries: list[dict[str, Any]]
) -> ProjectSchema:
    """Create any missing LCC fields and labels in the project, then resolve them.

    Also sets the project's label geometry to point. Safe to re-run. Fields in
    SHARED_LABELSETS are created on their owner field's labelset.
    """
    project = client.get_project(project_id)
    settings = project_settings(project)
    settings_id = settings["id"]

    if settings.get("label_geometry") != "point":
        print("Setting project label geometry to point")
        client.request(
            "PUT",
            f"/project_settings/{settings_id}",
            json={
                "name": settings["name"],
                "label_geometry": "point",
                "project_id": project_id,
            },
        )

    schema = resolve_schema(project)
    needed = required_labels(entries)
    labelsets_by_name = {
        labelset["name"]: labelset for labelset in settings.get("labelsets") or []
    }

    for order, name in enumerate(ALL_FIELDS):
        if name in schema.field_ids:
            continue
        field: dict[str, Any] = {
            "name": name,
            "display_name": FIELD_DISPLAY_NAMES[name],
            "data_type": "labelset" if is_labelset_field(name) else "text",
            "project_settings_id": settings_id,
            "required": False,
            "read_only": False,
            "display_order": order,
        }
        if is_labelset_field(name) and labelset_owner(name) != name:
            field["labelset_id"] = _field_labelset_id(
                client, project_id, labelset_owner(name)
            )
        elif is_labelset_field(name):
            labelset = labelsets_by_name.get(name)
            if labelset is None:
                print(f"Creating labelset {name!r}")
                labelset = client.create(
                    "/labelsets",
                    {
                        "name": name,
                        "display_name": FIELD_DISPLAY_NAMES[name],
                        "project_settings_id": settings_id,
                        "labels": [
                            {"name": label, "color": label_color(name, label, i)}
                            for i, label in enumerate(needed[name])
                        ],
                    },
                )
            field["labelset_id"] = labelset["id"]
        print(f"Creating field {name!r}")
        client.create("/annotation_metadata_fields", field)

    schema = resolve_schema(client.get_project(project_id))
    # Fields on one labelset share their label_ids dict, so each labelset is filled
    # once. A field in SHARED_LABELSETS that still has its own labelset (from an
    # older project) is filled separately.
    filled: set[int] = set()
    for name in ALL_FIELDS:
        if not is_labelset_field(name):
            continue
        existing = schema.label_ids[name]
        if id(existing) in filled:
            continue
        filled.add(id(existing))
        labels = needed[labelset_owner(name)]
        missing = [label for label in labels if label not in existing]
        if not missing:
            continue
        print(f"Adding labels to {name!r}: {missing}")
        labelset_id = _field_labelset_id(client, project_id, name)
        created = client.request(
            "POST",
            "/labels",
            json=[
                {
                    "name": label,
                    "color": label_color(name, label, len(existing) + i),
                    "labelset_id": labelset_id,
                }
                for i, label in enumerate(missing)
            ],
        )["records"]
        for record in created:
            existing[record["name"]] = record["id"]
    return schema


def _field_labelset_id(client: StudioClient, project_id: str, field_name: str) -> str:
    settings = project_settings(client.get_project(project_id))
    for field in settings.get("annotation_metadata_fields") or []:
        if field["name"] == field_name:
            return field["labelset_id"]
    raise ValueError(f"field {field_name!r} not found")
