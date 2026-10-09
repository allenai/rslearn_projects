"""Write annotations into rslearn window ``label`` layers."""

import shutil
from datetime import datetime, timezone
from typing import Any

from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat

LABEL_LAYER = "label"

# Copied from the annotation pool onto the label feature.
PROVENANCE_FIELDS = [
    "scene_id",
    "slice",
    "stratum",
    "tier",
    "split",
    "classifier_label",
    "classifier_prob_correct",
    "detector_score",
    "distance_to_coast_m",
    "land_cover_class",
    "selection_reason",
    "substituted_product",
]

# An existing label layer with the same values is left untouched.
IDENTITY_FIELDS = ("label", "reason", "note", "annotator")


def build_properties(record: dict, pool: dict, group: str) -> dict[str, Any]:
    """The label feature properties for one annotation event and its pool row."""
    return {
        "label": record["label"],
        "reason": record.get("reason"),
        "note": record.get("note", ""),
        "annotator": record.get("annotator", "unknown"),
        "labeled_at": (
            datetime.fromtimestamp(record["ts"], tz=timezone.utc).isoformat()
            if record.get("ts")
            else None
        ),
        "annotation_round": group,
        "lon": float(pool["longitude"]),
        "lat": float(pool["latitude"]),
        "ts": pool["ts"],
        **{field: pool.get(field) for field in PROVENANCE_FIELDS},
    }


def read_existing_label(
    window: Window, vector_format: GeojsonVectorFormat
) -> dict[str, Any] | None:
    """The annotation already on a window, or None if it has no label layer."""
    if not window.is_layer_completed(LABEL_LAYER):
        return None
    features = window.data.read_vector(LABEL_LAYER, vector_format)
    if not features:
        return None
    return features[0].properties


def is_current(existing: dict[str, Any] | None, properties: dict[str, Any]) -> bool:
    """Whether an existing label layer already carries this annotation."""
    if existing is None:
        return False
    return all(existing.get(field) == properties[field] for field in IDENTITY_FIELDS)


def write_label(
    window: Window, properties: dict[str, Any], vector_format: GeojsonVectorFormat
) -> bool:
    """Write the annotation to the window's label layer; False if already current."""
    if is_current(read_existing_label(window, vector_format), properties):
        return False
    feature = Feature(window.get_geometry(), properties)
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_vector(vector_format, [feature])
    window.mark_layer_completed(LABEL_LAYER)
    return True


def clear_label(window: Window) -> bool:
    """Remove a window's label layer (used for cleared and skipped labels).

    rslearn has no API for deleting a layer, so this removes the layer directory.
    """
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    if not layer_dir.exists():
        return False
    shutil.rmtree(str(layer_dir))
    return True
