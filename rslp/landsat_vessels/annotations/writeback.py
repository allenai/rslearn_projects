"""Writing annotations into rslearn window ``label`` layers.

Shared by the annotation app, which writes each label through as it is made, and by
``scripts/apply_round1_labels.py``, which reconciles the whole labels JSONL against the
dataset. Both must produce byte-identical label layers, so the decision about what a
label layer contains lives here and nowhere else.
"""

import shutil
from datetime import datetime, timezone
from typing import Any

from rslearn.dataset import Window
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat

LABEL_LAYER = "label"

# Labels that become a label layer. "skip" means "come back to this one", not a
# judgement, so it is never written; a window that had a label and is then skipped has
# its layer removed.
WRITTEN_LABELS = {"correct", "incorrect", "unsure"}
# What the classifier can train on. "unsure" is written but skipped at training time by
# the task's skip_unknown_categories, because "a human looked and could not tell" is
# evidence about the imagery rather than something to drop silently.
TRAINABLE_LABELS = {"correct", "incorrect"}

# Copied from the annotation pool onto the label feature so each window carries its own
# history: what the model said before a human looked, and why it was queued.
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

# Fields compared to decide whether an existing label layer is already current.
IDENTITY_FIELDS = ("label", "reason", "note", "annotator")


def build_properties(record: dict, pool: dict, group: str) -> dict[str, Any]:
    """The properties of the label feature for one annotation.

    Args:
        record: the annotation event (label, reason, note, annotator, ts).
        pool: the window's row from the annotation pool index.
        group: the window group, recorded so a window knows which round labelled it.
    """
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
    """Whether an existing label layer already carries this annotation.

    Compares the whole annotation rather than just the class, so a changed reason, note
    or annotator still reaches the window.
    """
    if existing is None:
        return False
    return all(existing.get(field) == properties[field] for field in IDENTITY_FIELDS)


def write_label(
    window: Window, properties: dict[str, Any], vector_format: GeojsonVectorFormat
) -> bool:
    """Write the annotation to the window's label layer.

    Returns True if it was written, False if the layer already carried exactly this
    annotation.
    """
    if is_current(read_existing_label(window, vector_format), properties):
        return False
    feature = Feature(window.get_geometry(), properties)
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_vector(vector_format, [feature])
    window.mark_layer_completed(LABEL_LAYER)
    return True


def clear_label(window: Window) -> bool:
    """Remove a window's label layer. Returns True if there was one to remove.

    Used when an annotation is cleared or changed to "skip", so the dataset never keeps a
    label the annotator has taken back. There is no storage API for deleting a layer, so
    this removes the layer directory (which holds both data.geojson and the completed
    marker) directly.
    """
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    if not layer_dir.exists():
        return False
    shutil.rmtree(str(layer_dir))
    return True
