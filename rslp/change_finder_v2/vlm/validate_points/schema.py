"""Shared data structures for the point-validation pipeline.

The pipeline passes JSON between three steps:

1. ``add_points`` reads a v2 annotation file and writes a :class:`PointSet`.
2. ``run_gemini`` reads that :class:`PointSet`, runs Gemini per point, and writes a
   :class:`PredictionSet` (the same points plus a ``prediction`` field).
3. ``compute_accuracy`` reads the :class:`PredictionSet` and reports metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Modes for adding points.
MODE_EVALUATION = "evaluation"
MODE_DEPLOYMENT = "deployment"

# Point labels / predictions.
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"


@dataclass
class PointRecord:
    """One point to validate, with its provenance in the source annotation file."""

    # Image-database key.
    lon: float
    lat: float
    year: int
    window_name: str

    # The change predicted/annotated at this point.
    predicted_change_date: str | None
    pre_category: str | None
    post_category: str | None

    # Ground-truth label ("positive"/"negative"), only set in evaluation mode.
    label: str | None

    # Provenance in the source v2 annotation file.
    annotation_group: str
    annotation_window_name: str
    point_type: str  # "positive" or "negative" (which list the point came from)
    point_index: int  # index within that list


@dataclass
class PointSet:
    """The output of ``add_points``: the set of points to validate."""

    mode: str
    image_db_path: str
    group: str
    points: list[PointRecord] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Write the point set to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> PointSet:
        """Load a point set from a JSON file."""
        data = json.loads(Path(path).read_text())
        points = [PointRecord(**point) for point in data.pop("points")]
        return PointSet(points=points, **data)


@dataclass
class PointPrediction:
    """A :class:`PointRecord` plus the model's validation result."""

    record: PointRecord
    prediction: str | None  # "positive"/"negative", or None if the model failed
    confidence: str | None
    reasoning: str | None
    image_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flatten the record and prediction into a single JSON-friendly dict."""
        out = asdict(self.record)
        out.update(
            prediction=self.prediction,
            confidence=self.confidence,
            reasoning=self.reasoning,
            image_dates=self.image_dates,
        )
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PointPrediction:
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        prediction = d.pop("prediction", None)
        confidence = d.pop("confidence", None)
        reasoning = d.pop("reasoning", None)
        image_dates = d.pop("image_dates", [])
        return PointPrediction(
            record=PointRecord(**d),
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            image_dates=image_dates,
        )


@dataclass
class PredictionSet:
    """The output of ``run_gemini``: points plus predictions."""

    mode: str
    image_db_path: str
    group: str
    predictions: list[PointPrediction] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Write the prediction set to a JSON file."""
        data = {
            "mode": self.mode,
            "image_db_path": self.image_db_path,
            "group": self.group,
            "predictions": [p.to_dict() for p in self.predictions],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str | Path) -> PredictionSet:
        """Load a prediction set from a JSON file."""
        data = json.loads(Path(path).read_text())
        predictions = [
            PointPrediction.from_dict(p) for p in data.pop("predictions")
        ]
        return PredictionSet(predictions=predictions, **data)
