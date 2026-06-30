"""Shared data structures for the category-tagging pipeline.

The pipeline passes JSON between two steps:

1. ``add_points`` reads a v2 annotation file and writes a :class:`PointSet`.
2. ``run_gemini`` reads that :class:`PointSet`, runs Gemini per point, and writes a
   :class:`CategorySet` (the same points plus the assigned categories).

Unlike the point-validation pipeline, the tagger has no ground-truth labels: every
point is an unlabeled change that Gemini is asked to categorize. Some points may later
be hand-corrected and used to iterate on the prompt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PointRecord:
    """One point to categorize, with its provenance in the source annotation file."""

    # Image-database key.
    lon: float
    lat: float
    year: int
    window_name: str

    # Context about the flagged change, used to prime the prompt.
    predicted_change_date: str | None
    pre_category: str | None
    post_category: str | None

    # Provenance in the source v2 annotation file.
    annotation_group: str
    annotation_window_name: str
    point_index: int  # index within the entry's positive_points list

    # Annotated change dates (ISO date strings), used to anchor image sampling and to
    # prime the prompt. None for older point sets that predate these fields.
    pre_change: str | None = None
    post_change: str | None = None
    first_observable: str | None = None

    # Hand-labeled ground-truth fine-grained change categories, when set. Used to
    # evaluate the model's predictions. None when the point is unlabeled.
    gt_pre_change_category: str | None = None
    gt_post_change_category: str | None = None
    gt_same_change_category: str | None = None


@dataclass
class PointSet:
    """The output of ``add_points``: the set of points to categorize."""

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
class CategoryPrediction:
    """A :class:`PointRecord` plus the model's category assignment."""

    record: PointRecord
    pre_change_category: str | None
    post_change_category: str | None
    same_change_category: str | None
    confidence: str | None
    reasoning: str | None
    flagged_for_review: bool = False
    image_dates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flatten the record and prediction into a single JSON-friendly dict."""
        out = asdict(self.record)
        out.update(
            pre_change_category=self.pre_change_category,
            post_change_category=self.post_change_category,
            same_change_category=self.same_change_category,
            flagged_for_review=self.flagged_for_review,
            confidence=self.confidence,
            reasoning=self.reasoning,
            image_dates=self.image_dates,
        )
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CategoryPrediction:
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        pre = d.pop("pre_change_category", None)
        post = d.pop("post_change_category", None)
        same = d.pop("same_change_category", None)
        flagged = d.pop("flagged_for_review", False)
        confidence = d.pop("confidence", None)
        reasoning = d.pop("reasoning", None)
        image_dates = d.pop("image_dates", [])
        return CategoryPrediction(
            record=PointRecord(**d),
            pre_change_category=pre,
            post_change_category=post,
            same_change_category=same,
            flagged_for_review=flagged,
            confidence=confidence,
            reasoning=reasoning,
            image_dates=image_dates,
        )


@dataclass
class CategorySet:
    """The output of ``run_gemini``: points plus assigned categories."""

    image_db_path: str
    group: str
    predictions: list[CategoryPrediction] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Write the category set to a JSON file."""
        data = {
            "image_db_path": self.image_db_path,
            "group": self.group,
            "predictions": [p.to_dict() for p in self.predictions],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: str | Path) -> CategorySet:
        """Load a category set from a JSON file."""
        data = json.loads(Path(path).read_text())
        predictions = [
            CategoryPrediction.from_dict(p) for p in data.pop("predictions")
        ]
        return CategorySet(predictions=predictions, **data)
