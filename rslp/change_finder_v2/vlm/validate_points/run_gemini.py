"""Run Gemini over the points produced by ``add_points`` to validate each one.

Assumes ``rslearn dataset prepare`` and ``materialize`` have already been run on the
image database. For each point this gathers one Sentinel-2 image and up to one
high-resolution aerial image per year (each captioned with its capture date), prompts
Gemini to decide whether the point has a genuine long-term change, and writes a
prediction set JSON.

Example:
    python -m rslp.change_finder_v2.vlm.validate_points.run_gemini \
        --points points.json \
        --output predictions.json
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from rslp.change_finder_v2.vlm.image_db import image_database
from rslp.change_finder_v2.vlm.image_db.image_database import AvailableImage

from .gemini import GeminiValidator
from .prompt import ImageRef, build_validation_prompt, label_image
from .schema import PointPrediction, PointRecord, PointSet, PredictionSet

logger = logging.getLogger(__name__)

# Month/day used as the per-year target so we pick a consistent season across years.
_TARGET_MONTH = 7
_TARGET_DAY = 1


def _select_one_per_year(
    images: list[AvailableImage], layer_name: str
) -> list[AvailableImage]:
    """Pick one image per calendar year for a layer, nearest to mid-year."""
    by_year: dict[int, list[AvailableImage]] = {}
    for image in images:
        if image.layer_name != layer_name or image.time_range is None:
            continue
        by_year.setdefault(image.time_range[0].year, []).append(image)

    selected: list[AvailableImage] = []
    for year in sorted(by_year):
        candidates = by_year[year]
        tzinfo = candidates[0].time_range[0].tzinfo
        target = datetime(year, _TARGET_MONTH, _TARGET_DAY, tzinfo=tzinfo)
        best = min(
            candidates, key=lambda im: abs((im.time_range[0] - target).days)
        )
        selected.append(best)
    return selected


def _build_image_refs(
    images: list[AvailableImage], s2_layer: str, highres_layer: str
) -> tuple[list[ImageRef], list[str]]:
    """Build chronologically-ordered, captioned image refs for one point."""
    selected = _select_one_per_year(images, s2_layer) + _select_one_per_year(
        images, highres_layer
    )
    selected.sort(key=lambda im: im.time_range[0])

    refs: list[ImageRef] = []
    dates: list[str] = []
    for image in selected:
        capture = image.time_range[0].date().isoformat()
        kind = "Sentinel-2" if image.layer_name == s2_layer else "Aerial (high-res)"
        caption = f"{kind} {capture}"
        refs.append(ImageRef(label=caption, png_bytes=label_image(image.array, caption)))
        dates.append(caption)
    return refs, dates


def _validate_point(
    record: PointRecord,
    validator: GeminiValidator,
    image_db_path: str,
    group: str,
    s2_layer: str,
    highres_layer: str,
) -> PointPrediction:
    """Gather imagery for one point and run the model."""
    images = image_database.list_available_images(
        image_db_path, record.lon, record.lat, record.year, group=group
    )
    refs, dates = _build_image_refs(images, s2_layer, highres_layer)
    if not refs:
        logger.warning(
            "no materialized images for %s; skipping", record.window_name
        )
        return PointPrediction(
            record=record,
            prediction=None,
            confidence=None,
            reasoning="no imagery available",
            image_dates=[],
        )

    prompt = build_validation_prompt(
        record.predicted_change_date, record.pre_category, record.post_category
    )
    result = validator.validate(prompt, refs)
    return PointPrediction(
        record=record,
        prediction=result.prediction,
        confidence=result.confidence,
        reasoning=result.reasoning,
        image_dates=dates,
    )


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Validate flagged points with Gemini using image-database imagery."
    )
    parser.add_argument(
        "--points", required=True, help="Point set JSON produced by add_points."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the prediction set JSON."
    )
    parser.add_argument(
        "--image-db-path",
        default=None,
        help="Override the image database path stored in the point set.",
    )
    parser.add_argument("--s2-layer", default="sentinel2", help="Sentinel-2 layer name.")
    parser.add_argument(
        "--highres-layer", default="esri", help="High-resolution layer name."
    )
    parser.add_argument("--project", default="earthsystem-dev-c3po")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N points."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent Gemini requests (threads).",
    )
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    point_set = PointSet.load(parsed.points)
    image_db_path = parsed.image_db_path or point_set.image_db_path
    records = point_set.points
    if parsed.limit is not None:
        records = records[: parsed.limit]

    validator = GeminiValidator(
        project=parsed.project, location=parsed.location, model=parsed.model
    )

    def validate_one(record: PointRecord) -> PointPrediction:
        try:
            return _validate_point(
                record,
                validator,
                image_db_path,
                point_set.group,
                parsed.s2_layer,
                parsed.highres_layer,
            )
        except Exception:  # noqa: BLE001 - keep going on per-point failures
            logger.warning("failed to validate %s", record.window_name, exc_info=True)
            return PointPrediction(
                record=record,
                prediction=None,
                confidence=None,
                reasoning="error during validation",
                image_dates=[],
            )

    predictions: list[PointPrediction] = []
    with ThreadPoolExecutor(max_workers=parsed.workers) as executor:
        for i, prediction in enumerate(executor.map(validate_one, records)):
            logger.info(
                "[%d/%d] validated %s",
                i + 1,
                len(records),
                prediction.record.window_name,
            )
            predictions.append(prediction)

    prediction_set = PredictionSet(
        mode=point_set.mode,
        image_db_path=str(image_db_path),
        group=point_set.group,
        predictions=predictions,
    )
    prediction_set.save(parsed.output)
    logger.info("Wrote %d predictions to %s", len(predictions), parsed.output)


if __name__ == "__main__":
    main()
