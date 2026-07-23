"""Run Claude (Anthropic API) over the points produced by ``add_points``.

This is the Anthropic counterpart of ``run_gemini``: it reuses the same image sampling,
captioning, and per-point orchestration and only swaps the model client for
:class:`AnthropicCategorizer`.

Assumes ``rslearn dataset prepare`` and ``materialize`` have already been run on the
image database. For each point this gathers one Sentinel-2 image (the least cloudy) and
up to one high-resolution aerial image per calendar quarter (each captioned with its
capture date), prompts Claude to assign fine-grained change categories, and writes a
category set JSON.

Example:
    ANTHROPIC_API_KEY=... python -m rslp.change_finder_v2.vlm.category_tagger.run_anthropic \
        --points points.json \
        --output categories.json
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from .anthropic import AnthropicCategorizer
from .run_gemini import _categorize_point
from .schema import CategoryPrediction, CategorySet, PointRecord, PointSet

logger = logging.getLogger(__name__)


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Categorize flagged points with Claude using image-database imagery."
    )
    parser.add_argument(
        "--points", required=True, help="Point set JSON produced by add_points."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the category set JSON."
    )
    parser.add_argument(
        "--image-db-path",
        default=None,
        help="Override the image database path stored in the point set.",
    )
    parser.add_argument(
        "--s2-layer", default="sentinel2", help="Sentinel-2 layer name."
    )
    parser.add_argument(
        "--highres-layer", default="esri", help="High-resolution layer name."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. Defaults to the ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N points."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent Anthropic requests (threads). Keep low to respect "
        "per-key rate limits.",
    )
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    point_set = PointSet.load(parsed.points)
    image_db_path = parsed.image_db_path or point_set.image_db_path
    records = point_set.points
    if parsed.limit is not None:
        records = records[: parsed.limit]

    categorizer = AnthropicCategorizer(api_key=parsed.api_key, model=parsed.model)

    def categorize_one(record: PointRecord) -> CategoryPrediction:
        try:
            return _categorize_point(
                record,
                categorizer,
                image_db_path,
                point_set.group,
                parsed.s2_layer,
                parsed.highres_layer,
            )
        except Exception:  # noqa: BLE001 - keep going on per-point failures
            logger.warning("failed to categorize %s", record.window_name, exc_info=True)
            return CategoryPrediction(
                record=record,
                pre_change_category=None,
                post_change_category=None,
                same_change_category=None,
                flagged_for_review=False,
                confidence=None,
                reasoning="error during categorization",
                image_dates=[],
            )

    predictions: list[CategoryPrediction] = []
    with ThreadPoolExecutor(max_workers=parsed.workers) as executor:
        for i, prediction in enumerate(executor.map(categorize_one, records)):
            logger.info(
                "[%d/%d] categorized %s -> pre=%s post=%s same=%s flagged=%s",
                i + 1,
                len(records),
                prediction.record.window_name,
                prediction.pre_change_category,
                prediction.post_change_category,
                prediction.same_change_category,
                prediction.flagged_for_review,
            )
            predictions.append(prediction)

    category_set = CategorySet(
        image_db_path=str(image_db_path),
        group=point_set.group,
        predictions=predictions,
    )
    category_set.save(parsed.output)
    logger.info("Wrote %d predictions to %s", len(predictions), parsed.output)


if __name__ == "__main__":
    main()
