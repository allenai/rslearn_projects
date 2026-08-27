"""Command-line interface for the LLM land-cover-change categorization pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .run import Pipeline, PipelineConfig, load_entries


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the categorization pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Categorize land-cover changes in a change_finder_v2 annotation "
            "JSON using Sentinel-2 + Wayback imagery and Gemini."
        )
    )
    parser.add_argument(
        "--json", required=True, help="path to the annotation JSON file"
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="directory for cached imagery and result JSON files",
    )
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model")
    parser.add_argument(
        "--project", default="earthsystem-dev-c3po", help="Vertex AI project"
    )
    parser.add_argument(
        "--location", default="global", help="Vertex AI location"
    )
    parser.add_argument(
        "--s2-date-tolerance-days",
        type=int,
        default=45,
        help="half-width of the Sentinel-2 date search window",
    )
    parser.add_argument(
        "--s2-clear-threshold",
        type=float,
        default=0.05,
        help=(
            "chip cloud/invalid fraction at or below which a Sentinel-2 scene is "
            "accepted immediately (closest such scene wins)"
        ),
    )
    parser.add_argument(
        "--s2-max-candidates",
        type=int,
        default=8,
        help="number of nearest-date Sentinel-2 scenes to cloud-score per chip",
    )
    parser.add_argument(
        "--wayback-zoom", type=int, default=18, help="Wayback tile zoom level"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many genuine processing attempts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reprocess entries even if a cached result exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch imagery and build prompts but do not call the model",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    # Quiet down noisy third-party loggers so pipeline progress is readable.
    for noisy in ("botocore", "boto3", "urllib3", "google_genai", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = PipelineConfig(
        cache_dir=Path(args.cache_dir),
        model=args.model,
        project=args.project,
        location=args.location,
        s2_date_tolerance_days=args.s2_date_tolerance_days,
        s2_clear_threshold=args.s2_clear_threshold,
        s2_max_candidates=args.s2_max_candidates,
        wayback_zoom=args.wayback_zoom,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    entries = load_entries(Path(args.json))
    pipeline = Pipeline(config)
    pipeline.run(entries, limit=args.limit)


if __name__ == "__main__":
    main()
