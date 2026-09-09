"""Run Gemini over the points produced by ``add_points`` to categorize each one.

Assumes ``rslearn dataset prepare`` and ``materialize`` have already been run on the
image database. For each point this gathers one Sentinel-2 image (the least cloudy) and
up to one high-resolution aerial image per calendar quarter (each captioned with its
capture date), prompts Gemini to assign fine-grained change categories, and writes a
category set JSON.

Example:
    python -m rslp.change_finder_v2.vlm.category_tagger.run_gemini \
        --points points.json \
        --output categories.json
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt

from rslp.change_finder_v2.vlm.image_db import image_database
from rslp.change_finder_v2.vlm.image_db.image_database import AvailableImage

from .gemini import GeminiCategorizer
from .prompt import ImageRef, build_category_prompt, label_image
from .schema import CategoryPrediction, CategorySet, PointRecord, PointSet

logger = logging.getLogger(__name__)

# Each segment before/after the change spans this long, and is split into this many
# even periods (so ~6-month periods) when sampling Sentinel-2.
_SEGMENT_SPAN = timedelta(days=730)  # ~2 years
_N_PERIODS = 4
# Max aerial (Esri) images kept per segment.
_MAX_ESRI_PER_SEGMENT = 4

# Center crop (native pixels) shown zoomed in the right panel, and center-point circle
# radius, per layer. Sentinel-2 chips are 64x64; high-res Esri rasters are 512x512.
_S2_CROP_SIZE = 16
_S2_CIRCLE_RADIUS = 4
_HIGHRES_CROP_SIZE = 32
_HIGHRES_CIRCLE_RADIUS = 16


def _center_clarity(chw_array: npt.NDArray) -> float:
    """Heuristic clarity of the image CENTER in [0, 1] (higher = clearer ground view).

    Penalizes cloud/haze/snow (bright, low-saturation pixels) and missing data
    (near-black pixels) in the central region. Pure heuristic on RGB.
    """
    arr = np.asarray(chw_array)[:3].astype(np.float32)
    _, h, w = arr.shape
    cy0, cy1 = int(h * 0.33), int(np.ceil(h * 0.67))
    cx0, cx1 = int(w * 0.33), int(np.ceil(w * 0.67))
    center = arr[:, cy0:cy1, cx0:cx1]
    r, g, b = center[0], center[1], center[2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    cloudy = (mn > 175.0) & ((mx - mn) < 35.0)
    black = mx < 12.0
    bad = cloudy | black
    return float(1.0 - bad.mean())


def _layer_images(
    images: list[AvailableImage], layer_name: str
) -> list[AvailableImage]:
    """Images for one layer that have a time range."""
    return [
        im for im in images if im.layer_name == layer_name and im.time_range is not None
    ]


def _even_periods(
    start: datetime, end: datetime, n: int
) -> list[tuple[datetime, datetime]]:
    """Split ``[start, end)`` into ``n`` equal periods (empty if end <= start)."""
    if end <= start or n <= 0:
        return []
    step = (end - start) / n
    return [(start + i * step, start + (i + 1) * step) for i in range(n)]


def _least_cloudy_in_period(
    images: list[AvailableImage], start: datetime, end: datetime
) -> AvailableImage | None:
    """Pick the clearest-center image whose start falls in ``[start, end)``."""
    candidates = [im for im in images if start <= im.time_range[0] < end]
    if not candidates:
        return None
    mid = start + (end - start) / 2
    return max(
        candidates,
        key=lambda im: (
            _center_clarity(im.array),
            -abs((im.time_range[0] - mid).days),
        ),
    )


def _spread_sample(images_sorted: list[AvailableImage], k: int) -> list[AvailableImage]:
    """Keep up to ``k`` images, maximally spread in time (farthest-point sampling).

    Seeds with the earliest and latest, then greedily adds the image with the largest
    minimum temporal distance to those already chosen. Drops closely-spaced images.
    """
    if len(images_sorted) <= k:
        return images_sorted
    times = [im.time_range[0] for im in images_sorted]
    chosen = [0, len(images_sorted) - 1]
    while len(chosen) < k:
        best_i, best_dist = None, -1.0
        for i in range(len(images_sorted)):
            if i in chosen:
                continue
            dist = min(abs((times[i] - times[j]).total_seconds()) for j in chosen)
            if dist > best_dist:
                best_dist, best_i = dist, i
        chosen.append(best_i)
    return [images_sorted[i] for i in sorted(chosen)]


def _segments(
    pre_change: datetime, post_change: datetime
) -> list[tuple[datetime, datetime]]:
    """The before / between / after segments around the change."""
    return [
        (pre_change - _SEGMENT_SPAN, pre_change),
        (pre_change, post_change),
        (post_change, post_change + _SEGMENT_SPAN),
    ]


def _select_s2(
    images: list[AvailableImage],
    layer_name: str,
    pre_change: datetime,
    post_change: datetime,
) -> list[AvailableImage]:
    """Least-cloudy Sentinel-2, one per even period within each segment."""
    s2 = _layer_images(images, layer_name)
    selected: list[AvailableImage] = []
    for seg_start, seg_end in _segments(pre_change, post_change):
        for p_start, p_end in _even_periods(seg_start, seg_end, _N_PERIODS):
            best = _least_cloudy_in_period(s2, p_start, p_end)
            if best is not None:
                selected.append(best)
    return selected


def _select_esri(
    images: list[AvailableImage],
    layer_name: str,
    pre_change: datetime,
    post_change: datetime,
) -> list[AvailableImage]:
    """Up to ``_MAX_ESRI_PER_SEGMENT`` spread-out aerial images per segment."""
    esri = _layer_images(images, layer_name)
    selected: list[AvailableImage] = []
    for seg_start, seg_end in _segments(pre_change, post_change):
        in_seg = sorted(
            (im for im in esri if seg_start <= im.time_range[0] < seg_end),
            key=lambda im: im.time_range[0],
        )
        selected.extend(_spread_sample(in_seg, _MAX_ESRI_PER_SEGMENT))
    return selected


def _build_image_refs(
    images: list[AvailableImage],
    s2_layer: str,
    highres_layer: str,
    pre_change: datetime,
    post_change: datetime,
) -> tuple[list[ImageRef], list[str]]:
    """Build chronologically-ordered, captioned image refs for one point.

    Samples imagery relative to the change dates (before / between / after).
    """
    selected = _select_s2(images, s2_layer, pre_change, post_change) + _select_esri(
        images, highres_layer, pre_change, post_change
    )
    selected.sort(key=lambda im: im.time_range[0])

    refs: list[ImageRef] = []
    dates: list[str] = []
    for image in selected:
        capture = image.time_range[0].date().isoformat()
        if image.layer_name == s2_layer:
            kind = "Sentinel-2"
            crop_size, circle_radius = _S2_CROP_SIZE, _S2_CIRCLE_RADIUS
        else:
            kind = "Aerial (high-res)"
            crop_size, circle_radius = _HIGHRES_CROP_SIZE, _HIGHRES_CIRCLE_RADIUS
        caption = f"{kind} {capture}"
        refs.append(
            ImageRef(
                label=caption,
                png_bytes=label_image(image.array, caption, crop_size, circle_radius),
            )
        )
        dates.append(caption)
    return refs, dates


def _parse_date(value: str) -> datetime:
    """Parse an ISO date string to a midnight-UTC datetime."""
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _categorize_point(
    record: PointRecord,
    categorizer: GeminiCategorizer,
    image_db_path: str,
    group: str,
    s2_layer: str,
    highres_layer: str,
) -> CategoryPrediction:
    """Gather imagery for one point and run the model."""
    images = image_database.list_available_images(
        image_db_path, record.lon, record.lat, record.year, group=group
    )
    if not record.pre_change or not record.post_change:
        raise ValueError(
            f"record {record.window_name} is missing pre_change/post_change dates"
        )
    pre_change = _parse_date(record.pre_change)
    post_change = _parse_date(record.post_change)
    refs, dates = _build_image_refs(
        images, s2_layer, highres_layer, pre_change, post_change
    )
    if not refs:
        logger.warning("no materialized images for %s; skipping", record.window_name)
        return CategoryPrediction(
            record=record,
            pre_change_category=None,
            post_change_category=None,
            same_change_category=None,
            flagged_for_review=False,
            confidence=None,
            reasoning="no imagery available",
            image_dates=[],
        )

    prompt = build_category_prompt(
        record.pre_change,
        record.first_observable,
        record.post_change,
        record.pre_category,
        record.post_category,
    )
    result = categorizer.categorize(prompt, refs)
    return CategoryPrediction(
        record=record,
        pre_change_category=result.pre_change_category,
        post_change_category=result.post_change_category,
        same_change_category=result.same_change_category,
        flagged_for_review=result.flagged_for_review,
        confidence=result.confidence,
        reasoning=result.reasoning,
        image_dates=dates,
    )


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Categorize flagged points with Gemini using image-database imagery."
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
    parser.add_argument("--project", default="earthsystem-dev-c3po")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--thinking-level",
        default=None,
        choices=["low", "high"],
        help=(
            "Thinking level for Gemini 3.x models (e.g. 'high' or 'low'). "
            "Leave unset to use the model default. Only set this for 3.x models; "
            "the 2.5 thinking control has a different structure."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N points."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Number of concurrent Gemini requests (threads).",
    )
    parsed = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)

    point_set = PointSet.load(parsed.points)
    image_db_path = parsed.image_db_path or point_set.image_db_path
    records = point_set.points
    if parsed.limit is not None:
        records = records[: parsed.limit]

    categorizer = GeminiCategorizer(
        project=parsed.project,
        location=parsed.location,
        model=parsed.model,
        thinking_level=parsed.thinking_level,
    )

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
