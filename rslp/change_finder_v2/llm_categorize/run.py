"""Orchestrate the LLM land-cover-change categorization pipeline.

For each annotation entry with a qualifying positive point, fetch Sentinel-2 and
ArcGIS Wayback imagery, prompt Gemini to assign fine-grained change categories,
cache the imagery, and write the resulting category list to the cache directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from .gemini import GeminiCategorizer
from .prompt import ImageRef, build_prompt
from .s2 import S2Fetcher
from .wayback import WaybackClient

logger = logging.getLogger(__name__)

# A positive point must have all of these set to be eligible.
REQUIRED_POINT_FIELDS = (
    "pre_change",
    "post_change",
    "first_date_change_noticeable",
)

# The extra "earlier baseline" Sentinel-2 chip targets ~9 months before the
# pre-change date, with a +/-3 month window so the clearest scene anywhere from
# 6 to 12 months before the change is used.
EARLIER_BASELINE_OFFSET_DAYS = 270
EARLIER_BASELINE_TOLERANCE_DAYS = 90


class Outcome(Enum):
    """The result of attempting to process one annotation entry."""

    NO_POINT = "no_point"
    CACHED = "cached"
    NO_IMAGERY = "no_imagery"
    DRY_RUN = "dry_run"
    DONE = "done"
    FAILED = "failed"


# Outcomes that count as a genuine processing attempt (towards --limit).
_ATTEMPTED = {Outcome.NO_IMAGERY, Outcome.DRY_RUN, Outcome.DONE, Outcome.FAILED}


@dataclass
class PipelineConfig:
    """Configuration for a categorization run."""

    cache_dir: Path
    model: str = "gemini-2.5-pro"
    project: str = "earthsystem-dev-c3po"
    location: str = "global"
    s2_date_tolerance_days: int = 45
    s2_clear_threshold: float = 0.05
    s2_max_candidates: int = 8
    wayback_zoom: int = 18
    overwrite: bool = False
    dry_run: bool = False


def select_positive_point(entry: dict) -> dict | None:
    """Return the first positive point with all required fields set, else None."""
    for point in entry.get("positive_points", []):
        if all(point.get(field) for field in REQUIRED_POINT_FIELDS):
            return point
    return None


def _s2_description(phase: str, target: date, capture: str | None) -> str:
    """Describe a Sentinel-2 chip; include the capture date when known."""
    if phase == "earlier baseline":
        text = (
            "Sentinel-2 (10m, upsampled to 128x128) baseline from 6-12 months "
            f"before the pre-change date, near {target.isoformat()}"
        )
    else:
        text = (
            f"Sentinel-2 (10m, upsampled to 128x128) near the {phase} date "
            f"{target.isoformat()}"
        )
    if capture:
        text += f" (actual capture {capture})"
    return text + "."


def _wayback_description(
    phase: str, target_label: str, capture: str | None
) -> str:
    """Describe a Wayback aerial image; include the capture date when known."""
    text = f"High-resolution aerial image {target_label} the {phase} date"
    if capture:
        text += f" (captured {capture})"
    return text + "."


class Pipeline:
    """Runs the categorization pipeline over a set of annotation entries."""

    def __init__(self, config: PipelineConfig) -> None:
        """Create the pipeline and its imagery/model clients.

        Args:
            config: the run configuration.
        """
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing imagery and model clients...")
        self.s2 = S2Fetcher(
            clear_threshold=config.s2_clear_threshold,
            max_candidates=config.s2_max_candidates,
        )
        self.wayback = WaybackClient()
        self.gemini: GeminiCategorizer | None = None
        if not config.dry_run:
            self.gemini = GeminiCategorizer(
                project=config.project,
                location=config.location,
                model=config.model,
            )
        logger.info("Clients ready (cache_dir=%s)", self.config.cache_dir)

    def _result_path(self, window_name: str) -> Path:
        return self.config.cache_dir / f"{window_name}.json"

    def _image_path(self, window_name: str, suffix: str) -> Path:
        return self.config.cache_dir / f"{window_name}_{suffix}.png"

    def _meta_path(self, window_name: str) -> Path:
        return self.config.cache_dir / f"{window_name}_meta.json"

    def _save_image(self, window_name: str, suffix: str, png_bytes: bytes) -> None:
        self._image_path(window_name, suffix).write_bytes(png_bytes)

    def _load_meta(self, window_name: str) -> dict:
        path = self._meta_path(window_name)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _gather_images(self, window_name: str, point: dict) -> list[ImageRef]:
        """Return labeled imagery for a point, reusing cached PNGs when present.

        Existing ``{window_name}_{suffix}.png`` files are reused as-is (no
        download) unless ``overwrite`` is set; only missing images are fetched.
        Capture dates are recorded in a ``{window_name}_meta.json`` sidecar so
        reused images keep the same descriptions as freshly fetched ones.
        """
        lon = float(point["lon"])
        lat = float(point["lat"])
        pre_change = date.fromisoformat(point["pre_change"])
        first_obs = date.fromisoformat(point["first_date_change_noticeable"])
        post_change = date.fromisoformat(point["post_change"])

        images: list[ImageRef] = []
        meta = self._load_meta(window_name)

        earlier_target = pre_change - timedelta(
            days=EARLIER_BASELINE_OFFSET_DAYS
        )
        s2_targets = [
            (
                "s2_earlier",
                "earlier baseline",
                earlier_target,
                EARLIER_BASELINE_TOLERANCE_DAYS,
            ),
            (
                "s2_pre",
                "pre-change",
                pre_change,
                self.config.s2_date_tolerance_days,
            ),
            (
                "s2_first",
                "first-observable",
                first_obs,
                self.config.s2_date_tolerance_days,
            ),
            (
                "s2_post",
                "post-change",
                post_change,
                self.config.s2_date_tolerance_days,
            ),
        ]
        for suffix, phase, target, tolerance in s2_targets:
            path = self._image_path(window_name, suffix)
            if path.exists() and not self.config.overwrite:
                logger.info(
                    "%s: reusing cached Sentinel-2 %s chip", window_name, phase
                )
                images.append(
                    ImageRef(
                        label=f"Sentinel-2 {phase} image",
                        description=_s2_description(
                            phase, target, meta.get(suffix, {}).get("capture_date")
                        ),
                        png_bytes=path.read_bytes(),
                    )
                )
                continue
            logger.info(
                "%s: fetching Sentinel-2 %s chip near %s",
                window_name,
                phase,
                target.isoformat(),
            )
            chip = self.s2.fetch_chip(lon, lat, target, tolerance)
            if chip is None:
                logger.warning(
                    "%s: no Sentinel-2 scene near %s (%s)",
                    window_name,
                    phase,
                    target.isoformat(),
                )
                continue
            capture = chip.item_date.isoformat()
            logger.info(
                "%s: got Sentinel-2 %s chip (capture %s, cloud %s)",
                window_name,
                phase,
                capture,
                f"{chip.cloud_fraction:.0%}"
                if chip.cloud_fraction is not None
                else "unknown",
            )
            meta[suffix] = {
                "capture_date": capture,
                "cloud_fraction": chip.cloud_fraction,
            }
            self._save_image(window_name, suffix, chip.png_bytes)
            images.append(
                ImageRef(
                    label=f"Sentinel-2 {phase} image",
                    description=_s2_description(phase, target, capture),
                    png_bytes=chip.png_bytes,
                )
            )

        wb_targets = [
            ("wayback_pre", "pre-change", "at/before", "pre"),
            ("wayback_post", "post-change", "at/after", "post"),
        ]
        need_fetch = any(
            not self._image_path(window_name, suffix).exists()
            or self.config.overwrite
            for suffix, _, _, _ in wb_targets
        )
        wb_images: dict[str, object | None] = {"pre": None, "post": None}
        if need_fetch:
            logger.info(
                "%s: fetching Wayback imagery (zoom %d)",
                window_name,
                self.config.wayback_zoom,
            )
            wb_pre, wb_post = self.wayback.find_images(
                lon, lat, pre_change, post_change, zoom=self.config.wayback_zoom
            )
            wb_images = {"pre": wb_pre, "post": wb_post}

        for suffix, phase, target_label, key in wb_targets:
            path = self._image_path(window_name, suffix)
            if path.exists() and not self.config.overwrite:
                logger.info(
                    "%s: reusing cached Wayback %s image", window_name, phase
                )
                images.append(
                    ImageRef(
                        label=f"High-resolution {phase} aerial image",
                        description=_wayback_description(
                            phase,
                            target_label,
                            meta.get(suffix, {}).get("capture_date"),
                        ),
                        png_bytes=path.read_bytes(),
                    )
                )
                continue
            wb = wb_images[key]
            if wb is None:
                logger.info(
                    "%s: no Wayback %s image available", window_name, phase
                )
                continue
            capture = wb.capture_date.isoformat()
            logger.info(
                "%s: got Wayback %s image (capture %s)",
                window_name,
                phase,
                capture,
            )
            meta[suffix] = {"capture_date": capture}
            self._save_image(window_name, suffix, wb.png_bytes)
            images.append(
                ImageRef(
                    label=f"High-resolution {phase} aerial image",
                    description=_wayback_description(phase, target_label, capture),
                    png_bytes=wb.png_bytes,
                )
            )

        if meta:
            self._meta_path(window_name).write_text(json.dumps(meta))

        return images

    def process_entry(self, entry: dict) -> tuple[Outcome, list[str]]:
        """Process one annotation entry.

        Returns:
            an (Outcome, categories) tuple; categories is empty unless the model
            was actually called.
        """
        window_name = entry.get("window_name")
        if not window_name:
            return Outcome.NO_POINT, []

        point = select_positive_point(entry)
        if point is None:
            logger.debug("%s: no qualifying positive point, skipping", window_name)
            return Outcome.NO_POINT, []

        result_path = self._result_path(window_name)
        if result_path.exists() and not self.config.overwrite:
            logger.info("%s: already cached, skipping", window_name)
            return Outcome.CACHED, []

        logger.info(
            "%s: processing point lon=%.6f lat=%.6f",
            window_name,
            float(point["lon"]),
            float(point["lat"]),
        )
        images = self._gather_images(window_name, point)
        if not images:
            logger.warning("%s: no imagery available, skipping", window_name)
            return Outcome.NO_IMAGERY, []

        prompt = build_prompt(
            lon=float(point["lon"]),
            lat=float(point["lat"]),
            pre_change=date.fromisoformat(point["pre_change"]),
            first_observable=date.fromisoformat(
                point["first_date_change_noticeable"]
            ),
            post_change=date.fromisoformat(point["post_change"]),
            pre_category=point.get("pre_category"),
            post_category=point.get("post_category"),
            images=images,
        )

        if self.config.dry_run or self.gemini is None:
            logger.info("%s: dry run, skipping model call", window_name)
            return Outcome.DRY_RUN, []

        result = self.gemini.categorize(prompt, images)
        logger.info(
            "%s: %s [confidence=%s] %s (%d prompt + %d output tokens)",
            window_name,
            result.categories,
            result.confidence or "unknown",
            result.summary,
            result.prompt_tokens,
            result.candidates_tokens,
        )
        result_path.write_text(
            json.dumps(
                {
                    "categories": result.categories,
                    "confidence": result.confidence,
                    "summary": result.summary,
                }
            )
        )
        return Outcome.DONE, result.categories

    def run(self, entries: list[dict], limit: int | None = None) -> None:
        """Process entries, stopping after ``limit`` genuine attempts."""
        logger.info(
            "Loaded %d entries; limit=%s",
            len(entries),
            limit if limit is not None else "none",
        )
        attempts = 0
        counts = {outcome: 0 for outcome in Outcome}
        for index, entry in enumerate(entries):
            if limit is not None and attempts >= limit:
                break
            try:
                outcome, _ = self.process_entry(entry)
            except Exception:  # noqa: BLE001 - never let one entry kill the run
                logger.exception(
                    "Failed to process entry %s; continuing",
                    entry.get("window_name", "<unknown>"),
                )
                outcome = Outcome.FAILED
            counts[outcome] += 1
            if outcome in _ATTEMPTED:
                attempts += 1
                logger.info(
                    "Progress: %d/%s attempts (scanned %d entries)",
                    attempts,
                    limit if limit is not None else "all",
                    index + 1,
                )
        logger.info(
            "Done. attempts=%d done=%d no_imagery=%d failed=%d cached=%d "
            "no_point=%d dry_run=%d",
            attempts,
            counts[Outcome.DONE],
            counts[Outcome.NO_IMAGERY],
            counts[Outcome.FAILED],
            counts[Outcome.CACHED],
            counts[Outcome.NO_POINT],
            counts[Outcome.DRY_RUN],
        )


def load_entries(json_path: Path) -> list[dict]:
    """Load the annotation entries from a JSON file."""
    with json_path.open() as f:
        return json.load(f)
