"""Iteration 2: iter1 hardening + multiple Sentinel-2 looks per year + calibration.

Rationale from the iter0/iter1 analysis:
- The model fails most on points where only Sentinel-2 is available (AUROC ~chance),
  and most false negatives are there; a single per-year mosaic is often cloudy/hazy.
- ``change_probability`` came back near-binary, so we ask explicitly for a graded,
  calibrated value.

Changes vs iter1:
- Show up to 3 Sentinel-2 images per year (the closest to mid-year), so at least one
  clear look is likely and clouds in one mosaic do not dominate.
- Keep up to one aerial image per year.
- Prompt tells the model that multiple same-year Sentinel-2 images may appear, to use
  the clearest and ignore cloudy ones, and to output a genuinely graded probability.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

from datetime import datetime

from .prompt import ImageRef, label_image
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year

_TARGET_MONTH = 7
_TARGET_DAY = 1
_MAX_S2_PER_YEAR = 3


def _select_k_per_year(metas, layer_name, k):
    """Up to k images per year for a layer, the k closest to mid-year."""
    by_year: dict[int, list] = {}
    for m in metas:
        if m.layer_name != layer_name:
            continue
        by_year.setdefault(m.time_range[0].year, []).append(m)
    chosen = []
    for year in sorted(by_year):
        cands = by_year[year]
        tz = cands[0].time_range[0].tzinfo
        target = datetime(year, _TARGET_MONTH, _TARGET_DAY, tzinfo=tz)
        cands.sort(key=lambda m: abs((m.time_range[0] - target).days))
        chosen.extend(cands[:k])
    return chosen


def _build_prompt(record) -> str:
    lines = [
        "You are an expert remote-sensing analyst. Your job is to decide whether a "
        "genuine, PERSISTENT land-cover change happened AT THE CENTER of a small area "
        "(about 640 m across), or whether any apparent difference is just imaging "
        "artifacts, seasonal/phenological variation, or differences between image "
        "sources.",
        "",
        "You are given a chronological time series of the SAME area, all centered on "
        "the exact point of interest. There are two DIFFERENT image sources, each "
        "captioned with its source and capture date:",
        "  - 'Sentinel-2': 10 m satellite imagery. You may be shown SEVERAL "
        "Sentinel-2 images from the same year; some may be cloudy or hazy. Use the "
        "CLEAREST images and ignore obscured ones. Color and brightness vary between "
        "dates.",
        "  - 'Aerial (high-res)': occasional high-resolution aerial imagery from a "
        "DIFFERENT sensor, often a different season, color balance, and resolution.",
        "",
        "CRITICAL: differences in appearance that come from the IMAGE SOURCE or "
        "conditions are NOT land-cover change. Do NOT call a change based on any of:",
        "  - clouds, cloud shadows, haze, smoke, or dust;",
        "  - snow or ice cover, frozen ground, or changing water level / sun glint on "
        "water;",
        "  - overall brightness/color/contrast differences, or Sentinel-2 vs aerial "
        "(resolution, sharpness, or color) differences;",
        "  - missing-data stripes, blur, or other compression/mosaic artifacts;",
        "  - seasonal vegetation (crop growth or harvest, leaf-on vs leaf-off, grass "
        "greening/browning) that recurs and is not a permanent conversion.",
        "",
        "A change-detection model flagged this point as possibly having a long-term "
        "change, but it is frequently WRONG, so do not assume a change exists.",
    ]

    details = []
    if record.predicted_change_date:
        details.append(
            f"the change was estimated around {record.predicted_change_date}"
        )
    if record.pre_category and record.post_category:
        details.append(
            f"the guessed transition is '{record.pre_category}' -> "
            f"'{record.post_category}'"
        )
    if details:
        lines += ["", "Reference (may be wrong): " + "; ".join(details) + "."]

    lines += [
        "",
        "Decide 'positive' ONLY if ALL of these hold:",
        "  1. a real land-cover conversion is visible at the CENTER of the images;",
        "  2. the post-change state APPEARS and then PERSISTS in multiple later, "
        "reasonably CLEAR images (not a one-image blip);",
        "  3. it cannot be explained by any of the artifacts or seasonal effects "
        "listed above.",
        "Otherwise answer 'negative'. Judge the change from the CLEAR images only; if "
        "you cannot tell because imagery is too obscured or ambiguous, answer "
        "'negative'.",
        "",
        "Also output 'change_probability': a calibrated number in [0, 1] for the "
        "probability that a genuine persistent land-cover change occurred at the "
        "center. Use the FULL range and make it graded: e.g. ~0.9-1.0 only when the "
        "change is obvious and well-corroborated across multiple clear images, "
        "~0.5-0.8 when a change is plausible but partly obscured or weakly supported, "
        "~0.2-0.4 when there is only a hint, and ~0.0-0.1 for clearly stable scenes. "
        "Avoid defaulting to exactly 0 or 1 unless you are certain.",
        "",
        "Give brief reasoning that cites specific dated images and notes which images "
        "were too cloudy/obscured to use, then your prediction, change_probability, "
        "and confidence.",
    ]
    return "\n".join(lines)


def build(record, metas, load_array):
    """Build the iter2 prompt and image list for one point."""
    selected = _select_k_per_year(metas, S2_LAYER, _MAX_S2_PER_YEAR)
    selected += select_one_per_year(metas, HR_LAYER)
    selected.sort(key=lambda m: m.time_range[0])

    images = []
    for m in selected:
        capture = m.time_range[0].date().isoformat()
        kind = "Sentinel-2" if m.layer_name == S2_LAYER else "Aerial (high-res)"
        caption = f"{kind} {capture}"
        images.append(
            ImageRef(label=caption, png_bytes=label_image(load_array(m), caption))
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
    }
    return _build_prompt(record), images, info
