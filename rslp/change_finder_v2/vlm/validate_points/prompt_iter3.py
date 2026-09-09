"""Iteration 3: honest baseline = iter1 prompt with the leaking reference removed.

Analysis of the iter0-2 runs found that the ``predicted_change_date`` /
``pre_category`` / ``post_category`` reference fields are populated ONLY for the
positive points in the evaluation subset (24/24 positives, 0/276 negatives). The
iter0-2 prompts appended a "Reference (may be wrong): ..." line whenever those fields
were set, so that line appeared exclusively on positives and effectively leaked the
label. To measure (and improve) Gemini's true imagery-based skill we drop that block
entirely here. Imagery selection is identical to iter1 (one Sentinel-2 image and up to
one aerial image per year, each nearest to mid-year) so this isolates the effect of
removing the leak.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

from .prompt import ImageRef, label_image
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year


def _build_prompt() -> str:
    lines = [
        "You are an expert remote-sensing analyst. Your job is to decide whether a "
        "genuine, PERSISTENT land-cover change happened AT THE CENTER of a small area "
        "(about 640 m across), or whether any apparent difference is just imaging "
        "artifacts, seasonal/phenological variation, or differences between image "
        "sources.",
        "",
        "You are given a short, chronological time series of the SAME area, all "
        "centered on the exact point of interest. There are two DIFFERENT image "
        "sources, each captioned with its source and capture date:",
        "  - 'Sentinel-2': 10 m satellite imagery. Color, brightness, and atmospheric "
        "haze vary a lot between dates.",
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
        "",
        "Decide 'positive' ONLY if ALL of these hold:",
        "  1. a real land-cover conversion is visible at the CENTER of the images;",
        "  2. the post-change state APPEARS and then PERSISTS in multiple later, "
        "reasonably CLEAR images (not a one-image blip);",
        "  3. it cannot be explained by any of the artifacts or seasonal effects "
        "listed above.",
        "Otherwise answer 'negative'. When the center is obscured by cloud/haze/snow "
        "in some images, judge the change from the CLEAR images only; if you cannot "
        "tell because imagery is too obscured or ambiguous, answer 'negative'.",
        "",
        "Also output 'change_probability': a calibrated number in [0, 1] for the "
        "probability that a genuine persistent land-cover change occurred at the "
        "center. Use values near 1.0 only when the change is obvious and "
        "well-corroborated; use values near 0.0 for stable scenes; use intermediate "
        "values when imagery is partly cloudy/ambiguous. Be conservative.",
        "",
        "Give brief reasoning that cites specific dated images and notes which images "
        "were too cloudy/obscured to use, then your prediction, change_probability, "
        "and confidence.",
    ]
    return "\n".join(lines)


def build(record, metas, load_array):
    """Build the iter3 prompt and image list for one point."""
    selected = select_one_per_year(metas, S2_LAYER) + select_one_per_year(
        metas, HR_LAYER
    )
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
    return _build_prompt(), images, info
