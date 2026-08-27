"""Iteration 4: iter3 honest prompt + center box marker + cloud-aware S2 selection.

Targets the two dominant iter1/iter3 error modes:
- False negatives from cloud cover: instead of one Sentinel-2 mosaic per year chosen by
  date (often cloudy), pick the 1-2 CLEAREST mosaics per year by scoring center clarity
  (see ``prompt_iter_util.select_clearest_per_year``). ~59 monthly mosaics exist per
  point, so a clear look almost always exists.
- Wrong location: draw a magenta box around the center of every image
  (``label_image_box``) and tell the model to judge only what is inside / next to it.

No reference fields (kept out, per the iter3 leak analysis). Imagery: up to 2 clear
Sentinel-2 per year + up to 1 aerial per year.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

from .prompt import ImageRef
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year
from .prompt_iter_util import label_image_box, select_clearest_per_year

_MAX_S2_PER_YEAR = 2


def _build_prompt() -> str:
    lines = [
        "You are an expert remote-sensing analyst. Your job is to decide whether a "
        "genuine, PERSISTENT land-cover change happened AT THE CENTER of a small area "
        "(about 640 m across), or whether any apparent difference is just imaging "
        "artifacts, seasonal/phenological variation, or differences between image "
        "sources.",
        "",
        "Every image has a MAGENTA BOX drawn around its center. The location of "
        "interest is INSIDE that box. Judge the change only from what is inside the "
        "box or immediately around it; ignore changes elsewhere in the frame. The box "
        "itself is an overlay, not part of the scene.",
        "",
        "You are given a chronological time series of the SAME area, all centered on "
        "the point of interest. There are two DIFFERENT image sources, each captioned "
        "with its source and capture date:",
        "  - 'Sentinel-2': 10 m satellite imagery. You may see SEVERAL Sentinel-2 "
        "images per year; they were chosen to be among the clearest available, but "
        "some haze or cloud may remain. Color and brightness vary between dates.",
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
        "  1. a real land-cover conversion is visible INSIDE the box;",
        "  2. the post-change state APPEARS and then PERSISTS in multiple later, "
        "reasonably CLEAR images (not a one-image blip);",
        "  3. it cannot be explained by any of the artifacts or seasonal effects "
        "listed above.",
        "Otherwise answer 'negative'. Judge the change from the CLEAR images only; if "
        "you cannot tell because imagery is too obscured or ambiguous, answer "
        "'negative'.",
        "",
        "Also output 'change_probability': a calibrated number in [0, 1] for the "
        "probability that a genuine persistent land-cover change occurred inside the "
        "box. Use the FULL range and make it graded: e.g. ~0.9-1.0 only when the "
        "change is obvious and corroborated across multiple clear images, ~0.5-0.8 "
        "when a change is plausible but partly obscured or weakly supported, ~0.2-0.4 "
        "when there is only a hint, and ~0.0-0.1 for clearly stable scenes. Avoid "
        "defaulting to exactly 0 or 1 unless you are certain.",
        "",
        "Give brief reasoning that cites specific dated images and notes which images "
        "were too cloudy/obscured to use, then your prediction, change_probability, "
        "and confidence.",
    ]
    return "\n".join(lines)


def build(record, metas, load_array):
    """Build the iter4 prompt and image list for one point."""
    selected = select_clearest_per_year(metas, S2_LAYER, _MAX_S2_PER_YEAR, load_array)
    selected += select_one_per_year(metas, HR_LAYER)
    selected.sort(key=lambda m: m.time_range[0])

    images = []
    for m in selected:
        capture = m.time_range[0].date().isoformat()
        kind = "Sentinel-2" if m.layer_name == S2_LAYER else "Aerial (high-res)"
        caption = f"{kind} {capture}"
        images.append(
            ImageRef(label=caption, png_bytes=label_image_box(load_array(m), caption))
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
    }
    return _build_prompt(), images, info
