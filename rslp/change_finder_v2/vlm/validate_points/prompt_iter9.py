"""Iteration 9: iter8 (exact-point crosshair + few-shot) + center-ZOOM preprocessing.

A single ~10 m point is hard to resolve in a ~640 m Sentinel-2 frame, so iter8's
crosshair still asks the model to read a tiny central patch. iter9 keeps everything from
iter8 but renders every image as a TWO-PANEL view: the full ~640 m context on the left
and a magnified crop of the exact center on the right (both with the crosshair). This
makes the labeled ~10 m point large enough to read while preserving context.

Few-shot exemplars are re-rendered in the same two-panel zoom form.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

import os

from .prompt import ImageRef
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year
from .prompt_iter8 import _build_prompt
from .prompt_iter_util import label_image_zoom_pair, select_clearest_per_year

_MAX_S2_PER_YEAR = 2
_CROP_FRAC = 0.4  # central ~256 m magnified in the zoom panel
_EX_DIR = "/data/favyenb/ai_agent_scratch/iter9_exemplars"

_EXEMPLARS = [
    (
        "ex1_positive_water.png",
        "POSITIVE",
        "In the zoom panel the exact point is dry grassland/bare ground early and becomes "
        "standing water later; the aerial confirms the point is submerged in a reservoir. "
        "The conversion is at the point, corroborated in the Sentinel-2 timeline, and "
        "persists.",
    ),
    (
        "ex2_positive_clearing.png",
        "POSITIVE",
        "The zoom panel shows the exact point on dark tree cover early and on cleared, "
        "lighter ground later; the aerial confirms cleared cropland/pasture at the point. "
        "A forest->cropland conversion at the point that persists.",
    ),
    (
        "ex3_negative_atmospheric.png",
        "NEGATIVE",
        "The zoom panel shows the same sparse shrubland at the point in both Sentinel-2 "
        "frames; they differ only in haze/color and the aerial only looks greener due to "
        "a different sensor/season. No conversion at the point -> NEGATIVE.",
    ),
    (
        "ex4_negative_stable.png",
        "NEGATIVE",
        "The zoom panel shows the exact point unchanged across all frames; nearby "
        "buildings/trees do not change the point's own land cover. Stable point -> "
        "NEGATIVE.",
    ),
]

_EXEMPLAR_IMAGES = []
for _fname, _answer, _why in _EXEMPLARS:
    with open(os.path.join(_EX_DIR, _fname), "rb") as _f:
        _EXEMPLAR_IMAGES.append((_answer, _why, _f.read()))

_ZOOM_NOTE = (
    "\n\nEVERY image is shown as TWO PANELS: the LEFT panel is the full ~640 m context "
    "and the RIGHT panel is a magnified crop of the EXACT center so you can read the ~10 "
    "m labeled point clearly. Both panels carry the same crosshair. Base your judgment on "
    "the exact point as seen in the RIGHT (zoom) panel, using the LEFT panel only for "
    "context."
)

_FEWSHOT_NOTE = (
    "\n\nBEFORE the actual case, you are shown FOUR WORKED EXAMPLES from OTHER, unrelated "
    "locations. Each example is one montage: left=early Sentinel-2, middle=late "
    "Sentinel-2, right=high-resolution aerial, and EACH of those is itself a "
    "context+zoom pair with the crosshair on the exact point. Each example is captioned "
    "with the CORRECT answer and reasoning. Study them to calibrate what counts as a "
    "persistent change AT the exact point. The examples are NOT the location you must "
    "judge. AFTER them you receive the FULL time series for the ACTUAL location; apply "
    "the same reasoning to the ACTUAL point ONLY."
)


def build(record, metas, load_array):
    """Build the iter9 prompt: zoom worked examples, then the per-point zoom query."""
    prompt = _build_prompt() + _ZOOM_NOTE + _FEWSHOT_NOTE

    images = []
    for i, (answer, why, png) in enumerate(_EXEMPLAR_IMAGES, start=1):
        label = (
            f"WORKED EXAMPLE {i} (different location) - CORRECT ANSWER: {answer}. {why} "
            "[montage of early Sentinel-2 | late Sentinel-2 | aerial, each shown as "
            "full-context + center-zoom with the crosshair on the exact point]"
        )
        images.append(ImageRef(label=label, png_bytes=png))

    selected = select_clearest_per_year(metas, S2_LAYER, _MAX_S2_PER_YEAR, load_array)
    selected += select_one_per_year(metas, HR_LAYER)
    selected.sort(key=lambda m: m.time_range[0])

    for j, m in enumerate(selected):
        capture = m.time_range[0].date().isoformat()
        kind = "Sentinel-2" if m.layer_name == S2_LAYER else "Aerial (high-res)"
        caption = f"{kind} {capture}"
        if j == 0:
            caption = (
                "=== ACTUAL LOCATION TO JUDGE (full time series follows) === " + caption
            )
        images.append(
            ImageRef(
                label=caption,
                png_bytes=label_image_zoom_pair(load_array(m), caption, _CROP_FRAC),
            )
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
        "n_exemplars": len(_EXEMPLAR_IMAGES),
    }
    return prompt, images, info
