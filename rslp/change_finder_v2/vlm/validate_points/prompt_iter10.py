"""Iteration 10: best-performing iter7 formula, but a SMALL point-sized box + center zoom.

Findings from iter8/iter9: making the prompt ultra-strict ("judge only the exact ~10 m
pixel; nearby change is negative") dropped AUROC from 0.947 (iter7) to 0.874 and did not
reduce false positives - the strict language, not the visualization, hurt ranking. iter7
(large box + four image few-shot) remained the best.

iter10 keeps iter7's proven prompt and few-shot examples but addresses the original
concern (the ~215 m box was far larger than the ~10 m point label) more gently:

- the marker is shrunk to a SMALL box (~75 m) centered on the labeled point, instead of
  the old ~215 m box;
- every image is rendered as a two-panel context+zoom view so the small box is readable;
- the prompt notes the box now marks the labeled point neighborhood and to focus there,
  WITHOUT the harsh exact-pixel/nearby-negative rules that hurt iter8/iter9.

Few-shot exemplars are re-rendered in the same small-box zoom form.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

import os

from .prompt import ImageRef
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year
from .prompt_iter6 import _build_prompt
from .prompt_iter_util import label_image_zoom_box, select_clearest_per_year

_MAX_S2_PER_YEAR = 2
_CROP_FRAC = 0.4
_BOX_FRAC = 0.12  # ~75 m box marking the labeled point neighborhood
_EX_DIR = "/data/favyenb/ai_agent_scratch/iter10_exemplars"

_EXEMPLARS = [
    (
        "ex1_positive_water.png",
        "POSITIVE",
        "Inside the small box the surface is dry grassland/bare ground in the early "
        "Sentinel-2 frame and becomes standing water later; the aerial confirms the box "
        "is submerged in a reservoir. The conversion is at the marked point, corroborated "
        "in the Sentinel-2 timeline, and persists.",
    ),
    (
        "ex2_positive_clearing.png",
        "POSITIVE",
        "Inside the box the cover is dark continuous trees early and cleared lighter "
        "ground later; the aerial confirms cleared cropland/pasture. A forest->cropland "
        "conversion at the marked point that persists.",
    ),
    (
        "ex3_negative_atmospheric.png",
        "NEGATIVE",
        "Inside the box it is the same sparse shrubland in both Sentinel-2 frames; they "
        "differ only in haze/color and the aerial only looks greener because it is a "
        "different sensor/season. No conversion at the point -> NEGATIVE.",
    ),
    (
        "ex4_negative_stable.png",
        "NEGATIVE",
        "Inside the box the surface is stable across all frames; nearby buildings/trees "
        "outside the box do not change the marked point's land cover. Stable -> NEGATIVE.",
    ),
]

_EXEMPLAR_IMAGES = []
for _fname, _answer, _why in _EXEMPLARS:
    with open(os.path.join(_EX_DIR, _fname), "rb") as _f:
        _EXEMPLAR_IMAGES.append((_answer, _why, _f.read()))

_BOX_NOTE = (
    "\n\nIMPORTANT - the magenta box is SMALL (about 75 m) and marks the specific labeled "
    "POINT, not a wide area. Judge the change at the marked point inside the box; a change "
    "that happens elsewhere in the frame but not at the boxed point does not count. Every "
    "image is shown as TWO PANELS: the LEFT is the full ~640 m context and the RIGHT is a "
    "magnified crop of the boxed point so you can read it clearly; both carry the same "
    "box. Base your judgment on the boxed point (use the RIGHT zoom panel for detail and "
    "the LEFT panel for context)."
)

_FEWSHOT_NOTE = (
    "\n\nBEFORE the actual case, you are shown FOUR WORKED EXAMPLES from OTHER, unrelated "
    "locations. Each is one montage: early Sentinel-2 | late Sentinel-2 | aerial, each as "
    "a context+zoom pair with the small box on the labeled point, captioned with the "
    "CORRECT answer and reasoning. Study them to calibrate what counts as a persistent "
    "change at the boxed point. The examples are NOT the location you must judge. AFTER "
    "them you receive the FULL time series for the ACTUAL location; apply the same "
    "reasoning to the ACTUAL boxed point ONLY."
)


def build(record, metas, load_array):
    """Build the iter10 prompt: small-box zoom worked examples, then the per-point query."""
    prompt = _build_prompt() + _BOX_NOTE + _FEWSHOT_NOTE

    images = []
    for i, (answer, why, png) in enumerate(_EXEMPLAR_IMAGES, start=1):
        label = (
            f"WORKED EXAMPLE {i} (different location) - CORRECT ANSWER: {answer}. {why} "
            "[montage of early Sentinel-2 | late Sentinel-2 | aerial, each as "
            "full-context + zoom with the small box on the labeled point]"
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
                png_bytes=label_image_zoom_box(
                    load_array(m), caption, _CROP_FRAC, _BOX_FRAC
                ),
            )
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
        "n_exemplars": len(_EXEMPLAR_IMAGES),
    }
    return prompt, images, info
