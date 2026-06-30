"""Iteration 7: iter6 + four image few-shot worked examples.

iter6 reached the best accuracy/F1 with prompt rules alone. iter7 adds four WORKED
EXAMPLES from held-out points (not in the 300-point eval subset), each a compact montage
(early Sentinel-2 | late Sentinel-2 | high-res aerial) captioned with the correct answer
and a short rationale, to calibrate the change-vs-noise boundary by demonstration:

- two POSITIVE examples (grassland -> reservoir water; forest -> cleared cropland) that
  show a real conversion corroborated within the Sentinel-2 timeline and persisting;
- two NEGATIVE examples (Sahel sparse shrubland that only differs by haze/season and
  source; a stable farmstead with trees and buildings present throughout) that show the
  atmospheric / cross-source / stable-scene traps that must NOT be called change.

The per-point query imagery is identical to iter6 (center box + cloud-aware Sentinel-2).

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

import os

from .prompt import ImageRef
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year
from .prompt_iter6 import _MAX_S2_PER_YEAR, _build_prompt
from .prompt_iter_util import label_image_box, select_clearest_per_year

_EX_DIR = "/data/favyenb/ai_agent_scratch/iter7_exemplars"

# (filename, correct answer, rationale shown to the model)
_EXEMPLARS = [
    (
        "ex1_positive_water.png",
        "POSITIVE",
        "The box is dry grassland/bare ground in the early Sentinel-2 frame, then becomes "
        "dark standing water in the later Sentinel-2 frame, and the aerial confirms the "
        "box is submerged in a reservoir. The new water state is corroborated within the "
        "Sentinel-2 timeline and persists, so this is a genuine grassland->water "
        "conversion.",
    ),
    (
        "ex2_positive_clearing.png",
        "POSITIVE",
        "The early Sentinel-2 frame shows dark, continuous tree cover inside the box; the "
        "later Sentinel-2 frame shows it opened to lighter cleared ground with field/track "
        "lines, and the aerial confirms cleared pasture/cropland where forest used to be. "
        "The conversion is visible within the Sentinel-2 timeline and persists -> genuine "
        "tree->cropland change.",
    ),
    (
        "ex3_negative_atmospheric.png",
        "NEGATIVE",
        "Both Sentinel-2 frames show the same sparse Sahel shrubland inside the box; they "
        "differ mainly in haze and color cast between dates, and the aerial only looks "
        "greener because it is a different sensor and season. Nothing is converted - the "
        "differences are atmospheric/seasonal and cross-source -> NEGATIVE.",
    ),
    (
        "ex4_negative_stable.png",
        "NEGATIVE",
        "The dark trees and bright farm buildings inside the box are present in both the "
        "early and late Sentinel-2 frames and in the aerial; no land-cover type is "
        "replaced by another. A stable farmstead with minor seasonal color change is NOT "
        "a land-cover change -> NEGATIVE.",
    ),
]

# Load exemplar montage bytes once at import.
_EXEMPLAR_IMAGES = []
for _fname, _answer, _why in _EXEMPLARS:
    with open(os.path.join(_EX_DIR, _fname), "rb") as _f:
        _EXEMPLAR_IMAGES.append((_answer, _why, _f.read()))

_FEWSHOT_NOTE = (
    "\n\nBEFORE the actual case, you are shown FOUR WORKED EXAMPLES from OTHER, "
    "unrelated locations. Each example is a single montage image: left = an early "
    "Sentinel-2 frame, middle = a late Sentinel-2 frame, right = a high-resolution "
    "aerial image, all with the same magenta box. Each example is captioned with the "
    "CORRECT answer and the reasoning. Study them to calibrate what does and does not "
    "count as a persistent land-cover change. The examples are NOT the location you must "
    "judge. AFTER the four examples you will receive the FULL time series for the ACTUAL "
    "location; apply the same reasoning and output your decision for the ACTUAL location "
    "ONLY."
)


def build(record, metas, load_array):
    """Build the iter7 prompt: four worked examples, then the per-point query."""
    prompt = _build_prompt() + _FEWSHOT_NOTE

    images = []
    for i, (answer, why, png) in enumerate(_EXEMPLAR_IMAGES, start=1):
        label = (
            f"WORKED EXAMPLE {i} (different location) - CORRECT ANSWER: {answer}. {why} "
            "[montage: left=early Sentinel-2, middle=late Sentinel-2, right=high-res "
            "aerial; magenta box = area to judge]"
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
            ImageRef(label=caption, png_bytes=label_image_box(load_array(m), caption))
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
        "n_exemplars": len(_EXEMPLAR_IMAGES),
    }
    return prompt, images, info
