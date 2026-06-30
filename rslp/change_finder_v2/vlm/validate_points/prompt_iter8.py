"""Iteration 8: pinpoint the EXACT center point (point labels are ~10 m, one S2 pixel).

The labels are POINT labels: they describe only the single ~10 m location (one Sentinel-2
pixel) at the frame center, not the ~215 m box that iter4-iter7 drew. That oversized box
caused false positives whenever any change appeared somewhere in the box even though the
exact labeled point did not change. iter8 keeps iter7's cloud-aware Sentinel-2 selection
and four image few-shot examples, but:

- replaces the box with a MAGENTA CROSSHAIR reticle that has an open center gap, so the
  exact target pixel stays visible and is unambiguously marked;
- rewrites the prompt to judge ONLY the exact crosshair point: change that is nearby but
  does not cover the exact point is NEGATIVE.

Few-shot exemplars are re-rendered with the same crosshair so the demonstration matches.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

import os

from .prompt import ImageRef
from .prompt_iter0 import HR_LAYER, S2_LAYER, select_one_per_year
from .prompt_iter_util import label_image_crosshair, select_clearest_per_year

_MAX_S2_PER_YEAR = 2
_EX_DIR = "/data/favyenb/ai_agent_scratch/iter8_exemplars"

_EXEMPLARS = [
    (
        "ex1_positive_water.png",
        "POSITIVE",
        "At the exact crosshair point the surface is dry grassland/bare ground in the "
        "early Sentinel-2 frame and becomes standing water in the later frames; the "
        "aerial confirms the point is submerged in a reservoir. The conversion is at the "
        "point itself, corroborated within the Sentinel-2 timeline, and persists.",
    ),
    (
        "ex2_positive_clearing.png",
        "POSITIVE",
        "The crosshair point is on dark continuous tree cover early and on cleared, "
        "lighter ground later; the aerial confirms the exact point became cleared "
        "cropland/pasture. The forest->cropland conversion happens AT the point and "
        "persists.",
    ),
    (
        "ex3_negative_atmospheric.png",
        "NEGATIVE",
        "The crosshair point is the same sparse shrubland in both Sentinel-2 frames; the "
        "frames differ only in haze/color and the aerial only looks greener because it is "
        "a different sensor/season. No conversion at the point -> NEGATIVE.",
    ),
    (
        "ex4_negative_stable.png",
        "NEGATIVE",
        "The crosshair point stays the same surface across all frames; nearby buildings "
        "and trees do not change the point's own land cover. A stable point with minor "
        "seasonal color change -> NEGATIVE.",
    ),
]

_EXEMPLAR_IMAGES = []
for _fname, _answer, _why in _EXEMPLARS:
    with open(os.path.join(_EX_DIR, _fname), "rb") as _f:
        _EXEMPLAR_IMAGES.append((_answer, _why, _f.read()))


def _build_prompt() -> str:
    lines = [
        "You are an expert remote-sensing analyst. You must decide whether a genuine, "
        "PERSISTENT land-cover change happened at ONE EXACT POINT - a single ~10 m "
        "location (one Sentinel-2 pixel) at the center of each image - or whether any "
        "apparent difference is just imaging artifacts, seasonal/phenological variation, "
        "a transient look-alike, or differences between image sources.",
        "",
        "Every image has a MAGENTA CROSSHAIR whose tick marks point inward to the EXACT "
        "point of interest at the center (the small open gap between the ticks is the "
        "target, about one Sentinel-2 pixel / ~10 m). The crosshair is an overlay, not "
        "part of the scene.",
        "",
        "MOST IMPORTANT: judge ONLY the EXACT crosshair point. The surrounding scene is "
        "context only. A change that happens NEARBY - even a few pixels away, or "
        "elsewhere in the frame - but does NOT cover the exact crosshair point is "
        "NEGATIVE. Ask specifically: did the land cover AT the crosshair point itself "
        "convert from one type to another and stay converted?",
        "",
        "You are given a chronological time series of the SAME area, all centered on the "
        "point, from two DIFFERENT sources, each captioned with its source and date:",
        "  - 'Sentinel-2': 10 m satellite imagery; SEVERAL clear-ish images per year. "
        "This series is your TIMELINE.",
        "  - 'Aerial (high-res)': occasional high-resolution aerial imagery from a "
        "DIFFERENT sensor/season/resolution. Use it to read in DETAIL what the land "
        "cover is at the point, not as a timeline.",
        "",
        "RULE 1 - same source only: a real change must be corroborated WITHIN one "
        "source's own time series (Sentinel-2 before vs after, OR aerial vs aerial). "
        "NEVER conclude change from an aerial-vs-Sentinel-2 difference alone.",
        "RULE 2 - persist to the end: the new state at the point must still be present in "
        "the LATEST clear images. If the point reverts to its earlier appearance, the "
        "difference is seasonal/transient -> NEGATIVE.",
        "RULE 3 - class conversion AT the point: the cover at the exact point must change "
        "from one type to a DIFFERENT type (forest->bare/crop/built, water->land, "
        "vegetation->buildings). Within-class fluctuation (plow/green/harvest of the same "
        "field, leaf-on/off, greening/browning) is NOT a conversion.",
        "",
        "Do NOT call a change based on any of these look-alikes:",
        "  - clouds, shadows, haze, smoke, dust, or brightness/color/contrast shifts;",
        "  - snow/ice, frozen ground;",
        "  - changing water level, shoreline, river width, flooding, wet-vs-dry soil, or "
        "sun glint;",
        "  - ephemeral river sandbars, braided channels, exposed riverbed, mudflats;",
        "  - a single small transient object (boat, vehicle, equipment, one building) "
        "appearing/disappearing for one date;",
        "  - seasonal vegetation: crop growth/harvest, tillage, leaf-on/off, "
        "greening/browning;",
        "  - missing-data stripes, blur, mosaic/compression artifacts;",
        "  - any change that does not cover the exact crosshair point.",
        "",
        "A change-detection model flagged this point, but it is frequently WRONG, so do "
        "not assume a change exists.",
        "",
        "Decide 'positive' ONLY if ALL hold: (1) a real land-cover CLASS conversion is "
        "visible AT the exact crosshair point; (2) it is corroborated within one source's "
        "timeline and PERSISTS through the latest clear images; (3) it is not explained "
        "by any artifact, seasonal effect, or transient look-alike. Otherwise 'negative'. "
        "Judge from CLEAR images only; if too obscured or ambiguous, answer 'negative'.",
        "",
        "Also output 'change_probability': a calibrated number in [0, 1] for a genuine "
        "persistent conversion AT the exact point. Use the full graded range (~0.9-1.0 "
        "obvious and persistent at the point; ~0.5-0.8 plausible but obscured/weak or "
        "uncertain it is exactly at the point; ~0.2-0.4 a hint; ~0.0-0.1 clearly stable "
        "or off-point/transient). Avoid exactly 0 or 1 unless certain.",
        "",
        "Give brief reasoning citing specific dated images (including the latest, to "
        "confirm persistence), state whether the change is AT the exact point or merely "
        "nearby, then your prediction, change_probability, and confidence.",
    ]
    return "\n".join(lines)


_FEWSHOT_NOTE = (
    "\n\nBEFORE the actual case, you are shown FOUR WORKED EXAMPLES from OTHER, unrelated "
    "locations. Each example is one montage image: left = early Sentinel-2, middle = late "
    "Sentinel-2, right = high-resolution aerial, all with the same crosshair on the exact "
    "point. Each is captioned with the CORRECT answer and the reasoning. Study them to "
    "calibrate what does and does not count as a persistent change AT the exact point. "
    "The examples are NOT the location you must judge. AFTER them you receive the FULL "
    "time series for the ACTUAL location; apply the same reasoning to the ACTUAL point "
    "ONLY."
)


def build(record, metas, load_array):
    """Build the iter8 prompt: four worked examples, then the per-point query."""
    prompt = _build_prompt() + _FEWSHOT_NOTE

    images = []
    for i, (answer, why, png) in enumerate(_EXEMPLAR_IMAGES, start=1):
        label = (
            f"WORKED EXAMPLE {i} (different location) - CORRECT ANSWER: {answer}. {why} "
            "[montage: left=early Sentinel-2, middle=late Sentinel-2, right=high-res "
            "aerial; crosshair = exact point]"
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
                label=caption, png_bytes=label_image_crosshair(load_array(m), caption)
            )
        )

    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
        "n_exemplars": len(_EXEMPLAR_IMAGES),
    }
    return prompt, images, info
