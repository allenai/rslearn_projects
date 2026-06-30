"""Iteration 6: iter5 + persistence-to-end and a transient/seasonal taxonomy.

iter4 fixed recall (cloud-aware Sentinel-2 + center box) and iter5 trimmed cross-source
false positives. The dominant remaining false positives are confident calls on
WITHIN-source differences that merely look like change but are transient or seasonal:
changing water level / shoreline / flooding, ephemeral riverbed sandbars and channels, a
single small object (building, boat, vehicle) appearing or disappearing for one date,
plowing/tillage of the same field, and deciduous leaf-off or harvest. iter5's
same-source rule does not catch these because they ARE corroborated within one source.

iter6 keeps iter4/iter5 imagery (center box + cloud-aware Sentinel-2) and the same-source
corroboration rule, and adds:

- a strict PERSISTENCE-TO-END test: the new state must still be present in the LAST
  clear images of the series; if the area ever reverts to its earlier appearance, it is
  seasonal/transient -> negative;
- an explicit taxonomy of transient look-alikes (water level, ephemeral channels, single
  transient object, tillage) on top of the existing artifact/season list;
- a requirement that the change be a CLASS conversion (e.g. forest->bare, field->built),
  not a within-class fluctuation of the same land cover.

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
        "artifacts, seasonal/phenological variation, a transient look-alike, or "
        "differences between image sources.",
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
        "some haze or cloud may remain. This series is your TIMELINE.",
        "  - 'Aerial (high-res)': occasional high-resolution aerial imagery from a "
        "DIFFERENT sensor, often a different season, color balance, and resolution. "
        "Use it to read in DETAIL what the land cover is, not as a timeline.",
        "",
        "RULE 1 - same source only: a real change must be corroborated WITHIN A SINGLE "
        "SOURCE'S OWN time series - Sentinel-2 before vs Sentinel-2 after, OR aerial vs "
        "aerial. NEVER conclude change from an aerial-vs-Sentinel-2 difference alone: "
        "an aerial image looking different from a Sentinel-2 image is expected "
        "(different sensor, season, resolution) and is NOT evidence of change.",
        "",
        "RULE 2 - the new state must PERSIST TO THE END: a genuine long-term change "
        "appears at some date and then STAYS in every later reasonably clear image, "
        "including the LAST images of the series. Before concluding 'positive', check "
        "the most recent clear images: if the area ever REVERTS to roughly its earlier "
        "appearance, the difference is seasonal or transient -> answer 'negative'. A "
        "one- or two-image deviation that later disappears is NOT a change.",
        "",
        "RULE 3 - it must be a CLASS conversion: the land cover must change from one "
        "type to a DIFFERENT type (e.g. forest -> bare ground / cropland / built, water "
        "-> land, vegetation -> buildings). Fluctuations of the SAME land cover (a field "
        "that is plowed then green then harvested, grass greening then browning, a "
        "forest losing then regrowing leaves) are NOT a conversion.",
        "",
        "Do NOT call a change based on any of these look-alikes (none is a persistent "
        "land-cover conversion):",
        "  - clouds, cloud shadows, haze, smoke, dust, or overall "
        "brightness/color/contrast differences between dates;",
        "  - snow or ice cover, frozen ground;",
        "  - changing WATER LEVEL, shoreline, river width, flooding/inundation, "
        "wet-vs-dry soil, or sun glint on water (water that advances or retreats and "
        "comes back is seasonal);",
        "  - EPHEMERAL river features: shifting sandbars, braided channels, exposed "
        "riverbed, mudflats;",
        "  - a SINGLE small transient object appearing or disappearing for one date (a "
        "boat, vehicle, parked equipment, a single building, a haystack) - this is not "
        "a land-cover conversion of the area;",
        "  - seasonal vegetation: crop growth or harvest, tillage/plowing of the same "
        "field, leaf-on vs leaf-off, grass greening/browning;",
        "  - missing-data stripes, blur, or compression/mosaic artifacts.",
        "",
        "A change-detection model flagged this point as possibly having a long-term "
        "change, but it is frequently WRONG, so do not assume a change exists.",
        "",
        "Decide 'positive' ONLY if ALL of these hold:",
        "  1. a real land-cover CLASS conversion is visible INSIDE the box (RULE 3);",
        "  2. it is corroborated within one source's timeline (RULE 1) and the "
        "post-change state PERSISTS through the latest clear images, with no reversion "
        "(RULE 2);",
        "  3. it cannot be explained by any artifact, seasonal effect, or transient "
        "look-alike listed above.",
        "Otherwise answer 'negative'. Judge from CLEAR images only; if you cannot tell "
        "because imagery is too obscured or ambiguous, answer 'negative'.",
        "",
        "Also output 'change_probability': a calibrated number in [0, 1] for the "
        "probability that a genuine persistent land-cover change occurred inside the "
        "box. Use the FULL range and make it graded: ~0.9-1.0 only when the conversion "
        "is obvious, corroborated across multiple clear Sentinel-2 images, AND still "
        "present in the latest images; ~0.5-0.8 when plausible but partly obscured, "
        "weakly supported, or you are unsure it persists to the end; ~0.2-0.4 for only "
        "a hint; ~0.0-0.1 for clearly stable scenes or transient/seasonal differences. "
        "Avoid defaulting to exactly 0 or 1 unless you are certain.",
        "",
        "Give brief reasoning that cites specific dated images (including the latest "
        "ones, to confirm persistence), notes which images were too cloudy/obscured to "
        "use, and names any transient/seasonal effect you ruled in or out; then your "
        "prediction, change_probability, and confidence.",
    ]
    return "\n".join(lines)


def build(record, metas, load_array):
    """Build the iter6 prompt and image list for one point."""
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
