"""Iteration 0: faithful replica of the production prompt/imagery selection.

Baseline for the prompt-tuning experiments. Selection: one Sentinel-2 image and up to
one high-resolution aerial image per calendar year (each nearest to July 1), captioned
with capture date, in chronological order. Prompt text matches ``prompt.py``.

Interface used by the experiment harness:
    build(record, metas, load_array) -> (prompt_text, images, info)
"""

from __future__ import annotations

from datetime import datetime

from .prompt import ImageRef, build_validation_prompt, label_image

_TARGET_MONTH = 7
_TARGET_DAY = 1
S2_LAYER = "sentinel2"
HR_LAYER = "esri"


def select_one_per_year(metas, layer_name):
    """One image per calendar year for a layer, nearest to mid-year."""
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
        chosen.append(min(cands, key=lambda m: abs((m.time_range[0] - target).days)))
    return chosen


def build(record, metas, load_array):
    """Build the baseline prompt and image list for one point."""
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

    prompt = build_validation_prompt(
        record.predicted_change_date, record.pre_category, record.post_category
    )
    info = {
        "n_s2_used": sum(1 for m in selected if m.layer_name == S2_LAYER),
        "n_aerial_used": sum(1 for m in selected if m.layer_name == HR_LAYER),
    }
    return prompt, images, info
