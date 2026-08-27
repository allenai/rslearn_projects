"""Category definitions and prompt construction for the category tagger.

The tagger asks Gemini to assign fine-grained land-cover-change categories to a flagged
point, shown as a short time series of imagery (one Sentinel-2 image and up to one
high-resolution aerial image per year, each captioned with its capture date).

Categories fall into three groups:

- ``PRE_CATEGORIES``: changes named for what was lost / the pre-change state being
  removed (e.g. ``deforestation``).
- ``POST_CATEGORIES``: changes named for what appeared / the post-change state (e.g.
  ``new_building``).
- ``SAME_CATEGORIES``: the land cover class stays the same with some variation (e.g.
  ``agricultural_activity``).

The model picks one ``PRE`` and/or one ``POST`` category, OR a single ``SAME`` category.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

# Pre-class changes: named for the pre-change state being lost/removed.
PRE_CATEGORIES: dict[str, str] = {
    "deforestation": "Loss of natural forest or tree cover (clearing of trees).",
    "urban_erosion": (
        "Removal/deconstruction of a building, road, or other developed "
        "infrastructure (the developed feature is taken away)."
    ),
    "removed_crop_structure": (
        "Removal/demolition of a greenhouse, polytunnel, or similar agricultural "
        "structure, with the land returning to cropland or open ground."
    ),
    "wetland_loss": (
        "Loss or draining of wetland; use only where the area was predominantly "
        "wetland before the change."
    ),
    "water_contract": (
        "A body of water receding or shrinking, e.g. a drying lake/reservoir, "
        "receding shoreline, or land reclaimed from water."
    ),
}

# Post-class changes: named for the post-change state that appears.
POST_CATEGORIES: dict[str, str] = {
    "vegetation_growth": (
        "An increase in tree or woody vegetation cover, e.g. grassland, bare "
        "ground, shrub, or cropland developing into forest, reforestation, a new "
        "tree plantation, or regrowth after a prior disturbance. Opposite of "
        "deforestation."
    ),
    "new_building": (
        "Construction of a new building or structure, including a newly paved "
        "parking lot."
    ),
    "new_road": "Construction of a new road, highway, or paved path.",
    "new_infrastructure": (
        "New developed infrastructure that is not a building or road, e.g. a solar "
        "farm, wind turbine, power transmission tower, or a new artificial lake or "
        "reservoir (such as one created behind a dam)."
    ),
    "new_crop_field": "A non-crop area being converted into a crop field.",
    "new_crop_structure": (
        "A new greenhouse, polytunnel, or similar agricultural structure built on "
        "existing cropland or farmland."
    ),
    "new_aquafarm": (
        "Creation of new aquaculture ponds (an aquafarm), e.g. cropland or a "
        "coastal area becoming fish or shrimp ponds. A new lake or reservoir is "
        "new_infrastructure, not this."
    ),
    "water_expand": (
        "A body of water growing or appearing, e.g. a newly flooded area, a "
        "shifted/newly formed river path, or rising lake/reservoir levels."
    ),
    "mining": "A new or expanding mine, quarry, or other extraction site.",
    "site_clearing": (
        "A LAST-RESORT category for an area that has been cleared (e.g. gravel or "
        "dirt surfacing) but where nothing has been constructed by the post-change "
        "date. Only use this if the change does not fit mining, new_road, "
        "new_building, or any other category above. If the cleared/bare area is an "
        "extraction site, a road, the footprint of a building, or any other "
        "identifiable development, use that category instead."
    ),
}

# Same-class changes: the land cover class stays the same with some variation.
SAME_CATEGORIES: dict[str, str] = {
    "agricultural_activity": (
        "Routine agricultural activity on existing farmland, e.g. harvesting tree "
        "crops, or crop planting/harvesting (cropland stays cropland)."
    ),
    "wildfire": (
        "A wildfire or burn scar where the land cover class is otherwise unchanged."
    ),
}

# Allowed values for each structured-output field (None means "not applicable").
ALLOWED_PRE = list(PRE_CATEGORIES.keys())
ALLOWED_POST = list(POST_CATEGORIES.keys())
ALLOWED_SAME = list(SAME_CATEGORIES.keys())

# Allowed confidence levels, in descending order.
CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Suggested pre-class categories keyed by the coarse pre-change (source) land cover.
# removed_crop_structure is handled separately since it depends on the destination.
PRE_BY_SOURCE: dict[str, list[str]] = {
    "tree": ["deforestation"],
    "urban/built-up": ["urban_erosion"],
    "water": ["water_contract"],
    "wetland (herbaceous)": ["wetland_loss"],
}

# Suggested post-class categories keyed by the coarse post-change (destination) cover.
POST_BY_DEST: dict[str, list[str]] = {
    "urban/built-up": [
        "new_building",
        "new_infrastructure",
        "new_road",
        "mining",
        "new_crop_structure",
    ],
    "water": ["mining", "new_infrastructure", "water_expand", "new_aquafarm"],
    "tree": ["vegetation_growth"],
    "wetland (herbaceous)": ["vegetation_growth"],
    "bare": ["mining", "new_infrastructure", "water_expand", "site_clearing"],
    "grassland": ["new_infrastructure", "site_clearing"],
    "crops": ["new_crop_field"],
}

# Land-cover classes among which a same-class agricultural_activity change can occur.
SAME_LAND_COVER = {"grassland", "crops", "bare", "shrub", "tree"}


def _suggested_categories(
    pre_cover: str | None, post_cover: str | None
) -> tuple[list[str], list[str], list[str]]:
    """Suggest likely (pre, post, same) categories for a coarse land-cover transition.

    Returns three ordered, de-duplicated lists. Empty lists mean no specific suggestion
    (e.g. when the coarse transition is unknown).
    """
    pre = list(PRE_BY_SOURCE.get(pre_cover, []))
    # A greenhouse/structure removal shows up as urban -> crops/grassland.
    if pre_cover == "urban/built-up" and post_cover in ("crops", "grassland"):
        pre.append("removed_crop_structure")

    post = list(POST_BY_DEST.get(post_cover, []))

    same: list[str] = []
    if pre_cover == "tree" and post_cover == "grassland":
        same.append("wildfire")
    if pre_cover in SAME_LAND_COVER and post_cover in SAME_LAND_COVER:
        same.append("agricultural_activity")

    return pre, post, same


_LABEL_STRIP_HEIGHT = 22
_MIN_DISPLAY_WIDTH = 256
# Black divider between the context and zoom panels.
_PANEL_GAP = 4

# Magenta (rare in natural imagery) marks where the model should look: a hollow box
# around the center crop region and a circle on the exact point of interest.
_BOX_COLOR = (255, 0, 255)
_BOX_WIDTH = 2


@dataclass
class ImageRef:
    """A labeled image to include in the prompt, in display order."""

    label: str
    png_bytes: bytes


def _get_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a legible font, falling back to PIL's default if needed."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def label_image(
    chw_array: npt.NDArray,
    caption: str,
    crop_size: int,
    circle_radius: int,
) -> bytes:
    """Render a CHW uint8 RGB array to a captioned two-panel PNG.

    The LEFT panel shows the full frame for context, with a hollow magenta box around
    the center crop region and a magenta circle on the exact point of interest. The
    RIGHT panel is just that center crop region, upscaled with nearest-neighbor so the
    point of interest fills the panel (no annotations). A caption strip is drawn beneath
    both panels.

    The box, circle, and crop are sized in NATIVE image pixels, so their on-screen size
    depends on how much each panel is upscaled.

    Args:
        chw_array: the image as a (C, H, W) array; the first three bands are used.
        caption: text drawn under the image (e.g. the image type and capture date).
        crop_size: side length (native pixels) of the center crop shown in the right
            panel, and of the box drawn on the left panel.
        circle_radius: radius (native pixels) of the center circle on the left panel.

    Returns:
        PNG-encoded bytes of the captioned two-panel image.
    """
    hwc = np.transpose(np.asarray(chw_array), (1, 2, 0))
    hwc = np.clip(hwc[:, :, :3], 0, 255).astype(np.uint8)
    base = Image.fromarray(hwc, "RGB")
    cx, cy = base.width / 2.0, base.height / 2.0
    half = crop_size / 2.0

    # Right panel: the center crop, taken before any annotations are drawn.
    left, top = int(round(cx - half)), int(round(cy - half))
    zoom = base.crop((left, top, left + crop_size, top + crop_size))

    # Upscale the context panel to a legible size first, then draw the box and circle on
    # it so the line width stays thin (in display pixels) regardless of the upscale.
    context = base.copy()
    scale = 1
    if context.width < _MIN_DISPLAY_WIDTH:
        scale = max(1, _MIN_DISPLAY_WIDTH // context.width)
        context = context.resize(
            (context.width * scale, context.height * scale), Image.NEAREST
        )
    draw = ImageDraw.Draw(context)
    draw.rectangle(
        [
            (cx - half) * scale,
            (cy - half) * scale,
            (cx + half) * scale,
            (cy + half) * scale,
        ],
        outline=_BOX_COLOR,
        width=_BOX_WIDTH,
    )
    draw.ellipse(
        [
            (cx - circle_radius) * scale,
            (cy - circle_radius) * scale,
            (cx + circle_radius) * scale,
            (cy + circle_radius) * scale,
        ],
        outline=_BOX_COLOR,
        width=_BOX_WIDTH,
    )

    panel_h = context.height
    zoom = zoom.resize((panel_h, panel_h), Image.NEAREST)

    combined_w = context.width + _PANEL_GAP + zoom.width
    out = Image.new("RGB", (combined_w, panel_h + _LABEL_STRIP_HEIGHT), (0, 0, 0))
    out.paste(context, (0, 0))
    out.paste(zoom, (context.width + _PANEL_GAP, 0))
    od = ImageDraw.Draw(out)
    od.text((4, panel_h + 4), caption, fill=(255, 255, 255), font=_get_font())

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def build_category_prompt(
    pre_change: str | None,
    first_observable: str | None,
    post_change: str | None,
    pre_category: str | None,
    post_category: str | None,
) -> str:
    """Build the text prompt for categorizing one flagged point.

    Args:
        pre_change: last date the area is still in its pre-change state.
        first_observable: first date the change becomes visible.
        post_change: date by which the change is complete.
        pre_category: the coarse land-cover category before the change, if known.
        post_category: the coarse land-cover category after the change, if known.

    Returns:
        the prompt text.
    """

    def _group_lines(categories: dict[str, str]) -> str:
        return "\n".join(f"- {name}: {desc}" for name, desc in categories.items())

    lines = [
        "You are analyzing a short time series of satellite and aerial imagery to "
        "categorize a land-cover change that occurred at a single location.",
        "",
        "You are given images of the SAME small area, all centered on the exact point of "
        "interest, in chronological order. EACH image has two panels side by side: the "
        "LEFT panel shows the wider context, with a MAGENTA BOX around the location of "
        "interest and a MAGENTA CIRCLE marking its exact center; the RIGHT panel is that "
        "same boxed region zoomed in. Judge the change at the marked point (inside the "
        "box/circle), using the zoomed-in right panel for detail and the left panel for "
        "context.",
        "",
        "IMPORTANT: The wider context (left panel) often contains large, eye-catching "
        "features OUTSIDE the magenta box -- big buildings, building complexes, parking "
        "lots, or other developments. These are DISTRACTORS, not the change you are "
        "categorizing. The actual change at the marked point is frequently smaller or "
        "subtler (e.g. a single road, a cleared strip, or one structure). Categorize "
        "ONLY what changes inside/around the magenta box and circle, and ignore "
        "everything outside it no matter how prominent it is.",
        "",
        "The imagery spans roughly two years before the change through two years after "
        "it: Sentinel-2 satellite images (10 m resolution, the least cloudy available) "
        "and, when available, high-resolution aerial images. Each image is captioned "
        "beneath it with its type and capture date.",
        "",
        "A change-detection model flagged this point as having a long-term change.",
    ]

    details = []
    if pre_change:
        details.append(f"the area is still in its pre-change state as of {pre_change}")
    if first_observable:
        details.append(f"the change first becomes visible around {first_observable}")
    if post_change:
        details.append(f"the change is complete by {post_change}")
    if pre_category and post_category:
        details.append(
            f"the predicted coarse transition is from '{pre_category}' to "
            f"'{post_category}'"
        )
    if details:
        lines.append("For reference, " + "; ".join(details) + ".")

    lines += [
        "",
        "Categorize the change using these category groups.",
        "",
        "Pre-class categories (named for what was lost / removed):",
        _group_lines(PRE_CATEGORIES),
        "",
        "Post-class categories (named for what appeared):",
        _group_lines(POST_CATEGORIES),
        "",
        "Same-class categories (the land cover class stays the same with variation):",
        _group_lines(SAME_CATEGORIES),
    ]

    pre_sugg, post_sugg, same_sugg = _suggested_categories(pre_category, post_category)
    if pre_sugg or post_sugg or same_sugg:
        lines += [
            "",
            f"Given the predicted coarse transition ('{pre_category}' to "
            f"'{post_category}'), the most likely categories are:",
        ]
        if pre_sugg:
            lines.append("- pre-class: " + ", ".join(pre_sugg))
        if post_sugg:
            lines.append("- post-class: " + ", ".join(post_sugg))
        if same_sugg:
            lines.append("- same-class: " + ", ".join(same_sugg))
        lines.append(
            "Pick the best-matching category from these suggestions. If the change "
            "clearly does not match any of them, do NOT force a category from the full "
            "lists above; set flagged_for_review to true instead (see below)."
        )

    has_suggestions = bool(pre_sugg or post_sugg or same_sugg)
    flag_rule = (
        "- If the change does not match one of the suggested categories above, or you "
        "cannot tell what changed, set flagged_for_review to true and leave all three "
        "category fields null so a human can review it. Otherwise set flagged_for_review "
        "to false."
        if has_suggestions
        else "- If none of the categories above reasonably describe the change, or you "
        "cannot tell what changed, set flagged_for_review to true and leave all three "
        "category fields null so a human can review it. Otherwise set flagged_for_review "
        "to false."
    )

    lines += [
        "",
        "Rules for choosing categories:",
        "- Either pick one pre-class category and/or one post-class category (a single "
        "change usually has a pre AND a post, e.g. tree -> building is 'deforestation' "
        "plus 'new_building'; pick just one side if only that side clearly applies), OR "
        "pick exactly one same-class category.",
        "- Do NOT combine a same-class category with a pre-class or post-class category.",
        "- Leave a field null when no category in that group applies.",
        flag_rule,
        "",
        "Focus on the marked point (the magenta box/circle, shown zoomed in the right "
        "panel) and compare across the years; ignore prominent features outside the box. "
        "The high-resolution aerial images are sharper, but their capture dates may be "
        "far from the change date; when that is the case, rely more on the Sentinel-2 "
        "images, which are usually closer in time.",
        "",
        "Provide your answer as the following fields:",
        "- reasoning: brief reasoning grounded in what the images show across the years.",
        "- pre_change_category: one pre-class category name, or null.",
        "- post_change_category: one post-class category name, or null.",
        "- same_change_category: one same-class category name, or null.",
        "- flagged_for_review: true if no category fits and a human should review, "
        "otherwise false.",
        "- confidence: your confidence in the categorization, one of 'high', 'medium', "
        "or 'low'.",
    ]
    return "\n".join(lines)
