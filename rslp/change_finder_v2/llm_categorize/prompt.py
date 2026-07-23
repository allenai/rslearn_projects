"""Category definitions and prompt construction for LLM categorization.

Defines the fine-grained land-cover-change categories and builds the text prompt
(interleaved with image labels) sent to the model for a single positive point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# The fine-grained change categories the model may assign. Order is preserved in
# the prompt and is also the set of allowed values for structured output.
CATEGORIES: dict[str, str] = {
    "deforestation": "Loss of natural forest or tree cover (clearing of trees).",
    "vegetation_growth": (
        "An increase in tree or woody vegetation cover, e.g. grassland, bare "
        "ground, shrub, or cropland developing into forest, reforestation, a "
        "new tree plantation, or regrowth after a prior disturbance. This is "
        "the opposite of deforestation."
    ),
    "new_building": (
        "Construction of a new building or structure, including a newly paved "
        "parking lot."
    ),
    "new_road": "Construction of a new road, highway, or paved path.",
    "new_infrastructure": (
        "New developed infrastructure that is not a building or road, e.g. a "
        "solar farm, wind turbine, power transmission tower, or a new "
        "artificial lake or reservoir (such as one created behind a dam)."
    ),
    "agricultural_activity": (
        "Routine agricultural activity on existing farmland, e.g. harvesting "
        "tree crops, or crop planting/harvesting."
    ),
    "new_crop_field": "A non-crop area being converted into a crop field.",
    "wildfire": "A wildfire or burn scar.",
    "urban_erosion": (
        "Removal/deconstruction of a building, road, or other infrastructure "
        "(the developed feature is taken away)."
    ),
    "new_aquafarm": (
        "Creation of new aquaculture ponds (an aquafarm), e.g. cropland or a "
        "coastal area becoming fish or shrimp ponds. A new lake or reservoir "
        "is new_infrastructure, not this."
    ),
    "site_clearing": (
        "A site that has been cleared (e.g. gravel or dirt surfacing) but where "
        "nothing has been constructed, even by the post-change date."
    ),
    "wetland_loss": (
        "Loss or draining of wetland; use only where the area was predominantly "
        "wetland before the change."
    ),
    "water_change": (
        "A change in the extent or course of a body of water, e.g. land "
        "reclamation from water, coastal erosion, a shifted or newly formed "
        "river path, or a newly flooded area. Use this only when the "
        "predominant change is to the water body itself; it generally should "
        "not be combined with new_road, new_building, or new_infrastructure "
        "(if development is the main change, prefer those instead)."
    ),
}

# Fallback label the model may output when no category clearly applies.
OTHER_CATEGORY = "other"

# Every value the model is allowed to output for a category.
ALLOWED_CATEGORIES = list(CATEGORIES.keys()) + [OTHER_CATEGORY]

# Allowed confidence levels, in descending order.
CONFIDENCE_LEVELS = ["high", "medium", "low"]



@dataclass
class ImageRef:
    """A labeled image to include in the prompt, in display order."""

    label: str
    description: str
    png_bytes: bytes


def build_prompt(
    lon: float,
    lat: float,
    pre_change: date,
    first_observable: date,
    post_change: date,
    pre_category: str | None,
    post_category: str | None,
    images: list[ImageRef],
) -> str:
    """Build the text prompt for a single positive point.

    Args:
        lon: longitude of the annotated point.
        lat: latitude of the annotated point.
        pre_change: annotated date before the change is visible.
        first_observable: annotated date the change first becomes noticeable.
        post_change: annotated date after the change is complete/visible.
        pre_category: coarse land-cover category before the change, if labeled.
        post_category: coarse land-cover category after the change, if labeled.
        images: the labeled images included with the request, in display order.

    Returns:
        the assembled prompt string.
    """
    category_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in CATEGORIES.items()
    )
    image_lines = "\n".join(
        f"- {img.label}: {img.description}" for img in images
    )

    transition = "unknown"
    if pre_category and post_category:
        transition = f"{pre_category} -> {post_category}"
    elif pre_category:
        transition = f"{pre_category} -> unknown"
    elif post_category:
        transition = f"unknown -> {post_category}"

    return f"""You are analyzing satellite and aerial imagery to categorize a \
land-cover change that occurred at a single location.

Location: longitude {lon:.6f}, latitude {lat:.6f}

Annotated dates:
- Before the change is visible (pre-change): {pre_change.isoformat()}
- Change first becomes noticeable: {first_observable.isoformat()}
- After the change (post-change): {post_change.isoformat()}

Annotated coarse land-cover transition: {transition}

The images are provided in this order:
{image_lines}

All images are centered on the same point (the location above). Compare the \
imagery across the dates to identify what changed. The high-resolution aerial \
images are sharper, but their capture dates may be far from the labeled change \
dates; when that is the case, rely more on the Sentinel-2 images, which are \
closer in time to the labeled dates, and do not conclude "no change" just \
because the aerial images look similar.

Choose the change category or categories from the following options:
{category_lines}

Guidance: the annotated coarse transition above usually maps to these change \
categories. Treat it as a strong prior and respect its direction, but rely on \
the imagery to decide:
- tree or shrub -> grassland, bare, crops, or urban/built-up: deforestation \
(loss of tree cover). It may also fit the destination (-> crops: new_crop_field; \
-> urban/built-up: new_building/new_road/new_infrastructure; -> bare: \
site_clearing). If the tree loss is from a fire and a burn scar is visible, use \
wildfire instead (tree -> grassland may be a wildfire).
- grassland, bare, crops, shrub, water, or urban/built-up -> tree or shrub: \
vegetation_growth (gain of tree/woody cover; this is never deforestation).
- grassland, bare, crops, shrub, or water -> urban/built-up: new_building, \
new_road, or new_infrastructure, whichever the imagery shows (water -> \
urban/built-up may also involve water_change from land reclamation).
- bare, grassland, water, or shrub -> crops: new_crop_field.
- grassland, crops, or shrub -> bare: site_clearing.
- urban/built-up -> grassland, bare, crops, shrub, or tree: urban_erosion \
(-> crops may also be new_crop_field).
- water -> bare or grassland: water_change (receding water or land reclamation).
- tree, grassland, crops, or bare -> water: water_change for a natural change \
(flooding or a shifted river); new_infrastructure if it is a reservoir behind a \
dam; new_aquafarm if it becomes regular aquaculture ponds.
- wetland (herbaceous) -> water, bare, crops, or grassland: wetland_loss.
- the same class -> itself (e.g. crops -> crops, tree -> tree): a within-class \
change such as agricultural_activity (cropland) or wildfire (burn scar), or \
seasonal differences; use "other" if nothing clearly changed.

Provide your answer as the following fields:
- summary: a single sentence describing what changed at this location.
- categories: the matching change category name(s) from the list above. Most \
locations have exactly one applicable category; only assign multiple if more \
than one clearly applies. If the change does not clearly fit any of the \
categories above, use "other".
- confidence: your confidence in the categorization, one of "high", "medium", \
or "low"."""
