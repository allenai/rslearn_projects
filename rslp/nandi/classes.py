"""Class definitions for Nandi County crop type and land cover mapping.

This module is the single source of truth for the class table. The rasterizer, the
training configs (via ``class_names``), and any prediction colormap all derive from
``CLASS_NAMES``, so class IDs cannot drift apart between the label rasters and the
model outputs.
"""

# The index of a name in this list is the class ID burned into the `label_raster`
# layer and predicted by the model. Order: annual crops, perennial crops, then
# natural / land cover classes. Appending is safe; reordering invalidates every
# existing label raster and checkpoint.
CLASS_NAMES = [
    "Maize",
    "Sugarcane",
    "Legumes",
    "Vegetables",
    "Coffee",
    "Tea",
    "Grassland",
    "Trees",
    "Shrubland",
    "Water",
    "Built-up",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Written to label_raster wherever no polygon covers the pixel center. It is kept
# outside [0, NUM_CLASSES) on purpose: SegmentationTask then masks these pixels out of
# both the loss and the metrics without the model spending a logit on them.
IGNORE_VALUE = 255

# Raw `Category` values found in the CGIAR ground truth and Studio annotations, mapped
# onto CLASS_NAMES. Values already equal to a class name need no entry here.
SOURCE_CATEGORY_MAP = {
    "Exoticetrees/forest": "Trees",
    "Nativetrees/forest": "Trees",
    "Exotic trees/forest": "Trees",
    "Native trees/forest": "Trees",
    "Tree cover": "Trees",
    "Builtup": "Built-up",
    "Built up": "Built-up",
}

# ESA WorldCover v200 codes we are willing to take weak labels from.
#
# Deliberately excluded: Tree cover (10), Cropland (40) and Grassland (30). Coffee and
# Tea both map to Tree cover or Cropland in WorldCover, so using those codes as labels
# would teach the model exactly the Trees/Coffee confusion we are trying to remove.
WORLDCOVER_CLASS_MAP = {
    20: "Shrubland",
    50: "Built-up",
    80: "Water",
}

# Display colour per class, indexed like CLASS_NAMES. Kept here so the label
# visualizations, any prediction colormap, and SegmentationTask(colors=...) all agree.
# Chosen so the classes most often confused with each other -- Coffee, Tea and Trees --
# sit far apart in hue rather than being three neighbouring greens.
CLASS_COLORS = [
    (230, 159, 0),  # Maize        amber
    (163, 217, 68),  # Sugarcane    lime
    (204, 121, 167),  # Legumes      orchid
    (240, 118, 175),  # Vegetables   pink
    (140, 43, 38),  # Coffee       deep red-brown
    (0, 158, 158),  # Tea          teal
    (222, 217, 116),  # Grassland    straw
    (20, 92, 45),  # Trees        dark green
    (150, 138, 84),  # Shrubland    olive
    (54, 111, 214),  # Water        blue
    (130, 130, 138),  # Built-up     grey
]

IGNORE_COLOR = (28, 28, 32)

# Per-class overrides for WorldCover blob extraction. Water covers only 2,982 pixels
# in all of Nandi County, so the default 2-pixel erosion and 16-pixel minimum leave
# just 9 blobs -- few enough that a spatial split can put every one of them in train.
WORLDCOVER_EXTRACTION_OVERRIDES = {
    80: {"erosion_iterations": 1, "min_blob_pixels": 4},
}

# Priority of each label source, used as the rasterization draw order: higher priority
# is drawn last and therefore wins where sources disagree.
SOURCE_PRIORITY = {
    "worldcover": 0,
    "studio": 1,
    "groundtruth": 2,
}


def normalize_category(raw: str) -> str | None:
    """Map a raw source category onto a name in CLASS_NAMES.

    Args:
        raw: the category string as it appears in the source data.

    Returns:
        the normalized class name, or None if the category is not one we model.
    """
    if raw is None:
        return None
    name = SOURCE_CATEGORY_MAP.get(raw.strip(), raw.strip())
    if name not in CLASS_IDS:
        return None
    return name
