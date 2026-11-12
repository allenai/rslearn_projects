"""Band layout of the LCC model output.

The model's ``output_change`` layer has OUTPUT_BANDS: 55 uint16 bands with
probabilities scaled 0-255 and the predicted change dates as days since
TIMESTAMP_EPOCH (see tasks.LCCMultiTask.process_output).
"""

from .timestamp_encoding import TIMESTAMP_EPOCH, UNIX_EPOCH

NUM_LC_CLASSES = 11
BINARY_CHANGE_BAND = 2
SRC_BAND_OFFSET = 3
DST_BAND_OFFSET = SRC_BAND_OFFSET + NUM_LC_CLASSES
# The timestamp section is two bands holding the predicted pre/post change dates
# encoded as integer days since TIMESTAMP_EPOCH (see timestamp_encoding.py).
TS_PRE_DAYS_BAND = DST_BAND_OFFSET + NUM_LC_CLASSES
TS_POST_DAYS_BAND = TS_PRE_DAYS_BAND + 1

# Land cover classes of the pre/post_category annotation fields and the src/dst
# heads (class 0 = nodata).
LC_CLASS_NAMES = [
    "nodata",
    "bare",
    "agriculture",
    "grassland",
    "shrub",
    "snow/ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland",
    "mangrove",
]

# Change-category classes of the pre/post_change_category annotation fields and the
# change-category heads. Class layout is [nodata, none, <options...>]: class 0 =
# nodata (masked, no point or neither field set); class 1 = "none" (this field unset
# but the other field is set); classes >= 2 = the annotated options. The annotation
# lab (olmoearth_studio/ui/labs/lcc-annotator/FIELDS.md) offers finer categories,
# which prepare.py maps onto these classes.
PRE_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "forest_loss",
    "removed_building",
    "wetland_loss",
    "water_to_land",
    "removed_crop_structure",
    "mangrove_loss",
]
POST_CHANGE_CATEGORY_NAMES = [
    "nodata",
    "none",
    "vegetation_growth",
    "new_building",
    "new_road",
    "new_infrastructure",
    "new_agriculture",
    "new_aquafarm",
    "site_clearing",
    "land_to_water",
    "new_mining_or_quarry",
    "new_crop_structure",
    "selective_logging",
    "landslide",
    "settlement",
    "agricultural_activity",
    "burned",
    "flooding",
    "storm_damage",
    "vegetation_dieback",
]

# Change-category output bands, in the same order as the model output and the
# output_change layer in config.json.
PRE_CHANGE_BANDS = [f"pre_change_{name}" for name in PRE_CHANGE_CATEGORY_NAMES]
POST_CHANGE_BANDS = [f"post_change_{name}" for name in POST_CHANGE_CATEGORY_NAMES]

# Band offsets of the two change-category sections within the output raster.
PRE_CHANGE_BAND_OFFSET = TS_POST_DAYS_BAND + 1
POST_CHANGE_BAND_OFFSET = PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)

OUTPUT_LAYER = "output_change"
OUTPUT_BANDS = [
    "binary_nodata",
    "binary_no_change",
    "binary_change",
    *(
        f"src_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    *(
        f"dst_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    "ts_pre_days",
    "ts_post_days",
    *PRE_CHANGE_BANDS,
    *POST_CHANGE_BANDS,
]

# Class index of the "none" category in both change category heads (index 0 is
# nodata).
NONE_CATEGORY_IDX = 1

# Add to a day value since TIMESTAMP_EPOCH to get days since 1970-01-01.
UNIX_EPOCH_DAY_OFFSET = (TIMESTAMP_EPOCH - UNIX_EPOCH).days
