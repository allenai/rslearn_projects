"""Shared constants for the annotation timestamp helper."""

SENTINEL2_LAYER = "sentinel2_monthly"
MODEL_INPUT_KEY = "sentinel2_l2a"
LABEL_LAYER = "label_timestamps"
OUTPUT_LAYER = "output_timestamps"
MANIFEST_FNAME = "timestamp_helper_manifest.json"

NUM_CROP_MONTHS = 60
WINDOW_SIZE = 32

TIMESTAMP_HEADS = ("pre", "first", "post")
DATE_FIELDS = {
    "pre": "pre_change",
    "first": "first_date_change_noticeable",
    "post": "post_change",
}
