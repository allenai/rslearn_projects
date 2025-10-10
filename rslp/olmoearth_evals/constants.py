"""Constants for the evaluation adapter."""

# The input to the evaluation adapter transform must correspond to these bands.
SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
LANDSAT_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
SENTINEL1_BANDS = ["vv", "vh"]
