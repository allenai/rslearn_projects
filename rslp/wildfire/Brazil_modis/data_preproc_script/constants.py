"""Constants used in the project."""

from pathlib import Path

PROJECT_PATH = Path(__file__).parent
CONFIG_PATH = PROJECT_PATH / "configs"

KEY_COLUMNS = [
    "id",
    "start_date",
    "end_date",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "center_x",
    "center_y",
    "geometry",
]
KEY_COLUMNS_SAMP = [
    "grid_id",
    "start_date",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "center_x",
    "center_y",
    "geometry",
    "fwinx_mean",
    "bucket",
]

EE_CRS = "EPSG:4326"

BANDS_10 = ["B4", "B3", "B2", "B8"]
BANDS_20 = ["B5", "B6", "B7", "B8A", "B11", "B12"]
BANDS_60 = ["B1", "B9", "B10"]
