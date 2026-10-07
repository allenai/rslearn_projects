"""Shared settings for the change alert experiments."""

from dataclasses import dataclass
from datetime import date, timedelta

# Change detection range: the model should detect changes up to this old.
DETECTION_RANGE = timedelta(days=90)
FREQUENT_PERIOD = timedelta(days=7)
INFREQUENT_PERIOD = timedelta(days=90)
NUM_SLOTS = 4

# The frequent layers cover 26 weeks so that the Recent model (20 weekly images) can
# fall back to older weeks; the History model only uses the last HISTORY_LOOKBACK of it.
FREQUENT_DURATION = timedelta(days=182)
# The History model takes its weekly images from this lookback before the latest
# weekly image, and the infrequent layers end this long before the slot end.
HISTORY_LOOKBACK = timedelta(days=60)
INFREQUENT_DURATION = timedelta(days=900)
# The test_history dataset only needs the frequent images within HISTORY_LOOKBACK.
TEST_HISTORY_FREQUENT_DURATION = timedelta(days=63)

# Test scenarios: the time series ends this many days after the change.
TEST_END_DAYS = [7, 45, 90]

# The number of images input to the model for each context: the history context has
# 8 quarterly and HISTORY_NUM_FREQUENT weekly images, the recent context has
# RECENT_NUM_FREQUENT weekly images.
HISTORY_NUM_INFREQUENT = 8
HISTORY_NUM_FREQUENT = 4
RECENT_NUM_FREQUENT = 20
NUM_TIMESTEPS = {
    "history": HISTORY_NUM_INFREQUENT + HISTORY_NUM_FREQUENT,
    "recent": RECENT_NUM_FREQUENT,
}

# Annotations are only used if their change date is within this range. Earlier changes
# do not have INFREQUENT_DURATION of Sentinel-2 L2A history before them. Later changes
# do not have imagery for the whole detection range yet: the cutoff is DETECTION_RANGE
# plus a week before 2026-10-07, when the datasets were created.
MIN_CHANGE_DATE = date(2019, 1, 1)
MAX_CHANGE_DATE = date(2026, 10, 7) - DETECTION_RANGE - timedelta(days=7)

WINDOW_SIZE = 64
RESOLUTION = 10

# Train/val/test fractions, assigned by hashing a SPLIT_CELL_SIZE (in pixels) grid cell.
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
SPLIT_CELL_SIZE = 1000

BANDS = [
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

# Same data source as data/olmoearth_lcc/lcc_model/config.json (on the
# favyen/20260407-change-finder branch): least cloudy scene
# first within each mosaic period.
DATA_SOURCE = {
    "class_path": "olmoearth_run.runner.tools.rslearn_data_sources.olmoearth_datasets.sentinel2_l2a.Sentinel2L2A",
    "ingest": False,
    "init_args": {
        "cache_dir": "cache/olmoearth_datasets",
        "harmonize": True,
        "provider_excludes": ["AWS_SENTINEL_2_L2A_COGS"],
        "provider_priority": ["PLANETARY_COMPUTER"],
        "query": {"sort_by": "CLOUD_COVER", "sort_direction": "ASC"},
        "timeout": "0:0:10",
    },
}

IMAGE_BAND_SETS = [
    {
        "bands": BANDS,
        "dtype": "uint16",
        "format": {"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    }
]

DATASET_ROOT = "/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261007"


@dataclass
class SourceDataset:
    """A Studio project to build a change alert dataset from."""

    # Studio project ID.
    project_id: str
    # Category names. Index 0 is reserved for nodata in the label raster, so class
    # IDs are 1 + the index in this list. The first category is the negative one.
    # Only categories with at least 50 annotations within the change date range are
    # used; annotations of other categories are skipped.
    categories: list[str]
    # Task source_file attributes to skip.
    skip_source_files: tuple[str, ...] = ()


SOURCE_DATASETS = {
    "forest_loss": SourceDataset(
        project_id="39088116-7327-4a1d-bce5-49cb55078c87",
        categories=[
            "none",
            "agriculture",
            "burned",
            "hurricane",
            "landslide",
            "logging",
            "mining",
            "river",
            "road",
        ],
    ),
    "lcc": SourceDataset(
        project_id="10af2d3e-12b1-41de-8ee2-af70255b9fa4",
        categories=[
            "none",
            "mining",
            "new_building",
            "new_crop_field",
            "new_road",
            "settlement",
            "site_clearing",
            "water_expand",
            "wildfire",
        ],
        # These batches have about five negatives per window.
        skip_source_files=(
            "annotations_original.json",
            "annotations_batch2_20260524.json",
            "annotations_batch3_20260606_with_timestamps.json",
        ),
    ),
    "mangrove": SourceDataset(
        project_id="0654f6b6-3ada-47e9-a8eb-18dd055412ee",
        categories=["no_change", "mangrove_loss"],
    ),
}


def dataset_path(source: str, kind: str) -> str:
    """Get the default rslearn dataset path.

    Args:
        source: the SOURCE_DATASETS key.
        kind: "train", "test_history", or "test_recent".
    """
    return f"{DATASET_ROOT}/{source}/{kind}"
