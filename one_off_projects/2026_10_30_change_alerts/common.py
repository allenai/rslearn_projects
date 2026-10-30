"""Shared settings for the change alert experiments (round 3, unified config)."""

from dataclasses import dataclass
from datetime import date, timedelta

from rslearn.change_alerts.layers import ChangeAlertTiming

# Weekly frequent images, of which the model uses the latest 4, and a 91-day detection
# range (13 weeks, so that a frequent period starts exactly at the change date).
TIMING = ChangeAlertTiming(
    frequent_period=timedelta(days=7),
    num_frequent=4,
    detection_range=timedelta(days=91),
)
# Quarterly infrequent images before the window start.
INFREQUENT_PERIOD = timedelta(days=90)
NUM_INFREQUENT = 8
INFREQUENT_DURATION = NUM_INFREQUENT * INFREQUENT_PERIOD
NUM_TIMESTEPS = NUM_INFREQUENT + TIMING.num_frequent

# Val and test scenarios: prediction windows whose time to alert on is this many days
# after the change.
END_DAYS = [7, 45, 90]

# Annotations are only used if their change date is within this range. These are the
# same dates as in round 2 (2026_10_07_change_alerts), so the windows are the same.
# Earlier changes do not have enough Sentinel-2 L2A history before them, and later
# changes did not have imagery for the whole detection range when the annotations were
# first exported.
MIN_CHANGE_DATE = date(2019, 1, 1)
MAX_CHANGE_DATE = date(2026, 7, 2)

WINDOW_SIZE = 64
RESOLUTION = 10

# Windows are spread over this many groups ("{source}_{k:02d}", by hashing the
# annotation ID) so that materialization can be split across jobs by group.
NUM_GROUPS = 16

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
# favyen/20260407-change-finder branch): least cloudy scene first within each mosaic
# period.
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

DATASET_ROOT = "/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261030"
# Round 2 datasets, used to test the new runs on the round 2 test scenarios.
ROUND2_DATASET_ROOT = "/weka/dfive-default/rslearn-eai/datasets/change_alerts/20261012"


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


def dataset_path(source: str) -> str:
    """Get the rslearn dataset path of a source (all splits are in one dataset)."""
    return f"{DATASET_ROOT}/{source}"
