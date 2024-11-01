"""Integration tests for the predict pipeline.
This should be refactored along with the prediction file into seperate components.
"""

import os
import uuid
from collections.abc import Generator

import pytest
from google.cloud import storage

from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    return "cropped_070W_10S_060W_00N.tif"


@pytest.fixture
def test_bucket() -> Generator[storage.Bucket, None, None]:
    """The test bucket."""
    bucket = storage.Client().bucket(os.environ.get("TEST_BUCKET", "rslearn-eai"))
    yield bucket
