import logging

import pytest
from rslearn.utils.jsonargparse import init_jsonargparse

from rslp.utils.mp import init_mp

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_init_mp() -> None:
    init_mp()


@pytest.fixture(scope="session", autouse=True)
def always_init_jsonargparse() -> None:
    init_jsonargparse()
