import logging

import pytest

from rslp.utils.mp import init_mp

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_init_mp() -> None:
    init_mp()
