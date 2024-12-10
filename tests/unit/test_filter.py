"""Unit tests for the NearInfraFilter class."""

from unittest.mock import patch

import numpy as np
import pytest

from rslp.log_utils import get_logger
from rslp.utils.filter import NearInfraFilter

logger = get_logger(__name__)


@pytest.fixture
def infra_coords() -> tuple[float, float]:
    """Fixture providing infrastructure coordinates extracted from the geojson file."""
    return (16.613, 103.381)


@pytest.fixture
def infra_filter(infra_coords: tuple[float, float]) -> NearInfraFilter:
    """Fixture providing a NearInfraFilter initialized with test coordinates."""
    infra_lat, infra_lon = infra_coords
    # Mocking funciton that actually fetches the infrastructure coordinates
    with patch("rslp.utils.filter.get_infra_latlons") as mock_get_infra_latlons:
        mock_get_infra_latlons.return_value = (
            np.array([infra_lat]),
            np.array([infra_lon]),
        )
    filter = NearInfraFilter()
    filter.infra_latlons = (np.array([infra_lat]), np.array([infra_lon]))
    return filter


def test_filter_exact_infrastructure_point(
    infra_coords: tuple[float, float], infra_filter: NearInfraFilter
) -> None:
    """Test that a point exactly on infrastructure is filtered out."""
    infra_lat, infra_lon = infra_coords
    logger.info(f"Infra coords: {infra_filter.infra_latlons}")

    assert infra_filter.should_filter(
        infra_lat, infra_lon
    ), "Detection should be filtered out as it is located on infrastructure."


def test_filter_near_infrastructure_point(
    infra_coords: tuple[float, float], infra_filter: NearInfraFilter
) -> None:
    """Test that a point close to infrastructure is filtered out."""
    infra_lat, infra_lon = infra_coords

    assert infra_filter.should_filter(
        infra_lat + 0.0001, infra_lon + 0.0001
    ), "Detection should be filtered out as it is too close to infrastructure."


def test_keep_point_far_from_infrastructure(
    infra_coords: tuple[float, float], infra_filter: NearInfraFilter
) -> None:
    """Test that a point far from infrastructure is not filtered out."""
    infra_lat, infra_lon = infra_coords

    assert not infra_filter.should_filter(
        infra_lat + 0.5, infra_lon + 0.5
    ), "Detection should be kept as it is far from infrastructure."
