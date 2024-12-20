import json
import pathlib

import pytest

from rslp.utils.filter import NearInfraFilter

TEST_INFRA_LON = 1.234
TEST_INFRA_LAT = 5.678


class TestNearInfraFilter:
    @pytest.fixture
    def single_point_infra_filter(self, tmp_path: pathlib.Path) -> NearInfraFilter:
        geojson_data = {
            "type": "FeatureCollection",
            "properties": {},
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [TEST_INFRA_LON, TEST_INFRA_LAT],
                    },
                }
            ],
        }
        fname = tmp_path / "data.geojson"
        with fname.open("w") as f:
            json.dump(geojson_data, f)

        return NearInfraFilter(infra_url=str(fname))

    def test_exactly_on_infra(self, single_point_infra_filter: NearInfraFilter) -> None:
        # Test when detection is exactly on infrastructure.
        # The coordinates are directly extracted from the geojson file.
        # Since this point is exactly an infrastructure point, the filter should discard it (return True)
        assert single_point_infra_filter.should_filter(
            TEST_INFRA_LAT,
            TEST_INFRA_LON,
        ), "Detection should be filtered out as it is located on infrastructure."

    def test_close_to_infra(self, single_point_infra_filter: NearInfraFilter) -> None:
        # Test when detection is close to infrastructure.
        assert single_point_infra_filter.should_filter(
            TEST_INFRA_LAT + 0.0001, TEST_INFRA_LON + 0.0001
        ), "Detection should be filtered out as it is too close to infrastructure."

    def test_far_from_infra(self, single_point_infra_filter: NearInfraFilter) -> None:
        # Test when detection is far from infrastructure.
        assert not single_point_infra_filter.should_filter(
            TEST_INFRA_LAT + 0.5, TEST_INFRA_LON + 0.5
        ), "Detection should be kept as it is far from infrastructure."
