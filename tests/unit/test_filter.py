from rslp.utils.filter import NearInfraFilter


def test_near_infra_filter() -> None:
    # Test case 1: Detection is exactly on infrastructure.
    # The coordinates are directly extracted from the geojson file.
    infra_lat = 16.613
    infra_lon = 103.381

    filter = NearInfraFilter()

    # Since this point is exactly an infrastructure point, the filter should discard it (return True)
    assert filter.should_discard(
        infra_lat, infra_lon
    ), "Detection should be discarded as it is located on infrastructure."

    # Test case 2: Detection is close to infrastructure.
    assert filter.should_discard(
        infra_lat + 0.001, infra_lon + 0.001
    ), "Detection should be discarded as it is too close to infrastructure."

    # Test case 3: Detection is far from infrastructure.
    assert not filter.should_discard(
        infra_lat + 0.5, infra_lon + 0.5
    ), "Detection should be kept as it is far from infrastructure."
