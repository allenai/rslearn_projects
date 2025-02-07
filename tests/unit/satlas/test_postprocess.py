"""Test Satlas post-processing."""

from typing import Any

from rslp.satlas.postprocess import apply_nms


class TestApplyNMS:
    """Test the apply_nms function."""

    DISTANCE_THRESHOLD = 1

    def make_feature(self, lon: float, lat: float, score: float) -> dict[str, Any]:
        """Helper function to create a GeoJSON Feature dict."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "score": score,
            },
        }

    def test_keep_far_away(self) -> None:
        """Ensure that points sufficiently far away from each other are retained."""
        features = [
            self.make_feature(0, 0, 1),
            self.make_feature(self.DISTANCE_THRESHOLD * 2, 0, 1),
        ]
        result = apply_nms(features, distance_threshold=self.DISTANCE_THRESHOLD)
        assert len(result) == 2

    def test_remove_two_of_three(self) -> None:
        """With three close together points, remove the two lower confidence ones."""
        features = [
            self.make_feature(0, 0, 0.5),
            self.make_feature(0.1, 0.1, 0.6),  # best one
            self.make_feature(0.2, 0.2, 0.4),
        ]
        result = apply_nms(features, distance_threshold=self.DISTANCE_THRESHOLD)
        assert len(result) == 1
        feature = result[0]
        assert feature["geometry"]["coordinates"] == [0.1, 0.1]
