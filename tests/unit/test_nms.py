import shapely
from rslearn.utils import Feature, STGeometry

from rslp.utils.nms import NMSDistanceMerger


def test_distance_nms() -> None:
    metadata: dict = {"bounds": [0, 0], "projection": "EPSG:4326"}

    # Test with no boxes provided.
    features: list = []
    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
    merged_features = merger.merge(features)
    # Expected: No boxes, so the result should be an empty list.
    assert len(merged_features) == 0

    # Test NMS with 3 boxes, single class.
    features = []
    for box, score, class_id in zip(
        [
            [10, 10, 20, 20],  # Box 0
            [12, 12, 22, 22],  # Box 1, overlaps with Box 0
            [30, 30, 40, 40],  # Box 2, separate
        ],
        [0.9, 0.85, 0.8],
        [0, 0, 0],
    ):
        shp = shapely.box(
            metadata["bounds"][0] + float(box[0]),
            metadata["bounds"][1] + float(box[1]),
            metadata["bounds"][0] + float(box[2]),
            metadata["bounds"][1] + float(box[3]),
        )
        geom = STGeometry(metadata["projection"], shp, None)
        properties = {"score": float(score), "category": class_id}
        features.append(Feature(geom, properties))

    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
    merged_features = merger.merge(features)

    # Expected: Box 0 (highest score) and Box 2 (no overlap) should be kept.
    expected_indices = [0, 2]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)

    # Test with a smaller threshold where all boxes should be kept.
    merger = NMSDistanceMerger(
        grid_size=10, distance_threshold=1
    )  # Smaller threshold to avoid suppression
    merged_features = merger.merge(features)
    # Expected: All boxes should be kept.
    expected_indices = [0, 1, 2]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)

    # Test with box coordinates minx/maxx being positive and miny/maxy being negative.
    features = []
    for box, score, class_id in zip(
        [
            [10, -20, 20, -10],  # Box 0
            [12, -22, 22, -12],  # Box 1, overlaps with Box 0
            [30, -40, 40, -30],  # Box 2, separate
        ],
        [0.9, 0.85, 0.8],
        [0, 0, 0],
    ):
        shp = shapely.box(
            metadata["bounds"][0] + float(box[0]),
            metadata["bounds"][1] + float(box[1]),
            metadata["bounds"][0] + float(box[2]),
            metadata["bounds"][1] + float(box[3]),
        )
        geom = STGeometry(metadata["projection"], shp, None)
        properties = {"score": float(score), "category": class_id}
        features.append(Feature(geom, properties))

    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
    merged_features = merger.merge(features)

    # Expected: Box 0 (highest score) and Box 2 (no overlap) should be kept.
    expected_indices = [0, 2]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)

    # Test with multiple classes where NMS should be performed per class.
    features = []
    for box, score, class_id in zip(
        [
            [10, 10, 20, 20],  # Class 0, Box 0
            [12, 12, 22, 22],  # Class 0, Box 1 (overlapping with Box 0)
            [10, 10, 20, 20],  # Class 1, Box 2
            [12, 12, 22, 22],  # Class 1, Box 3 (overlapping with Box 2)
        ],
        [0.9, 0.85, 0.8, 0.95],
        [0, 0, 1, 1],
    ):
        shp = shapely.box(
            metadata["bounds"][0] + float(box[0]),
            metadata["bounds"][1] + float(box[1]),
            metadata["bounds"][0] + float(box[2]),
            metadata["bounds"][1] + float(box[3]),
        )
        geom = STGeometry(metadata["projection"], shp, None)
        properties = {"score": float(score), "category": class_id}
        features.append(Feature(geom, properties))

    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
    merged_features = merger.merge(features)
    # Expected: For Class 0, Box 0 kept (higher score); Box 1 suppressed.
    # For Class 1, Box 3 kept (higher score); Box 2 suppressed.
    expected_indices = [0, 3]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)

    # Test with multiple classes where NMS should be performed class-agnostic.
    features = []
    for box, score, class_id in zip(
        [
            [10, 10, 20, 20],  # Class 0, Box 0
            [12, 12, 22, 22],  # Class 0, Box 1 (overlapping with Box 0)
            [10, 10, 20, 20],  # Class 1, Box 2
            [12, 12, 22, 22],  # Class 1, Box 3 (overlapping with Box 2)
        ],
        [0.9, 0.85, 0.8, 0.95],
        [0, 0, 1, 1],
    ):
        shp = shapely.box(
            metadata["bounds"][0] + float(box[0]),
            metadata["bounds"][1] + float(box[1]),
            metadata["bounds"][0] + float(box[2]),
            metadata["bounds"][1] + float(box[3]),
        )
        geom = STGeometry(metadata["projection"], shp, None)
        properties = {"score": float(score), "category": class_id}
        features.append(Feature(geom, properties))

    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5, class_agnostic=True)
    merged_features = merger.merge(features)
    # Expected: Box 3 kept (highest score); Box 0, Box 1, and Box 2 suppressed.
    expected_indices = [3]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)

    # Test with equal scores and overlapping boxes.
    features = []
    for box, score, class_id in zip(
        [
            [10, 10, 20, 20],  # Box 0
            [12, 12, 22, 22],  # Box 1 (overlapping with Box 0)
        ],
        [0.9, 0.9],
        [0, 0],
    ):
        shp = shapely.box(
            metadata["bounds"][0] + float(box[0]),
            metadata["bounds"][1] + float(box[1]),
            metadata["bounds"][0] + float(box[2]),
            metadata["bounds"][1] + float(box[3]),
        )
        geom = STGeometry(metadata["projection"], shp, None)
        properties = {"score": float(score), "category": class_id}
        features.append(Feature(geom, properties))

    merger = NMSDistanceMerger(grid_size=10, distance_threshold=5)
    merged_features = merger.merge(features)
    # Expected: Box 0 kept because it has a lower index (tie-breaking).
    expected_indices = [0]
    assert len(merged_features) == len(expected_indices)
    assert all(features[i] in merged_features for i in expected_indices)
