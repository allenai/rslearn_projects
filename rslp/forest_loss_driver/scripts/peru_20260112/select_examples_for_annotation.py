"""Select examples for this new Peru annotation.

Based on predictions in Peru over five-year period:
- Select 100 for each of logging/burned/none/river/airstrip
- Select 500 where max(probs) < 0.5
"""

import random
from datetime import datetime

from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

PREDICTION_FNAME = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/inference/dataset_20260109/events_from_studio_jobs.geojson"
OUTPUT_DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/peru_20260112/rslearn_dataset_for_selected_events/"
TARGET_GROUP = "20260112_peru"
RARE_CATEGORIES = ["logging", "burned", "none", "river", "airstrip"]
PROB_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 1000 / 111111
WINDOW_SIZE = 128


if __name__ == "__main__":
    # Load predictions.
    predictions = GeojsonVectorFormat().decode_from_file(UPath(PREDICTION_FNAME))

    # Create candidates for the different selection criteria.
    by_class_options: dict[str, list[Feature]] = {
        category: [] for category in RARE_CATEGORIES
    }
    by_prob_options: list[Feature] = []
    for feat in predictions:
        category = feat.properties["new_label"]
        if category in RARE_CATEGORIES:
            by_class_options[category].append(feat)
        elif max(feat.properties["probs"]) < PROB_THRESHOLD:
            by_prob_options.append(feat)

    for category, candidates in by_class_options.items():
        print(f"got {len(candidates)} options by class for category={category}")
    print(f"got {len(by_prob_options)} options by prob")

    # Select windows, we make sure their center points are at least 500 m away from
    # each other.
    grid_index = GridIndex(size=DISTANCE_THRESHOLD)
    selected: list[Feature] = []

    def contains_bbox(box: tuple[float, float, float, float]) -> bool:
        """Check whether the box intersects a point in grid_index."""
        for other in grid_index.query(box):
            if (
                other[0] > box[0]
                and other[1] > box[1]
                and other[0] < box[2]
                and other[1] < box[3]
            ):
                return True
        return False

    def add_random_sample_of_features(features: list[Feature], max_count: int) -> int:
        """Add a random sample of windows from the list to the selected set."""
        # Add up to max_count from the features list.
        random.shuffle(features)
        cur_selected: list[Feature] = []
        for feat in features:
            center_point = feat.geometry.to_projection(WGS84_PROJECTION).shp.centroid
            if contains_bbox(
                (
                    center_point.x - DISTANCE_THRESHOLD,
                    center_point.y - DISTANCE_THRESHOLD,
                    center_point.x + DISTANCE_THRESHOLD,
                    center_point.y + DISTANCE_THRESHOLD,
                )
            ):
                continue

            cur_selected.append(feat)
            grid_index.insert(
                (center_point.x, center_point.y, center_point.x, center_point.y),
                (center_point.x, center_point.y),
            )
            if len(cur_selected) >= max_count:
                break

        selected.extend(cur_selected)
        return len(cur_selected)

    for category, candidates in by_class_options.items():
        count = add_random_sample_of_features(candidates, 100)
        print(f"by class category={category} picked {count}/{len(candidates)} windows")
    count = add_random_sample_of_features(by_prob_options, 500)
    print(f"by prob picked {count}/{len(by_prob_options)} windows")
    print(f"got {len(selected)} total to remap")

    # Create windows in the destination dataset for these features.
    dataset = Dataset(UPath(OUTPUT_DATASET_PATH))
    dst_proj = Projection(CRS.from_epsg(3857), 9.554628535647032, -9.554628535647032)
    random.shuffle(selected)
    for idx, feat in enumerate(selected):
        wgs84_geom = feat.geometry.to_projection(WGS84_PROJECTION)
        lon = wgs84_geom.shp.centroid.x
        lat = wgs84_geom.shp.centroid.y
        predicted_category = feat.properties["new_label"]
        window_name = f"[#{idx}]_{lon:.04f}_{lat:.04f}_predicted:{predicted_category}"

        # Get bounds in our WebMercator projection.
        dst_geom = feat.geometry.to_projection(dst_proj)
        dst_bounds = (
            int(dst_geom.shp.centroid.x) - WINDOW_SIZE // 2,
            int(dst_geom.shp.centroid.y) - WINDOW_SIZE // 2,
            int(dst_geom.shp.centroid.x) + WINDOW_SIZE // 2,
            int(dst_geom.shp.centroid.y) + WINDOW_SIZE // 2,
        )

        ts = datetime.fromisoformat(feat.properties["oe_start_time"])
        window = Window(
            storage=dataset.storage,
            group=TARGET_GROUP,
            name=window_name,
            projection=dst_proj,
            bounds=dst_bounds,
            time_range=(ts, ts),
        )
        window.save()
