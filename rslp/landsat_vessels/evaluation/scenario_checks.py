"""Evaluate the scene-level predictions against the expected results."""

import json

import pandas as pd
from upath import UPath

RESULT_JSON_DIR = "gs://rslearn-eai/projects/landsat_evaluation/scenario_checks/jsons/"
TARGET_CSV_PATH = "gs://rslearn-eai/projects/landsat_evaluation/scenario_checks/csv/landsat_targets.csv"


if __name__ == "__main__":
    # Load the target CSV file
    target_df = pd.read_csv(UPath(TARGET_CSV_PATH))
    count_pass = 0
    count_fail = 0
    for _, row in target_df.iterrows():
        scene_id = row["scene_id"]
        low = row["low"]
        high = row["high"]
        # Load the JSON file for the scene
        json_path = UPath(RESULT_JSON_DIR) / f"{scene_id}.json"
        with json_path.open() as f:
            data = json.load(f)
        # Count the number of detections
        count_detections = len(data)
        # Check if the number of detections is within the expected range
        if low <= count_detections <= high:
            count_pass += 1
            print(
                f"Pass: {scene_id} \n"
                f"Description: {row['description']} \n"
                f"Location: {row['location']}, Latitude: {row['latitude']}, Longitude: {row['longitude']} \n"
                f"Expected detections between {low} and {high}, Actual detections: {count_detections} \n"
            )
        else:
            count_fail += 1
            print(
                f"Fail: {scene_id} \n"
                f"Description: {row['description']} \n"
                f"Location: {row['location']}, Latitude: {row['latitude']}, Longitude: {row['longitude']} \n"
                f"Expected detections between {low} and {high}, Actual detections: {count_detections} \n"
            )
    print(
        f"Pass: {count_pass}, Fail: {count_fail}, Pass %: {count_pass / (count_pass + count_fail) * 100:.2f}"
    )
