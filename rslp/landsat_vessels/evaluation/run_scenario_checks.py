"""Run the full predict pipeline on all scenario-check scenes and evaluate."""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from upath import UPath

from rslp.landsat_vessels.predict_pipeline import predict_pipeline
from rslp.utils.mp import init_mp

TARGET_CSV_PATH = "gs://rslearn-eai/projects/landsat_evaluation/scenario_checks/csv/landsat_targets.csv"


def main() -> None:
    """Run predict pipeline on all scenario-check scenes and print results."""
    init_mp()

    target_df = pd.read_csv(UPath(TARGET_CSV_PATH))
    print(f"Loaded {len(target_df)} scenario-check scenes\n")

    output_dir = Path(tempfile.mkdtemp(prefix="landsat_v2_eval_"))
    print(f"Saving results to {output_dir}\n")

    count_pass = 0
    count_fail = 0
    count_error = 0

    for idx, row in target_df.iterrows():
        scene_id = row["scene_id"]
        low = row["low"]
        high = row["high"]
        scratch = str(output_dir / "scratch" / scene_id)

        print(f"[{idx + 1}/{len(target_df)}] {scene_id} ({row['description']})")

        try:
            detections = predict_pipeline(
                scene_id=scene_id,
                scratch_path=scratch,
            )
            count_detections = len(detections)

            json_path = output_dir / f"{scene_id}.json"
            with open(json_path, "w") as f:
                json.dump([d.to_dict() for d in detections], f)

            passed = low <= count_detections <= high
            status = "PASS" if passed else "FAIL"
            if passed:
                count_pass += 1
            else:
                count_fail += 1

            print(
                f"  {status}: {count_detections} detections "
                f"(expected [{low}, {high}])\n"
            )
        except Exception as e:
            count_error += 1
            print(f"  ERROR: {e}\n")
        finally:
            # Clean up scratch to save disk space between scenes
            shutil.rmtree(scratch, ignore_errors=True)

    print("=" * 60)
    print(f"Results: {count_pass} pass, {count_fail} fail, {count_error} error")
    if count_pass + count_fail > 0:
        print(
            f"Pass rate: {count_pass / (count_pass + count_fail) * 100:.1f}% "
            f"(excluding errors)"
        )
    print(f"Output JSONs saved to: {output_dir}")


if __name__ == "__main__":
    main()
