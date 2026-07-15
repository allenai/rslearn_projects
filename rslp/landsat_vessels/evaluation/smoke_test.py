"""Baseline smoke test (Dec 2025 code) on T1/T2 scenario-check scenes.

This runs the same scenes as the current-branch smoke test, but against the Dec 2025
pipeline code. Use it with rslearn 0.0.18 installed to check whether the classifier
false-positive regression is caused by the rslearn upgrade.
"""

import json
import shutil

from rslp.landsat_vessels.predict_pipeline import predict_pipeline
from rslp.utils.mp import init_mp

SCENES = [
    ("LC09_L1GT_129107_20241104_20241104_02_T2", [0, 10], "Mostly ice"),
    ("LC09_L1TP_001090_20241103_20241103_02_T1", [0, 10], "Mostly whitecaps"),
    ("LC09_L1TP_193021_20241104_20241104_02_T1", [20, 50], "Some vessels"),
    ("LC09_L1TP_170084_20241103_20241103_02_T1", [20, 50], "Some vessels"),
    ("LC09_L1TP_177081_20241104_20241104_02_T1", [0, 10], "Mostly whitecaps"),
    (
        "LC09_L1TP_010012_20241102_20241102_02_T1",
        [0, 10],
        "Mostly islands with some ice",
    ),
    ("LC09_L1TP_193030_20241104_20241104_02_T1", [20, 50], "Some vessels"),
]


def main() -> None:
    """Run the Dec 2025 pipeline on T1/T2 scenes and print pass/fail results."""
    init_mp()

    results = []
    for scene_id, (low, high), desc in SCENES:
        scratch = f"/tmp/eval_baseline_{scene_id}"
        json_path = f"/tmp/eval_baseline_{scene_id}.json"
        print(f"[RUNNING] {scene_id} ({desc})...")
        try:
            detections = predict_pipeline(
                scene_id=scene_id,
                scratch_path=scratch,
            )
            count = len(detections)
            with open(json_path, "w") as f:
                json.dump(detections, f, default=str)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append((scene_id, desc, low, high, -1, "ERROR"))
            continue
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        passed = low <= count <= high
        status = "PASS" if passed else "FAIL"
        results.append((scene_id, desc, low, high, count, status))
        print(f"  {status}: {count} detections (expected [{low}, {high}])\n")

    print("=" * 90)
    print(f"{'Scene':<55} {'Expected':>10} {'Actual':>8} {'Result':>8}")
    print("-" * 90)
    for scene_id, desc, low, high, count, status in results:
        print(f"{scene_id:<55} [{low},{high}]{count:>8} {status:>8}")

    passes = sum(1 for *_, s in results if s == "PASS")
    fails = sum(1 for *_, s in results if s == "FAIL")
    errors = sum(1 for *_, s in results if s == "ERROR")
    print("-" * 90)
    print(f"Pass: {passes}, Fail: {fails}, Error: {errors}")


if __name__ == "__main__":
    main()
