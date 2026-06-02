"""Smoke test: run the predict pipeline on T1/T2 scenario-check scenes."""

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

ALREADY_RAN: dict[str, tuple[int, int, int]] = {}


def main() -> None:
    """Run predict pipeline on T1/T2 scenes and print pass/fail results."""
    init_mp()

    results = []
    for scene_id, (low, high), desc in SCENES:
        if scene_id in ALREADY_RAN:
            det, cls, fb = ALREADY_RAN[scene_id]
            print(f"[CACHED] {scene_id}: detector={det} classifier={cls} feedback={fb}")
        else:
            scratch = f"/tmp/eval_{scene_id}"
            json_path = f"/tmp/eval_{scene_id}.json"
            print(f"[RUNNING] {scene_id} ({desc})...")
            try:
                result = predict_pipeline(
                    scene_id=scene_id,
                    scratch_path=scratch,
                )
                det = result.detector_count
                cls = result.classifier_count
                fb = result.feedback_classifier_count
                with open(json_path, "w") as f:
                    json.dump([d.to_dict() for d in result.detections], f)
            except Exception as e:
                print(f"  ERROR: {e}\n")
                results.append((scene_id, desc, low, high, -1, -1, -1, "ERROR"))
                continue
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

        passed = low <= fb <= high
        status = "PASS" if passed else "FAIL"
        results.append((scene_id, desc, low, high, det, cls, fb, status))
        print(
            f"  {status}: detector={det} classifier={cls} feedback={fb} "
            f"(expected [{low}, {high}])\n"
        )

    print("=" * 115)
    print(
        f"{'Scene':<55} {'Expected':>10}"
        f" {'Detector':>10} {'Classifier':>12} {'Feedback':>10} {'Result':>8}"
    )
    print("-" * 115)
    for scene_id, desc, low, high, det, cls, fb, status in results:
        print(
            f"{scene_id:<55} [{low},{high}]" f"{det:>10}{cls:>12}{fb:>10} {status:>8}"
        )

    passes = sum(1 for *_, s in results if s == "PASS")
    fails = sum(1 for *_, s in results if s == "FAIL")
    errors = sum(1 for *_, s in results if s == "ERROR")
    print("-" * 115)
    print(f"Pass: {passes}, Fail: {fails}, Error: {errors}")


if __name__ == "__main__":
    main()
