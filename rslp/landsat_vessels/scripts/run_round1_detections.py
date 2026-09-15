"""Run the vessel pipeline locally over the round-1 scene sample.

Loops over the scenes from sample_scenes.py and runs the predict pipeline
(detector + classifier) on each, in annotation mode (include_rejected=True, so
classifier-rejected candidates are kept with their prob for the manifest).

Resumable: scenes whose JSON already exists are skipped. Failures are logged to
failures.log in the output directory and the run continues. Per-scene scratch
datasets are deleted after each scene to bound disk use.

Usage:
    export RSLP_PREFIX=/weka/dfive-default/rslearn-eai/   # model checkpoints
    # plus AWS credentials for the usgs-landsat bucket
    python -m rslp.landsat_vessels.scripts.run_round1_detections \
        --scene_csv /weka/dfive-default/yawenz/landsat/scene_sample_v1.csv \
        --out_dir /weka/dfive-default/yawenz/landsat/round1_detections \
        --scratch_dir /projects/round1_scratch
"""

import argparse
import csv
import os
import shutil
import time
import traceback

from rslp.landsat_vessels.predict_pipeline import predict_pipeline


def main() -> None:
    """Run the pipeline over all sampled scenes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--scratch_dir",
        required=True,
        help="local scratch for per-scene datasets; each scene's dir is deleted "
        "after it finishes (ephemeral disk is fine)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N scenes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    json_dir = os.path.join(args.out_dir, "json")
    crops_dir = os.path.join(args.out_dir, "crops")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)
    failures_path = os.path.join(args.out_dir, "failures.log")

    with open(args.scene_csv) as f:
        scenes = list(csv.DictReader(f))
    if args.limit:
        scenes = scenes[: args.limit]

    done = skipped = failed = 0
    for i, scene in enumerate(scenes, 1):
        scene_id = scene["scene_id"]
        json_path = os.path.join(json_dir, f"{scene_id}.json")
        if os.path.exists(json_path) and not args.overwrite:
            skipped += 1
            continue

        scratch = os.path.join(args.scratch_dir, scene_id)
        start = time.time()
        try:
            result = predict_pipeline(
                scene_id=scene_id,
                json_path=json_path,
                crop_path=os.path.join(crops_dir, scene_id),
                scratch_path=scratch,
                include_rejected=True,
            )
            done += 1
            print(
                f"[{i}/{len(scenes)}] {scene_id} ({scene['stratum']}): "
                f"{result.detector_count} candidates, "
                f"{result.classifier_count} passed classifier, "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
        except Exception:
            failed += 1
            with open(failures_path, "a") as f:
                f.write(f"=== {scene_id} ===\n{traceback.format_exc()}\n")
            print(
                f"[{i}/{len(scenes)}] {scene_id} FAILED after "
                f"{time.time() - start:.0f}s (see failures.log)",
                flush=True,
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"finished: {done} ok, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
