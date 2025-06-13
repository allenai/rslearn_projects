"""Write Sentinel-2 vessel detection tasks to a Beaker queue."""

import json
import random

from upath import UPath

from rslp.common.worker import write_jobs

DEFAULT_BATCH_SIZE = 30


def check_scene_done(out_path: UPath, scene_id: str) -> tuple[str, bool]:
    """Checks whether the scene ID is done processing already.

    It is determined based on existence of output JSON file for that scene.

    Args:
        out_path: the directory where output JSON files should appear.
        scene_id: the scene ID to check.

    Returns:
        whether the job was completed
    """
    return scene_id, (out_path / (scene_id + ".json")).exists()


def write_entries(
    queue_name: str,
    json_fname: str,
    json_out_dir: str,
    geojson_out_dir: str | None = None,
    crop_out_dir: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    count: int | None = None,
) -> None:
    """Write Sentinel-2 vessel detection tasks to Beaker queue.

    Args:
        queue_name: the Beaker queue to write to.
        json_fname: the JSON filename containing a list of string Sentinel-2 scene IDs
            to process.
        json_out_dir: where to write the JSON results. This directory is also used to
            exclude tasks that have already been completed.
        geojson_out_dir: where to write the GeoJSON results.
        crop_out_dir: where to write vessel crops.
        batch_size: how many Sentinel-2 scenes to process in parallel on one GPU.
        count: limit to writing this many tasks.
    """
    # Load the scene IDs from the JSON file and see which ones are not done yet.
    with open(json_fname) as f:
        scene_ids: list[str] = json.load(f)

    out_path = UPath(json_out_dir)
    done_scene_ids = set()
    for fname in out_path.iterdir():
        if not fname.name.endswith(".json"):
            continue
        done_scene_ids.add(fname.name.split(".json")[0])
    missing_scene_ids = [
        scene_id for scene_id in scene_ids if scene_id not in done_scene_ids
    ]

    # Run up to count tasks.
    if count and len(missing_scene_ids) > count * batch_size:
        run_scene_ids = random.sample(missing_scene_ids, count * batch_size)
    else:
        run_scene_ids = missing_scene_ids

    print(
        f"Got {len(scene_ids)} total scenes, {len(missing_scene_ids)} pending, running {len(run_scene_ids)} of them"
    )
    random.shuffle(run_scene_ids)
    args_list = []
    for i in range(0, len(run_scene_ids), batch_size):
        batch = run_scene_ids[i : i + batch_size]
        tasks: list[dict] = []
        for scene_id in batch:
            tasks.append(
                dict(
                    scene_id=scene_id,
                    json_path=f"{json_out_dir}{scene_id}.json",
                    geojson_path=f"{geojson_out_dir}{scene_id}.geojson"
                    if geojson_out_dir
                    else None,
                    crop_path=f"{crop_out_dir}{scene_id}" if crop_out_dir else None,
                )
            )
        args_list.append(
            [
                "--tasks",
                json.dumps(tasks),
            ]
        )
    write_jobs(queue_name, "sentinel2_vessels", "predict", args_list)
