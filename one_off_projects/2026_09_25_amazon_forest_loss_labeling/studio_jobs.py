"""Run the forest loss driver models on Studio over the selected events.

This follows the Studio job handling in the olmoearth_projects forest loss driver
deploy pipeline (olmoearth_projects.projects.forest_loss_driver.deploy): the events are
converted to centroid points and submitted in chunks of EVENTS_PER_STUDIO_JOB, with
min_window_success_rate=0.5. Each chunk is run with each model.

Job IDs are saved to {out_dir}/studio_job_ids.json after each submission, so
re-running "launch" skips jobs that were already started.

Example:

    STUDIO_API_KEY=... python studio_jobs.py launch --out_dir /path/to/out
    STUDIO_API_KEY=... python studio_jobs.py status --out_dir /path/to/out
    STUDIO_API_KEY=... python studio_jobs.py download --out_dir /path/to/out
"""

import argparse
import json
from collections import Counter

from rslearn.utils.fsspec import open_atomic
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.projects.forest_loss_driver.deploy import (
    EVENTS_PER_STUDIO_JOB,
    get_prediction_result,
    simplify_features_to_centroids,
)
from olmoearth_projects.utils.studio_client import StudioClient

# "Forest Loss Driver Model" project, in the org with the forest loss labeling projects.
PROJECT_ID = "b4e2fdb0-f99d-468f-b103-5dd312b80124"
MODELS = {
    "utm": "d3db659a-fe9e-4749-9c14-d7088d18bbb8",  # 20260924_forest_loss_driver_utm_config.yaml_02
    "peru_phase2": "dc7772b0-4c50-40f3-9177-4bd4922da4e4",  # 20260401_forest_loss_driver_peru_phase2_config.yaml_airstripfix_02
}
JOB_NAME_PREFIX = "amazon_labeling_20260925"

# Refuse to start jobs if the selection is unexpectedly large.
MAX_EVENTS = 200000


def get_chunks(out_dir: UPath) -> list[list[dict]]:
    """Load the selected events and split them into per-job chunks of point features.

    Each feature gets an event_id property (its index in selected_events.geojson) so
    outputs can be matched back to the events and across models.
    """
    with (out_dir / "selected_events.geojson").open() as f:
        features = json.load(f)["features"]
    if len(features) > MAX_EVENTS:
        raise ValueError(
            f"got {len(features)} selected events which is more than {MAX_EVENTS}"
        )
    for event_id, feat in enumerate(features):
        feat["properties"]["event_id"] = event_id
    features = simplify_features_to_centroids(features)
    return [
        features[i : i + EVENTS_PER_STUDIO_JOB]
        for i in range(0, len(features), EVENTS_PER_STUDIO_JOB)
    ]


def load_job_ids(out_dir: UPath) -> dict[str, str]:
    """Load the mapping from job name to Studio job ID."""
    fname = out_dir / "studio_job_ids.json"
    if not fname.exists():
        return {}
    with fname.open() as f:
        return json.load(f)


def run_launch(cli_args: argparse.Namespace) -> None:
    """Start one Studio job per (model, chunk) that has not been started yet."""
    out_dir = UPath(cli_args.out_dir)
    chunks = get_chunks(out_dir)
    num_events = sum(len(chunk) for chunk in chunks)
    print(f"{num_events} events in {len(chunks)} chunks per model")

    job_ids = load_job_ids(out_dir)
    client = None if cli_args.dry_run else StudioClient.from_env()
    for model_name, model_id in MODELS.items():
        for chunk_idx, chunk in enumerate(chunks):
            job_name = f"{JOB_NAME_PREFIX}_{model_name}_chunk_{chunk_idx}"
            if job_name in job_ids:
                print(f"skipping {job_name} (already started as {job_ids[job_name]})")
                continue
            geojson = {"type": "FeatureCollection", "properties": {}, "features": chunk}
            if client is None:
                print(
                    f"[dry run] would start {job_name} with {len(chunk)} features, "
                    f"{len(json.dumps(geojson)) / 2**20:.1f} MB, first feature "
                    f"{json.dumps(chunk[0])}"
                )
                continue
            job_id = client.create_prediction(
                project_id=PROJECT_ID,
                model_id=model_id,
                name=job_name,
                geojson=geojson,
                # Same as the deploy pipeline: some events may not have enough
                # Sentinel-2 images, so only require half of the windows to succeed.
                min_window_success_rate=0.5,
            )
            print(f"started {job_name} as {job_id}", flush=True)
            job_ids[job_name] = job_id
            with open_atomic(out_dir / "studio_job_ids.json", "w") as f:
                json.dump(job_ids, f, indent=2)


def run_status(cli_args: argparse.Namespace) -> None:
    """Print the status of each started job."""
    out_dir = UPath(cli_args.out_dir)
    client = StudioClient.from_env()
    statuses: Counter[str] = Counter()
    for job_name, job_id in load_job_ids(out_dir).items():
        record = client.get_prediction(job_id)
        statuses[record["status"]] += 1
        print(
            job_name,
            job_id,
            record["status"],
            record.get("progress"),
            record.get("error_message") or "",
        )
    print(dict(statuses))


def run_download(cli_args: argparse.Namespace) -> None:
    """Download the results of completed jobs to {out_dir}/studio_outputs/."""
    out_dir = UPath(cli_args.out_dir)
    dst_dir = out_dir / "studio_outputs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    client = StudioClient.from_env()
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    for job_name, job_id in load_job_ids(out_dir).items():
        dst_fname = dst_dir / f"{job_name}.geojson"
        if dst_fname.exists():
            continue
        status = client.get_prediction(job_id)["status"]
        if status != "completed":
            print(f"{job_name} is {status}, skipping")
            continue
        features = get_prediction_result(client, job_id)
        vector_format.encode_to_file(dst_fname, features)
        print(f"downloaded {len(features)} features for {job_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["launch", "status", "download"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--dry_run", action="store_true", help="launch: only print what would be sent"
    )
    cli_args = parser.parse_args()
    {"launch": run_launch, "status": run_status, "download": run_download}[
        cli_args.stage
    ](cli_args)
