"""Launch Beaker jobs to prepare + materialize the national PASTIS2 dataset at scale.

National generation is thousands of windows x 12 Planetary-Computer monthly composites --
far beyond a foreground run. This fans it out over N Beaker jobs using the repo's canonical
helper (`rslp.common.beaker_data_materialization.launch_jobs`).

How the parallelism works (no explicit sharding needed): every job runs the *same*
`rslearn dataset <verb> --group <group>` command against the *same* weka dataset root.
rslearn shuffles the window order per job and skips windows already marked "completed"
(per-window, per-layer marker files), so N jobs racing on one group distribute the work
lock-free and disjointly -- safe because each window writes to its own directory.

Prereqs:
  * `build_windows.py` already ran, creating group `rpg_<year>` in the dataset (on weka).
  * The Beaker image has rslearn + planetary_computer + pystac_client installed.
  * Dataset root is a weka path both this launcher and the jobs can see.

Run with the rslearn_projects venv (has rslp + beaker):
  <rslearn_projects>/.venv/bin/python launch_national.py \
    --dataset <weka>/pastis2/national_ds \
    --group rpg_2019 --num-jobs 16 --step both   # --image defaults to a verified rslp image
"""

from __future__ import annotations

import argparse

from beaker import BeakerJobPriority
from rslp.common.beaker_data_materialization import launch_jobs

# Shared flags for both verbs. Keep --workers modest per job since many jobs hit the
# Planetary Computer STAC concurrently (rate limits); retries ride out transient errors.
_COMMON = [
    "--root", "{ds_path}",
    "--workers", "32",
    "--no-use-initial-job",
    "--retry-max-attempts", "8",
    "--retry-backoff-seconds", "60",
    "--ignore-errors",
]


def _command(verb: str) -> list[str]:
    return ["rslearn", "dataset", verb, *_COMMON]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="weka dataset root (shared, has config.json + windows)")
    ap.add_argument("--group", default="rpg_2019")
    # Any rslp image built from rslearn_projects/Dockerfile has rslearn[extra], which
    # includes planetary_computer + pystac_client. This default is verified to import
    # them; bump to a newer favyen/rslpomp<date> as they're published.
    ap.add_argument("--image", default="favyen/rslpomp20260702a",
                    help="Beaker image with rslearn + planetary_computer + pystac_client")
    ap.add_argument("--num-jobs", type=int, default=16)
    ap.add_argument("--clusters", nargs="+", default=["ai2/saturn", "ai2/jupiter", "ai2/neptune"])
    ap.add_argument("--priority", default="high", choices=[p.name for p in BeakerJobPriority])
    ap.add_argument("--step", choices=["prepare", "materialize", "both"], default="both",
                    help="prepare finds PC items per window; materialize downloads/composites."
                         " 'both' launches prepare first, then materialize.")
    args = ap.parse_args()

    priority = BeakerJobPriority[args.priority]
    verbs = ["prepare", "materialize"] if args.step == "both" else [args.step]
    for verb in verbs:
        print(f"=== launching {args.num_jobs} '{verb}' jobs on group {args.group} ===")
        launch_jobs(
            image=args.image,
            ds_path=args.dataset,
            group=args.group,
            clusters=args.clusters,
            num_jobs=args.num_jobs,
            command=_command(verb),
            priority=priority,
        )
    print(
        "\nAfter materialize completes, run the label + tensor steps (cheap, no download):\n"
        f"  python rasterize_labels.py --dataset {args.dataset} --group {args.group}\n"
        f"  python make_tensors.py --dataset {args.dataset} --group {args.group} "
        "--out data/tensors_national --year 2019"
    )


if __name__ == "__main__":
    main()
