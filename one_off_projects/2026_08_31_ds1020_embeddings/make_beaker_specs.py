"""Generate the Beaker experiment specs for the ds1020 embedding run.

Three specs, mirroring the 2026_08_19 AOI run's task layout but sharded by
window group because there are 58,825 windows rather than seven:

- windows.yaml   one 0-GPU task that creates every window (STAGES="windows").
                 Runs once, before anything else. Not preemptible: it must not
                 restart halfway through interleaved with the sharded jobs.
- cand.yaml      one 1-GPU task per shard group running
                 STAGES="prepare materialize check predict" with the cand_ndvi
                 arm (output layer "output").
- distilled.yaml one 1-GPU task per shard group running STAGES="predict" with
                 the distilled arm (output layer "output_distilled"). Launch
                 after cand.yaml finishes: it reuses the imagery those jobs
                 materialized.

Usage:
    beaker dataset create one_off_projects/2026_08_31_ds1020_embeddings \
        --name ds1020-embeddings-20260831 --workspace ai2/earth-systems
    python make_beaker_specs.py --mount gabrielt/ds1020-embeddings-20260831
    beaker experiment create specs/windows.yaml
    # when it finishes:
    beaker experiment create specs/cand.yaml
    # when those finish:
    beaker experiment create specs/distilled.yaml

Pass --groups to regenerate specs for a subset (e.g. re-running failed shards).
"""

import argparse
import csv
import math
import pathlib

import yaml

# The image and helios pin the AOI run verified: patrickj/rslpomp-geozarr-20260810
# plus helios at c7c7936b6, which loads both the cand_ndvi and the distilled
# checkpoint through rslearn's OlmoEarth wrapper.
IMAGE = "01KZN3PWGWRF9FY6Z3T1KCHTYJ"
HELIOS_COMMIT = "c7c7936b6"
DEFAULT_DS = "/weka/dfive-default/gabrielt/ds1020_embeddings_20260831"

INSTALL_HELIOS = f"""set -e
# The image ships olmoearth_pretrain 0.1.1, whose EncoderConfig predates the
# fields these checkpoints carry; install helios at the commit the cand_ndvi
# eval sweeps pin. --no-deps: the image already has torch 2.7.1+cu128 and
# ai2-olmo-core 2.3.0.
git clone --filter=blob:none "https://$GITHUB_TOKEN@github.com/allenai/olmoearth_pretrain.git" /opt/helios
git -C /opt/helios checkout {HELIOS_COMMIT}
git -C /opt/helios submodule update --init --recursive
uv pip install --system --break-system-packages --no-deps /opt/helios
"""


def make_command(ds: str, stages: str, extra: str = "") -> str:
    return (
        INSTALL_HELIOS
        + f"""
export DS={ds}
export PROJ_DIR=/ds1020
export STAGES="{stages}"
{extra}
# The image's olmoearth_shared predates sun_elevation/sun_azimuth on Landsat
# items and forbids extra fields, which fails every Landsat item.
python3 /ds1020/patch_olmoearth_shared.py
bash /ds1020/run_pipeline.sh
"""
    )


def make_task(
    name: str, command: str, mount: str, gpus: int, preemptible: bool
) -> dict:
    return {
        "name": name,
        "image": {"beaker": IMAGE},
        "command": ["bash", "-c", command],
        "datasets": [
            {"mountPath": "/weka/dfive-default", "source": {"weka": "dfive-default"}},
            {"mountPath": "/ds1020", "source": {"beaker": mount}},
        ],
        "envVars": [
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
            # The Landsat assets are in the public gs://gcp-public-data-landsat;
            # GDAL cannot complete an OAuth2 exchange with the workspace GCP
            # credential, so read the bucket unsigned. AWS credentials stay for
            # the requester-pays usgs-landsat fallback.
            {"name": "GS_NO_SIGN_REQUEST", "value": "YES"},
            {"name": "GITHUB_TOKEN", "secret": "gabrielt_GITHUB_TOKEN"},
            {"name": "AWS_ACCESS_KEY_ID", "secret": "AWS_ACCESS_KEY_ID"},
            {"name": "AWS_SECRET_ACCESS_KEY", "secret": "AWS_SECRET_ACCESS_KEY"},
            {"name": "OEDATASETS_API_URL", "value": "https://datasets.olmoearth.allenai.org"},
            {"name": "DATASETS_API_TOKEN", "secret": "LCC_DATASETS_API_TOKEN"},
        ],
        "result": {"path": "/outputs"},
        "resources": {"gpuCount": gpus, "sharedMemory": "256 GiB"},
        "context": {"priority": "urgent", "preemptible": preemptible},
        "constraints": {"cluster": ["ai2/jupiter", "ai2/saturn-cirrascale"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mount", required=True, help="Beaker dataset with this directory's files"
    )
    parser.add_argument("--ds", default=DEFAULT_DS, help="dataset root on weka")
    parser.add_argument("--shard_size", type=int, default=2000)
    parser.add_argument(
        "--groups", nargs="*", default=None, help="only these shard groups"
    )
    parser.add_argument("--out_dir", default="specs")
    args = parser.parse_args()

    here = pathlib.Path(__file__).parent
    groups = []
    for prefix, fname in [
        ("y2017", "ds1020_consolidated_survey_points_2017.csv"),
        ("fixed", "ds1020_consolidated_survey_points_fixed.csv"),
    ]:
        with open(here / fname) as f:
            n = sum(1 for _ in csv.DictReader(f))
        groups += [
            f"{prefix}_{i:02d}" for i in range(math.ceil(n / args.shard_size))
        ]
    if args.groups:
        unknown = set(args.groups) - set(groups)
        if unknown:
            raise SystemExit(f"unknown groups: {sorted(unknown)}")
        groups = args.groups

    out_dir = here / args.out_dir
    out_dir.mkdir(exist_ok=True)

    specs = {
        "windows.yaml": {
            "version": "v2",
            "budget": "ai2/atec-olmoearth",
            "description": "ds1020: create the 58,825 224x224 survey-point windows (run once, first)",
            "tasks": [
                make_task(
                    "ds1020_windows",
                    make_command(args.ds, "windows"),
                    args.mount,
                    gpus=0,
                    preemptible=False,
                )
            ],
        },
        "cand.yaml": {
            "version": "v2",
            "budget": "ai2/atec-olmoearth",
            "description": (
                "ds1020: materialize S2+S1+Landsat (3x30d mosaics) and run cand_ndvi "
                "(224 px windows, patch_size=1, crop 16, overlap 4), one task per shard group"
            ),
            "tasks": [
                make_task(
                    f"ds1020_cand_{g}",
                    make_command(
                        args.ds,
                        "prepare materialize check predict",
                        extra=f'export SHARD_GROUPS="{g}"\n',
                    ),
                    args.mount,
                    gpus=1,
                    preemptible=True,
                )
                for g in groups
            ],
        },
        "distilled.yaml": {
            "version": "v2",
            "budget": "ai2/atec-olmoearth",
            "description": (
                "ds1020: distilled lin_sup768_w1_d128 forward passes over the already-"
                "materialized windows, one task per shard group (launch after cand.yaml)"
            ),
            "tasks": [
                make_task(
                    f"ds1020_distilled_{g}",
                    make_command(
                        args.ds,
                        "predict",
                        extra=(
                            f'export SHARD_GROUPS="{g}"\n'
                            "export MODEL_CONFIG=distilled.yaml\n"
                            "# The distilled arm's 128 dims come from the detached student\n"
                            "# (projected_registers), not the first 128 register dims.\n"
                            "export OE_PROJECTED_REGISTER_DIM=128\n"
                            "python3 /ds1020/patch_rslearn_projected_registers.py\n"
                        ),
                    ),
                    args.mount,
                    gpus=1,
                    preemptible=True,
                )
                for g in groups
            ],
        },
    }
    for fname, spec in specs.items():
        path = out_dir / fname
        with open(path, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, width=100)
        print(f"wrote {path} ({len(spec['tasks'])} tasks)")


if __name__ == "__main__":
    main()
