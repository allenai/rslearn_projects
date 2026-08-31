#!/bin/bash
# ds1020 survey-point embedding tiles: materialize S2 + S1 + Landsat (three
# 30-day mosaics per window) for the 224x224 windows and run one forward pass
# per arm, writing a 128-band int8 embedding GeoTIFF per window.
#
# Same environment expectations as the 2026_08_19 AOI run: rslearn (with the
# register-bottleneck output flag), olmoearth_run[runner], olmoearth_pretrain and
# rslearn_projects importable -- e.g. the large-scale embeddings Beaker image
# after running patch_olmoearth_shared.py (and
# patch_rslearn_projected_registers.py for the distilled arm).
set -euo pipefail

# Dataset root; weka keeps it next to the checkpoints.
DS=${DS:?set DS to the dataset root}
# Which stages to run. The windows stage must run exactly once (one job or
# locally) before the sharded jobs; the sharded jobs run the rest per group.
STAGES=${STAGES:-windows prepare materialize check predict}
# Optional space-separated shard groups (e.g. "fixed_07 fixed_08"). When empty,
# every group is processed. Groups are assigned by make_windows.py: y2017_00..
# for the per-observation-date CSV, fixed_00.. for the June-2018 CSV.
# (Not named GROUPS: that is a special read-only bash variable, and assignments
# to it are silently ignored.)
SHARD_GROUPS=${SHARD_GROUPS:-}
# cand_ndvi.yaml writes layer "output"; distilled.yaml writes "output_distilled"
# (and needs OE_PROJECTED_REGISTER_DIM=128 plus the projected-registers patch).
MODEL_CONFIG=${MODEL_CONFIG:-cand_ndvi.yaml}
DATASET_CONFIG=${DATASET_CONFIG:-config_90d.json}
# Directory holding this project's configs, CSVs and scripts (the Beaker dataset
# mount, or this directory when run locally).
PROJ_DIR=${PROJ_DIR:-one_off_projects/2026_08_31_ds1020_embeddings}
SHARD_SIZE=${SHARD_SIZE:-2000}
PREPARE_WORKERS=${PREPARE_WORKERS:-32}
MATERIALIZE_WORKERS=${MATERIALIZE_WORKERS:-128}

has_stage() { echo " $STAGES " | grep -q " $1 "; }

mkdir -p "$DS"
cp "$PROJ_DIR/$DATASET_CONFIG" "$DS/config.json"

group_args=()
if [ -n "$SHARD_GROUPS" ]; then
    # prepare/materialize --group is nargs="*".
    read -r -a group_args <<< "$SHARD_GROUPS"
fi

if has_stage windows; then
    # 6,994 windows with per-row observation dates; window start = first day of
    # the month before the observation month.
    python3 "$PROJ_DIR/make_windows.py" --root "$DS" --prefix y2017 \
        --shard_size "$SHARD_SIZE" \
        --csv "$PROJ_DIR/ds1020_consolidated_survey_points_2017.csv"
    # 51,831 windows, every date 2018-06-15, so every window starts 2018-05-01
    # (May/June/July 2018).
    python3 "$PROJ_DIR/make_windows.py" --root "$DS" --prefix fixed \
        --shard_size "$SHARD_SIZE" \
        --csv "$PROJ_DIR/ds1020_consolidated_survey_points_fixed.csv"
fi

if has_stage prepare; then
    prepare_args=(--root "$DS" --workers "$PREPARE_WORKERS"
        --retry-max-attempts 10 --retry-backoff-seconds 10)
    [ -n "$SHARD_GROUPS" ] && prepare_args+=(--group "${group_args[@]}")
    rslearn dataset prepare "${prepare_args[@]}"
fi

if has_stage materialize; then
    materialize_args=(--root "$DS" --workers "$MATERIALIZE_WORKERS" --no-use-initial-job
        --retry-max-attempts 10 --retry-backoff-seconds 10)
    [ -n "$SHARD_GROUPS" ] && materialize_args+=(--group "${group_args[@]}")
    rslearn dataset materialize "${materialize_args[@]}"
fi

if has_stage check; then
    # S1 and Landsat are required:false in the model config, so a window that
    # failed to materialize one of them silently gets embeddings from the
    # remaining modalities. Count what each group actually got, and fail if any
    # window is missing Sentinel-2 outright (it is required:true, so predict
    # would die on it anyway).
    export DS SHARD_GROUPS
    python3 - <<'EOF'
import os, sys

ds = os.environ["DS"]
only = set(os.environ.get("SHARD_GROUPS", "").split())
missing_s2 = []
for group in sorted(os.listdir(os.path.join(ds, "windows"))):
    if only and group not in only:
        continue
    gdir = os.path.join(ds, "windows", group)
    counts = {"sentinel2_l2a": 0, "sentinel1": 0, "landsat": 0}
    n = 0
    for w in os.scandir(gdir):
        if not w.is_dir():
            continue
        n += 1
        layers = set()
        layer_dir = os.path.join(w.path, "layers")
        if os.path.isdir(layer_dir):
            layers = {e.name.split(".")[0] for e in os.scandir(layer_dir)}
        for key in counts:
            if key in layers:
                counts[key] += 1
        if "sentinel2_l2a" not in layers:
            missing_s2.append(f"{group}/{w.name}")
    print(f"{group}: {n} windows, with>=1 mosaic:", counts)
if missing_s2:
    print(f"ERROR: {len(missing_s2)} windows have no sentinel2_l2a layer:", file=sys.stderr)
    for name in missing_s2[:50]:
        print(f"  {name}", file=sys.stderr)
    sys.exit(1)
EOF
fi

if has_stage predict; then
    predict_args=(
        --config "$PROJ_DIR/$MODEL_CONFIG"
        --data.init_args.path "$DS"
    )
    if [ -n "$SHARD_GROUPS" ]; then
        groups_list=$(echo "$SHARD_GROUPS" | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd, -)
        predict_args+=(--data.init_args.predict_config.groups "[$groups_list]")
        echo "=== predicting groups: [$groups_list]"
    fi
    rslearn model predict "${predict_args[@]}"
fi

echo "=== done ($STAGES); embeddings are per-window GeoTIFFs under $DS/windows/*/*/layers/ (see $MODEL_CONFIG for the output layer)"
