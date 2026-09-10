#!/usr/bin/env bash
# Fan out the PASTIS2 dense DROM materialize over N Beaker jobs via Gantry (the Beaker
# path proven to work in this env; the rslp beaker helper is broken by a deprecated
# `budget` field). Every job runs the SAME `rslearn dataset prepare/materialize --root
# <national_ds> --group rpg_2019`; rslearn shuffles window order per job and skips
# windows already marked completed, so N jobs distribute the ~2,742 dense windows
# lock-free. CPU/network only (no GPU). Resumable: re-run to fill gaps.
#
# Usage:
#   NUM_JOBS=8 bash launch_pastis2_dense.sh
#   STEP=materialize NUM_JOBS=12 bash launch_pastis2_dense.sh
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter}"
WORKSPACE="${WORKSPACE:-ai2/earth-systems}"
BUDGET="${BUDGET:-}"
BEAKER_IMAGE="${BEAKER_IMAGE:-beaker-py/gantry}"
GPUS="${GPUS:-0}"
CPUS="${CPUS:-32}"
MEMORY="${MEMORY:-128GiB}"
SHARED_MEMORY="${SHARED_MEMORY:-32GiB}"
PRIORITY="${PRIORITY:-high}"
TASK_TIMEOUT="${TASK_TIMEOUT:-96h}"

WEKA_BUCKET="${WEKA_BUCKET:-dfive-default}"
WEKA_MOUNT="${WEKA_MOUNT:-/weka/dfive-default}"
PIPERW_ROOT="${PIPERW_ROOT:-${WEKA_MOUNT}/piperw}"
# gantry must run from a git repo; the job installs rslearn from weka and reads the
# dataset by absolute path, so the uploaded repo is incidental.
GANTRY_REPO="${GANTRY_REPO:-${PIPERW_ROOT}/rslearn_projects}"

DS="${DS:-${PIPERW_ROOT}/dev/rslearn_projects/pastis2/data/national_ds}"
GROUP="${GROUP:-rpg_2019}"
WORKERS="${WORKERS:-32}"
RETRY="${RETRY:-8}"
BACKOFF="${BACKOFF:-30}"
NUM_JOBS="${NUM_JOBS:-8}"
STEP="${STEP:-both}"   # prepare | materialize | both
DRY_RUN="${DRY_RUN:-0}"

_step_cmds() {
  local out=""
  if [[ "${STEP}" == "prepare" || "${STEP}" == "both" ]]; then
    out+="rslearn dataset prepare --root '${DS}' --group '${GROUP}' --workers ${WORKERS} --retry-max-attempts ${RETRY} --retry-backoff-seconds ${BACKOFF} --ignore-errors"$'\n'
  fi
  if [[ "${STEP}" == "materialize" || "${STEP}" == "both" ]]; then
    out+="rslearn dataset materialize --root '${DS}' --group '${GROUP}' --workers ${WORKERS} --retry-max-attempts ${RETRY} --retry-backoff-seconds ${BACKOFF} --ignore-errors"$'\n'
  fi
  printf '%s' "${out}"
}

job_cmd=$(cat <<EOF
set -euo pipefail
cd "${PIPERW_ROOT}"
if ! python -m pip --version >/dev/null 2>&1; then python -m ensurepip --upgrade || true; fi
python -m pip install --upgrade pip
python -m pip install -U 'jsonargparse[signatures]>=4.27.7'
cd "${PIPERW_ROOT}/rslearn" && python -m pip install -e '.[extra]'
$(_step_cmds)
EOF
)

launch_one() {
  local name="$1"
  local args=(
    run --yes --allow-dirty --name "${name}" --task-name materialize
    --cluster "${CLUSTER}" --gpus "${GPUS}" --cpus "${CPUS}"
    --memory "${MEMORY}" --shared-memory "${SHARED_MEMORY}"
    --priority "${PRIORITY}" --task-timeout "${TASK_TIMEOUT}"
    --beaker-image "${BEAKER_IMAGE}" --weka "${WEKA_BUCKET}:${WEKA_MOUNT}"
    --exec-method bash
  )
  [[ -n "${WORKSPACE}" ]] && args+=(--workspace "${WORKSPACE}")
  [[ -n "${BUDGET}" ]] && args+=(--budget "${BUDGET}")
  echo "=== ${name} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY_RUN gantry ${args[*]} -- <cmd>"; return; fi
  ( cd "${GANTRY_REPO}" && gantry "${args[@]}" -- "${job_cmd}" )
}

echo "Fanning out ${NUM_JOBS} '${STEP}' jobs on ${DS} (group ${GROUP})"
ts=$(date +%Y%m%d_%H%M%S)
for i in $(seq 1 "${NUM_JOBS}"); do
  launch_one "pastis2_drom_dense_${ts}_${i}"
done
