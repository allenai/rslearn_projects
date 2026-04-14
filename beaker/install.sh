#!/usr/bin/env bash
set -euo pipefail

ROOT="/weka/dfive-default/piperw"
cd "${ROOT}/olmoearth_pretrain" && pip install -e .
cd "${ROOT}/rslearn" && pip install -e ".[extra]"
cd "${ROOT}/rslearn_projects" && pip install -r requirements.txt
pip uninstall -y torch torchvision torchaudio || true
pip install torch==2.10.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd "${ROOT}/rslearn_projects"
exec rslearn model fit --config data/landslide/model.yaml "$@"
