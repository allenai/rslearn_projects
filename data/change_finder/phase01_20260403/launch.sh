#!/bin/bash
# Launch all phase 1 experiments (architecture/recipe/augmentation) on Beaker.
# Usage: bash data/change_finder/phase01_20260403/launch.sh IMAGE_NAME [CLUSTER]
#
# Example:
#   bash data/change_finder/phase01_20260403/launch.sh favyen/rslp_image ai2/jupiter

set -euo pipefail

IMAGE_NAME="${1:?Usage: $0 IMAGE_NAME [CLUSTER]}"
CLUSTER="${2:-ai2/jupiter}"
CONFIG_DIR="data/change_finder/phase01_20260403"
WEKA='{"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Launching $(basename "$config" .yaml) ..."
    python -m rslp.main common beaker_train \
        --image_name "$IMAGE_NAME" \
        --cluster+="$CLUSTER" \
        --config_path "$config" \
        --gpus 1 \
        --weka_mounts+="$WEKA"
done

echo "All phase 1 experiments launched."
