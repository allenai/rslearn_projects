#!/bin/bash
# Boots one GCE GPU VM as an embeddings queue worker.
#
# Beaker's executor supplies the container runtime, the env set, a large /dev/shm, a
# raised fd limit and log shipping. On a bare VM each is ours, and each was found by a
# run failing without it, so none are optional.
#
# Every setting below is overridable per instance:
#   gcloud compute instances create ... \
#     --metadata-from-file=startup-script=startup.sh \
#     --metadata=embed-image-tag=rc-20261001b,embed-queue=patrickj/conus-2024
set -uo pipefail
exec > >(logger -t embedworker -s) 2>&1

meta() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/$1" 2>/dev/null
}
# Metadata value, or the given default when the attribute is absent.
attr() { meta "instance/attributes/$1" || printf '%s' "$2"; }

# Defaults to the VM's own project, so the script carries no project id.
PROJECT="$(attr embed-project "$(meta project/project-id)")"
REGION="$(attr embed-region us-central1)"
AR_REPO="$(attr embed-ar-repo olmoearth-run)"
IMAGE_NAME="$(attr embed-image-name rslp-embeddings)"
IMAGE_TAG="$(attr embed-image-tag rc-20260930d)"
CKPT_GCS="$(attr embed-checkpoint-gcs gs://ai2-helios-us-central1/checkpoints/v1_3_release_v2)"
# Queue entries carry this path, baked in by the supervisor at enqueue time, so the
# checkpoint has to land here rather than somewhere tidier.
CKPT_DIR="$(attr embed-checkpoint-dir /weka/dfive-default/gabrielt/helios/v1_3_release_v2)"
QUEUE="$(attr embed-queue patrickj/global-2025-v2)"
DATASETS_API_URL="$(attr embed-datasets-api-url https://datasets.olmoearth.allenai.org)"
SECRET_PROJECT="$(attr embed-secret-project "$PROJECT")"
SEC_DATASETS="$(attr embed-secret-datasets-token olmoearth-datasets-api-token)"
SEC_AWS_KEY="$(attr embed-secret-aws-key-id aws-access-key-id)"
SEC_AWS_SECRET="$(attr embed-secret-aws-secret aws-secret-access-key)"
SEC_BEAKER="$(attr embed-secret-beaker-token patrickj-beaker-token)"
# The embeddings-writer service account key, the same one Beaker jobs mount, so every
# writer to the store is one identity. The VM's own account only reads secrets and
# pulls the image.
SEC_GCP_CREDS="$(attr embed-secret-gcp-credentials olmoearth-embeddings-gcp-credentials)"
SHM_SIZE="$(attr embed-shm-size 16g)"
# Appended to every entry this worker runs, overriding what the supervisor baked
# in. A GPU-memory knob like --batch_size follows the hardware, not the job.
EXTRA_ARGS="$(attr embed-worker-extra-args "")"
NOFILE="$(attr embed-nofile 65535:524288)"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "image=$IMAGE queue=$QUEUE secrets_from=$SECRET_PROJECT"

echo "=== docker ==="
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
fi
# The DLVM image ships the driver and nvidia-container-runtime but no engine to drive
# them, so the runtime has to be registered with docker explicitly.
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "=== ops agent ==="
if ! systemctl is-active --quiet google-cloud-ops-agent; then
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
fi
cat >/etc/google-cloud-ops-agent/config.yaml <<'CFG'
logging:
  receivers:
    docker_containers:
      type: files
      include_paths:
        - /var/lib/docker/containers/*/*-json.log
  service:
    pipelines:
      docker_pipeline:
        receivers: [docker_containers]
metrics:
  service:
    pipelines:
      default_pipeline:
        receivers: [hostmetrics]
CFG
systemctl restart google-cloud-ops-agent

echo "=== checkpoint ==="
mkdir -p "$CKPT_DIR"
gcloud storage cp -r "${CKPT_GCS}/*" "$CKPT_DIR/"
ls -la "$CKPT_DIR"

echo "=== secrets ==="
# Fetched with the VM's own service account, so no credential is placed on the image
# or in instance metadata.
sec() { gcloud secrets versions access latest --secret="$1" --project="$SECRET_PROJECT"; }
umask 077
GCP_CREDS_FILE=/etc/embedworker-gcp-credentials.json
# Refuse to start rather than fall back to the VM's account: a worker writing as the
# wrong identity fails on its first marker, or worse, succeeds where it should not.
if ! sec "$SEC_GCP_CREDS" >"$GCP_CREDS_FILE" || [ ! -s "$GCP_CREDS_FILE" ]; then
  echo "could not read secret $SEC_GCP_CREDS from $SECRET_PROJECT; not starting the worker"
  exit 1
fi
cat >/etc/embedworker.env <<ENV
OEDATASETS_API_URL=${DATASETS_API_URL}
GS_USER_PROJECT=${PROJECT}
GCLOUD_PROJECT=${PROJECT}
GOOGLE_CLOUD_PROJECT=${PROJECT}
RSLP_PREFIX=project_data/
MKL_THREADING_LAYER=GNU
DATASETS_API_TOKEN=$(sec "$SEC_DATASETS")
AWS_ACCESS_KEY_ID=$(sec "$SEC_AWS_KEY")
AWS_SECRET_ACCESS_KEY=$(sec "$SEC_AWS_SECRET")
BEAKER_TOKEN=$(sec "$SEC_BEAKER")
RSLP_WORKER_NAME=gce_$(hostname)
RSLP_WORKER_EXTRA_ARGS=${EXTRA_ARGS}
GOOGLE_APPLICATION_CREDENTIALS=/etc/credentials/gcp_credentials.json
ENV

echo "=== worker ==="
docker pull "$IMAGE"
# --shm-size: the 64MB default kills DataLoader workers outright.
# --ulimit nofile: the 1024 default exhausts fds in the fd sharing strategy and the
#   loader dies with EOFError in recvfds.
# --restart: the worker exits after its idle timeout, and the queue is kept shallow on
#   purpose, so an exit is routine. Without a restart the GPU idles and bills until the
#   Flex Start window closes.
docker run -d --name embedworker \
  --gpus all \
  --restart unless-stopped \
  --env-file /etc/embedworker.env \
  --shm-size="$SHM_SIZE" \
  --ulimit "nofile=${NOFILE}" \
  -v /weka:/weka \
  -v "${GCP_CREDS_FILE}:/etc/credentials/gcp_credentials.json:ro" \
  "$IMAGE" \
  python -m rslp.main common worker --queue_name "$QUEUE"

echo "=== started ==="
docker ps --format '{{.Names}} {{.Status}}'
