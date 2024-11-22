#!/bin/bash
cleanup() {
    if [ -n "$VM_NAME" ]; then
        echo -e "\nCleaning up VM $VM_NAME..."
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet || true
    fi
    exit 1
}

# Set up trap for SIGINT (CTRL+C)
trap cleanup SIGINT
# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --project-id      GCP project ID (default: skylight-proto-1)"
    echo "  --zone            GCP zone (default: us-west1-b)"
    echo "  --machine-type    VM machine type (default: e2-micro)"
    echo "  --docker-image    Docker image to run"
    echo "  --command         Command to run in container on the vm"
    echo "  --user            User (default: henryh)"
    echo "  --ghcr-user       GitHub Container Registry user (default: allenai)"
    echo "  --delete          Delete VM after completion (yes/no)"
    echo "  --beaker-token    Beaker token"
    echo "  --beaker-addr     Beaker address"
    echo "  --beaker-username Beaker username associated with the token"
    echo "  --rslp-project    rslp project name (e.g forest_loss_driver)"
    echo "  --rslp-prefix     rslp prefix"
    echo "  --workflow        workflow name (e.g predict_pipeline) to run on beaker"
    echo "  --gpu-count       Number of GPUs to use"
    echo "  --shared-memory   Amount of shared memory"
    echo "  --cluster         Cluster to use"
    echo "  --priority        Priority level"
    echo "  --task-name       Name of the task"
    echo "  --budget          Budget to use"
    echo "  --workspace       Workspace name"
    exit 1
}

# Default values
PROJECT_ID="skylight-proto-1"
ZONE="us-west1-b"
MACHINE_TYPE="e2-micro"
IMAGE_FAMILY="debian-11"
IMAGE_PROJECT="debian-cloud"
USER="henryh"
GHCR_USER="allenai"
DELETE_VM="no"
GPU_COUNT="1"
SHARED_MEMORY="64Gib"
CLUSTER="ai2/jupiter-cirrascale-2"
PRIORITY="normal"
TASK_NAME="forest_loss_driver_inference_$(uuidgen | cut -c1-8)"
BUDGET="ai2/d5"
WORKSPACE="ai2/earth-systems"

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --project-id)
            shift
            PROJECT_ID="$1"
            ;;
        --zone)
            shift
            ZONE="$1"
            ;;
        --machine-type)
            shift
            MACHINE_TYPE="$1"
            ;;
        --docker-image)
            shift
            DOCKER_IMAGE="$1"
            ;;
        --command)
            shift
            COMMAND="$1"
            ;;
        --user)
            shift
            USER="$1"
            ;;
        --ghcr-user)
            shift
            GHCR_USER="$1"
            ;;
        --delete)
            shift
            DELETE_VM="$1"
            ;;
        --beaker-token)
            shift
            BEAKER_TOKEN="$1"
            ;;
        --beaker-addr)
            shift
            BEAKER_ADDR="$1"
            ;;
        --beaker-username)
            shift
            BEAKER_USERNAME="$1"
            ;;
        --service-account)
            shift
            SERVICE_ACCOUNT="$1"
            ;;
        --rslp-project)
            shift
            RSLP_PROJECT="$1"
            ;;
        --workflow)
            shift
            WORKFLOW="$1"
            ;;
        --rslp-prefix)
            shift
            RSLP_PREFIX="$1"
            ;;
        --gpu-count)
            shift
            GPU_COUNT="$1"
            ;;
        --shared-memory)
            shift
            SHARED_MEMORY="$1"
            ;;
        --cluster)
            shift
            CLUSTER="$1"
            ;;
        --priority)
            shift
            PRIORITY="$1"
            ;;
        --task-name)
            shift
            TASK_NAME="$1"
            ;;
        --budget)
            shift
            BUDGET="$1"
            ;;
        --workspace)
            shift
            WORKSPACE="$1"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$DOCKER_IMAGE" ]; then
    echo "Error: --docker-image is required"
    usage
fi

if [ -z "$COMMAND" ]; then
    echo "Error: --command is required"
    usage
fi

if [ -z "$GHCR_PAT" ]; then
    echo "Error: GHCR_PAT environment variable must be set"
    exit 1
fi

# Generate VM name
VM_NAME="test-vm-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-4)"

# Rest of your existing create_vm function...
create_vm() {
    local vm_name="$1"
    local project_id="$2"
    local zone="$3"
    local machine_type="$4"
    local image_family="$5"
    local image_project="$6"
    local ghcr_pat="$7"
    local ghcr_user="$8"
    local user="$9"
    local docker_image="${10}"
    local command="${11}"
    local beaker_token="${12}"
    local beaker_addr="${13}"
    local beaker_username="${14}"
    local service_account="${15}"
    local rslp_project="${16}"
    local gpu_count="${17}"
    local shared_memory="${18}"
    local cluster="${19}"
    local priority="${20}"
    local task_name="${21}"
    local budget="${22}"
    local workspace="${23}"
    local rslp_prefix="${24}"
    local workflow="${25}"
    echo "Creating VM $vm_name in project $project_id..."
    gcloud compute instances create "$vm_name" \
        --project="$project_id" \
        --zone="$zone" \
        --machine-type="$machine_type" \
        --service-account="$service_account" \
        --scopes=cloud-platform \
        --metadata=ghcr-token="$ghcr_pat",ghcr-user="$ghcr_user",user="$user",docker-image="$docker_image",command="$command",beaker-token="$beaker_token",beaker-addr="$beaker_addr",beaker_username="$beaker_username",rslp-project="$rslp_project",gpu-count="$gpu_count",shared-memory="$shared_memory",cluster="$cluster",priority="$priority",task-name="$task_name",budget="$budget",workspace="$workspace",rslp-prefix="$rslp_prefix",workflow="$workflow" \
        --metadata-from-file=startup-script=<(echo '#!/bin/bash
        sudo apt-get update
        sudo apt-get install -y docker.io
        sudo systemctl start docker
        export USER=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/user)
        sudo usermod -aG docker $USER
        export GHCR_TOKEN=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/ghcr-token)
        export GHCR_USER=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/ghcr-user)
        export DOCKER_IMAGE=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker-image)
        export COMMAND=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/command)
        echo $GHCR_TOKEN | sudo docker login ghcr.io -u $GHCR_USER --password-stdin
        sudo docker pull $DOCKER_IMAGE
        echo "Docker image pulled" && \
        sudo docker run -e CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token) $DOCKER_IMAGE /bin/bash -c "$COMMAND" && \
        echo "Data Extraction Complete" && \
        export BEAKER_TOKEN=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/beaker-token) && \
        export BEAKER_ADDR=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/beaker-addr) && \
        curl -s 'https://beaker.org/api/v3/release/cli?os=linux&arch=amd64' | sudo tar -zxv -C /usr/local/bin ./beaker && \
        export IMAGE_ID=$(docker images --format "{{.ID}}" $DOCKER_IMAGE | head -n 1) && \
        export BEAKER_IMAGE_NAME=$(echo $DOCKER_IMAGE | tr '/' '_' | tr ':' '_' | tr -cd '[:alnum:]-') && \
        echo "Creating Beaker image" && \
        beaker image create $IMAGE_ID --name $BEAKER_IMAGE_NAME && \
        echo "Image uploaded to Beaker" && \
        export BEAKER_USERNAME=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/beaker_username) && \
        export GPU_COUNT=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/gpu-count) && \
        export SHARED_MEMORY=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/shared-memory) && \
        export CLUSTER=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/cluster) && \
        export PRIORITY=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/priority) && \
        export TASK_NAME=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/task-name) && \
        export BUDGET=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/budget) && \
        export WORKSPACE=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/workspace) && \
        export RSLP_PREFIX=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/rslp-prefix) && \
        export RSLP_PROJECT=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/rslp-project) && \
        export WORKFLOW=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/workflow) && \
        export INFERENCE_JOB_LAUNCH_COMMAND="python rslp/$RSLP_PROJECT/job_launcher.py \
            --project $RSLP_PROJECT \
            --workflow $WORKFLOW \
            --image $BEAKER_USERNAME/$BEAKER_IMAGE_NAME \
            --gpu_count $GPU_COUNT \
            --shared_memory $SHARED_MEMORY \
            --cluster $CLUSTER \
            --priority $PRIORITY \
            --task_name $TASK_NAME \
            --budget $BUDGET \
            --workspace $WORKSPACE" && \
        echo "Launching inference job on Beaker" && \
        docker run -e BEAKER_TOKEN=$BEAKER_TOKEN \
            -e BEAKER_ADDR=$BEAKER_ADDR \
            -e RSLP_PREFIX=$RSLP_PREFIX \
            $DOCKER_IMAGE /bin/bash -c "$INFERENCE_JOB_LAUNCH_COMMAND" && \
        echo "Model inference launched!" && \
        ') \
        --image-family="$image_family" \
        --image-project="$image_project" \
        --boot-disk-size=200GB

    echo "Done!"
}

# Echo all the variables being passed in (DEBUGGING ONLY)
echo "VM_NAME: $VM_NAME"
echo "PROJECT_ID: $PROJECT_ID"
echo "ZONE: $ZONE"
echo "MACHINE_TYPE: $MACHINE_TYPE"
echo "IMAGE_FAMILY: $IMAGE_FAMILY"
echo "IMAGE_PROJECT: $IMAGE_PROJECT"
echo "GHCR_PAT: $GHCR_PAT"
echo "GHCR_USER: $GHCR_USER"
echo "USER: $USER"
echo "DOCKER_IMAGE: $DOCKER_IMAGE"
echo "COMMAND: $COMMAND"
echo "BEAKER_TOKEN: $BEAKER_TOKEN"
echo "BEAKER_ADDR: $BEAKER_ADDR"
echo "SERVICE_ACCOUNT: $SERVICE_ACCOUNT"
echo "GPU_COUNT: $GPU_COUNT"
echo "SHARED_MEMORY: $SHARED_MEMORY"
echo "CLUSTER: $CLUSTER"
echo "PRIORITY: $PRIORITY"
echo "TASK_NAME: $TASK_NAME"
echo "BUDGET: $BUDGET"
echo "WORKSPACE: $WORKSPACE"
echo "RSLP_PREFIX: $RSLP_PREFIX"

# Create the VM
create_vm "$VM_NAME" "$PROJECT_ID" "$ZONE" "$MACHINE_TYPE" "$IMAGE_FAMILY" "$IMAGE_PROJECT" "$GHCR_PAT" "$GHCR_USER" "$USER" "$DOCKER_IMAGE" "$COMMAND" "$BEAKER_TOKEN" "$BEAKER_ADDR" "$BEAKER_USERNAME" "$SERVICE_ACCOUNT" "$RSLP_PROJECT" "$GPU_COUNT" "$SHARED_MEMORY" "$CLUSTER" "$PRIORITY" "$TASK_NAME" "$BUDGET" "$WORKSPACE" "$WORKFLOW"

# Handle VM deletion if requested
if [[ "$DELETE_VM" == "yes" ]]; then
    echo "Deleting VM $VM_NAME..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
else
    echo "VM $VM_NAME retained for further testing."
fi

echo "Done!"
