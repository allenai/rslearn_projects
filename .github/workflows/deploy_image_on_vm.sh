#!/bin/bash

# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --project-id      GCP project ID (default: skylight-proto-1)"
    echo "  --zone            GCP zone (default: us-west1-b)"
    echo "  --machine-type    VM machine type (default: e2-micro)"
    echo "  --docker-image    Docker image to run"
    echo "  --command         Command to run in container"
    echo "  --user            User (default: henryh)"
    echo "  --ghcr-user       GitHub Container Registry user (default: allenai)"
    echo "  --delete          Delete VM after completion (yes/no)"
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

    echo "Creating VM $vm_name in project $project_id..."
    gcloud compute instances create "$vm_name" \
        --project="$project_id" \
        --zone="$zone" \
        --machine-type="$machine_type" \
        --metadata=ghcr-token="$ghcr_pat",ghcr-user="$ghcr_user",user="$user",docker-image="$docker_image",command="$command" \
        --metadata-from-file=startup-script=<(echo '#! /bin/bash
        sudo apt-get update
        sudo apt-get install -y docker.io
        sudo systemctl start docker
        export USER=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/user)
        sudo usermod -aG docker $USER && \
        export GHCR_TOKEN=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/ghcr-token) && \
        export GHCR_USER=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/ghcr-user) && \
        export DOCKER_IMAGE=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker-image) && \
        export COMMAND=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/command | base64 --decode) && \
        echo $GHCR_TOKEN | sudo docker login ghcr.io -u $GHCR_USER --password-stdin && \
        sudo docker pull $DOCKER_IMAGE && \
        echo "Docker image pulled" && \
        sudo docker run -d $DOCKER_IMAGE /bin/bash -c "$COMMAND" && \
        echo "Docker container Pulled and Running"
        ') \
        --image-family="$image_family" \
        --image-project="$image_project" \
        --boot-disk-size=200GB \

    echo "Done!"
}

# Echo all the variables being passed in
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

# Create the VM
create_vm "$VM_NAME" "$PROJECT_ID" "$ZONE" "$MACHINE_TYPE" "$IMAGE_FAMILY" "$IMAGE_PROJECT" "$GHCR_PAT" "$GHCR_USER" "$USER" "$DOCKER_IMAGE" "$COMMAND"

# Handle VM deletion if requested
if [[ "$DELETE_VM" == "yes" ]]; then
    echo "Deleting VM $VM_NAME..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
else
    echo "VM $VM_NAME retained for further testing."
fi
