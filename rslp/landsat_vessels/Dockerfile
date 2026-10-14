# Base image
FROM base-image:latest

# Environment variables
ENV PYTHONPATH="/opt/rslearn_projects:${PYTHONPATH}"
ENV LANDSAT_PORT=5555

# Make port 5555 available to the world outside this container
EXPOSE $LANDSAT_PORT

# Download marine infrastructure GeoJSON for vessel detection filtering.
RUN wget -q -O /opt/rslearn_projects/marine_infrastructure.geojson \
    https://storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson
ENV MARINE_INFRA_PATH=/opt/rslearn_projects/marine_infrastructure.geojson

# Download detector, classifier, and attribute checkpoints.
RUN mkdir -p /project_data/projects/landsat_vessels/data_20240924_model_20240924_imagenet_patch512_flip_03/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/landsat_vessels/detector/best.ckpt \
        -O /project_data/projects/landsat_vessels/data_20240924_model_20240924_imagenet_patch512_flip_03/best.ckpt && \
    mkdir -p /project_data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_20260908d/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_20260908d/best.ckpt \
        -O /project_data/projects/landsat_vessel_classification_v2/olmoearth_base_layerdecay_20260908d/best.ckpt && \
    mkdir -p /project_data/projects/2026_09_17_vessel_attribute_l1/landsat_l1/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_09_17_vessel_attribute_l1/landsat_l1/best.ckpt \
        -O /project_data/projects/2026_09_17_vessel_attribute_l1/landsat_l1/best.ckpt
ENV RSLP_PREFIX=/project_data

# Bake the ImageNet Swin-v2-B weights into the torch hub cache. The Swin encoder loads
# them when the model is built, before the checkpoint overwrites them, so the container
# does not download them at startup.
ENV TORCH_HOME=/opt/torch
RUN python3 -c "from torchvision.models import Swin_V2_B_Weights; Swin_V2_B_Weights.IMAGENET1K_V1.get_state_dict(progress=False)"

# Bake the pretrained OlmoEarth configs and weights into the Hugging Face cache: v1.2
# Base for the classifier and v1 Base for the attribute model. Each encoder loads its
# model by model_id when it is built, before the checkpoint overwrites the weights, so
# with HF_HUB_OFFLINE set the container never reaches Hugging Face at startup.
ENV HF_HOME=/opt/huggingface
RUN python3 -c "from huggingface_hub import hf_hub_download; from olmoearth_pretrain_minimal import ModelID; [hf_hub_download(repo_id=m.repo_id(), filename=f) for m in (ModelID.OLMOEARTH_V1_2_BASE, ModelID.OLMOEARTH_V1_BASE) for f in ('config.json', 'weights.pth')]"
ENV HF_HUB_OFFLINE=1

# Run app.py when the container launches
CMD ["python3", "rslp/landsat_vessels/api_main.py"]
