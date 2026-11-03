# Base image
FROM base-image:latest

# Environment variables
ENV PYTHONPATH="/opt/rslearn_projects:${PYTHONPATH}"
ENV SENTINEL2_PORT=5555

# Make port 5555 available to the world outside this container
EXPOSE $SENTINEL2_PORT

# Download marine infrastructure GeoJSON for vessel detection filtering.
RUN wget -q -O /opt/rslearn_projects/marine_infrastructure.geojson \
    https://storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson
ENV MARINE_INFRA_PATH=/opt/rslearn_projects/marine_infrastructure.geojson

# Download detector and attribute checkpoints.
RUN mkdir -p /project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/ && \
    mkdir -p /project_data/projects/2026_09_17_vessel_attribute_l1_01/sentinel2_l1/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessels/data_20250213_02_all_bands/best.ckpt \
        -O /project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/best.ckpt && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_09_17_vessel_attribute_l1_01/sentinel2_l1/best.ckpt \
        -O /project_data/projects/2026_09_17_vessel_attribute_l1_01/sentinel2_l1/best.ckpt
ENV RSLP_PREFIX=/project_data

# Bake the ImageNet Swin-v2-B weights into the torch hub cache. The Swin encoder loads
# them when the model is built, before the checkpoint overwrites them, so the container
# does not download them at startup.
# The attribute model's SatlasPretrain encoder still downloads its weights from Hugging
# Face at startup, since satlaspretrain_models fetches them directly with no cache.
ENV TORCH_HOME=/opt/torch
RUN python3 -c "from torchvision.models import Swin_V2_B_Weights; Swin_V2_B_Weights.IMAGENET1K_V1.get_state_dict(progress=False)"

# Run app.py when the container launches
CMD ["python3", "rslp/sentinel2_vessels/api_main.py"]
