# Base image
FROM base-image:latest

# Environment variables
ENV PYTHONPATH="/opt/rslearn_projects:${PYTHONPATH}"
ENV NISAR_PORT=5555

# Make port 5555 available to the world outside this container
EXPOSE $NISAR_PORT

# Download marine infrastructure GeoJSON for vessel detection filtering.
RUN wget -q -O /opt/rslearn_projects/marine_infrastructure.geojson \
    https://storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson
ENV MARINE_INFRA_PATH=/opt/rslearn_projects/marine_infrastructure.geojson

# Download the detector checkpoint. The path must match project_name/run_name in
# DETECT_MODEL_CONFIG (data/nisar_vessels/config_olmoearth_tiny.yaml), since that is
# where rslearn looks for the weights under RSLP_PREFIX.
RUN mkdir -p /project_data/projects/2026_10_12_nisar_vessels/data_20261013_olmoearth_tiny_00/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_10_12_nisar_vessels/data_20261013_olmoearth_tiny_00/best.ckpt \
        -O /project_data/projects/2026_10_12_nisar_vessels/data_20261013_olmoearth_tiny_00/best.ckpt
ENV RSLP_PREFIX=/project_data

# Bake the pretrained OlmoEarth v1.2-Tiny config and weights into the Hugging Face cache.
# The encoder loads them by model_id when the detector is built, before the checkpoint
# overwrites the weights, so with HF_HUB_OFFLINE set the container never reaches Hugging
# Face at startup.
ENV HF_HOME=/opt/huggingface
RUN python3 -c "from huggingface_hub import hf_hub_download; from olmoearth_pretrain_minimal import ModelID; [hf_hub_download(repo_id=ModelID.OLMOEARTH_V1_2_TINY.repo_id(), filename=f) for f in ('config.json', 'weights.pth')]"
ENV HF_HUB_OFFLINE=1

# Run the API server when the container launches
CMD ["python3", "rslp/nisar_vessels/api_main.py"]
