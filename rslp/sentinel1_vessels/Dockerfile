# Base image
FROM base-image:latest

# Environment variables
ENV PYTHONPATH="/opt/rslearn_projects:${PYTHONPATH}"
ENV SENTINEL1_PORT=5555

# Make port 5555 available to the world outside this container
EXPOSE $SENTINEL1_PORT

# Download marine infrastructure GeoJSON for vessel detection filtering.
RUN wget -q -O /opt/rslearn_projects/marine_infrastructure.geojson \
    https://storage.googleapis.com/satlas-explorer-public/outputs/marine/latest.geojson
ENV MARINE_INFRA_PATH=/opt/rslearn_projects/marine_infrastructure.geojson

# Download detector and attribute checkpoints.
RUN mkdir -p /project_data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/best.ckpt \
        -O /project_data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/best.ckpt && \
    mkdir -p /project_data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/best.ckpt \
        -O /project_data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/best.ckpt
ENV RSLP_PREFIX=/project_data

# Run app.py when the container launches
CMD ["python3", "rslp/sentinel1_vessels/api_main.py"]
