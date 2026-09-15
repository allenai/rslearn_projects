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
# DETECT_MODEL_CONFIG (data/nisar_vessels/config_satlas.yaml), since that is where
# rslearn looks for the weights under RSLP_PREFIX.
# TODO: no NISAR detector checkpoint has been published under
# gs://ai2-rslearn-projects-data/projects/nisar_vessels/ yet. Point this at the chosen
# run's best.ckpt (and update DETECT_MODEL_CONFIG if it is not the satlas run) before
# deploying this image.
RUN mkdir -p /project_data/projects/2026_08_28_nisar_vessels/data_20260828_satlas_02/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_08_28_nisar_vessels/data_20260828_satlas_02/best.ckpt \
        -O /project_data/projects/2026_08_28_nisar_vessels/data_20260828_satlas_02/best.ckpt
ENV RSLP_PREFIX=/project_data

# Run the API server when the container launches
CMD ["python3", "rslp/nisar_vessels/api_main.py"]
