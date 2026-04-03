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
    mkdir -p /project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessels/data_20250213_02_all_bands/checkpoints/best.ckpt \
        -O /project_data/projects/sentinel2_vessels/data_20250213_02_all_bands/best.ckpt && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/checkpoints/best.ckpt \
        -O /project_data/projects/sentinel2_vessel_attribute/data_20250205_regress_00/best.ckpt
ENV RSLP_PREFIX=/project_data

# Run app.py when the container launches
CMD ["python3", "rslp/sentinel2_vessels/api_main.py"]
