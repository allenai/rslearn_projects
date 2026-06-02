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
    mkdir -p /project_data/projects/landsat_vessel_classification_v2/olmoearth_base_v1.1_final_20260601/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/landsat_vessel_classification_v2/olmoearth_base_v1.1_final_20260601/best.ckpt \
        -O /project_data/projects/landsat_vessel_classification_v2/olmoearth_base_v1.1_final_20260601/best.ckpt && \
    mkdir -p /project_data/projects/2026_04_22_landsat_vessel_attribute/olmoearth_base_freeze_newdata_01/ && \
    wget -q https://storage.googleapis.com/ai2-rslearn-projects-data/projects/2026_04_22_landsat_vessel_attribute/olmoearth_base_freeze_newdata_01/best.ckpt \
        -O /project_data/projects/2026_04_22_landsat_vessel_attribute/olmoearth_base_freeze_newdata_01/best.ckpt
ENV RSLP_PREFIX=/project_data

# Run app.py when the container launches
CMD ["python3", "rslp/landsat_vessels/api_main.py"]
