# Landsat Vessel Detection API

The Landsat Vessel Detection API provides a way to apply the Landsat scenes for vessel detection. This guide explains how to set up and use the API, including running it locally or using prebuilt Docker images hosted on [GitHub Container Registry (GHCR)](https://github.com/allenai/rslearn_projects/pkgs/container/landsat-vessel-detection) and [Google Container Registry (GCR)](https://console.cloud.google.com/gcr/images/skylight-proto-1?referrer=search&inv=1&invt=Abh22Q&project=skylight-proto-1).


## Overview
- **Model Name**: Landsat Vessel Detection
- **Model Version**: `v0.0.1`
- **Tag**: `landsat_vessels_v0.0.1`
- **Last Updated**: `2024-11-19`


## Setting Up the Environment

First, create an `.env` file in the directory that you are running the API from, including the following environment variables:

```bash
# Required
LANDSAT_HOST=<host_address>
LANDSAT_PORT=<port_number>
RSLP_PREFIX=<rslp_prefix>
GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_key>
# Optional
AWS_ACCESS_KEY_ID=<aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
```

- `LANDSAT_HOST` and `LANDSAT_PORT` are required to configure the host and port for the Landsat service.
- `RSLP_PREFIX` is required to specify the prefix of the GCS bucket where model checkpoints are stored.
- `GOOGLE_APPLICATION_CREDENTIALS` is required for fetching model checkpoints from GCS bucket, also used for fetching downloaded Landsat scenes from GCS bucket. The service account key file should have the `storage.admin` role.
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are optional, and only required when `scene_id` is used to fetch Landsat scenes from AWS S3 bucket.


## Running the API server Locally

   ```python
   python rslp/landsat_vessels/api_main.py
   ```

## Using Docker Images for API Deployment

Prebuilt Docker images are available on both GHCR and GCR. Use the following steps to pull and run the image:

### GHCR image

1. Pull the image from GHCR.

    ```bash
    docker pull ghcr.io/allenai/landsat-vessel-detection:v0.0.1
    ```

2. Run the container. Note that you need to replace the `<port_number>` and `<path_to_service_account_key>` with the actual `LANDSAT_PORT` and path to your local service account key file, and keep the other arguments unchanged.

    ```bash
    docker run --gpus all \
    -e LANDSAT_PORT=<port_number> \
    --rm -p ${LANDSAT_PORT}:${LANDSAT_PORT} \
    -e GOOGLE_APPLICATION_CREDENTIALS=/path/in/container/key.json \
    -v <path_to_service_account_key>:/path/in/container/key.json \
    --env-file .env \
    ghcr.io/allenai/landsat-vessel-detection:v0.0.1
    ```

### GCR image

1. Pull the image from GCR.

    ```bash
    docker pull gcr.io/skylight-proto-1/landsat-vessel-detection:v0.0.1
    ```

2. Run the container.

    ```bash
    docker run --gpus all \
    -e GOOGLE_APPLICATION_CREDENTIALS=/path/in/container/key.json \
    -v /path/to/your/local/key.json:/path/in/container/key.json \
    --env-file .env \
    gcr.io/skylight-proto-1/landsat-vessel-detection:v0.0.1
    ```

## Making Requests to the API

Once the API server is running, you can send requests to the `/detections` endpoint to perform vessel detection. The API accepts several types of payloads, depending on the source of your Landsat scene:

1. Fetch Landsat Scene from AWS S3 Bucket:

    Provide the `scene_id` to retrieve the Landsat scene directly from the AWS S3 bucket.

    Payload Example:
    ```json
    {
        "scene_id": scene_id
    }
    ```

2. Fetch Zipped Landsat Scene from Local or GCS Storage:

    Provide the `scene_zip_path` to specify the path to a zipped Landsat scene stored locally or in a GCS bucket (for the Skylight team).

    Payload Example:
    ```json
    {
        "scene_zip_path": "gs://your_bucket/your_scene.zip"
    }
    ```

3. Fetch Unzipped Landsat Scene from Local or GCS Storage:

    Provide the image_files dictionary to specify paths to individual band files of the unzipped Landsat scene, either locally or in a GCS bucket.

    Payload Example:
    ```json
    {
        "image_files": {
            "B2": "path/to/B2.TIF",
            "B3": "path/to/B3.TIF",
            "B4": "path/to/B4.TIF",
            "B5": "path/to/B5.TIF",
            "B6": "path/to/B6.TIF",
            "B7": "path/to/B7.TIF",
            "B8": "path/to/B8.TIF"
        }
    }
    ```

You can send requests using `curl` or `requests` library.

Example:

```bash
curl -X POST http://${LANDSAT_HOST}:${LANDSAT_PORT}/detections -H "Content-Type: application/json" -d '{"scene_zip_path": "gs://test-bucket-rslearn/Landsat/LC08_L1TP_162042_20241103_20241103_02_RT.zip"}'
```

The API will respond with the vessel detection results in JSON format.


## Auto Documentation

This API has enabled Swagger UI and ReDoc.

You can access the Swagger UI at `http://<your_address>:<port_number>/docs` and ReDoc at `http://<your_address>:<port_number>/redoc` for a detailed documentation of the API. If you are running this API on VM, you will need to open the port to the public.
