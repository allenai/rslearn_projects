# Landsat Vessel Detection API

The Landsat Vessel Detection API provides a way to apply the Landsat scenes for vessel detection. This guide explains how to set up and use the API, including running it locally or using prebuilt Docker images hosted on [GitHub Container Registry (GHCR)](https://github.com/allenai/rslearn_projects/pkgs/container/landsat-vessel-detection) and [Google Container Registry (GCR)](https://console.cloud.google.com/gcr/images/skylight-proto-1?referrer=search&inv=1&invt=Abh22Q&project=skylight-proto-1).


## Overview
- **Model Name**: Landsat Vessel Detection
- **Model Version**: `v0.0.1`
- **Tag**: `landsat_vessels_v0.0.1`
- **Last Updated**: `2024-11-21`


## Setting Up the Environment

First, create an `.env` file in the directory that you are running the API or Docker container from, including the following environment variables:

```bash
# Required
RSLP_PREFIX=<rslp_prefix>
GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_key>

# Optional (with default values)
LANDSAT_HOST=<host_address>
LANDSAT_PORT=<port_number>

# Optional (only if you are fetching Landsat scenes from AWS S3 bucket)
AWS_ACCESS_KEY_ID=<aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
```

- `RSLP_PREFIX` is required to specify the prefix of the GCS bucket where model checkpoints are stored.
- `LANDSAT_HOST` and `LANDSAT_PORT` are optional, and used to configure the host and port for the Landsat service. The default values are `0.0.0.0` and `5555`.
- `GOOGLE_APPLICATION_CREDENTIALS` is required for fetching model checkpoints from GCS bucket, also used for fetching downloaded Landsat scenes from GCS bucket. The service account key file should have the `storage.admin` role.
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are optional, and only required when `scene_id` is used to fetch Landsat scenes from AWS S3 bucket.


## Running the API server Locally

   ```python
   python rslp/landsat_vessels/api_main.py
   ```

This will start the API server on the specified host and port, and will rewrite the environment variables in the `.env` file.

## Using Docker Images for API Deployment

Prebuilt Docker images are available on both GHCR and GCR. Use the following steps to pull and run the image (make sure the `.env` file is in the same directory as the Dockerfile, and at least 15GB of shared memory is available):

### GHCR image

1. Pull the image from GHCR.

    ```bash
    docker pull ghcr.io/allenai/landsat-vessel-detection:v0.0.1
    ```

2. Run the container. Note that you need to replace the `<port_number>` and `<path_to_service_account_key>` with the actual `LANDSAT_PORT` (if you use the default port, set it to `5555`) and path to your local service account key file, and keep the other arguments unchanged.

    ```bash
    docker run \
    --rm -p <port_number>:<port_number> \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json \
    -v <path_to_service_account_key>:/app/credentials/key.json \
    --env-file .env \
    --shm-size=15g \
    --gpus all \
    ghcr.io/allenai/landsat-vessel-detection:v0.0.1
    ```

### GCR image

1. Pull the image from GCR.

    ```bash
    docker pull gcr.io/skylight-proto-1/landsat-vessel-detection:v0.0.1
    ```

2. Run the container. Note that you need to replace the `<port_number>` and `<path_to_service_account_key>` with the actual `LANDSAT_PORT` (if you use the default port, set it to `5555`) and path to your local service account key file, and keep the other arguments unchanged.

    ```bash
    docker run \
    --rm -p <port_number>:<port_number> \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json \
    -v <path_to_service_account_key>:/app/credentials/key.json \
    --env-file .env \
    --shm-size=15g \
    --gpus all \
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

Example with `curl`:

```bash
curl -X POST http://${LANDSAT_HOST}:${LANDSAT_PORT}/detections -H "Content-Type: application/json" -d '{"scene_zip_path": "gs://test-bucket-rslearn/Landsat/LC08_L1TP_162042_20241103_20241103_02_RT.zip"}'
```

The API will respond with the vessel detection results in JSON format.

Note that the above example uses a test zip file, which is a cropped Landsat scene, not a full scene. To run the API on a full scene, you can use the command below:

```bash
curl -X POST http://${LANDSAT_HOST}:${LANDSAT_PORT}/detections -H "Content-Type: application/json" -d '{"scene_id": "LC09_L1GT_106084_20241002_20241002_02_T2"}'
```


## Auto Documentation

This API has enabled Swagger UI and ReDoc.

You can access the Swagger UI at `http://<your_address>:<port_number>/docs` and ReDoc at `http://<your_address>:<port_number>/redoc` for a detailed documentation of the API. If you are running this API on VM, the `<your_address>` should be the public IP address of the VM, and you also need to open the `<port_number>` to the public.
