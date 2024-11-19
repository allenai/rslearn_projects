# Landsat Vessel Detection API

The Landsat Vessel Detection API provides a way to apply the Landsat scenes for vessel detection. This guide explains how to set up and use the API, including running it locally or using prebuilt Docker images hosted on [GitHub Container Registry (GHCR)](https://github.com/allenai/rslearn_projects/pkgs/container/landsat-vessel-detection) and [Google Container Registry (GCR)](https://console.cloud.google.com/gcr/images/skylight-proto-1?referrer=search&inv=1&invt=Abh22Q&project=skylight-proto-1).

---

## Setting Up the Environment

Define the following environment variables to configure the host and port for the Landsat service:

   ```bash
   export LANDSAT_HOST=<host_address>
   export LANDSAT_PORT=<port_number>
   ```

## Running the API server Locally

   ```python
   python rslp/landsat_vessels/api_main.py
   ```

## Using Docker Images for API Deployment

A prebuilt Docker image is available on both GHCR and GCR. Use the following steps to pull and run the image:

### GHCR image

1. Pull the image from GHCR.

    ```bash
    docker pull ghcr.io/allenai/rslearn_projects/landsat-vessel-detection:latest
    ```
2. Run the container.

    ```bash
    docker run --rm -p <local_port>:<container_port> \
    -e LANDSAT_HOST=<host_address> \
    -e LANDSAT_PORT=<port_number> \
    ghcr.io/allenai/rslearn_projects/landsat-vessel-detection:latest
    ```

### GCR image

1. Pull the image from GCR.

    ```bash
    docker pull gcr.io/skylight-proto-1/landsat-vessel-detection:latest
    ```

2. Run the container.

    ```bash
    docker run --rm -p <local_port>:<container_port> \
    -e LANDSAT_HOST=<host_address> \
    -e LANDSAT_PORT=<port_number> \
    gcr.io/skylight-proto-1/landsat-vessel-detection:latest
    ```

## Making Requests to the API

Once the API server is running, you can send requests to the `/detections` endpoint. The request payload should include the path to the Landsat scene ZIP file in a GCS bucket.

For example, use the following sample payload in a POST request:

```json
{
    "scene_zip_path": "gs://your_bucket/your_scene.zip"
}
```

The response will contain the vessel detection results in JSON format.

Sample request can be found in `rslp/landsat_vessels/scripts/sample_request.py`

<!-- TODO: enable autodoc for the API server.

TODO: add one more integration test.  -->
