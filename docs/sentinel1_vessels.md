Sentinel-1 Vessel Detection
---------------------------

The Sentinel-1 vessel detection model detects ships in Sentinel-1 IW GRDH scenes using
SAR imagery (VV and VH polarisations). It uses the target scene along with two
historical images (from approximately 60 and 90 days prior) to help distinguish vessels
from fixed infrastructure.


Inference
---------

First, download the model checkpoints to the `RSLP_PREFIX` directory.

    cd rslearn_projects
    mkdir -p project_data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/best.ckpt -O project_data/projects/sentinel1_vessels/data_20250521_model_20250530_satlaspretrain_unfreeze4_13/best.ckpt

    mkdir -p project_data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/best.ckpt -O project_data/projects/sentinel1_vessel_attribute/data_20260330_swinb_01/best.ckpt

The easiest way to apply the model is using the prediction pipeline in
`rslp/sentinel1_vessels/predict_pipeline.py`. It accepts a Sentinel-1 scene ID and
automatically downloads the scene images from AWS (via the Copernicus data source for
scene metadata lookup). Historical images with the best spatial overlap are
automatically selected.

    mkdir output_crops
    mkdir scratch_dir
    python -m rslp.main sentinel1_vessels predict --tasks '[{"scene_id": "S1C_IW_GRDH_1SDV_20250610T051814_20250610T051839_002716_00599D_6955.SAFE", "geojson_path": "out.geojson", "crop_path": "output_crops/"}]' --scratch_path scratch_dir/

Then, `out.geojson` will contain a GeoJSON of detected ships while `output_crops` will
contain corresponding VV and VH crops centered around those ships.

Alternatively, you can provide local GeoTIFF file paths for the target and historical
images instead of a scene ID. Each image requires VV and VH band files. The historical
images must have the same orbit direction as the target image.

    python -m rslp.main sentinel1_vessels predict --tasks '[{"image": {"vv": "/path/to/target_vv.tif", "vh": "/path/to/target_vh.tif"}, "historical1": {"vv": "/path/to/h1_vv.tif", "vh": "/path/to/h1_vh.tif"}, "historical2": {"vv": "/path/to/h2_vv.tif", "vh": "/path/to/h2_vh.tif"}, "geojson_path": "out.geojson"}]' --scratch_path scratch_dir/


Training
--------

The object detection model applies a SwinB backbone on each of three images (the target
scene and two historical scenes). The features from the target scene are concatenated
with the mean-pooled features across the historical scenes. The concatenated features
are then passed to a feature pyramid network and Faster R-CNN detection head. The
historical scenes help to differentiate between vessels and fixed infrastructure.

Use the command below to train the model:

    rslearn model fit --config data/sentinel1_vessels/config.yaml --data.init_args.path /path/to/sentinel1_vessels_dataset/

To visualize outputs on the validation set:

    mkdir vis
    rslearn model test --config data/sentinel1_vessels/config.yaml --data.init_args.path /path/to/sentinel1_vessels_dataset/ --model.init_args.visualize_dir vis/ --load_checkpoint_mode=best


Docker Container with FastAPI
-----------------------------

We also have a Docker container that exposes a FastAPI interface to apply vessel
detection on Sentinel-1 scenes. This section explains how to setup the API.

### Run the Docker container

The Docker container includes the model weights and marine infrastructure data. Run
the container:

```bash
export SENTINEL1_PORT=5555
docker run \
    --rm -p $SENTINEL1_PORT:$SENTINEL1_PORT \
    -e SENTINEL1_PORT=$SENTINEL1_PORT \
    --shm-size=15g \
    --gpus all \
    ghcr.io/allenai/sentinel1-vessel-detection:latest
```

### Auto Documentation

This API has enabled Swagger UI (`http://<your_address>:<port_number>/docs`) and ReDoc (`http://<your_address>:<port_number>/redoc`).

### Making Requests

Process a scene by its Sentinel-1 scene ID:

```bash
curl -X POST http://localhost:${SENTINEL1_PORT}/detections -H "Content-Type: application/json" -d '{"scene_id": "S1C_IW_GRDH_1SDV_20250610T051814_20250610T051839_002716_00599D_6955.SAFE"}'
```

The API will respond with the vessel detection results in JSON format.

Alternatively, process the scene by providing paths to VV/VH GeoTIFFs for the target
and two historical images. The paths must be accessible from inside the Docker
container.

```bash
curl -X POST http://localhost:${SENTINEL1_PORT}/detections -H "Content-Type: application/json" -d '{
    "image": {"vv": "/path/to/target_vv.tif", "vh": "/path/to/target_vh.tif"},
    "historical1": {"vv": "/path/to/h1_vv.tif", "vh": "/path/to/h1_vh.tif"},
    "historical2": {"vv": "/path/to/h2_vv.tif", "vh": "/path/to/h2_vh.tif"}
}'
```


Vessel Attribute Prediction
---------------------------

The vessel attribute prediction model predicts the vessel type, length, width, speed,
and heading of each detected vessel. It uses a separate model configured in
`data/sentinel1_vessel_attribute/config.yaml`. The predicted values are available under
the "attributes" key of the JSON or GeoJSON vessel object.

Attribute prediction and near-marine-infrastructure filtering (which removes detections
within 50 m of known marine infrastructure) are always applied during `predict_pipeline`,
both when using the CLI and the Docker API.
